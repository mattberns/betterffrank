"""Third-party ranking hints on the draft board: Justin Boone's overall ranks
and Establish The Run's take/avoid list.

Two facts about these two files shape everything here.

**They are hand-keyed name lists, not feeds.** ETR writes "Ceedee Lamb",
"Travis Ettiene" and "Jeremiah Love"; Boone writes "JAC" for the Jacksonville
defense where the board says "Jacksonville Jaguars DST". So the ids are
resolved ONCE, by `resolve`, and written back into the JSON beside each name.
The site then joins on the id. A rename on either side after that shows up as
an unresolved row in `report` — a number that has to change — instead of a
hint that quietly stops appearing on the board.

**They are presentation only.** Nothing here reaches a feature, the choice
model, the VORP curve or the simulation; the columns are attached in
`cli.cmd_site`, downstream of `train`, for the sole purpose of being rendered.
Same rule the parent repo applies to VONA. A hint list is one analyst's
opinion with no walk-forward evidence behind it, and there is no honest way to
tune on it: ETR's 35 rows are 2026-only and Boone's ranks have no history in
this repo at all.

Identity written into the files, per row:

  ``id``   the board's ``gsis_id`` — a real nflverse id (``00-0039139``) or the
           FantasyPros placeholder its rookies carry (``LOV121782``). ``null``
           where the board has none, which on this board means every K and
           every D/ST.
  ``key``  the board's own canonical name key, so the K and D/ST rows still
           have an exact join key. Copied verbatim FROM the board rather than
           recomputed, which is what makes the join exact rather than a second
           name match at render time.

Both are ``null`` on a row that is genuinely not on the board — Boone ranks 303
players and the FantasyPros board is 350 deep, but the two lists disagree about
who those are, and 34 of Boone's names (all at his rank 208 or worse) are
nowhere on it. A written-out ``null`` records "checked, absent" and keeps
`resolve` idempotent; a missing field means the file was never resolved, and
`check` refuses that.

THE SEASON IS PINNED HERE. Neither file carries one, and both are current-board
artifacts, so `HINT_SEASON` says which board they describe and `attach` no-ops
on any other. Same reasoning as `config.ECR_SOURCE_BY_SEASON`: a file whose
contents change series or season must not silently reroute a board.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .adp_match import canon
from .config import RAW

BOONE_JSON = RAW / "boone.json"
ETR_JSON = RAW / "etr.json"

# Which board these two files describe. See the module docstring.
HINT_SEASON = 2026

# Hint-side spelling -> the board's canonical key. A FIFTH `norm_name` consumer
# and deliberately its own table: `adp_match.ALIASES` maps real league-wide
# naming differences that both the ADP boards and ESPN's draft history need,
# and folding one analyst's typos into it would apply them to the model's own
# joins. Suffixes, punctuation and accents are already handled by `norm_name`;
# only genuine misspellings belong here.
ALIASES = {
    "jeremiah love": "jeremiyah love",     # ETR; board says Jeremiyah
    "travis ettiene": "travis etienne",    # ETR
}

# Boone's D/ST rows are keyed by team abbreviation, and he uses a different one
# than `board.TEAM_BY_CITY_NICK` for two franchises. Skill rows never match on
# team, so this is only ever consulted for a defense.
TEAM_ALIASES = {"JAC": "JAX", "WAS": "WSH"}

BOONE_COLS = ("boone_rank",)
ETR_COLS = ("etr_take", "etr_round")
HINT_COLS = BOONE_COLS + ETR_COLS


# --------------------------------------------------------------- resolution ---

class _Index:
    """The board, keyed the three ways a hint row can name a player.

    There is deliberately NO initial-and-surname pass, which `adp_match` has
    and `board.link_picks` gets away with. It was tried: against this board its
    single extra match was Boone's "Trevor Etienne" (CAR, not on the board)
    landing on Travis Etienne Jr. (NO), who was already matched from his own
    row. For a hint column a wrong player is worse than a blank one, and a
    303-name list against a 350-name board does not need the reach.
    """

    def __init__(self, board: pl.DataFrame):
        self.by_key_pos: dict[tuple[str, str], dict] = {}
        self.by_key: dict[str, list[dict]] = {}
        self.by_dst: dict[str, dict] = {}
        for r in board.iter_rows(named=True):
            key = r["key"]
            if key is None:
                continue
            self.by_key_pos[(key, r["position"])] = r
            self.by_key.setdefault(key, []).append(r)
            if r["position"] == "DST" and r["pro_team"]:
                self.by_dst[r["pro_team"]] = r

    def find(self, name: str, position: str | None, team: str | None):
        """(board row, how). `how` is 'dst' / 'name' / 'name-nopos' /
        'ambiguous' / 'none' and is reported, never discarded."""
        if position == "DST":
            abbr = TEAM_ALIASES.get(team or "", team)
            hit = self.by_dst.get(abbr)
            return (hit, "dst") if hit else (None, "none")
        key = canon(name)
        key = ALIASES.get(key, key)
        if key is None:
            return None, "none"
        if position is not None:
            hit = self.by_key_pos.get((key, position))
            if hit:
                return hit, "name"
        cands = self.by_key.get(key) or []
        if len(cands) == 1:
            # ETR gives no position, so this is its ONLY pass. Ambiguity is
            # returned rather than resolved by guessing.
            return cands[0], "name" if position is not None else "name-nopos"
        if len(cands) > 1:
            return None, "ambiguous"
        return None, "none"


def _load_json(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path.name}: expected a JSON array of records")
    return rows


def _write_json(path: Path, rows: list[dict]) -> None:
    """One record per line, the shape these files already have, so a
    re-resolution diffs row by row instead of as one reflowed blob."""
    body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(f"[\n{body}\n]\n")


def _name_of(row: dict) -> str:
    """Boone calls it `player`, ETR calls it `name`. Neither file is ours."""
    for field in ("player", "name"):
        if field in row:
            return row[field]
    raise ValueError(f"no name field in hint row: {row}")


def resolve(path: Path, board: pl.DataFrame, write: bool = True) -> dict:
    """Write `id` and `key` into every row of a hint file, in place.

    Idempotent: the fields are recomputed from the board and overwritten, so
    running this against an already-resolved file rewrites it byte-identically
    unless the board moved.
    """
    rows = _load_json(path)
    idx = _Index(board)
    hows: dict[str, int] = {}
    claimed: dict[int, str] = {}
    collisions = []
    for row in rows:
        name = _name_of(row)
        hit, how = idx.find(name, row.get("position"), row.get("team"))
        hows[how] = hows.get(how, 0) + 1
        row["id"] = None if hit is None else hit["gsis_id"]
        row["key"] = None if hit is None else hit["key"]
        if hit is not None:
            pid = int(hit["pid"])
            if pid in claimed:
                collisions.append((claimed[pid], name, hit["name"]))
            claimed[pid] = name
    if collisions:
        raise ValueError(
            f"{path.name}: two hint rows resolved to one board row — "
            + "; ".join(f"{a!r} and {b!r} both -> {c!r}" for a, b, c in collisions)
        )
    if write:
        _write_json(path, rows)
    return {
        "path": path,
        "rows": len(rows),
        "matched": len(claimed),
        "how": dict(sorted(hows.items())),
        "unmatched": [
            {"name": _name_of(r), "position": r.get("position"),
             "team": r.get("team"), "rank": r.get("rank"), "round": r.get("round")}
            for r in rows if r["id"] is None and r["key"] is None
        ],
    }


def resolve_all(board: pl.DataFrame, boone_path: Path = BOONE_JSON,
                etr_path: Path = ETR_JSON, write: bool = True) -> list[dict]:
    return [resolve(p, board, write=write) for p in (boone_path, etr_path)]


# ------------------------------------------------------------------- joining ---

def load(boone_path: Path = BOONE_JSON, etr_path: Path = ETR_JSON) -> pl.DataFrame:
    """Both files as ONE frame keyed (hint_id, hint_key), the shape `attach`
    left-joins. A player can appear in both lists; the join is on the board
    row, so he gets one row here with both sets of columns filled."""
    boone = _load_json(boone_path)
    etr = _load_json(etr_path)
    for label, rows in (("boone.json", boone), ("etr.json", etr)):
        bad = [i for i, r in enumerate(rows) if "id" not in r or "key" not in r]
        if bad:
            raise ValueError(
                f"{label}: {len(bad)} row(s) carry no `id`/`key` — resolve the "
                f"file first with `uv run python -m tendies hints`"
            )

    def frame(rows, cols) -> pl.DataFrame:
        keep = [r for r in rows if not (r["id"] is None and r["key"] is None)]
        return pl.DataFrame(
            {
                "hint_id": [r["id"] for r in keep],
                "hint_key": [r["key"] for r in keep],
                **{name: [get(r) for r in keep] for name, (get, _) in cols.items()},
            },
            schema={"hint_id": pl.Utf8, "hint_key": pl.Utf8,
                    **{name: dt for name, (_, dt) in cols.items()}},
        )

    b = frame(boone, {"boone_rank": (lambda r: r["rank"], pl.Int32)})
    e = frame(etr, {"etr_take": (lambda r: r["take"], pl.Utf8),
                    "etr_round": (lambda r: r["round"], pl.Int32)})
    for label, df in (("boone.json", b), ("etr.json", e)):
        dup = df.filter(pl.col("hint_key").is_duplicated())
        if dup.height:
            raise ValueError(f"{label}: duplicate hint keys {dup['hint_key'].to_list()}")
    return b.join(e, on=["hint_id", "hint_key"], how="full", coalesce=True)


def attach(board: pl.DataFrame, boone_path: Path = BOONE_JSON,
           etr_path: Path = ETR_JSON, season: int | None = None,
           verbose: bool = True) -> pl.DataFrame:
    """Left-join the hint columns onto one season's board.

    LEFT, and asserted to stay left: the board is the population and a hint is
    an optional annotation on it, so the row count cannot move and no row is
    dropped for lacking a hint. Joined on `gsis_id` first and on the board's
    `key` second — the same ladder `board.link_picks` uses, and the `key` rung
    is what carries K and D/ST, which have no gsis id on this board.
    """
    for col in HINT_COLS:
        if col in board.columns:
            raise ValueError(f"board already carries `{col}`; attach ran twice")
    seasons = board["season"].unique().to_list()
    if len(seasons) != 1:
        raise ValueError(f"attach takes one season's board, got {sorted(seasons)}")
    season = int(season if season is not None else seasons[0])
    blank = board.with_columns(
        boone_rank=pl.lit(None, dtype=pl.Int32),
        etr_take=pl.lit(None, dtype=pl.Utf8),
        etr_round=pl.lit(None, dtype=pl.Int32),
    )
    if season != HINT_SEASON:
        if verbose:
            print(f"  hints: none for {season} (boone/etr describe the "
                  f"{HINT_SEASON} board) — columns attached empty")
        return blank

    h = load(boone_path, etr_path)
    by_id = h.filter(pl.col("hint_id").is_not_null()).select(
        pl.col("hint_id").alias("gsis_id"), *HINT_COLS
    )
    by_key = h.filter(pl.col("hint_id").is_null()).select(
        pl.col("hint_key").alias("key"), *HINT_COLS
    )
    out = (
        blank.drop(HINT_COLS)
        .join(by_id, on="gsis_id", how="left")
        .join(by_key, on="key", how="left", suffix="_k")
        .with_columns(
            **{c: pl.coalesce(c, f"{c}_k") for c in HINT_COLS}
        )
        .drop([f"{c}_k" for c in HINT_COLS])
    )
    if out.height != board.height:
        raise ValueError(
            f"hint join changed the board height {board.height} -> {out.height}; "
            "a hint id or key matches more than one board row"
        )
    if verbose:
        nb = out["boone_rank"].is_not_null().sum()
        ne = out["etr_take"].is_not_null().sum()
        print(f"  hints: Boone rank on {nb}/{out.height} board rows · "
              f"ETR take/avoid on {ne}")
    return out.select(*board.columns, *HINT_COLS)


# ------------------------------------------------------------------ reporting ---

def report(board: pl.DataFrame, boone_path: Path = BOONE_JSON,
           etr_path: Path = ETR_JSON, write: bool = True) -> None:
    """Resolve both files and print what matched, what did not, and why.

    The unmatched list is printed in full on purpose. It is the only place a
    misspelling on either side becomes visible, and it is short.
    """
    for stats in resolve_all(board, boone_path, etr_path, write=write):
        path = stats["path"]
        print(f"\n{path.name}: {stats['matched']}/{stats['rows']} matched "
              f"to the {HINT_SEASON} board · {stats['how']}")
        for u in stats["unmatched"]:
            where = f"rank {u['rank']}" if u["rank"] is not None else f"round {u['round']}"
            pos = f" {u['position']}" if u["position"] else ""
            team = f" {u['team']}" if u["team"] else ""
            print(f"    not on the board: {u['name']}{pos}{team} ({where})")
        if write:
            print(f"    wrote {path}")
