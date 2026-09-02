"""The per-season player universe: every player who could be drafted.

Three things make this more than a parquet read.

**K and D/ST are restored.** They are 13.2% of this league's picks and they
consume real roster slots, so an availability simulation without them is wrong.
The parent repo's `fp_adp_half.parquet` drops them at a `KEEP_POS` filter, but
they survive in the raw capture (`data/raw/fp_adp/half_<season>.json`, ~23 K and
~30 D/ST per season), so this module parses the raw board directly and keeps all
six positions.

**`pid` is the board's identity and `pid == adp_rank - 1` always.** The whole
board is one ADP-sorted array in Python and again in JavaScript; availability is
a byte per pid. Every downstream structure — choice sets, the taken mask, the
simulation — indexes on it. Sorting is `(adp, name)` so the tiebreak is stable
rather than row-order dependent.

**Picks are linked to board rows here, once.** ESPN and FantasyPros name players
differently, and D/ST worst of all: ESPN says "Ravens D/ST" (or `BAL` via the
negative-id fallback), FantasyPros says "Baltimore Ravens DST". Both sides are
resolved to the current NFL team abbreviation, which works because both sources
restate historical rows under current team identity (the same reason Robby
Anderson appears as "Robbie Chosen" on every board).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

from . import attrs, rookies
from .adp_match import canon
from .espn import PRO_TEAMS

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
SKILL = ("QB", "RB", "WR", "TE")

# Real-time ADP cell: "96 -47" (rank then move vs AVG), "1" (unchanged), or an
# em dash where the recent-draft sample has never taken the player. Only the
# leading rank is used; the move is the site's own presentation.
_REALTIME_RE = re.compile(r"^(\d+)")

# ESPN writes "D/ST"; FantasyPros writes "DST". Canonical form here is "DST".
ESPN_POS = {"D/ST": "DST", "DEF": "DST"}

# "Baltimore Ravens DST" -> BAL. Built once; both sources use current names.
TEAM_BY_CITY_NICK = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LAR", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WSH",
}
# ESPN's D/ST display name is the nickname alone ("Ravens D/ST").
TEAM_BY_NICK = {city.split()[-1]: abbr for city, abbr in TEAM_BY_CITY_NICK.items()}
TEAM_BY_NICK["49ers"] = "SF"


def _pos_of(cell: str) -> tuple[str, int | None]:
    """'DST4' -> ('DST', 4); 'WR12' -> ('WR', 12); 'QB' -> ('QB', None)."""
    head = "".join(c for c in cell if c.isalpha())
    tail = "".join(c for c in cell if c.isdigit())
    return ESPN_POS.get(head, head), (int(tail) if tail else None)


def dst_team(name: str) -> str | None:
    """Team abbreviation for a defense, from either source's naming."""
    base = name.replace(" DST", "").replace(" D/ST", "").strip()
    hit = TEAM_BY_CITY_NICK.get(base.lower())
    if hit:
        return hit
    return TEAM_BY_NICK.get(base.split()[-1] if base else "")


def parse_raw(path: Path, realtime: bool = False) -> pl.DataFrame:
    """Raw FantasyPros capture -> every row, ALL SIX positions.

    Mirrors the parent's `bff/fp_adp.build_table` row parse without its
    `KEEP_POS` filter. The AVG column is located through the header list
    because 2025-2026 captures carry an extra trailing column — that extra
    column is the REAL-TIME ADP, which `realtime` reads instead of AVG (see
    config.REALTIME_ADP_SEASONS for which seasons may use it and why).

    A player the recent-draft sample has never taken keeps his AVG rather than
    dropping off the board, and `adp_col` records which column each row came
    from. Both scales are pick numbers, so the two interleave sanely; the
    fallbacks are deep by construction (2026: 72 of 355, none above AVG 178).
    """
    payload = json.loads(path.read_text())
    headers = payload["headers"]
    avg_i = headers.index("AVG")
    classes = payload.get("classes") or []
    rt_i = classes.index("realtime") if realtime and "realtime" in classes else None
    if realtime and rt_i is None:
        raise ValueError(
            f"{path.name}: real-time ADP asked for, but the capture has no "
            f"`realtime` column (classes: {classes}). Re-fetch it in the parent "
            f"repo: uv run python -m bff.fp_adp --fetch --format half "
            f"--seasons <season> --refresh"
        )
    rows = []
    for p in payload["players"]:
        cells = p["cells"]
        if len(cells) <= avg_i:
            continue
        try:
            adp = float(cells[avg_i])
        except (TypeError, ValueError):
            continue
        adp_col = "avg"
        if rt_i is not None and rt_i < len(cells):
            m_rt = _REALTIME_RE.match(cells[rt_i].strip())
            if m_rt:
                adp, adp_col = float(m_rt.group(1)), "realtime"
        position, pos_rank = _pos_of(cells[2])
        if position not in POSITIONS:
            continue
        team_bye = p.get("team_bye") or ""
        bye = None
        if "(" in team_bye:
            digits = team_bye.split("(")[-1].rstrip(")").strip()
            bye = int(digits) if digits.isdigit() else None
        team = team_bye.split("(")[0].strip() or None
        name = p["name"]
        if position == "DST":
            team = dst_team(name) or team
        rows.append(
            {
                "name": name,
                "position": position,
                "pro_team": team,
                "bye": bye,
                "adp": adp,
                "adp_col": adp_col,
                "fp_pos_rank": pos_rank,
            }
        )
    return pl.DataFrame(rows)


def build(
    season: int,
    raw_dir: Path,
    skill_board: pl.DataFrame | None = None,
    ecr: pl.DataFrame | None = None,
    rookie_maps: tuple[pl.DataFrame, pl.DataFrame] | None = None,
    birth_maps: tuple[pl.DataFrame, pl.DataFrame] | None = None,
    actuals_path: Path | None = None,
    realtime: bool = False,
) -> pl.DataFrame:
    """The season board, one ADP-sorted array with `pid == adp_rank - 1`.

    `skill_board` (the parent's curated `fp_adp_half.parquet`) supplies gsis ids
    for QB/RB/WR/TE; it is joined on, never used to filter, so the raw capture
    stays the single source of who is on the board.

    `rookie_maps` is `rookies.entry_year_map(...)` and `birth_maps` is
    `attrs.birth_date_map(...)`. Absent, the columns they feed are still
    written, at their neutral value, so the schema never depends on what was
    passed in and a downstream feature is inert rather than missing.
    """
    raw = parse_raw(raw_dir / f"half_{season}.json", realtime=realtime)
    if raw.is_empty():
        raise ValueError(f"no rows parsed for {season}")

    board = raw.with_columns(key=pl.col("name").map_elements(canon, return_dtype=pl.Utf8))

    if skill_board is not None:
        ids = (
            skill_board.filter(pl.col("season") == season)
            .select(
                key=pl.col("name").map_elements(canon, return_dtype=pl.Utf8),
                position=pl.col("position"),
                gsis_id=pl.col("gsis_id"),
            )
            .unique(subset=["key", "position"], keep="first")
        )
        board = board.join(ids, on=["key", "position"], how="left")
    else:
        board = board.with_columns(gsis_id=pl.lit(None, dtype=pl.Utf8))

    if ecr is not None:
        e = (
            ecr.filter(pl.col("season") == season)
            .select("gsis_id", "ecr_rank")
            .drop_nulls("gsis_id")
            .unique(subset=["gsis_id"], keep="first")
        )
        board = board.join(e, on="gsis_id", how="left")
    else:
        board = board.with_columns(ecr_rank=pl.lit(None, dtype=pl.Int32))

    board = (
        board.sort(["adp", "name"])
        .with_row_index("pid")
        .with_columns(
            pid=pl.col("pid").cast(pl.Int32),
            season=pl.lit(season, dtype=pl.Int32),
            adp_rank=(pl.col("pid") + 1).cast(pl.Int32),
        )
        .with_columns(
            pos_rank=pl.col("adp_rank").rank("ordinal").over("position").cast(pl.Int32)
        )
    )

    # After `season` exists, because rookie means "entry year == THIS season"
    # and age is measured at THIS season's kickoff.
    if rookie_maps is not None:
        board = rookies.attach(board, rookie_maps)
    else:
        board = board.with_columns(
            rookie=pl.lit(0, dtype=pl.Int8), rookie_match=pl.lit("none")
        )
    if birth_maps is not None:
        board = attrs.attach_age(board, birth_maps)
    else:
        board = board.with_columns(
            age=pl.lit(0.0), has_age=pl.lit(0, dtype=pl.Int8)
        )
    # durability reads `rookie`, so it comes after it
    if actuals_path is not None:
        board = attrs.attach_durability(board, actuals_path)
    else:
        board = board.with_columns(durability=pl.lit(0.0))

    assert (board["pid"] == board["adp_rank"] - 1).all(), "pid/adp_rank invariant broken"
    return board.select(
        "season", "pid", "name", "position", "pro_team", "bye", "adp", "adp_col",
        "adp_rank", "pos_rank", "fp_pos_rank", "ecr_rank", "gsis_id", "key",
        "rookie", "rookie_match", "age", "has_age", "durability",
    )


def pick_keys(picks: pl.DataFrame) -> pl.DataFrame:
    """Add the join columns that match a pick to a board row."""
    return picks.with_columns(
        position=pl.col("position").replace(ESPN_POS),
        key=pl.col("player_name").map_elements(canon, return_dtype=pl.Utf8),
    ).with_columns(
        dst_team=pl.when(pl.col("position") == "DST")
        .then(
            pl.coalesce(
                pl.col("player_name").map_elements(
                    lambda n: dst_team(n or ""), return_dtype=pl.Utf8
                ),
                pl.col("pro_team"),
            )
        )
        .otherwise(None)
    )


def link_picks(board: pl.DataFrame, picks: pl.DataFrame) -> pl.DataFrame:
    """Attach `pid` to each pick. Three keys in order: gsis_id, then
    (name, position), then team abbreviation for defenses."""
    p = pick_keys(picks)
    b = board.with_columns(
        dst_team=pl.when(pl.col("position") == "DST").then(pl.col("pro_team")).otherwise(None)
    )

    by_gsis = b.filter(pl.col("gsis_id").is_not_null()).select(
        "season", "gsis_id", pid_g="pid"
    ).unique(subset=["season", "gsis_id"], keep="first")
    by_name = b.select("season", "key", "position", pid_n="pid").unique(
        subset=["season", "key", "position"], keep="first"
    )
    by_dst = b.filter(pl.col("dst_team").is_not_null()).select(
        "season", "dst_team", pid_d="pid"
    ).unique(subset=["season", "dst_team"], keep="first")

    out = (
        p.join(by_gsis, on=["season", "gsis_id"], how="left")
        .join(by_name, on=["season", "key", "position"], how="left")
        .join(by_dst, on=["season", "dst_team"], how="left")
        .with_columns(pid=pl.coalesce("pid_g", "pid_n", "pid_d").cast(pl.Int32))
        .drop("pid_g", "pid_n", "pid_d")
    )
    return out


def coverage(linked: pl.DataFrame) -> pl.DataFrame:
    return (
        linked.group_by("season")
        .agg(
            picks=pl.len(),
            on_board=pl.col("pid").is_not_null().sum(),
            rate=pl.col("pid").is_not_null().mean().round(4),
        )
        .sort("season")
    )
