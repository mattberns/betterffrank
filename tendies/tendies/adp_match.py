"""Join drafted players to the market ADP board for that season.

A board is one row per (season, player) with an `adp` and an `adp_rank`. The
PRIMARY board here is FFC half-PPR, which matches how this league scores; the
parent repo also builds the full-PPR board and that is attached as a SECOND,
separately-named set of columns.

Why two boards. FFC's half-PPR history is shallower than its PPR twin and some
seasons are pruned at the source: the 2022 half board is 117 players deep and
does not contain Austin Ekeler at all, who went 3rd overall in this league;
2025 half is missing David Njoku and Jauan Jennings. Rather than record those
as "no ADP", the full-PPR rank rides along in `*_ppr` columns, and
`adp_rank_filled` / `adp_source` say exactly which board each number came
from. Half-PPR is never silently replaced — you can always read the pure
half-PPR column.

Names are the only shared key, so matching runs in passes and each pick
records WHICH pass hit it, in `adp_match` — a match rate is not a number to
take on trust:

  name     normalized full name within the season, position agrees
  alias    ditto, after an explicit rename in ALIASES
  initial  unique (first initial, last name, position) inside the season
  none     no ADP row: off-board player, K, or D/ST

Sign convention, stated once: `adp_delta = overall_pick - adp_rank`.
POSITIVE means the team took him LATER than the market did (value); NEGATIVE
means a reach. Null when there is no ADP row to compare to.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

# variant -> canonical, applied to BOTH sides so either spelling folds to the
# same key. Only real naming differences belong here: suffixes, punctuation and
# accents are already handled by norm_name, and an entry that duplicates that
# work is what silently BROKE Kenneth Walker III in testing. Extend freely.
ALIASES = {
    # Legal name changes are the sharp edge here: ADP sources key on a player
    # id and retroactively restate old boards under the CURRENT name, while
    # ESPN's draft history keeps the name as it was drafted. All four boards
    # call the 2019-2021 Robby Anderson "Robbie Chosen".
    "robby anderson": "robbie chosen",
    "robbie anderson": "robbie chosen",
    "mitch trubisky": "mitchell trubisky",
    "joshua palmer": "josh palmer",
    "gabe davis": "gabriel davis",
    "chigoziem okonkwo": "chig okonkwo",
    "cameron akers": "cam akers",
    "marquise brown": "hollywood brown",
    "jeff wilson": "jeffery wilson",
    "elijah mitchell": "eli mitchell",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
NO_ADP_POSITIONS = {"K", "D/ST", "OTHER"}
BOARD_COLS = ["adp_name", "adp", "adp_rank", "adp_pos_rank", "gsis_id", "adp_match"]


def norm_name(name: str | None) -> str | None:
    """Fourth cousin of the parent repo's norm_name variants; kept separate."""
    if not name:
        return None
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[.'`’\"]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    parts = [p for p in s.split() if p not in SUFFIXES]
    return " ".join(parts) or None


def canon(name: str | None) -> str | None:
    """Normalized name, then folded through ALIASES. The one join key."""
    n = norm_name(name)
    return ALIASES.get(n, n) if n else None


def _initial_key(n: str | None) -> str | None:
    if not n:
        return None
    parts = n.split()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else None


def load_board(path, seasons: list[int]) -> pl.DataFrame:
    board = pl.read_parquet(path)
    need = {"season", "name", "position", "adp", "adp_rank"}
    missing = need - set(board.columns)
    if missing:
        raise ValueError(f"ADP board {path} is missing columns: {sorted(missing)}")
    board = board.filter(pl.col("season").is_in(seasons))
    if "gsis_id" not in board.columns:
        board = board.with_columns(gsis_id=pl.lit(None, dtype=pl.Utf8))
    return (
        board.select(
            "season", "position", "adp", "adp_rank", "gsis_id",
            adp_name=pl.col("name"),
            key=pl.col("name").map_elements(canon, return_dtype=pl.Utf8),
        )
        .with_columns(
            adp_pos_rank=pl.col("adp").rank("ordinal").over("season", "position").cast(pl.Int64)
        )
        # A board can list one name twice (cross-position listings); the
        # earliest ADP is the one a drafter was looking at.
        .sort("adp_rank")
        .unique(subset=["season", "key", "position"], keep="first")
    )


def _match_one(picks: pl.DataFrame, board: pl.DataFrame) -> pl.DataFrame:
    """picks + BOARD_COLS, keyed by the pick (season, overall_pick)."""
    cols = [c for c in BOARD_COLS if c != "adp_match"]
    p = picks.with_columns(
        key=pl.col("player_name").map_elements(canon, return_dtype=pl.Utf8),
        plain=pl.col("player_name").map_elements(norm_name, return_dtype=pl.Utf8),
    )

    out = p.join(
        board.select("season", "key", "position", *cols),
        on=["season", "key", "position"], how="left",
    ).with_columns(
        adp_match=pl.when(pl.col("adp_rank").is_null()).then(pl.lit("none"))
        .when(pl.col("key") != pl.col("plain")).then(pl.lit("alias"))
        .otherwise(pl.lit("name"))
    )

    init = (
        board.with_columns(ikey=pl.col("key").map_elements(_initial_key, return_dtype=pl.Utf8))
        .drop_nulls("ikey")
        .filter(pl.len().over("season", "ikey", "position") == 1)
        .select(["season", "ikey", "position"] + [pl.col(c).alias(f"i_{c}") for c in cols])
    )
    return (
        out.with_columns(ikey=pl.col("key").map_elements(_initial_key, return_dtype=pl.Utf8))
        .join(init, on=["season", "ikey", "position"], how="left")
        .with_columns(
            [pl.coalesce(pl.col(c), pl.col(f"i_{c}")).alias(c) for c in cols]
            + [
                pl.when((pl.col("adp_match") == "none") & pl.col("i_adp_rank").is_not_null())
                .then(pl.lit("initial")).otherwise(pl.col("adp_match")).alias("adp_match")
            ]
        )
        .drop(["ikey", "key", "plain"] + [f"i_{c}" for c in cols])
    )


def attach(
    picks: pl.DataFrame,
    board: pl.DataFrame,
    secondaries: list[tuple[str, pl.DataFrame]] | None = None,
    primary_tag: str = "half",
) -> pl.DataFrame:
    """Primary board in `adp_*`; each extra board in its own `adp_*_<tag>`.

    `adp_rank_filled` / `adp_delta_filled` coalesce down the chain in the order
    given and `adp_source` names the board each value came from, so a filled
    number is never anonymous. No board is ever silently substituted for
    another: the primary columns stay pure.
    """
    out = _match_one(picks, board)
    tags: list[str] = []

    for tag, sec_board in secondaries or []:
        tags.append(tag)
        sec = _match_one(picks, sec_board).select(
            ["season", "overall_pick"] + [pl.col(c).alias(f"{c}_{tag}") for c in BOARD_COLS]
        )
        out = out.join(sec, on=["season", "overall_pick"], how="left").with_columns(
            gsis_id=pl.coalesce("gsis_id", f"gsis_id_{tag}")
        ).drop(f"gsis_id_{tag}")

    out = out.with_columns(
        adp_delta=(pl.col("overall_pick") - pl.col("adp_rank")).cast(pl.Int64),
        **{
            f"adp_delta_{t}": (pl.col("overall_pick") - pl.col(f"adp_rank_{t}")).cast(pl.Int64)
            for t in tags
        },
    )
    if tags:
        source = pl.when(pl.col("adp_rank").is_not_null()).then(pl.lit(primary_tag))
        for t in tags:
            source = source.when(pl.col(f"adp_rank_{t}").is_not_null()).then(pl.lit(t))
        out = out.with_columns(
            adp_rank_filled=pl.coalesce("adp_rank", *[f"adp_rank_{t}" for t in tags]),
            adp_delta_filled=pl.coalesce("adp_delta", *[f"adp_delta_{t}" for t in tags]),
            adp_source=source.otherwise(pl.lit(None, dtype=pl.Utf8)),
        )
    return out


def board_columns(tags: list[str]) -> list[str]:
    """Column order for the primary board plus each extra board, then the
    coalesced convenience columns."""
    cols = ["adp", "adp_rank", "adp_pos_rank", "adp_delta", "adp_match", "adp_name"]
    for t in tags:
        cols += [f"{c}_{t}" for c in
                 ("adp", "adp_rank", "adp_pos_rank", "adp_delta", "adp_match")]
    return cols + (["adp_rank_filled", "adp_delta_filled", "adp_source"] if tags else [])


def coverage(joined: pl.DataFrame) -> pl.DataFrame:
    """Match rate by season over picks a board could plausibly carry."""
    skill = joined.filter(~pl.col("position").is_in(list(NO_ADP_POSITIONS)))
    aggs = [
        pl.len().alias("picks"),
        pl.col("adp_rank").is_not_null().mean().round(3).alias("primary_rate"),
        pl.col("adp_rank").is_not_null().filter(pl.col("overall_pick") <= 100)
        .mean().round(3).alias("primary_top100"),
    ]
    if "adp_rank_filled" in skill.columns:
        aggs.append(
            pl.col("adp_rank_filled").is_not_null().mean().round(3).alias("filled_rate")
        )
    return skill.group_by("season").agg(aggs).sort("season")


def unmatched(joined: pl.DataFrame, max_pick: int | None = None) -> pl.DataFrame:
    col = "adp_rank_filled" if "adp_rank_filled" in joined.columns else "adp_rank"
    q = joined.filter(
        pl.col(col).is_null() & ~pl.col("position").is_in(list(NO_ADP_POSITIONS))
    )
    if max_pick:
        q = q.filter(pl.col("overall_pick") <= max_pick)
    return q.select("season", "overall_pick", "player_name", "position", "pro_team").sort(
        "season", "overall_pick"
    )
