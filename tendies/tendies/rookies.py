"""Rookie status for board players, from NFL entry year.

A rookie is a player whose first NFL season IS this season. That is a static
player attribute crossed with the season, so it is a preseason fact and cannot
leak an outcome — but it has to be read from the right place, and the two
obvious places are both wrong here.

**`gsis_id` is null for exactly the population this measures.** The board joins
ids from the parent repo's `fp_adp_half.parquet`, and nflverse has no gsis id
for a player who has not played yet; `dataset.check_ecr` already says so in its
own comment (2026 reads 0.894 id coverage against 0.98-1.00 elsewhere). Keying
rookie status on gsis alone silently marks the whole rookie class as veterans.

**FantasyPros ids do not rescue it.** Joining the raw capture's `fp_id` to
`db_playerids.fantasypros_id` matches 77.5% of the 2026 board and finds **zero**
rookies: the 82 unmatched rows ARE the rookie class (Jeremiyah Love, Carnell
Tate, Fernando Mendoza, ...). A match rate that looks acceptable overall can be
0% on the subgroup you are measuring, which is the whole reason this module
records which pass hit each row.

So the source is the parent repo's RAW nflverse roster files,
`data/raw/context/roster_<season>.parquet`, which carry `entry_year` for every
player who ever appeared on a roster — including the current class, whose
placeholder gsis ids the parent's build nulls out but whose NAMES are real. The
map is built across ALL seasons at once and keyed on the player, not on
season-t roster membership, so a board player who never makes a roster still
resolves. `entry_year` is effectively invariant per player (3 conflicts in
12,255); `min()` settles those.

Measured coverage on the boards this league drafts from: 0-2.5% unresolved,
and **zero** unresolved inside the top 100 of any board. `check` asserts that
top-100 property rather than a global rate, because the deep tail of a 400-name
board is not what a manager is choosing between.

K and D/ST resolve to `rookie = 0` by construction. A defense has no entry
year; a kicker usually does, and gets a real answer.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .adp_match import canon

# The roster files the parent repo keeps. 2010 is the earliest; the map only
# needs to be deep enough that any player on a board has been seen once.
FIRST_ROSTER_SEASON = 2010

SKILL = ("QB", "RB", "WR", "TE")

# Rows above this board rank must all resolve. Deep-tail names on a 400-player
# board are camp bodies nobody is choosing between; the head is not.
CHECK_DEPTH = 100


def _roster_frame(context_dir: Path, last_season: int) -> pl.DataFrame:
    frames = []
    for season in range(FIRST_ROSTER_SEASON, last_season + 1):
        path = Path(context_dir) / f"roster_{season}.parquet"
        if not path.exists():
            continue
        frames.append(
            pl.read_parquet(path).select("gsis_id", "full_name", "entry_year")
        )
    if not frames:
        raise FileNotFoundError(
            f"no roster_<season>.parquet under {context_dir}; rookie status "
            "needs the parent repo's nflverse rosters"
        )
    return pl.concat(frames).drop_nulls("entry_year")


def entry_year_map(context_dir: Path, last_season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(by_gsis, by_name). Two lookups, tried in that order.

    The name map drops any name carrying more than one distinct entry year
    rather than guessing which player it is — an ambiguous name resolves to
    `none` and is visible in `rookie_match`, not silently wrong.

    That filter is load-bearing, and cross-era collisions are real: "Kaleb
    Johnson" is a 2025 rookie AND an earlier player, so the name carries two
    entry years and is dropped here. He still resolves, because `attach` tries
    gsis first and the id join is unambiguous. Abstaining is the safe direction
    — a dropped name reads as `rookie = 0`, which is what an unknown player
    already is — and `test_name_fallback_never_contradicts_the_id_join` pins
    the two passes to the same answer wherever both have one.
    """
    rosters = _roster_frame(context_dir, last_season)
    by_gsis = (
        rosters.drop_nulls("gsis_id")
        .group_by("gsis_id")
        .agg(entry_gsis=pl.col("entry_year").min())
    )
    by_name = (
        rosters.with_columns(
            key=pl.col("full_name").map_elements(canon, return_dtype=pl.Utf8)
        )
        .drop_nulls("key")
        .group_by("key")
        .agg(entry_name=pl.col("entry_year").min(), n=pl.col("entry_year").n_unique())
        .filter(pl.col("n") == 1)
        .drop("n")
    )
    return by_gsis, by_name


def attach(board: pl.DataFrame, maps: tuple[pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """Add `rookie` (Int8) and `rookie_match` (gsis / name / none).

    `key` is the board's canonical name, already computed by `board.build`.
    An unresolved row scores `rookie = 0` — the conservative default, and the
    one `check` exists to keep honest.
    """
    by_gsis, by_name = maps
    out = (
        board.join(by_gsis, on="gsis_id", how="left")
        .join(by_name, on="key", how="left")
        .with_columns(entry=pl.coalesce("entry_gsis", "entry_name"))
        .with_columns(
            rookie=(pl.col("entry") == pl.col("season"))
            .fill_null(False)
            .cast(pl.Int8),
            rookie_match=pl.when(pl.col("entry_gsis").is_not_null())
            .then(pl.lit("gsis"))
            .when(pl.col("entry_name").is_not_null())
            .then(pl.lit("name"))
            .otherwise(pl.lit("none")),
        )
        .drop("entry_gsis", "entry_name", "entry")
    )
    return out


def coverage(boards: pl.DataFrame) -> pl.DataFrame:
    """Per-season resolution, skill positions only (K/DST are not the point)."""
    return (
        boards.filter(pl.col("position").is_in(SKILL))
        .group_by("season")
        .agg(
            skill=pl.len(),
            by_gsis=(pl.col("rookie_match") == "gsis").sum(),
            by_name=(pl.col("rookie_match") == "name").sum(),
            unresolved=(pl.col("rookie_match") == "none").sum(),
            rookies=pl.col("rookie").sum(),
            head_unresolved=(
                (pl.col("rookie_match") == "none") & (pl.col("adp_rank") <= CHECK_DEPTH)
            ).sum(),
        )
        .sort("season")
    )


def check(boards: pl.DataFrame, depth: int = CHECK_DEPTH) -> None:
    """Every skill player inside the top `depth` of a board must resolve.

    Measured at zero failures on 2019-2026 when this was written. A failure
    means a name the roster files spell differently, so fix it with an
    `adp_match.ALIASES` entry — never by lowering `depth`.
    """
    bad = boards.filter(
        pl.col("position").is_in(SKILL)
        & (pl.col("adp_rank") <= depth)
        & (pl.col("rookie_match") == "none")
    )
    if bad.height:
        rows = bad.select("season", "name", "position", "adp_rank").rows()
        raise AssertionError(
            f"{bad.height} board players inside the top {depth} have no NFL "
            f"entry year: {rows[:10]}"
        )
