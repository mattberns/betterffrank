"""Player attributes the model has a per-manager taste for: age, durability,
and whether this seat drafted him last year.

`rookies.py` handles entry year; this is the rest of the same idea, and the
three columns here exist for the same reason that one does. Each is a fact
about a PLAYER that a drafter can plausibly have a standing preference about,
so each gets a pooled coefficient plus a per-manager deviation (see
`train.PLAYER_CHANNELS`).

## age

From `birth_date` in the same raw nflverse roster files the entry year comes
from, resolved by `gsis_id` then by normalised name, taken at **1 September of
the draft season**. 52 of 10,262 players carry two birth dates across seasons;
`min()` settles them, the same rule `rookies` uses for entry year.

Centred within (season, position), so the number means "older or younger than
the typical player at his position this year" rather than "old". That is the
right frame because the player stage only ever chooses WITHIN a position: a
27-year-old running back and a 27-year-old quarterback are not the same
proposition. A missing birth date centres to 0, i.e. imputes the typical age
rather than asserting anything (97 of 3,045 skill rows, 3 of them inside any
board's top 100).

`age` and `is_rookie` are not redundant. Rookie is a rare 8% indicator at the
extreme tail of age; age carries information on every pick.

## durability

Share of the PRIOR season's games the player actually appeared in, from
`actuals.parquet`, centred within (season, position).

Three cases, and the middle one is the whole point:

* **A prior-season row.** `games / schedule`, clipped to [0, 1]. The clip is
  load-bearing: a player traded mid-season can dodge two byes and appear in 18
  games against a 17-game schedule (Rashid Shaheed, 2025).
* **No prior-season row, not a rookie.** He played zero fantasy-relevant games:
  injured all year, on a practice squad, or out of the league. Scored as 0.0
  before centring, because that is exactly the signal a drafter reacts to. 308
  of 2,609 non-rookie skill rows, 12 of 839 inside a top 100.
* **A rookie.** There is no prior season to measure, so he is scored AT the
  centre (deviation 0) and left to `is_rookie` to describe. Scoring him 0.0
  would call every rookie maximally fragile.

Schedule length follows the season being MEASURED, not the season being
drafted: 16 games through 2020, 17 from 2021.

## prev_owner

The franchise that drafted this player in the PREVIOUS season's draft, or null.
`was_mine` in `features.py` is this column compared against the seat on the
clock, so the board carries one string per player and the seat-specific part
stays in the feature builder.

81-89% of a season's picks reappear on the next season's board; the rest
retired, were cut, or fell off a 300-400 name board. Two consequences worth
knowing:

* Board seasons **2018 and 2019 have no prior league draft at all**, so the
  column is entirely null there. Fold 2020 trains on 2019 alone, which means
  `was_mine` has zero variance in that fold and its coefficients are fitted at
  0 by the ridge. That is the same situation the parent repo documents for
  `vs_adp` in its own first fold, and it resolves itself from fold 2021 on.
* This is a LEAGUE fact, not a market fact, so unlike everything else on the
  board it cannot be computed for a season the league did not play.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .adp_match import canon

FIRST_ROSTER_SEASON = 2010
SKILL = ("QB", "RB", "WR", "TE")

# Games in a regular season. The 17th was added in 2021.
def schedule_length(season: int) -> int:
    return 17 if season >= 2021 else 16


# --------------------------------------------------------------------- age ---


def birth_date_map(
    context_dir: Path, last_season: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(by_gsis, by_name), same two-pass shape as `rookies.entry_year_map`."""
    frames = []
    for season in range(FIRST_ROSTER_SEASON, last_season + 1):
        path = Path(context_dir) / f"roster_{season}.parquet"
        if path.exists():
            frames.append(
                pl.read_parquet(path).select("gsis_id", "full_name", "birth_date")
            )
    if not frames:
        raise FileNotFoundError(f"no roster_<season>.parquet under {context_dir}")
    rosters = pl.concat(frames).drop_nulls("birth_date")
    by_gsis = (
        rosters.drop_nulls("gsis_id")
        .group_by("gsis_id")
        .agg(birth_gsis=pl.col("birth_date").min())
    )
    by_name = (
        rosters.with_columns(
            key=pl.col("full_name").map_elements(canon, return_dtype=pl.Utf8)
        )
        .drop_nulls("key")
        .group_by("key")
        .agg(birth_name=pl.col("birth_date").min(), n=pl.col("birth_date").n_unique())
        .filter(pl.col("n") == 1)
        .drop("n")
    )
    return by_gsis, by_name


def attach_age(board: pl.DataFrame, maps: tuple[pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    by_gsis, by_name = maps
    return (
        board.join(by_gsis, on="gsis_id", how="left")
        .join(by_name, on="key", how="left")
        .with_columns(birth=pl.coalesce("birth_gsis", "birth_name"))
        .with_columns(
            age_raw=(
                pl.date(pl.col("season"), 9, 1) - pl.col("birth")
            ).dt.total_days() / 365.25
        )
        .with_columns(
            age=(
                pl.col("age_raw") - pl.col("age_raw").mean().over("season", "position")
            ).fill_null(0.0)
        )
        .with_columns(has_age=pl.col("birth").is_not_null().cast(pl.Int8))
        .drop("birth_gsis", "birth_name", "birth", "age_raw")
    )


# -------------------------------------------------------------- durability ---


def attach_durability(board: pl.DataFrame, actuals_path: Path) -> pl.DataFrame:
    """Share of last season's games played, centred within (season, position).

    Rookies sit at the centre by construction — see the module docstring.
    """
    prev = (
        pl.read_parquet(actuals_path)
        .select("gsis_id", "season", "games")
        .with_columns(season=pl.col("season").cast(pl.Int32) + 1)
        .unique(subset=["season", "gsis_id"], keep="first")
        .rename({"games": "prev_games"})
    )
    sched = pl.DataFrame(
        {
            "season": pl.Series([s for s in range(2011, 2031)], dtype=pl.Int32),
            "prev_sched": [schedule_length(s - 1) for s in range(2011, 2031)],
        }
    )
    out = (
        board.join(prev, on=["season", "gsis_id"], how="left")
        .join(sched, on="season", how="left")
        .with_columns(
            played=(pl.col("prev_games").fill_null(0) / pl.col("prev_sched"))
            .clip(0.0, 1.0)
        )
    )
    # The centre is taken over players who HAVE a prior season, so rookies do
    # not drag it down; then rookies are placed exactly on it.
    out = out.with_columns(
        centre=pl.col("played")
        .filter(pl.col("rookie") == 0)
        .mean()
        .over("season", "position")
    ).with_columns(
        durability=pl.when(pl.col("rookie") == 1)
        .then(0.0)
        .otherwise(pl.col("played") - pl.col("centre"))
        .fill_null(0.0)
    )
    return out.drop("prev_games", "prev_sched", "played", "centre")


# ------------------------------------------------------------- prev owner ---


def attach_prev_owner(board: pl.DataFrame, picks: pl.DataFrame) -> pl.DataFrame:
    """The franchise that drafted each player in the PREVIOUS season's draft."""
    from .board import pick_keys

    p = pick_keys(picks).select("season", "franchise_id", "gsis_id", "key", "position")
    p = p.with_columns(season=(pl.col("season").cast(pl.Int32) + 1))
    by_gsis = (
        p.drop_nulls("gsis_id")
        .unique(subset=["season", "gsis_id"], keep="first")
        .select("season", "gsis_id", owner_g="franchise_id")
    )
    by_name = p.unique(subset=["season", "key", "position"], keep="first").select(
        "season", "key", "position", owner_n="franchise_id"
    )
    return (
        board.join(by_gsis, on=["season", "gsis_id"], how="left")
        .join(by_name, on=["season", "key", "position"], how="left")
        .with_columns(prev_owner=pl.coalesce("owner_g", "owner_n"))
        .drop("owner_g", "owner_n")
    )


# ------------------------------------------------------------------ checks ---


def coverage(boards: pl.DataFrame) -> pl.DataFrame:
    return (
        boards.filter(pl.col("position").is_in(SKILL))
        .group_by("season")
        .agg(
            skill=pl.len(),
            no_age=(pl.col("has_age") == 0).sum(),
            head_no_age=((pl.col("has_age") == 0) & (pl.col("adp_rank") <= 100)).sum(),
            prev_owned=pl.col("prev_owner").is_not_null().sum(),
        )
        .sort("season")
    )


def check(boards: pl.DataFrame, depth: int = 100) -> None:
    """Age must resolve for the head of every board, the way rookie status does.

    Durability needs no equivalent: "no prior-season row" is a real answer for a
    real player, not a failed join.
    """
    bad = boards.filter(
        pl.col("position").is_in(SKILL)
        & (pl.col("adp_rank") <= depth)
        & (pl.col("has_age") == 0)
    )
    if bad.height > 5:
        rows = bad.select("season", "name", "position", "adp_rank").rows()
        raise AssertionError(
            f"{bad.height} board players inside the top {depth} have no birth "
            f"date: {rows[:10]}"
        )
