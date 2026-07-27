"""Two zero-fetch candidate blocks, keyed (season, gsis_id), 2011-2026.

Registered in bff.model.CANDIDATE_BLOCKS as `sos` and `trajectory`.
Selection on the 2012-2017 tune window only (bff.select_features). All
features preseason-known for season t:

  sos_dvp        Mean over the player's season-t REG opponents of the
                 opponent defense's PRIOR-year (t-1) PPR points allowed per
                 game to the player's position, league-centered per
                 (season, position). Positive = soft schedule. Season-t
                 schedule is a preseason fact (released each spring); the
                 defensive strength is a t-1 outcome. Null (->0) for
                 FA/no-team rows.
  sos_dvp_first4 Same, restricted to the player's first four REG opponents
                 (the early slate that drives draft-day streaming value).
  yrs_exp        Season t minus draft_year (crosswalk). Static fact. 0 for
                 rookies / unknown.
  yrs_since_peak (t-1) minus the season of the player's career-best ppg_ppr
                 through t-1. 0 if peaked last year or no prior.
  career_best_ppg Max ppg_ppr over seasons < t (0 if none). A level feature:
                 the player's demonstrated ceiling.
  last_was_career_best 1 if the t-1 ppg_ppr equalled the career best (broke
                 out / peaked last season). Regression-vs-continuation signal.

Career history spans source seasons 1999-2025: actuals.parquet (2010+) plus
legacy_career_ppg() aggregating raw player_stats_{1999..2009}.parquet
(fetched from the nflverse player_stats release, same as 2010+). Do NOT
extend actuals.parquet itself -- it feeds training targets and backtest
ground truth, and pre-2010 seasons are never scored.

Leakage: sos uses season-t opponents (preseason schedule) x t-1 defense
outcomes; trajectory uses only seasons < t plus static draft_year.

Build: uv run python -m bff.schedule_trajectory_features
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bff.context_data import norm_team

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW_STATS = ROOT / "data" / "raw" / "stats"
CTX = ROOT / "data" / "raw" / "context"
CROSSWALK = ROOT / "data" / "raw" / "db_playerids.csv"

SEASONS = range(2011, 2027)
FANTASY_POS = ["QB", "RB", "WR", "TE"]
POS_MAP = {"HB": "RB", "FB": "RB"}


def build_pool() -> pl.DataFrame:
    adp = pl.read_parquet(PROC / "adp.parquet")
    pool = adp.filter(
        pl.col("season").is_between(2011, 2026)
        & pl.col("position").is_in(FANTASY_POS)
        & pl.col("gsis_id").is_not_null()
    ).with_columns(
        pl.when(pl.col("team").is_null() | (pl.col("team") == "FA"))
        .then(None).otherwise(norm_team("team")).alias("team")
    ).select("season", "gsis_id", "name", "position", "team")
    assert pool.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return pool


# ------------------------------------------------------------------ SOS

def _weekly_frames() -> pl.DataFrame:
    """REG weekly rows (season, opponent_team, position, fantasy_points_ppr,
    week) for source seasons 2010-2025."""
    frames = []
    for y in range(2010, 2025):
        w = pl.read_parquet(RAW_STATS / f"player_stats_{y}.parquet")
        frames.append(w.select(
            pl.col("season").cast(pl.Int64), "season_type",
            "opponent_team", "position", "fantasy_points_ppr", "week",
        ))
    w25 = pl.read_parquet(RAW_STATS / "stats_player_week_2025.parquet")
    frames.append(w25.select(
        pl.col("season").cast(pl.Int64), "season_type",
        "opponent_team", "position", "fantasy_points_ppr", "week",
    ))
    return pl.concat(frames).filter(
        (pl.col("season_type") == "REG")
        & pl.col("opponent_team").is_not_null()
        & pl.col("position").is_in(FANTASY_POS)
    ).with_columns(
        norm_team("opponent_team").alias("def_team"),
        pl.col("position").replace(POS_MAP).alias("pos"),
    )


def defense_vs_position() -> pl.DataFrame:
    """(source season, def_team, pos, dvp_c): PPR allowed per game to a
    position, league-centered per (season, pos). Points allowed / games the
    defense played (distinct REG weeks)."""
    w = _weekly_frames()
    games = w.group_by("season", "def_team").agg(
        pl.col("week").n_unique().alias("n_games")
    )
    dvp = (
        w.group_by("season", "def_team", "pos")
        .agg(pl.col("fantasy_points_ppr").sum().alias("fp"))
        .join(games, on=["season", "def_team"])
        .with_columns((pl.col("fp") / pl.col("n_games")).alias("dvp"))
        .with_columns(
            (pl.col("dvp") - pl.col("dvp").mean().over("season", "pos")).alias("dvp_c")
        )
        .select("season", "def_team", "pos", "dvp_c")
    )
    return dvp


def schedule() -> pl.DataFrame:
    """(season, team, opp, week) for every REG matchup, both directions,
    canonical team codes. Seasons 2010-2026."""
    g = pl.read_csv(CTX / "games.csv", infer_schema_length=20000).filter(
        pl.col("game_type") == "REG"
    )
    home = g.select("season", "week",
                    norm_team("home_team").alias("team"),
                    norm_team("away_team").alias("opp"))
    away = g.select("season", "week",
                    norm_team("away_team").alias("team"),
                    norm_team("home_team").alias("opp"))
    return pl.concat([home, away]).filter(pl.col("season").is_in(list(SEASONS)))


def sos_features(pool: pl.DataFrame) -> pl.DataFrame:
    dvp = defense_vs_position().with_columns(
        (pl.col("season") + 1).alias("season")  # t-1 dvp -> target season t
    )
    sched = schedule()
    # opponent's t-1 dvp to the player's position, per (season t, team, pos)
    pos_teams = pool.filter(pl.col("team").is_not_null()).select(
        "season", "team", "position"
    ).unique().with_columns(pl.col("position").replace(POS_MAP).alias("pos"))
    sch_pos = sched.join(pos_teams, on=["season", "team"], how="inner").join(
        dvp.rename({"def_team": "opp"}), on=["season", "opp", "pos"], how="left"
    )
    agg = sch_pos.group_by("season", "team", "position").agg(
        pl.col("dvp_c").mean().alias("sos_dvp"),
        pl.col("dvp_c").filter(pl.col("week") <= 4).mean().alias("sos_dvp_first4"),
    )
    return pool.join(agg, on=["season", "team", "position"], how="left").select(
        "season", "gsis_id", "sos_dvp", "sos_dvp_first4"
    )


# ----------------------------------------------------------- TRAJECTORY

def legacy_career_ppg() -> pl.DataFrame:
    """(gsis_id, season, ppg_ppr) for source seasons 1999-2009, aggregated from
    raw weekly stats exactly like bff.actuals.build_weekly (games = distinct REG
    weeks). Career-history ONLY: actuals.parquet deliberately stays 2010+ (it
    feeds training targets and the backtest ground truth; pre-2010 seasons are
    never scored). Without this, tune-fold players had 2-7 visible career
    seasons vs 8-15 in test folds -- yrs_since_peak pool mean ramped 0.33
    (2012) -> 1.5 (2025) purely from window truncation (fixed 2026-07-27)."""
    frames = []
    for y in range(1999, 2010):
        f = RAW_STATS / f"player_stats_{y}.parquet"
        if not f.exists():
            continue
        w = pl.read_parquet(f).filter(pl.col("season_type") == "REG")
        frames.append(
            w.group_by("player_id", "season").agg(
                games=pl.col("week").n_unique(),
                pts_ppr=pl.col("fantasy_points_ppr").sum(),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={"gsis_id": pl.String, "season": pl.Int64, "ppg_ppr": pl.Float64}
        )
    return (
        pl.concat(frames)
        .select(
            pl.col("player_id").alias("gsis_id"),
            pl.col("season").cast(pl.Int64),
            (pl.col("pts_ppr") / pl.col("games")).alias("ppg_ppr"),
        )
    )


def trajectory_features(pool: pl.DataFrame) -> pl.DataFrame:
    actuals = pl.concat([
        legacy_career_ppg(),
        pl.read_parquet(PROC / "actuals.parquet").select(
            "gsis_id", pl.col("season").cast(pl.Int64), "ppg_ppr"
        ),
    ])
    xw = pl.read_csv(CROSSWALK, null_values=["NA"], infer_schema_length=10000).filter(
        pl.col("gsis_id").is_not_null()
    ).select("gsis_id", "draft_year").unique("gsis_id")

    # career stats through t-1: for each target season t, aggregate the
    # player's actuals rows in seasons < t
    tgt = pool.select("season", "gsis_id").unique()
    hist = tgt.join(actuals, on="gsis_id", how="left", suffix="_a").filter(
        pl.col("season_a") < pl.col("season")
    )
    career = hist.group_by("season", "gsis_id").agg(
        pl.col("ppg_ppr").max().alias("career_best_ppg"),
        pl.col("season_a").filter(
            pl.col("ppg_ppr") == pl.col("ppg_ppr").max()
        ).max().alias("peak_season"),
        pl.col("ppg_ppr").filter(pl.col("season_a") == pl.col("season") - 1)
        .first().alias("prev_ppg"),
    ).with_columns(
        (pl.col("season") - 1 - pl.col("peak_season")).alias("yrs_since_peak"),
        (
            (pl.col("prev_ppg") == pl.col("career_best_ppg"))
            & pl.col("prev_ppg").is_not_null()
        ).cast(pl.Int8).alias("last_was_career_best"),
    )
    out = (
        pool.select("season", "gsis_id")
        .join(career, on=["season", "gsis_id"], how="left")
        .join(xw, on="gsis_id", how="left")
        .with_columns(
            (pl.col("season") - pl.col("draft_year")).clip(0, 20)
            .cast(pl.Float64).alias("yrs_exp"),
            pl.col("career_best_ppg").fill_null(0.0),
            pl.col("yrs_since_peak").fill_null(0).clip(0, 20).cast(pl.Float64),
            pl.col("last_was_career_best").fill_null(0),
        )
        .with_columns(pl.col("yrs_exp").fill_null(0.0))
        .select("season", "gsis_id", "yrs_exp", "yrs_since_peak",
                "career_best_ppg", "last_was_career_best")
    )
    return out


FEATS = ["sos_dvp", "sos_dvp_first4", "yrs_exp", "yrs_since_peak",
         "career_best_ppg", "last_was_career_best"]


def build() -> pl.DataFrame:
    pool = build_pool()
    out = (
        pool.join(sos_features(pool), on=["season", "gsis_id"], how="left")
        .join(trajectory_features(pool), on=["season", "gsis_id"], how="left")
        .sort("season", "gsis_id")
    )
    assert out.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return out


def main() -> None:
    out = build()
    out.write_parquet(PROC / "sched_traj_features.parquet")
    print(f"wrote {out.height} rows, seasons {out['season'].min()}-{out['season'].max()}")
    for label, df in [
        ("2012-2017 (tune)", out.filter(pl.col("season").is_between(2012, 2017))),
        ("2018-2025 (test)", out.filter(pl.col("season").is_between(2018, 2025))),
        ("2026", out.filter(pl.col("season") == 2026)),
    ]:
        print(f"\n== coverage {label} (n={df.height}) ==")
        for c in FEATS:
            nn = 1.0 - df[c].null_count() / df.height
            print(f"  {c:22s} non-null {nn:5.1%}  mean {df[c].cast(pl.Float64).mean():+8.3f}")
    # sanity: hardest/softest 2026 schedules for RBs
    rb = out.filter((pl.col("season") == 2026)).join(
        build_pool().filter(pl.col("season") == 2026).select("gsis_id", "name", "position"),
        on="gsis_id", how="left")
    print("\n2026 softest RB schedules (top sos_dvp):")
    print(rb.filter(pl.col("position") == "RB").sort("sos_dvp", descending=True)
          .select("name", "sos_dvp", "sos_dvp_first4").head(5))
    print("2026 highest career_best_ppg:")
    print(out.filter(pl.col("season") == 2026).join(
        build_pool().filter(pl.col("season")==2026).select("gsis_id","name"), on="gsis_id")
        .sort("career_best_ppg", descending=True).select("name","career_best_ppg","yrs_exp","yrs_since_peak").head(5))


if __name__ == "__main__":
    main()
