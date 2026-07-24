"""Zero-fetch candidate feature blocks, keyed (season, gsis_id), 2011-2026.

One row per season-t ADP-pool player (QB/RB/WR/TE from data/processed/adp.parquet).
Four 2-column blocks, all computed from data ALREADY in the repo (no new raw
sources). Registered in bff.model.CANDIDATE_BLOCKS as coach_scheme / qb_rush /
ol_proxy / adp_gap; selection on the 2012-2017 tune window only. Timing per
feature (every one preseason-known for season t):

  coach_pass_oe       Season-t week-1 coach (preseason fact, ctx_week1_coaches)
                      x his CAREER pass-rate-over-league at week-1-coach stops
                      in seasons < t (past outcomes). 0 = no prior HC history.
  coach_pass_shift    coach_pass_oe minus the team's own t-1 centered pass
                      rate, only when the week-1 coach CHANGED t-1 -> t and
                      the new coach has history; else 0. The scheme delta the
                      new play-caller brings to this roster.
  qb_rush_ypg         QB rows only: own t-1 REG rushing yards per game
                      (Konami floor). 0 for non-QBs and no-prior QBs.
  expqb_rush_wrte     WR/TE rows only: the season-t EXPECTED QB's
                      (ctx_team_qb.expected_qb_gsis, a preseason ADP fact)
                      t-1 rushing ypg. Rushing QBs depress pass-catcher
                      volume. 0 when no expected QB or he has no t-1 stats.
  ol_sack_rate_prior  Team t-1 sacks / dropbacks (pbp), league-centered per
                      source season, joined on the season-t ADP team.
  ol_stuff_rate_prior Team t-1 stuffed-run rate (rush attempts with
                      yards_gained <= 0, kneels excluded) / attempts,
                      centered likewise. Both are t-1 outcomes on a
                      preseason-known team assignment (same timing pattern
                      as team_fp_prior).
  adp_gap_ahead       log1p(own adp_rank - the same-team-same-position
                      teammate directly ahead in season-t ADP). 0 for
                      depth-1 players (depth_rank_adp already marks them).
  adp_gap_behind      log1p(next same-team-same-position teammate behind -
                      own adp_rank), capped at log1p(100) when no teammate
                      behind exists in the ADP table (secure role). Pure
                      season-t market facts.

Leakage: coach history / QB rushing / OL rates use only seasons < t; the
week-1 coach, expected QB, and ADP gaps are season-t PRESEASON facts.

Build: uv run python -m bff.candidate_features
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW_STATS = ROOT / "data" / "raw" / "stats"
RAW_PBP = ROOT / "data" / "raw" / "pbp"

SEASONS = range(2011, 2027)
FANTASY_POS = ["QB", "RB", "WR", "TE"]
GAP_CAP = 100.0


def build_pool() -> pl.DataFrame:
    adp = pl.read_parquet(PROC / "adp.parquet")
    pool = adp.filter(
        pl.col("season").is_between(2011, 2026)
        & pl.col("position").is_in(FANTASY_POS)
        & pl.col("gsis_id").is_not_null()
    ).with_columns(
        pl.when(pl.col("team").is_null() | (pl.col("team") == "FA"))
        .then(None).otherwise(pl.col("team")).alias("team")
    ).select("season", "gsis_id", "name", "position", "team", "adp_rank")
    assert pool.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return pool


# ---------------------------------------------------------------- coach_scheme

def team_pass_centered() -> pl.DataFrame:
    """(season, team, pass_c): team pass rate targets/(targets+carries),
    centered on the league mean of that season. Seasons 2010-2025."""
    vol = pl.read_parquet(PROC / "ctx_team_volume.parquet")
    t = vol.group_by("season", "team").agg(
        pl.col("team_targets").first(), pl.col("team_carries").first()
    ).with_columns(
        (pl.col("team_targets") / (pl.col("team_targets") + pl.col("team_carries")))
        .alias("pass_rate")
    ).with_columns(
        (pl.col("pass_rate") - pl.col("pass_rate").mean().over("season")).alias("pass_c")
    )
    return t.select("season", "team", "pass_c")


def coach_features(pool: pl.DataFrame) -> pl.DataFrame:
    coaches = pl.read_parquet(PROC / "ctx_week1_coaches.parquet")
    pass_c = team_pass_centered()

    # coach-season observations: the pass_c his team produced that season
    obs = coaches.join(pass_c, on=["season", "team"], how="inner").select(
        "week1_coach", pl.col("season").alias("obs_season"), "pass_c"
    )
    # for each (target season t, team): the week-1 coach + change flag
    tgt = coaches.join(
        coaches.select((pl.col("season") + 1).alias("season"), "team",
                       pl.col("week1_coach").alias("prev_coach")),
        on=["season", "team"], how="left",
    ).with_columns(
        (pl.col("week1_coach") != pl.col("prev_coach"))
        .fill_null(False).alias("chg")
    )
    # career mean over seasons < t (16 seasons x 32 teams: join+filter is cheap)
    hist = (
        tgt.join(obs, on="week1_coach", how="left")
        .filter(pl.col("obs_season") < pl.col("season"))
        .group_by("season", "team")
        .agg(pl.col("pass_c").mean().alias("coach_pass_oe"))
    )
    out = (
        tgt.join(hist, on=["season", "team"], how="left")
        .join(pass_c.select((pl.col("season") + 1).alias("season"), "team",
                            pl.col("pass_c").alias("prev_pass_c")),
              on=["season", "team"], how="left")
        .with_columns(
            pl.when(pl.col("chg") & pl.col("coach_pass_oe").is_not_null()
                    & pl.col("prev_pass_c").is_not_null())
            .then(pl.col("coach_pass_oe") - pl.col("prev_pass_c"))
            .otherwise(0.0)
            .alias("coach_pass_shift"),
            pl.col("coach_pass_oe").fill_null(0.0),
        )
        .select("season", "team", "coach_pass_oe", "coach_pass_shift")
    )
    return pool.join(out, on=["season", "team"], how="left").select(
        "season", "gsis_id", "coach_pass_oe", "coach_pass_shift"
    )


# -------------------------------------------------------------------- qb_rush

def qb_rush_by_source_season() -> pl.DataFrame:
    """(source season, gsis_id, rush_ypg) for QBs, REG weeks, 2010-2025."""
    frames = []
    for y in range(2010, 2025):
        w = pl.read_parquet(RAW_STATS / f"player_stats_{y}.parquet")
        frames.append(
            w.filter((pl.col("season_type") == "REG") & (pl.col("position") == "QB"))
            .group_by("season", "player_id")
            .agg((pl.col("rushing_yards").sum() / pl.len()).alias("rush_ypg"))
        )
    w25 = pl.read_parquet(RAW_STATS / "stats_player_week_2025.parquet")
    frames.append(
        w25.filter((pl.col("season_type") == "REG") & (pl.col("position") == "QB"))
        .group_by("season", "player_id")
        .agg((pl.col("rushing_yards").sum() / pl.len()).alias("rush_ypg"))
    )
    return (
        pl.concat([f.with_columns(pl.col("season").cast(pl.Int32)) for f in frames])
        .rename({"player_id": "gsis_id"})
    )


def qb_rush_features(pool: pl.DataFrame) -> pl.DataFrame:
    rush = qb_rush_by_source_season().with_columns(
        (pl.col("season") + 1).alias("season")  # -> target season t
    )
    own = pool.filter(pl.col("position") == "QB").join(
        rush, on=["season", "gsis_id"], how="left"
    ).select("season", "gsis_id", pl.col("rush_ypg").alias("qb_rush_ypg"))

    tq = pl.read_parquet(PROC / "ctx_team_qb.parquet").select(
        "season", "team", "expected_qb_gsis"
    )
    exp = (
        pool.filter(pl.col("position").is_in(["WR", "TE"]))
        .join(tq, on=["season", "team"], how="left")
        .join(rush.rename({"gsis_id": "expected_qb_gsis"}),
              on=["season", "expected_qb_gsis"], how="left")
        .select("season", "gsis_id", pl.col("rush_ypg").alias("expqb_rush_wrte"))
    )
    return pool.select("season", "gsis_id").join(
        own, on=["season", "gsis_id"], how="left"
    ).join(exp, on=["season", "gsis_id"], how="left")


# ------------------------------------------------------------------- ol_proxy

def ol_by_source_season() -> pl.DataFrame:
    """(source season, team, sack_rate_c, stuff_rate_c) from REG pbp,
    league-centered per source season. 2010-2025."""
    frames = []
    cols = ["season_type", "posteam", "sack", "qb_dropback", "rush_attempt",
            "qb_kneel", "yards_gained"]
    for y in range(2010, 2026):
        p = pl.read_parquet(RAW_PBP / f"play_by_play_{y}.parquet", columns=cols)
        reg = p.filter((pl.col("season_type") == "REG") & pl.col("posteam").is_not_null())
        frames.append(
            reg.group_by("posteam").agg(
                (pl.col("sack").sum() / pl.col("qb_dropback").sum()).alias("sack_rate"),
                (
                    ((pl.col("rush_attempt") == 1) & (pl.col("qb_kneel") == 0)
                     & (pl.col("yards_gained") <= 0)).sum()
                    / ((pl.col("rush_attempt") == 1) & (pl.col("qb_kneel") == 0)).sum()
                ).alias("stuff_rate"),
            ).with_columns(pl.lit(y, dtype=pl.Int32).alias("season"))
        )
    ol = pl.concat(frames).rename({"posteam": "team"})
    return ol.with_columns(
        (pl.col("sack_rate") - pl.col("sack_rate").mean().over("season"))
        .alias("sack_rate_c"),
        (pl.col("stuff_rate") - pl.col("stuff_rate").mean().over("season"))
        .alias("stuff_rate_c"),
    ).select("season", "team", "sack_rate_c", "stuff_rate_c")


def ol_features(pool: pl.DataFrame) -> pl.DataFrame:
    ol = ol_by_source_season().with_columns((pl.col("season") + 1).alias("season"))
    return pool.join(ol, on=["season", "team"], how="left").select(
        "season", "gsis_id",
        pl.col("sack_rate_c").alias("ol_sack_rate_prior"),
        pl.col("stuff_rate_c").alias("ol_stuff_rate_prior"),
    )


# -------------------------------------------------------------------- adp_gap

def adp_gap_features(pool: pl.DataFrame) -> pl.DataFrame:
    teamed = pool.filter(pl.col("team").is_not_null()).sort(
        "season", "team", "position", "adp_rank"
    ).with_columns(
        pl.col("adp_rank").shift(1).over("season", "team", "position").alias("ahead"),
        pl.col("adp_rank").shift(-1).over("season", "team", "position").alias("behind"),
    ).select(
        "season", "gsis_id",
        pl.when(pl.col("ahead").is_null()).then(0.0)
        .otherwise((pl.col("adp_rank") - pl.col("ahead")).cast(pl.Float64).log1p())
        .alias("adp_gap_ahead"),
        pl.when(pl.col("behind").is_null()).then(pl.lit(GAP_CAP).log1p())
        .otherwise(
            (pl.col("behind") - pl.col("adp_rank")).cast(pl.Float64)
            .clip(upper_bound=GAP_CAP).log1p()
        )
        .alias("adp_gap_behind"),
    )
    return pool.select("season", "gsis_id").join(
        teamed, on=["season", "gsis_id"], how="left"
    )


# ----------------------------------------------------------------------- main

def build() -> pl.DataFrame:
    pool = build_pool()
    out = (
        pool.select("season", "gsis_id", "name", "position", "team")
        .join(coach_features(pool), on=["season", "gsis_id"], how="left")
        .join(qb_rush_features(pool), on=["season", "gsis_id"], how="left")
        .join(ol_features(pool), on=["season", "gsis_id"], how="left")
        .join(adp_gap_features(pool), on=["season", "gsis_id"], how="left")
        .sort("season", "gsis_id")
    )
    assert out.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return out


FEATS = ["coach_pass_oe", "coach_pass_shift", "qb_rush_ypg", "expqb_rush_wrte",
         "ol_sack_rate_prior", "ol_stuff_rate_prior",
         "adp_gap_ahead", "adp_gap_behind"]


def coverage_report(out: pl.DataFrame) -> None:
    for label, df in [
        ("2012-2017 (tune)", out.filter(pl.col("season").is_between(2012, 2017))),
        ("2018-2025 (test)", out.filter(pl.col("season").is_between(2018, 2025))),
        ("2026", out.filter(pl.col("season") == 2026)),
    ]:
        print(f"\n== coverage {label} (n={df.height}) ==")
        for c in FEATS:
            nn = 1.0 - df[c].null_count() / df.height
            nz = (df[c].fill_null(0.0) != 0).sum() / df.height
            print(f"  {c:22s} non-null {nn:5.1%}  nonzero {nz:5.1%}  "
                  f"mean {df[c].cast(pl.Float64).mean():+8.4f}")


def sanity(out: pl.DataFrame) -> None:
    def show(season, sub, cols):
        row = out.filter((pl.col("season") == season) & pl.col("name").str.contains(sub))
        print(f"\n{season} {sub}:")
        print(row.select(["name", "position", "team"] + cols))

    show(2026, "Lamar Jackson", ["qb_rush_ypg"])
    show(2026, "Jayden Daniels", ["qb_rush_ypg"])
    show(2026, "Zay Flowers", ["expqb_rush_wrte"])          # BAL: Lamar's rush ypg
    show(2026, "Gibbs", ["adp_gap_ahead", "adp_gap_behind"])  # DET committee
    show(2026, "Bijan", ["adp_gap_ahead", "adp_gap_behind"])  # bell cow
    show(2016, "Jordan Matthews", ["coach_pass_oe", "coach_pass_shift"])  # PHI: Pederson
    show(2020, "Amari Cooper", ["coach_pass_oe", "coach_pass_shift"])     # DAL: McCarthy
    # extremes for eyeballing
    top = out.filter((pl.col("season") == 2026) & (pl.col("position") == "QB")).sort(
        "qb_rush_ypg", descending=True).head(5)
    print("\n2026 top-5 qb_rush_ypg:")
    print(top.select("name", "team", "qb_rush_ypg"))


def main() -> None:
    out = build()
    out.write_parquet(PROC / "candidate_features.parquet")
    print(f"wrote {out.height} rows, seasons {out['season'].min()}-{out['season'].max()}")
    coverage_report(out)
    sanity(out)


if __name__ == "__main__":
    main()
