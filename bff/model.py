"""The shipped model: ridge residual over an ECR-first market anchor,
converted to predicted VORP.

Score(t, player) = market-implied expectation of log1p(season PPR pts)
                 + shrink * ridge-predicted residual,
then converted to PREDICTED VORP via the leakage-safe historical
points-by-positional-rank curve (bff.vorp, seasons < t only).

Market anchor: log(ecr_rank) when preseason ECR exists for the season
(2015-2025 + 2026), log(adp_rank) otherwise (pre-2015), ordinally re-ranked
per season. No blend weight. ADP enters separately as a signal:
vs_adp = log((ecr_rank+10)/(adp_rank+10)), 0.0 exactly when ECR is missing;
positive means the market drafts him earlier than the experts rank him.

Features (51): base bias set (age curve, ppg mismatch, games missed, rookie
pedigree, TD share, team change, draft pedigree) + vs_adp + 17 preseason
context features + 3 position interactions + the curated 8-feature
opportunity block + the 11-column EXP_FEATURES expansion (season-level ppg
trend, career durability, injury-report history, draft round buckets,
contract commitment) selected block-wise on the 2012-2017 tune window
(kept: trend/injury/draft_capital/contracts; rejected: redzone, snaps,
vegas, landing_spot, and the whole 2026-07-24 zero-fetch round —
coach_scheme/qb_rush/ol_proxy/adp_gap — under the streaming metric). Drops
vs the legacy feature set are justified only by feature-feature correlation
or near-constancy (no outcome data). See reports/REPORT.md for all measured
numbers.

Leakage: predictions for season t use only seasons < t stats/outcomes,
season-t PRESEASON facts (ADP, ECR, April draft, offseason rosters, week-1
coaches, contracts signed <= t) and static attributes. Hyperparameters
(ridge alpha, shrink) are tuned ONLY on walk-forward validation seasons
2012-2017 on the frozen grid, re-derived deterministically every run (no
stored params). The tuner scores spearman_vorp -- the same decision metric
used everywhere else. Test seasons 2018-2025 are never used for decisions.

Run eval:  uv run python -m bff.model
           -> data/processed/preds_model.parquet (2018-2025, score = pred VORP)
Baselines: uv run python -m bff.model --baselines
           -> data/processed/preds_adp.parquet, preds_ecr.parquet
Run 2026:  uv run python -m bff.model --season 2026
           -> data/processed/preds_model_2026.parquet,
              reports/rankings_2026.csv, reports/steals_2026.csv
Eval:      uv run python -m bff.backtest data/processed/preds_model.parquet --name model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bff.backtest import POSITIONS, REPL_RANKS, build_pool, eval_season
from bff.vorp import build_curve, curve_at, pool_market_ranks

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW_STATS = ROOT / "data" / "raw" / "stats"
CROSSWALK = ROOT / "data" / "raw" / "db_playerids.csv"
assert PROC.exists(), f"processed data dir not found: {PROC}"

# FIRST_TARGET = the first season the model may TRAIN on. It is a training-set
# boundary only; the first SCORED season is TUNE_SEASONS[0].
#
# It is 2012 because 2012 is the first season with a real preseason ECR board.
# FantasyPros' PPR cheatsheet has no Wayback capture before 2012-06-10 (checked
# via the CDX index 2026-07-25), so 2011 ECR does not exist and never will.
#
# 2011 was a training season until 2026-07-25 and was REMOVED as contaminated
# input. Do not restore it to buy back score. The argument for keeping it was
# that a training row only supplies (market rank, outcome), so an ADP anchor is
# merely a noisier instrument. That argument is WRONG FOR THIS ARCHITECTURE.
# The model is an anchor plus a learned RESIDUAL, and a residual is defined
# relative to its anchor: on 2011 rows the ridge learns "how wrong was ADP",
# and at serve time that correction is applied to an ECR anchor. ADP and ECR
# have different systematic biases -- that is the entire premise of `vs_adp`
# being a feature that carries signal -- so this is a train/serve mismatch, not
# added noise. Worse, `vs_adp` is 0.0 on those rows, which does not mean
# "unknown": it asserts that the experts and the market agreed EXACTLY, on 160
# players for whom no expert board exists. That is fabricated feature content,
# and it was 14% of the 2018 fold's training rows and ~50% of the earliest
# folds'.
#
# Removing it costs a lot and the cost is reported, not hidden: it deletes fold
# 2012 (empty train window, so the tune window is 2013-2017 / 5 folds) and the
# tune-window edge over both baselines collapses to roughly zero. See
# reports/REPORT.md for the numbers. A metric that improves when fabricated
# rows are added is not evidence those rows belong; it is evidence the edge
# was partly resting on them.
#
# Shifting the whole protocol forward (tune 2013-2018, test 2019-2025) to buy a
# sixth ECR-clean fold is likewise REJECTED, and not on a close call: it would
# move 2018 -- the model's only ADP loss, already scored eleven times -- out of
# the test set, turning the ADP record into 7-of-7 by deletion. Test seasons
# are finite; tune folds cost only compute.
FIRST_TARGET = 2012
EVAL_SEASONS = range(2018, 2026)         # test set: 2018-2025 (S=8)
TUNE_SEASONS = tuple(range(2013, 2018))  # validation: 2013-2017 (5 walk-forward folds)

OUT_EVAL = PROC / "preds_model.parquet"
OUT_2026 = PROC / "preds_model_2026.parquet"
OUT_RANK = ROOT / "reports" / "rankings_2026.csv"
OUT_STEALS = ROOT / "reports" / "steals_2026.csv"

STEAL_MIN_DELTA = 24
STEAL_MAX_ADP_RANK = 120

BASE_FEATURES = [
    "age_c", "age_c2", "age_rb", "age_qb",
    "ppg_mismatch", "games_missed", "rookie_pedigree",
    "td_share_c", "team_change", "has_prior", "draft_ovr_log",
]
MARKET_FEATURES = ["vs_adp"]
# which vs_adp transformation ships: "vs_adp_off" = log((ecr+10)/(adp+10)).
# Chosen on the 2012-2017 tune window (clip 0.4515 / log 0.4513 / off
# 0.4512 — noise-level tie; vs_adp is live in only 2 of 6 folds) and on
# principle: the raw log diff explodes mechanically at the top of RANK data
# (2026 Chase, ECR 1 / ADP 4 -> -1.39 = 5.4 sigma, which pushed ECR's #1 to
# board #17); clip keeps that gradient up to a hard kink. Alternatives
# (vs_adp_log/pct/clip) stay built in build_dataset for re-tests.
VS_ADP_VARIANT = "vs_adp_off"
CTX_FEATURES = [
    "vacated_target_share", "vacated_carry_share",
    "arriving_vet_usage", "draft_competition",
    "qb_change", "qb_quality_delta", "qb_rookie", "qb_expected_missing",
    "coach_change",
    "team_fp_prior", "team_fp_prior_z", "team_pass_fp_share_prior",
    "team_pass_rate_prior",
    "returning_target_competition", "returning_carry_competition",
    "depth_rank_adp", "is_rookie",
]
INTERACTIONS = ["qb_delta_wrte", "vacated_tgt_wrte", "vacated_carry_rb"]
OPP_FEATURES = [
    "opp_target_share", "opp_air_yards_share", "opp_ts_slope", "opp_ts_l4f4",
    "opp_cs_l6_delta", "opp_fp_oe_pg", "opp_td_per_opp_vs_pos", "opp_boom_rate",
]
# Feature expansion selected block-wise on the 2012-2017 tune window (joint
# 0.4513 vs 0.4456 baseline, +0.0057; see reports/REPORT.md). Kept blocks:
# trend, injury, contracts, draft_capital. Rejected: redzone, vegas, snaps
# (negative deltas), landing_spot (hurt the joint).
EXP_FEATURES = [
    "ppg_delta", "career_missed_rate",                                # trend
    "inj_weeks_listed_l2y", "inj_soft_tissue_l2y", "inj_recurrence",  # injury
    "apy_cap_pct", "contract_year", "rookie_deal_yr",                 # contracts
    "draft_r1", "draft_r23", "rb_early_rookie",                       # draft capital
]
# 2026-07-24 zero-fetch round: verdicts under the QB12/TE12 metric were
# coach_scheme +0.0023 (provisionally promoted), others rejected. The
# REPL_RANKS streaming change (QB8/TE8) redefined the metric BEFORE the ship
# decision, so the promotion was rolled back and all four blocks re-derived
# on the tune window under the new metric. See reports/REPORT.md §4.
SCHEME_FEATURES = ["coach_pass_oe", "coach_pass_shift"]
# 2026-07-24 zero-fetch round 2: trajectory SHIPPED as a lean 2-column block
# (tune +0.0059, the strongest new block since the original expansion). The
# other two trajectory candidates (yrs_exp r=0.80 vs age_c, career_best_ppg
# r=0.79 vs has_prior) were dead weight — dropping both RAISED the tune mean
# (0.4605 -> 0.4617), so only the two low-collinearity, genuinely-new columns
# ship. `sos` rejected (tune -0.0068: schedule is real info but noise at the
# season level, defenses regress). See reports/REPORT.md §4.
TRAJ_FEATURES = ["yrs_since_peak", "last_was_career_best"]
FEATURES = (BASE_FEATURES + MARKET_FEATURES + CTX_FEATURES + INTERACTIONS
            + OPP_FEATURES + EXP_FEATURES + TRAJ_FEATURES)

# Candidate feature blocks for tune-window selection (2012-2017 ONLY; see
# reports/REPORT.md). Columns exist in build_dataset()
# regardless; a block enters FEATURES only if it wins on the tune window.
CANDIDATE_BLOCKS: dict[str, list[str]] = {
    "redzone": ["rz_target_share", "rz_gl_carry_share", "rz_ez_target_share",
                "rz_opp_pg", "rz_td_minus_expected"],
    "snaps": ["snap_share", "snap_share_l4f4", "snap_share_max4"],
    "injury": ["inj_weeks_listed_l2y", "inj_soft_tissue_l2y", "inj_recurrence"],
    "contracts": ["apy_cap_pct", "contract_year", "rookie_deal_yr"],
    "vegas": ["vegas_wins_c", "vegas_wins_delta"],
    # rookie_log_pick dropped: |r| = 0.917 vs rookie_pedigree (collinearity
    # rule; see reports/REPORT.md)
    "draft_capital": ["draft_r1", "draft_r23", "rb_early_rookie"],
    "trend": ["ppg_delta", "career_missed_rate"],
    "landing_spot": ["rookie_x_vacated"],
    # 2026-07-24 zero-fetch candidates (bff/candidate_features.py)
    "coach_scheme": ["coach_pass_oe", "coach_pass_shift"],
    "qb_rush": ["qb_rush_ypg", "expqb_rush_wrte"],
    "ol_proxy": ["ol_sack_rate_prior", "ol_stuff_rate_prior"],
    "adp_gap": ["adp_gap_ahead", "adp_gap_behind"],
    # 2026-07-24 zero-fetch round 2 (bff/schedule_trajectory_features.py)
    "sos": ["sos_dvp", "sos_dvp_first4"],
    "trajectory": ["yrs_exp", "yrs_since_peak", "career_best_ppg",
                   "last_was_career_best"],
    # 2026-07-24 preseason player props (bff/props.py -> props_features.parquet).
    # Only markets with all six tune seasons are candidates; the three TD-leader
    # markets start in 2017 (one tune season) and comeback in 2014, so they are
    # deliberately NOT offered here. Three nested variants: the model can only
    # use what the tune window can judge.
    "props": ["prop_mvp", "prop_oroy", "prop_pass_yds", "prop_rush_yds",
              "prop_rec_yds"],
    "props_dense": ["prop_mvp", "prop_rush_yds", "prop_rec_yds"],
    "props_lean": ["prop_rush_yds", "prop_rec_yds"],
}


def load_td_shares() -> pl.DataFrame:
    """Prev-season TD points share per (season, gsis_id) from weekly data."""
    frames = []
    for yr in range(2010, 2025):
        df = pl.read_parquet(RAW_STATS / f"player_stats_{yr}.parquet").filter(
            pl.col("season_type") == "REG"
        )
        frames.append(
            df.group_by("player_id", "season").agg(
                pl.col("passing_tds").sum().alias("pass_td"),
                pl.col("rushing_tds").sum().alias("rush_td"),
                pl.col("receiving_tds").sum().alias("rec_td"),
                pl.col("fantasy_points_ppr").sum().alias("pts"),
            )
        )
    # 2025 exists only as a season-level aggregate (REG only) with the same
    # TD/pts columns; used solely as prior-year info for season-2026 scoring.
    s25 = RAW_STATS / "stats_player_reg_2025.parquet"
    if s25.exists():
        frames.append(
            pl.read_parquet(s25).select(
                "player_id", "season",
                pl.col("passing_tds").alias("pass_td"),
                pl.col("rushing_tds").alias("rush_td"),
                pl.col("receiving_tds").alias("rec_td"),
                pl.col("fantasy_points_ppr").alias("pts"),
            ).with_columns(pl.col("season").cast(pl.Int32))
        )
    frames = [f.with_columns(pl.col("season").cast(pl.Int32)) for f in frames]
    td = pl.concat(frames).with_columns(
        (
            (4.0 * pl.col("pass_td") + 6.0 * (pl.col("rush_td") + pl.col("rec_td")))
            / pl.col("pts").clip(lower_bound=1.0)
        ).alias("td_share")
    )
    return td.select(
        pl.col("player_id").alias("gsis_id"), "season",
        pl.col("td_share").clip(0.0, 1.0),
    )


def build_dataset() -> pl.DataFrame:
    """One row per (season, gsis_id) in the ADP universe: anchor, vs_adp,
    target, and every regression feature (plus internal columns like rookie)."""
    adp = pl.read_parquet(PROC / "adp.parquet").filter(
        pl.col("gsis_id").is_not_null() & pl.col("position").is_in(POSITIONS)
    )
    ecr = pl.read_parquet(PROC / "ecr.parquet").filter(
        pl.col("gsis_id").is_not_null()
    ).select("season", "gsis_id", pl.col("ecr_rank").cast(pl.Int64))
    ecr = ecr.sort("ecr_rank").unique(subset=["season", "gsis_id"], keep="first")
    actuals = pl.read_parquet(PROC / "actuals.parquet").select(
        "gsis_id", pl.col("season").cast(pl.Int64), "team", "games",
        "pts_ppr", "ppg_ppr",
    )
    td = load_td_shares().with_columns(pl.col("season").cast(pl.Int64))
    xw = pl.read_csv(CROSSWALK, null_values=["NA"]).filter(
        pl.col("gsis_id").is_not_null()
    ).select("gsis_id", "birthdate", "draft_year", "draft_ovr", "draft_round").unique(
        subset=["gsis_id"], keep="first"
    )

    df = adp.select(
        "season", "gsis_id", "position", pl.col("team").alias("adp_team"),
        "adp", "adp_rank",
    )
    df = df.join(ecr.with_columns(pl.col("season").cast(pl.Int64)),
                 on=["season", "gsis_id"], how="left")

    # anchor: ECR when it exists, ADP otherwise. No blend weight.
    df = df.with_columns(
        pl.when(pl.col("ecr_rank").is_not_null())
        .then(pl.col("ecr_rank").cast(pl.Float64).log())
        .otherwise(pl.col("adp_rank").cast(pl.Float64).log())
        .alias("mkt_val")
    ).with_columns(
        pl.col("mkt_val").rank(method="ordinal").over("season").alias("mkt_rank")
    ).with_columns(pl.col("mkt_rank").cast(pl.Float64).log().alias("log_rank"))

    # ADP as a signal, not an anchor. Candidate transformations of the same
    # expert/market disagreement (all 0.0 exactly when ECR is missing; all
    # keep the sign convention: positive = market drafts him earlier than
    # the experts rank him). The raw log-difference explodes mechanically at
    # the top of rank data (ECR 1 / ADP 4 -> -1.39, a 5-sigma outlier for a
    # 3-spot disagreement); selection among these runs on the tune window.
    has_ecr = pl.col("ecr_rank").is_not_null()
    e = pl.col("ecr_rank").cast(pl.Float64)
    a = pl.col("adp_rank").cast(pl.Float64)
    df = df.with_columns(
        pl.when(has_ecr).then(e.log() - a.log()).otherwise(0.0)
        .alias("vs_adp_log"),
        pl.when(has_ecr)
        .then((e - a) / pl.when(has_ecr).then(1).otherwise(0).sum().over("season"))
        .otherwise(0.0)
        .alias("vs_adp_pct"),
        pl.when(has_ecr).then(((e + 10.0) / (a + 10.0)).log()).otherwise(0.0)
        .alias("vs_adp_off"),
        pl.when(has_ecr).then((e.log() - a.log()).clip(-0.7, 0.7)).otherwise(0.0)
        .alias("vs_adp_clip"),
    )
    # the shipped definition (see VS_ADP_VARIANT below)
    df = df.with_columns(pl.col(VS_ADP_VARIANT).alias("vs_adp"))

    # current-season outcome (target; used only for seasons < eval season)
    df = df.join(
        actuals.select("gsis_id", "season", pl.col("pts_ppr").alias("tgt_pts")),
        on=["season", "gsis_id"], how="left",
    ).with_columns(pl.col("tgt_pts").fill_null(0.0))

    # prior-season stats
    prev = actuals.select(
        "gsis_id", (pl.col("season") + 1).alias("season"),
        pl.col("team").alias("prev_team"), pl.col("games").alias("prev_games"),
        pl.col("ppg_ppr").alias("prev_ppg"),
    )
    df = df.join(prev, on=["season", "gsis_id"], how="left")
    df = df.join(
        td.select("gsis_id", (pl.col("season") + 1).alias("season"),
                  pl.col("td_share").alias("prev_td_share")),
        on=["season", "gsis_id"], how="left",
    )

    # static attributes
    df = df.join(xw, on="gsis_id", how="left")
    df = df.with_columns(
        (
            (pl.date(pl.col("season"), 9, 1) - pl.col("birthdate").str.to_date(strict=False))
            .dt.total_days() / 365.25
        ).alias("age")
    )

    # prev-season schedule length: games_missed measures absence from the
    # PRIOR season, so the 17-game era applies from target season 2022 on
    # (season 2021's prior year, 2020, was a 16-game schedule)
    prev_sched = pl.when(pl.col("season") >= 2022).then(17).otherwise(16)
    df = df.with_columns(
        pl.col("age").fill_null(pl.col("age").median().over("position")),
        pl.col("prev_games").fill_null(0).cast(pl.Float64),
        pl.col("prev_ppg").fill_null(0.0),
        pl.col("prev_td_share").fill_null(0.0),
        (pl.col("draft_year") == pl.col("season")).fill_null(False).alias("is_rk_draft"),
        pl.col("prev_games").is_not_null().alias("_dummy"),
    ).with_columns(
        pl.col("age").fill_null(26.0),
        (pl.col("prev_ppg") > 0).cast(pl.Float64).alias("has_prior"),
    ).with_columns(
        # rookie is internal-only (r = -1.00 with has_prior); it stays a
        # dataframe column because rookie_pedigree keys off it
        (pl.col("is_rk_draft") | (pl.col("has_prior") == 0.0))
        .cast(pl.Float64).alias("rookie"),
    ).with_columns(
        (pl.col("age") - 26.0).alias("age_c"),
        ((pl.col("age") - 26.0) ** 2).alias("age_c2"),
        ((pl.col("age") - 26.0) * (pl.col("position") == "RB").cast(pl.Float64)).alias("age_rb"),
        ((pl.col("age") - 26.0) * (pl.col("position") == "QB").cast(pl.Float64)).alias("age_qb"),
        (prev_sched - pl.col("prev_games")).clip(0, 17)
        .mul(pl.col("has_prior")).alias("games_missed"),
        pl.col("draft_ovr").fill_null(270).cast(pl.Float64).log().alias("draft_ovr_log"),
        (pl.col("rookie") * (270.0 - pl.col("draft_ovr").fill_null(270).cast(pl.Float64)).clip(0, 270) / 270.0)
        .alias("rookie_pedigree"),
        (
            (pl.col("adp_team") != pl.col("prev_team")).fill_null(False)
            & (pl.col("has_prior") == 1.0)
        ).cast(pl.Float64).alias("team_change"),
        (pl.col("prev_td_share") - pl.col("prev_td_share").mean().over("position", "season"))
        .alias("td_share_c"),
        pl.col("tgt_pts").log1p().alias("log_pts"),
    )

    # preseason context features (17 kept columns) + position interactions
    ctx = pl.read_parquet(PROC / "context_features.parquet").select(
        ["season", "gsis_id"] + CTX_FEATURES
    ).with_columns([pl.col(c).cast(pl.Float64) for c in CTX_FEATURES])
    df = df.join(ctx, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in CTX_FEATURES]
    )
    is_wrte = pl.col("position").is_in(["WR", "TE"]).cast(pl.Float64)
    is_rb = (pl.col("position") == "RB").cast(pl.Float64)
    df = df.with_columns(
        (pl.col("qb_quality_delta") * is_wrte).alias("qb_delta_wrte"),
        (pl.col("vacated_target_share") * is_wrte).alias("vacated_tgt_wrte"),
        (pl.col("vacated_carry_share") * is_rb).alias("vacated_carry_rb"),
    )

    # prior-year weekly opportunity features (curated 8); nulls -> 0.0 (neutral)
    opp = pl.read_parquet(PROC / "opportunity_features.parquet").select(
        ["season", "gsis_id"] + OPP_FEATURES
    ).with_columns([pl.col(c).cast(pl.Float64) for c in OPP_FEATURES])
    df = df.join(opp, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in OPP_FEATURES]
    )

    # --- candidate blocks (2026-07-23 feature expansion; selection on the
    # 2012-2017 tune window only, see CANDIDATE_BLOCKS) ---

    # red-zone location features (t-1 pbp)
    rz_cols = CANDIDATE_BLOCKS["redzone"]
    rz = pl.read_parquet(PROC / "redzone_features.parquet").select(
        ["season", "gsis_id"] + rz_cols
    ).with_columns([pl.col(c).cast(pl.Float64) for c in rz_cols])
    df = df.join(rz, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in rz_cols]
    )

    # situational features (snaps t-1, injuries t-2..t-1, contracts <= t,
    # vegas preseason t). vegas_wins is per-season centered BEFORE the
    # zero-fill so a missing line imputes to league-average, not 0 wins.
    sit_raw = ["snap_share", "snap_share_l4f4", "snap_share_max4",
               "inj_weeks_listed_l2y", "inj_soft_tissue_l2y", "inj_recurrence",
               "apy_cap_pct", "contract_year", "rookie_deal_yr",
               "vegas_wins", "vegas_wins_delta"]
    sit = pl.read_parquet(PROC / "situation_features.parquet").select(
        ["season", "gsis_id"] + sit_raw
    ).with_columns([pl.col(c).cast(pl.Float64) for c in sit_raw])
    sit = sit.with_columns(
        (pl.col("vegas_wins") - pl.col("vegas_wins").mean().over("season"))
        .alias("vegas_wins_c")
    ).drop("vegas_wins")
    sit_cols = [c for c in sit.columns if c not in ("season", "gsis_id")]
    df = df.join(sit, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in sit_cols]
    )

    # draft-capital sharpening (static crosswalk facts)
    df = df.with_columns(
        (pl.col("rookie")
         * (np.log(271.0) - pl.col("draft_ovr").fill_null(271).cast(pl.Float64).log()))
        .alias("rookie_log_pick"),
        (pl.col("draft_round") == 1).fill_null(False).cast(pl.Float64).alias("draft_r1"),
        pl.col("draft_round").is_in([2, 3]).fill_null(False).cast(pl.Float64).alias("draft_r23"),
        (pl.col("rookie") * (pl.col("position") == "RB").cast(pl.Float64)
         * (pl.col("draft_round") <= 2).fill_null(False).cast(pl.Float64))
        .alias("rb_early_rookie"),
    )

    # season-granularity trend: t-1 ppg minus t-2 ppg (0 when either missing)
    prev2 = actuals.select(
        "gsis_id", (pl.col("season") + 2).alias("season"),
        pl.col("ppg_ppr").alias("prev2_ppg"),
    )
    df = df.join(prev2, on=["season", "gsis_id"], how="left").with_columns(
        pl.when(pl.col("has_prior") == 1.0)
        .then(pl.col("prev_ppg") - pl.col("prev2_ppg").fill_null(pl.col("prev_ppg")))
        .otherwise(0.0)
        .alias("ppg_delta")
    ).drop("prev2_ppg")

    # career missed-games rate through t-1 (seasons with an actuals row only;
    # a fully-missed season is invisible to actuals -- documented limitation)
    sched = pl.when(pl.col("season") >= 2021).then(17).otherwise(16)
    cum = (
        actuals.sort("gsis_id", "season")
        .with_columns(sched.alias("sched"))
        .with_columns(
            pl.col("games").cum_sum().over("gsis_id").alias("cum_games"),
            pl.col("sched").cum_sum().over("gsis_id").alias("cum_sched"),
        )
        .select(
            "gsis_id", pl.col("season").alias("asof_season"),
            (1.0 - pl.col("cum_games") / pl.col("cum_sched")).alias("career_missed_rate"),
        )
        .sort("gsis_id", "asof_season")
    )
    df = (
        df.with_columns((pl.col("season") - 1).alias("_prev_season"))
        .sort("gsis_id", "_prev_season")
        .join_asof(cum, left_on="_prev_season", right_on="asof_season",
                   by="gsis_id", strategy="backward")
        .drop("_prev_season", "asof_season")
        .with_columns(pl.col("career_missed_rate").fill_null(0.0))
    )

    # zero-fetch candidate blocks (2026-07-24: coach_scheme / qb_rush /
    # ol_proxy / adp_gap; built by bff/candidate_features.py from data
    # already in the repo). Zero-fill = neutral, same convention as above.
    cand_cols = (CANDIDATE_BLOCKS["coach_scheme"] + CANDIDATE_BLOCKS["qb_rush"]
                 + CANDIDATE_BLOCKS["ol_proxy"] + CANDIDATE_BLOCKS["adp_gap"])
    cand = pl.read_parquet(PROC / "candidate_features.parquet").select(
        ["season", "gsis_id"] + cand_cols
    ).with_columns([pl.col(c).cast(pl.Float64) for c in cand_cols])
    df = df.join(cand, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in cand_cols]
    )

    # zero-fetch round 2 (2026-07-24): sos + trajectory blocks, built by
    # bff/schedule_trajectory_features.py. Zero-fill neutral.
    st_cols = CANDIDATE_BLOCKS["sos"] + CANDIDATE_BLOCKS["trajectory"]
    st = pl.read_parquet(PROC / "sched_traj_features.parquet").select(
        ["season", "gsis_id"] + st_cols
    ).with_columns([pl.col(c).cast(pl.Float64) for c in st_cols])
    df = df.join(st, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in st_cols]
    )

    # preseason player-prop boards (2026-07-24; bff/props.py). Zero-fill is
    # meaningful here and NOT merely neutral: an unpriced player is one the
    # market did not treat as a contender, which is information. But a missing
    # SOURCE zero-fills identically -- and season 2026 has no boards at all, so
    # every prop column is 0 on the live board. See CLAUDE.md "Player props".
    prop_cols = sorted({c for k in ("props", "props_dense", "props_lean")
                        for c in CANDIDATE_BLOCKS[k]})
    props = pl.read_parquet(PROC / "props_features.parquet").select(
        ["season", "gsis_id"] + prop_cols
    ).with_columns([pl.col(c).cast(pl.Float64) for c in prop_cols])
    df = df.join(props, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in prop_cols]
    )

    # landing spot: rookie draft capital INTO vacated volume
    df = df.with_columns(
        (pl.col("rookie_pedigree")
         * (pl.col("vacated_target_share") + pl.col("vacated_carry_share")))
        .alias("rookie_x_vacated")
    )
    return df


def implied_expectation(train: pl.DataFrame, score_df: pl.DataFrame) -> np.ndarray:
    """Per-position OLS of log1p(pts) on [1, log_rank, log_rank^2] from train,
    applied to score_df."""
    out = np.zeros(score_df.height)
    for pos in POSITIONS:
        tr = train.filter(pl.col("position") == pos)
        X = np.column_stack([
            np.ones(tr.height),
            tr["log_rank"].to_numpy(),
            tr["log_rank"].to_numpy() ** 2,
        ])
        beta, *_ = np.linalg.lstsq(X, tr["log_pts"].to_numpy(), rcond=None)
        mask = (score_df["position"] == pos).to_numpy()
        lr = score_df["log_rank"].to_numpy()[mask]
        # clamp to the decreasing branch of the quadratic so implied value is
        # monotone non-increasing in market rank within position
        if abs(beta[2]) > 1e-12:
            vertex = -beta[1] / (2.0 * beta[2])
            lr = np.maximum(lr, vertex) if beta[2] < 0 else np.minimum(lr, vertex)
        out[mask] = beta[0] + beta[1] * lr + beta[2] * lr**2
    return out


def fit_predict(df: pl.DataFrame, eval_season: int, ridge_alpha: float,
                shrink: float, return_contrib: bool = False,
                features: list[str] | None = None):
    """Walk-forward fit on seasons < eval_season, score eval_season.

    features defaults to the shipped FEATURES list; selection experiments
    pass FEATURES + a candidate block. ppg_mismatch must be present and is
    ALWAYS moved to the last matrix column (fragility contract).

    With return_contrib=True also returns per-player ridge contributions
    (c_<feature> = shrink * scaled z-feature * coef, in log-points units),
    the fitted model, and feat_order.
    """
    if features is None:
        features = FEATURES
    assert "ppg_mismatch" in features
    train = df.filter(
        (pl.col("season") >= FIRST_TARGET) & (pl.col("season") < eval_season)
    )
    test = df.filter(pl.col("season") == eval_season)
    if test.height == 0:
        empty = pl.DataFrame(schema={"season": pl.Int64, "gsis_id": pl.Utf8,
                                     "score": pl.Float64})
        return (empty, None, None, None) if return_contrib else empty

    tr_implied = implied_expectation(train, train)
    te_implied = implied_expectation(train, test)
    resid = train["log_pts"].to_numpy() - tr_implied

    sched_tr = np.where(train["season"].to_numpy() >= 2021, 17.0, 16.0)
    sched_te = np.where(test["season"].to_numpy() >= 2021, 17.0, 16.0)
    tr_mismatch = train["prev_ppg"].to_numpy() - np.expm1(tr_implied) / sched_tr
    te_mismatch = test["prev_ppg"].to_numpy() - np.expm1(te_implied) / sched_te
    tr_mismatch *= train["has_prior"].to_numpy()
    te_mismatch *= test["has_prior"].to_numpy()

    feats = [f for f in features if f != "ppg_mismatch"]
    # contract: ppg_mismatch is ALWAYS the last matrix column
    Xtr = np.column_stack([train.select(feats).to_numpy(), tr_mismatch])
    Xte = np.column_stack([test.select(feats).to_numpy(), te_mismatch])
    feat_order = feats + ["ppg_mismatch"]

    # StandardScaler's zero-variance fallback (scale_=1) is load-bearing:
    # vs_adp has zero train variance for eval season 2012 (it trains on 2011
    # alone and ECR starts 2012), so its coefficient is fit as 0 there and it
    # self-activates from 2013. Before the 2026-07-25 Wayback backfill ECR
    # started in 2015 and this held through eval season 2015, leaving vs_adp
    # dead in 4 of the 6 tune folds.
    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=ridge_alpha)
    model.fit(scaler.transform(Xtr), np.clip(resid, -4.0, 4.0))
    Xte_scaled = scaler.transform(Xte)
    pred_resid = model.predict(Xte_scaled)

    score = te_implied + shrink * pred_resid
    preds = test.select("season", "gsis_id").with_columns(pl.Series("score", score))
    if not return_contrib:
        return preds
    contrib = shrink * Xte_scaled * model.coef_[None, :]
    cdf = test.select("gsis_id").with_columns(
        [pl.Series(f"c_{f}", contrib[:, i]) for i, f in enumerate(feat_order)]
        + [pl.Series("raw_ppg_mismatch", te_mismatch)]
    )
    return preds, cdf, model, feat_order


def tune(df: pl.DataFrame, features: list[str] | None = None,
         quiet: bool = False) -> tuple[float, float, float]:
    """Pick (ridge_alpha, shrink) on walk-forward validation seasons 2012-2017.

    Frozen grid (never expanded, even if the winner lands on an edge; edge
    landings are noted in REPORT.md). Metric: spearman_vorp -- the same
    decision metric used everywhere else, scored through the canonical
    to_vorp -> backtest.eval_season pipeline. Validation seasons all end
    <= 2017, strictly before the 2018-2025 test set. Leakage-safe:
    build_curve(hist, s) restricts to seasons < s. Deterministic; re-run
    every invocation, no stored params.
    """
    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_market_ranks(adp, ecr, actuals)
    pools = {s: build_pool(s, adp, ecr)[0] for s in TUNE_SEASONS}

    best, best_val = (10.0, 0.5), -np.inf
    for alpha in (3.0, 10.0, 30.0, 100.0, 300.0):
        for w in (0.3, 0.5, 0.7, 1.0):
            vals = []
            for s in TUNE_SEASONS:
                preds = fit_predict(df, s, alpha, w, features=features)
                vorp_preds = to_vorp(preds, s, adp, ecr, hist)
                vals.append(eval_season(pools[s], vorp_preds, actuals, s)["spearman_vorp"])
            m = float(np.mean(vals))
            if m > best_val:
                best_val, best = m, (alpha, w)
    assert best_val > -np.inf
    if not quiet:
        print(f"tuned: alpha={best[0]}, shrink={best[1]}, "
              f"{TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]} mean spearman_vorp={best_val:.4f}")
    return best[0], best[1], best_val


def to_vorp(preds: pl.DataFrame, season: int, adp: pl.DataFrame, ecr: pl.DataFrame,
            hist: pl.DataFrame) -> pl.DataFrame:
    """Ordinal scores -> predicted VORP via CURVE(season) built from seasons < season."""
    curves = build_curve(hist, season)
    assert all(pos in curves for pos in POSITIONS), f"no prior curve for {season}"
    if season == 2026:
        # full 2026 ADP pool (185 matched players)
        pool = adp.filter(
            (pl.col("season") == 2026) & pl.col("gsis_id").is_not_null()
            & pl.col("position").is_in(POSITIONS)
        ).select("gsis_id", "position", pl.col("adp_rank").alias("tiebreak_rank"))
        pool = pool.sort("tiebreak_rank").unique(subset=["gsis_id"], keep="first")
    else:
        pool, _ = build_pool(season, adp, ecr)
    df = pool.join(
        preds.filter(pl.col("season") == season).select("gsis_id", "score"),
        on="gsis_id", how="inner",
    )
    out = []
    for pos in POSITIONS:
        sub = df.filter(pl.col("position") == pos).sort(
            ["score", "tiebreak_rank"], descending=[True, False]
        )
        if sub.height == 0:
            continue
        c = curves[pos]
        repl_pts = curve_at(c, REPL_RANKS[pos])
        vorp = np.array([curve_at(c, r) - repl_pts for r in range(1, sub.height + 1)])
        out.append(sub.with_columns(pl.Series("vorp", vorp)))
    return pl.concat(out).select(
        pl.lit(season).cast(pl.Int64).alias("season"), "gsis_id",
        pl.col("vorp").alias("score"),
    )


def report_coefs(df: pl.DataFrame, alpha: float) -> None:
    """Standardized ridge coefficients from a full-history fit (targets
    2011-2025; shrink=1.0 for reporting)."""
    _, _, model, feat_order = fit_predict(df, 2026, alpha, 1.0, return_contrib=True)
    coefs = dict(zip(feat_order, model.coef_))
    print("\n=== standardized ridge coefficients (full-history fit, targets 2011-2025) ===")
    for f in feat_order:
        if f in CTX_FEATURES or f in INTERACTIONS:
            tag = " [ctx]"
        elif f in OPP_FEATURES:
            tag = " [opp]"
        elif f in MARKET_FEATURES:
            tag = " [mkt]"
        elif f in EXP_FEATURES:
            tag = " [exp]"
        else:
            tag = ""
        print(f"  {f:32s} {coefs[f]:+.4f}{tag}")


def write_baselines() -> None:
    """VORP-ordered market baselines through the same to_vorp conversion.

    preds_adp.parquet: score = -adp_rank for pool-eligible ADP rows,
    seasons 2015-2025. preds_ecr.parquet: score = -ecr_rank for seasons with
    preseason ECR (2021-2025) after the standard ECR dedupe; pool players
    without an ECR row get no pred row and are bottomed by the backtest's
    unscored rule (intended).
    """
    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_market_ranks(adp, ecr, actuals)

    frames = []
    for season in EVAL_SEASONS:
        p = adp.filter(
            (pl.col("season") == season) & pl.col("gsis_id").is_not_null()
            & pl.col("position").is_in(POSITIONS)
        ).select(
            pl.col("season").cast(pl.Int64), "gsis_id",
            (-pl.col("adp_rank").cast(pl.Float64)).alias("score"),
        )
        frames.append(to_vorp(p, season, adp, ecr, hist))
    out = pl.concat(frames)
    out.write_parquet(PROC / "preds_adp.parquet")
    print(f"wrote {PROC / 'preds_adp.parquet'} ({out.height} rows, seasons "
          f"{out['season'].min()}-{out['season'].max()})")

    e = ecr.filter(pl.col("gsis_id").is_not_null()).select(
        pl.col("season").cast(pl.Int64), "gsis_id", pl.col("ecr_rank").cast(pl.Int64)
    ).sort("ecr_rank").unique(subset=["season", "gsis_id"], keep="first")
    frames = []
    for season in sorted(set(e["season"].unique().to_list()) & set(EVAL_SEASONS)):
        p = e.filter(pl.col("season") == season).select(
            "season", "gsis_id", (-pl.col("ecr_rank").cast(pl.Float64)).alias("score")
        )
        frames.append(to_vorp(p, season, adp, ecr, hist))
    out = pl.concat(frames)
    out.write_parquet(PROC / "preds_ecr.parquet")
    print(f"wrote {PROC / 'preds_ecr.parquet'} ({out.height} rows, seasons "
          f"{out['season'].min()}-{out['season'].max()})")


def rank_by_vorp(scored: pl.DataFrame, curves: dict, score_col: str,
                 rank_name: str) -> pl.DataFrame:
    """Per-position order by score_col -> curve VORP -> overall rank."""
    out = []
    for pos in POSITIONS:
        sub = scored.filter(pl.col("position") == pos).sort(
            [score_col, "adp_rank"], descending=[True, False]
        ).with_columns(pl.arange(1, pl.len() + 1).alias("pos_rank"))
        c = curves[pos]
        repl_pts = curve_at(c, REPL_RANKS[pos])
        vorp = np.array([curve_at(c, r) - repl_pts for r in range(1, sub.height + 1)])
        out.append(sub.with_columns(pl.Series("pred_vorp", vorp)))
    return (
        pl.concat(out)
        .sort(["pred_vorp", "adp_rank"], descending=[True, False])
        .with_columns(pl.arange(1, pl.len() + 1).alias(rank_name))
    )


def reason_string(row: dict) -> str:
    """Plain-language 'why' from the biggest positive ridge contributions,
    checked against raw feature values so the phrase matches the player."""
    parts: list[tuple[float, str]] = []

    # --- base drivers ---
    age_c = sum(row[f"c_{f}"] for f in ("age_c", "age_c2", "age_rb", "age_qb"))
    if age_c > 0.01:
        side = "young for the position" if row["age_c"] < 0 else "age profile"
        parts.append((age_c, f"favorable age curve ({side})"))

    c = row["c_ppg_mismatch"]
    if c > 0.01 and row["raw_ppg_mismatch"] > 0:
        parts.append((c, "outproduced his market rank last season "
                         f"(+{row['raw_ppg_mismatch']:.1f} PPR pts/game vs market-implied)"))

    c = row["c_games_missed"]
    if c > 0.01 and row["games_missed"] <= 2:
        parts.append((c, "played a near-full season last year (durability)"))

    c = row["c_rookie_pedigree"] + row["c_is_rookie"]
    if c > 0.01 and row["rookie"] > 0:
        parts.append((c, "rookie with early-draft pedigree"))

    c = row["c_td_share_c"]
    if c > 0.01 and row["td_share_c"] < 0:
        parts.append((c, "scoring was not TD-dependent last year (low regression risk)"))

    # --- market driver: gated on the raw value, so it is inert whenever the
    # season has no ECR (vs_adp == 0.0 exactly, e.g. all of 2026) ---
    c = row["c_vs_adp"]
    if c > 0.01 and row["vs_adp"] < 0:
        parts.append((c, "experts rank him better than his draft cost"))

    # --- context drivers ---
    c = row["c_arriving_vet_usage"]
    if c > 0.01 and row["arriving_vet_usage"] <= 0.02:
        parts.append((c, "no veteran arrivals competing for his role [context]"))

    c = row["c_draft_competition"]
    if c > 0.01 and row["draft_competition"] <= 0.0:
        parts.append((c, "no April draft capital spent at his position [context]"))

    c = row["c_qb_quality_delta"] + row["c_qb_delta_wrte"]
    if c > 0.01 and row["qb_quality_delta"] > 0:
        parts.append((c, "improved expected QB play [context]"))

    c = (row["c_vacated_target_share"] + row["c_vacated_carry_share"]
         + row["c_vacated_tgt_wrte"] + row["c_vacated_carry_rb"])
    if c > 0.01 and (row["vacated_target_share"] > 0.05
                     or row["vacated_carry_share"] > 0.05):
        parts.append((c, "vacated volume from departures [context]"))

    c = (row["c_team_fp_prior"] + row["c_team_fp_prior_z"]
         + row["c_team_pass_fp_share_prior"] + row["c_team_pass_rate_prior"])
    if c > 0.01:
        parts.append((c, "team offensive environment [context]"))

    # --- opportunity drivers (curated subset) ---
    c = row["c_opp_target_share"]
    if c > 0.008 and row["opp_target_share"] >= 0.18:
        parts.append((c, f"{row['opp_target_share']*100:.0f}% target share "
                         "last season [opp]"))

    c_vel = (row["c_opp_ts_slope"] + row["c_opp_ts_l4f4"])
    if c_vel > 0.008 and row["opp_ts_l4f4"] > 0.01:
        parts.append((c_vel, "target share "
                             f"{row['opp_ts_l4f4']*100:+.0f} pts over final 4 games "
                             "vs first 4 [opp velocity]"))
    elif c_vel > 0.008 and row["opp_ts_slope"] > 0:
        parts.append((c_vel, "target share trending up through last season "
                             "[opp velocity]"))

    c = row["c_opp_cs_l6_delta"]
    if c > 0.008 and row["opp_cs_l6_delta"] > 0.01:
        parts.append((c, "carry share rising over the final 6 games [opp velocity]"))

    # fp-over-expected is structural for QBs (opportunity model excludes pass
    # attempts), so only cite it as a player-specific reason for RB/WR/TE
    c = row["c_opp_fp_oe_pg"]
    if c > 0.008 and row["opp_fp_oe_pg"] > 0 and row["position"] != "QB":
        parts.append((c, "produced over his opportunity volume "
                         f"(+{row['opp_fp_oe_pg']:.1f} PPR pts/game) [opp]"))

    c = row["c_opp_td_per_opp_vs_pos"]
    if c > 0.008 and row["opp_td_per_opp_vs_pos"] > 0:
        parts.append((c, "TD-efficient on real opportunity last season [opp]"))

    c = row["c_opp_boom_rate"]
    if c > 0.008 and 0 < row["opp_boom_rate"] < 0.25:
        parts.append((c, "steady week-to-week, not spike-week dependent [opp]"))

    parts.sort(key=lambda t: -t[0])
    why = "; ".join(p for _, p in parts[:3])
    scarcity = (f"{row['position']}{row['pos_rank']} slot priced above ADP "
                f"(+{row['scarcity_gain']} ranks from positional scarcity, "
                f"{row['feature_gain']:+d} from features)")
    return f"{why}; {scarcity}" if why else scarcity


def run_future_season(df: pl.DataFrame, season: int, alpha: float,
                      shrink: float) -> None:
    """2026 deliverables: preds parquet + rankings + steals with reasons."""
    preds, contrib, _, _ = fit_predict(df, season, alpha, shrink,
                                       return_contrib=True)

    adp_all = pl.read_parquet(PROC / "adp.parquet")
    ecr_all = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_market_ranks(adp_all, ecr_all, actuals)
    curves = build_curve(hist, season)
    assert all(pos in curves for pos in POSITIONS)

    if season not in set(ecr_all["season"].unique().to_list()):
        print(f"\nWARNING: no preseason ECR exists for {season} "
              f"(ecr.parquet covers "
              f"{ecr_all['season'].min()}-{ecr_all['season'].max()}). "
              f"The {season} anchor is ADP-only and vs_adp = 0 for every "
              "player; this list is effectively the pre-2021 model "
              "configuration, and the beat-ECR evidence does not directly "
              "certify it.\n")

    adp = adp_all.filter(pl.col("season") == season)
    scored = adp.join(preds.select("gsis_id", "score"), on="gsis_id", how="inner")

    ranked = rank_by_vorp(scored, curves, "score", "our_rank").with_columns(
        (pl.col("adp_rank") - pl.col("our_rank")).alias("delta")
    )

    # the preds parquet and the rankings CSV come from the same in-memory
    # pred_vorp array (assert exact identity pre-write)
    vorp_out = ranked.select(
        pl.lit(season).cast(pl.Int64).alias("season"), "gsis_id",
        pl.col("pred_vorp").alias("score"),
    )
    assert float((vorp_out["score"] - ranked["pred_vorp"]).abs().max()) == 0.0
    vorp_out.write_parquet(OUT_2026)
    print(f"wrote {OUT_2026} ({vorp_out.height} rows, season {season})")

    # anchor-only ranking (ADP alone) through the same curve, to decompose
    # delta into scarcity repricing vs feature residual. adp_rank stays the
    # decomposition anchor: for 2026 it IS the model anchor (no ECR).
    anchor = rank_by_vorp(
        scored.with_columns((-pl.col("adp_rank").cast(pl.Float64)).alias("neg_adp")),
        curves, "neg_adp", "anchor_rank",
    ).select("gsis_id", "anchor_rank")
    ranked = ranked.join(anchor, on="gsis_id", how="left").with_columns(
        (pl.col("adp_rank") - pl.col("anchor_rank")).alias("scarcity_gain"),
        (pl.col("anchor_rank") - pl.col("our_rank")).alias("feature_gain"),
    )

    out = ranked.select(
        "our_rank", pl.col("name").alias("player"), "position", "team",
        "adp", "adp_rank", "delta", pl.col("pred_vorp").round(4).alias("score"),
    )
    out.write_csv(OUT_RANK)
    print(f"wrote {OUT_RANK} ({out.height} rows)")

    # post-write consistency: CSV score is round(pred_vorp, 4); same row order
    csv_scores = pl.read_csv(OUT_RANK)["score"].to_numpy()
    pq_scores = pl.read_parquet(OUT_2026)["score"].to_numpy()
    assert float(np.abs(csv_scores - np.round(pq_scores, 4)).max()) < 1e-9

    # sanity gates: investigate a failure, never weaken silently
    n_qb_top15 = out.head(15).filter(pl.col("position") == "QB").height
    first_qb = out.filter(pl.col("position") == "QB")["our_rank"].min()
    top3_pos = out.head(3)["position"].to_list()
    bad_pos = out.filter(~pl.col("position").is_in(POSITIONS)).height
    print(f"gates: QBs in top 15 = {n_qb_top15} (<=2), first QB #{first_qb}, "
          f"top3 = {top3_pos}, non-QB/RB/WR/TE rows = {bad_pos}")
    assert n_qb_top15 <= 2 and bad_pos == 0
    assert all(p in ("RB", "WR") for p in top3_pos), top3_pos

    # steals
    raw_cols = ["gsis_id", "age_c", "games_missed", "rookie", "td_share_c",
                "team_change", "vs_adp", "arriving_vet_usage",
                "draft_competition", "qb_quality_delta",
                "vacated_target_share", "vacated_carry_share",
                "opp_target_share", "opp_ts_slope", "opp_ts_l4f4",
                "opp_cs_l6_delta", "opp_fp_oe_pg", "opp_td_per_opp_vs_pos",
                "opp_boom_rate"]
    feat_raw = df.filter(pl.col("season") == season).select(raw_cols)
    steals = (
        ranked.filter(
            (pl.col("adp_rank") <= STEAL_MAX_ADP_RANK)
            & (pl.col("delta") >= STEAL_MIN_DELTA)
        )
        .join(contrib, on="gsis_id", how="left")
        .join(feat_raw, on="gsis_id", how="left")
    )
    reasons = [reason_string(r) for r in steals.iter_rows(named=True)]
    steals = steals.with_columns(pl.Series("reason", reasons)).select(
        "our_rank", pl.col("name").alias("player"), "position", "team", "adp",
        "adp_rank", "delta", pl.col("pred_vorp").round(4).alias("score"), "reason",
    ).sort("our_rank")
    steals.write_csv(OUT_STEALS)
    print(f"wrote {OUT_STEALS} ({steals.height} rows)")

    with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=200):
        print(out.head(25))
        print(steals)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit/score the shipped model.")
    ap.add_argument("--season", type=int, default=None,
                    help="score a single future season (e.g. 2026) instead of eval seasons")
    ap.add_argument("--baselines", action="store_true",
                    help="write VORP-ordered ADP/ECR baseline preds and exit")
    args = ap.parse_args()

    if args.baselines:
        write_baselines()
        return

    df = build_dataset()
    alpha, shrink, _ = tune(df)

    if args.season is not None:
        run_future_season(df, args.season, alpha, shrink)
        return

    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_market_ranks(adp, ecr, actuals)
    frames = []
    for t in EVAL_SEASONS:
        preds = fit_predict(df, t, alpha, shrink)
        frames.append(to_vorp(preds, t, adp, ecr, hist))
    out = pl.concat(frames)
    out.write_parquet(OUT_EVAL)
    print(f"wrote {OUT_EVAL} ({out.height} rows, seasons "
          f"{out['season'].min()}-{out['season'].max()})")

    report_coefs(df, alpha)


if __name__ == "__main__":
    main()
