"""opp_residual: v3 — v2 context_residual architecture + weekly-opportunity features.

Score(t, player) = market-implied expectation of log1p(season PPR pts)
                 + w * ridge-predicted residual,
then converted to PREDICTED VORP via the leakage-safe historical curve
(bff.vorp, seasons < t only).

Market anchor: FROZEN at v2's — 0.7*log(ecr_rank) + 0.3*log(adp_rank) when
preseason ECR exists (2021+), log(adp_rank) alone otherwise. Not re-tuned.

Features: v2 set (v1 + context + position interactions) PLUS opportunity
features from data/processed/opportunity_features.parquet (t-1 REG weekly:
levels, opportunity-production divergence, velocity, stability, QB volume).

Feature-bloat guards (a priori, not tuned):
  * exact/near-exact linear combos dropped: opp_ppg (= opp_fp_exp_pg +
    opp_fp_oe_pg), opp_wopr (= 1.5*opp_target_share + 0.7*opp_air_yards_share),
    raw opp_ypt / opp_td_per_opp (their _vs_pos versions kept; the raw ones
    differ only by a season-position constant).
  * nulls (rookies, volume guards, games<6 velocity) filled with 0.0 =
    neutral/none; has_prior_weekly / opp_short_season / opp_has_velocity
    flags let the model separate "missing" from "zero".
Feature-bloat guards (tuned ONLY on 2012-2014 walk-forward, like alpha/shrink):
  * pre-2015-tuned feature pruning: the opp block enters as one of a few
    predeclared candidate subsets (full / curated / levels+divergence /
    velocity / divergence+velocity), chosen on 2012-2014.
  * grouped shrinkage: after standardization the opp block is scaled by
    g <= 1 before the ridge, i.e. an effectively larger alpha on the new
    block than on the proven v1/v2 features.

Hyperparameter protocol: the (alpha, shrink) grid is v2's frozen grid
(alpha in 3..300, shrink in 0.3..1.0) — not expanded — so the only new
tuned knobs are the opp subset and g. (ridge_alpha, subset, opp_scale,
shrink) tuned ONLY on walk-forward seasons 2012-2014 (weekly data starts
2010, so t-1 weekly exists for all tuning targets).

Leakage: season-t predictions use only seasons < t stats/outcomes, season-t
preseason facts, static attributes.

Run eval:  uv run python -m bff.models.opp_residual
           -> data/processed/preds_opp_residual.parquet (2015-2025, score = pred VORP)
Run 2026:  uv run python -m bff.models.opp_residual --season 2026
           -> data/processed/preds_opp_residual_2026.parquet
Eval:      uv run python -m bff.backtest data/processed/preds_opp_residual.parquet --name opp_residual
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bff.models.context_residual import (
    ALL_FEATURES as V2_FEATURES,
    build_dataset_v2,
    to_vorp,
)
from bff.models.market_residual import FIRST_TARGET, PROC, implied_expectation
from bff.vorp import pool_actual_ranks

OUT_EVAL = PROC / "preds_opp_residual.parquet"
OUT_2026 = PROC / "preds_opp_residual_2026.parquet"
EVAL_SEASONS = range(2015, 2026)

OPP_LEVELS = [
    "opp_games", "opp_target_share", "opp_air_yards_share", "opp_carry_share",
    "opp_targets_pg", "opp_carries_pg",
]
OPP_DIVERGENCE = [
    "opp_racr", "opp_ypt_vs_pos", "opp_td_per_opp_vs_pos",
    "opp_epa_per_target", "opp_epa_per_carry",
    "opp_fp_exp_pg", "opp_fp_oe_pg",
]
OPP_VELOCITY = [
    f"opp_{m}_{s}"
    for m in ("ts", "wopr", "cs", "tpg", "att")
    for s in ("slope", "l6_delta", "l4f4")
]
OPP_STABILITY = ["opp_ts_std", "opp_boom_rate"]
OPP_QB = ["opp_attempts_pg", "opp_pass_air_pg", "opp_epa_per_att"]
OPP_FLAGS = ["has_prior_weekly", "opp_short_season", "opp_has_velocity"]

OPP_FEATURES = (
    OPP_LEVELS + OPP_DIVERGENCE + OPP_VELOCITY + OPP_STABILITY + OPP_QB + OPP_FLAGS
)
# dropped as exact/near-exact linear combos of kept columns (see module docstring)
OPP_DROPPED = ["opp_ppg", "opp_wopr", "opp_ypt", "opp_td_per_opp"]

# curated a-priori hypothesis set: opportunity level, role velocity,
# production-over-opportunity regression, TD luck, spike-week ability
OPP_CURATED = [
    "opp_target_share", "opp_air_yards_share", "opp_ts_slope", "opp_ts_l4f4",
    "opp_cs_l6_delta", "opp_fp_oe_pg", "opp_td_per_opp_vs_pos", "opp_boom_rate",
    "has_prior_weekly",
]

OPP_SUBSETS: dict[str, list[str]] = {
    "full": OPP_FEATURES,
    "curated": OPP_CURATED,
    "lev+div": OPP_LEVELS + OPP_DIVERGENCE + OPP_FLAGS,
    "velocity": OPP_VELOCITY + ["opp_has_velocity", "has_prior_weekly"],
    "div+vel": OPP_DIVERGENCE + OPP_VELOCITY + OPP_FLAGS,
}

ALL_FEATURES = V2_FEATURES + OPP_FEATURES


def build_dataset_v3() -> pl.DataFrame:
    """v2 dataset joined to opportunity features; opp nulls -> 0.0 (neutral)."""
    df = build_dataset_v2()
    opp = pl.read_parquet(PROC / "opportunity_features.parquet").select(
        ["season", "gsis_id"] + OPP_FEATURES
    ).with_columns([pl.col(c).cast(pl.Float64) for c in OPP_FEATURES])
    return df.join(opp, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in OPP_FEATURES]
    )


def fit_predict(df: pl.DataFrame, eval_season: int, ridge_alpha: float,
                opp_scale: float, shrink: float, opp_feats: list[str],
                return_model: bool = False):
    train = df.filter(
        (pl.col("season") >= FIRST_TARGET) & (pl.col("season") < eval_season)
    )
    test = df.filter(pl.col("season") == eval_season)
    if test.height == 0:
        empty = pl.DataFrame(schema={"season": pl.Int64, "gsis_id": pl.Utf8,
                                     "score": pl.Float64})
        return (empty, None, None) if return_model else empty

    tr_implied = implied_expectation(train, train)
    te_implied = implied_expectation(train, test)
    resid = train["log_pts"].to_numpy() - tr_implied

    sched_tr = np.where(train["season"].to_numpy() >= 2021, 17.0, 16.0)
    sched_te = np.where(test["season"].to_numpy() >= 2021, 17.0, 16.0)
    tr_mismatch = train["prev_ppg"].to_numpy() - np.expm1(tr_implied) / sched_tr
    te_mismatch = test["prev_ppg"].to_numpy() - np.expm1(te_implied) / sched_te
    tr_mismatch *= train["has_prior"].to_numpy()
    te_mismatch *= test["has_prior"].to_numpy()

    feats = [f for f in V2_FEATURES if f != "ppg_mismatch"] + opp_feats
    Xtr = np.column_stack([train.select(feats).to_numpy(), tr_mismatch])
    Xte = np.column_stack([test.select(feats).to_numpy(), te_mismatch])
    feat_order = feats + ["ppg_mismatch"]

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    # grouped shrinkage: down-scale the opp block => stronger effective ridge
    # penalty on the new features than on the proven v1/v2 block
    opp_mask = np.array([f in opp_feats for f in feat_order])
    Xtr[:, opp_mask] *= opp_scale
    Xte[:, opp_mask] *= opp_scale

    model = Ridge(alpha=ridge_alpha)
    model.fit(Xtr, np.clip(resid, -4.0, 4.0))
    pred_resid = model.predict(Xte)

    score = te_implied + shrink * pred_resid
    preds = test.select("season", "gsis_id").with_columns(pl.Series("score", score))
    if return_model:
        return preds, model, feat_order
    return preds


def tune(df: pl.DataFrame) -> tuple[float, float, float, str]:
    """Pick (ridge_alpha, opp_scale, shrink, opp_subset) on walk-forward seasons
    2012-2014 only. Alpha/shrink grids are v2's frozen grids."""
    from scipy.stats import spearmanr
    actuals = pl.read_parquet(PROC / "actuals.parquet").select(
        "gsis_id", pl.col("season").cast(pl.Int64), "pts_ppr"
    )
    best, best_val = (100.0, 0.5, 0.5, "full"), -np.inf
    for subset, opp_feats in OPP_SUBSETS.items():
        for alpha in (3.0, 10.0, 30.0, 100.0, 300.0):
            for g in (0.25, 0.5, 1.0):
                for w in (0.3, 0.5, 0.7, 1.0):
                    vals = []
                    for s in (2012, 2013, 2014):
                        preds = fit_predict(df, s, alpha, g, w, opp_feats)
                        pool = df.filter((pl.col("season") == s)
                                         & (pl.col("adp_rank") <= 150))
                        j = pool.select("gsis_id").join(preds, on="gsis_id").join(
                            actuals.filter(pl.col("season") == s),
                            on="gsis_id", how="left",
                        ).with_columns(pl.col("pts_ppr").fill_null(0.0))
                        vals.append(spearmanr(j["score"].to_numpy(),
                                              j["pts_ppr"].to_numpy()).statistic)
                    m = float(np.mean(vals))
                    if m > best_val:
                        best_val, best = m, (alpha, g, w, subset)
    print(f"tuned: alpha={best[0]}, opp_scale={best[1]}, shrink={best[2]}, "
          f"subset={best[3]}, pre2015 spearman={best_val:.4f}")
    return best


def report_opp_coefs(df: pl.DataFrame, alpha: float, opp_scale: float,
                     opp_feats: list[str], label: str) -> None:
    """Standardized ridge coefficients from a full-history fit (targets 2011-2025).
    Opp coefficients are reported as coef * opp_scale = effect per 1 SD of the
    original (unscaled) feature."""
    _, model, feat_order = fit_predict(df, 2026, alpha, opp_scale, 1.0, opp_feats,
                                       return_model=True)
    coefs = dict(zip(feat_order, model.coef_))
    print(f"\n=== standardized ridge coefficients, {label} "
          f"(full-history fit, targets 2011-2025) ===")
    print(f"(opp block reported as coef * opp_scale={opp_scale}; "
          f"dropped as collinear: {OPP_DROPPED})")
    for f in feat_order:
        if f in opp_feats:
            tag = " [opp-vel]" if f in OPP_VELOCITY else " [opp]"
            val = coefs[f] * opp_scale
        else:
            tag, val = "", coefs[f]
        print(f"  {f:32s} {val:+.4f}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None,
                    help="score a single future season (e.g. 2026) instead of eval seasons")
    args = ap.parse_args()

    df = build_dataset_v3()
    alpha, g, shrink, subset = tune(df)
    opp_feats = OPP_SUBSETS[subset]

    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_actual_ranks(adp, ecr, actuals)

    if args.season is not None:
        preds = fit_predict(df, args.season, alpha, g, shrink, opp_feats)
        vorp = to_vorp(preds, args.season, adp, ecr, hist)
        vorp.write_parquet(OUT_2026)
        print(f"wrote {OUT_2026} ({vorp.height} rows, season {args.season})")
    else:
        frames = []
        for t in EVAL_SEASONS:
            preds = fit_predict(df, t, alpha, g, shrink, opp_feats)
            frames.append(to_vorp(preds, t, adp, ecr, hist))
        out = pl.concat(frames)
        out.write_parquet(OUT_EVAL)
        print(f"wrote {OUT_EVAL} ({out.height} rows, seasons "
              f"{out['season'].min()}-{out['season'].max()})")

    # shipped model's coefficients, then a diagnostic all-opp-features fit so
    # every opportunity feature's standardized coefficient is visible
    report_opp_coefs(df, alpha, g, opp_feats, f"SHIPPED subset={subset}")
    report_opp_coefs(df, alpha, g, OPP_FEATURES, "DIAGNOSTIC all opp features")


if __name__ == "__main__":
    main()
