"""context_residual: v2 of market_residual — same ridge-residual architecture,
ECR-primary market anchor, old features PLUS preseason context features.

Score(t, player) = market-implied expectation of log1p(season PPR pts)
                 + w * ridge-predicted residual,
then converted to PREDICTED VORP via the leakage-safe historical
points-by-positional-rank curve (bff.vorp, seasons < t only).

Market anchor: when preseason ECR exists for the season (2021+), blend
0.7 * log(ecr_rank) + 0.3 * log(adp_rank); otherwise log(adp_rank) alone.
The 70/30 ECR-primary weight is fixed A PRIORI, not tuned: pre-2015 tuning
seasons have no ECR (so the weight cannot be tuned walk-forward without
touching eval seasons), and standalone ECR beats standalone ADP on every
held-out VORP metric we have (ecr_vorp 0.5298 vs adp_vorp 0.5006 mean
spearman_vorp, 2021-2025) — ECR is compiled by analysts closer to the season
and reacts to preseason news faster than crowd ADP, so it gets the majority
weight; ADP keeps 30% because the crowd retains real information (cost side)
and full 100% ECR would discard it.

Features: v1 set (age curve, ppg mismatch, games missed, rookie + pedigree,
TD share, team change, pedigree) + ALL context features from
data/processed/context_features.parquet + three position interactions:
qb_quality_delta x WR/TE, vacated_target_share x WR/TE,
vacated_carry_share x RB.

Leakage: for target season t only seasons < t stats/outcomes, season-t
preseason facts (ADP, ECR, April draft, offseason rosters, week-1 coaches)
and static attributes. Hyperparams tuned ONLY on walk-forward seasons
2012-2014.

Run eval:  uv run python -m bff.models.context_residual
           -> data/processed/preds_context_residual.parquet (2015-2025, score = pred VORP)
Run 2026:  uv run python -m bff.models.context_residual --season 2026
           -> data/processed/preds_context_residual_2026.parquet
Eval:      uv run python -m bff.backtest data/processed/preds_context_residual.parquet --name context_residual
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bff.backtest import POSITIONS, REPL_RANKS, build_pool
from bff.models.market_residual import (
    FEATURES as V1_FEATURES,
    FIRST_TARGET,
    PROC,
    build_dataset,
    implied_expectation,
)
from bff.vorp import build_curve, curve_at, pool_actual_ranks

OUT_EVAL = PROC / "preds_context_residual.parquet"
OUT_2026 = PROC / "preds_context_residual_2026.parquet"
EVAL_SEASONS = range(2015, 2026)

CTX_FEATURES = [
    "team_missing",
    "vacated_target_share", "vacated_carry_share", "vacated_rec_fp_share",
    "arriving_vet_usage", "draft_competition",
    "qb_change", "qb_quality_delta", "qb_rookie", "qb_expected_missing",
    "coach_change",
    "team_fp_prior", "team_fp_prior_z", "team_pass_fp_share_prior",
    "team_pass_rate_prior",
    "returning_target_competition", "returning_carry_competition",
    "depth_rank_adp", "is_rookie", "is_new_team",
]
INTERACTIONS = ["qb_delta_wrte", "vacated_tgt_wrte", "vacated_carry_rb"]
ALL_FEATURES = V1_FEATURES + CTX_FEATURES + INTERACTIONS


def build_dataset_v2() -> pl.DataFrame:
    """v1 dataset, re-anchored ECR-primary (70/30), joined to context features."""
    df = build_dataset()
    # re-blend the market anchor: ECR-primary 70/30 in log-rank space
    df = df.with_columns(
        pl.when(pl.col("ecr_rank").is_not_null())
        .then(0.3 * pl.col("adp_rank").cast(pl.Float64).log()
              + 0.7 * pl.col("ecr_rank").cast(pl.Float64).log())
        .otherwise(pl.col("adp_rank").cast(pl.Float64).log())
        .alias("mkt_val")
    ).with_columns(
        pl.col("mkt_val").rank(method="ordinal").over("season").alias("mkt_rank")
    ).with_columns(pl.col("mkt_rank").cast(pl.Float64).log().alias("log_rank"))

    ctx = pl.read_parquet(PROC / "context_features.parquet").select(
        ["season", "gsis_id"] + CTX_FEATURES
    ).with_columns([pl.col(c).cast(pl.Float64) for c in CTX_FEATURES])
    df = df.join(ctx, on=["season", "gsis_id"], how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in CTX_FEATURES]
    )
    is_wrte = pl.col("position").is_in(["WR", "TE"]).cast(pl.Float64)
    is_rb = (pl.col("position") == "RB").cast(pl.Float64)
    return df.with_columns(
        (pl.col("qb_quality_delta") * is_wrte).alias("qb_delta_wrte"),
        (pl.col("vacated_target_share") * is_wrte).alias("vacated_tgt_wrte"),
        (pl.col("vacated_carry_share") * is_rb).alias("vacated_carry_rb"),
    )


def fit_predict(df: pl.DataFrame, eval_season: int, ridge_alpha: float,
                shrink: float, return_model: bool = False):
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

    feats = [f for f in ALL_FEATURES if f != "ppg_mismatch"]
    Xtr = np.column_stack([train.select(feats).to_numpy(), tr_mismatch])
    Xte = np.column_stack([test.select(feats).to_numpy(), te_mismatch])
    feat_order = feats + ["ppg_mismatch"]

    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=ridge_alpha)
    model.fit(scaler.transform(Xtr), np.clip(resid, -4.0, 4.0))
    pred_resid = model.predict(scaler.transform(Xte))

    score = te_implied + shrink * pred_resid
    preds = test.select("season", "gsis_id").with_columns(pl.Series("score", score))
    if return_model:
        return preds, model, feat_order
    return preds


def tune(df: pl.DataFrame) -> tuple[float, float]:
    """Pick (ridge_alpha, shrink) on pre-eval walk-forward seasons 2012-2014 only."""
    from scipy.stats import spearmanr
    actuals = pl.read_parquet(PROC / "actuals.parquet").select(
        "gsis_id", pl.col("season").cast(pl.Int64), "pts_ppr"
    )
    best, best_val = (10.0, 0.5), -np.inf
    for alpha in (3.0, 10.0, 30.0, 100.0, 300.0):
        for w in (0.3, 0.5, 0.7, 1.0):
            vals = []
            for s in (2012, 2013, 2014):
                preds = fit_predict(df, s, alpha, w)
                pool = df.filter((pl.col("season") == s) & (pl.col("adp_rank") <= 150))
                j = pool.select("gsis_id").join(preds, on="gsis_id").join(
                    actuals.filter(pl.col("season") == s), on="gsis_id", how="left"
                ).with_columns(pl.col("pts_ppr").fill_null(0.0))
                vals.append(spearmanr(j["score"].to_numpy(), j["pts_ppr"].to_numpy()).statistic)
            m = float(np.mean(vals))
            if m > best_val:
                best_val, best = m, (alpha, w)
    print(f"tuned: alpha={best[0]}, shrink={best[1]}, pre2015 spearman={best_val:.4f}")
    return best


def to_vorp(preds: pl.DataFrame, season: int, adp: pl.DataFrame, ecr: pl.DataFrame,
            hist: pl.DataFrame) -> pl.DataFrame:
    """Ordinal scores -> predicted VORP via CURVE(season) built from seasons < season."""
    curves = build_curve(hist, season)
    assert all(pos in curves for pos in POSITIONS), f"no prior curve for {season}"
    if season == 2026:
        # full 2026 ADP pool (matches rank_2026's 185-player pool)
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


def report_context_coefs(df: pl.DataFrame, alpha: float) -> dict[str, float]:
    """Standardized ridge coefficients from a full-history fit (targets 2011-2025)."""
    _, model, feat_order = fit_predict(df, 2026, alpha, 1.0, return_model=True)
    coefs = dict(zip(feat_order, model.coef_))
    print("\n=== standardized ridge coefficients (full-history fit, targets 2011-2025) ===")
    for f in feat_order:
        tag = " [ctx]" if f in CTX_FEATURES + INTERACTIONS else ""
        print(f"  {f:32s} {coefs[f]:+.4f}{tag}")
    return {f: float(coefs[f]) for f in CTX_FEATURES + INTERACTIONS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None,
                    help="score a single future season (e.g. 2026) instead of eval seasons")
    args = ap.parse_args()

    df = build_dataset_v2()
    alpha, shrink = tune(df)

    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_actual_ranks(adp, ecr, actuals)

    if args.season is not None:
        preds = fit_predict(df, args.season, alpha, shrink)
        vorp = to_vorp(preds, args.season, adp, ecr, hist)
        vorp.write_parquet(OUT_2026)
        print(f"wrote {OUT_2026} ({vorp.height} rows, season {args.season})")
    else:
        frames = []
        for t in EVAL_SEASONS:
            preds = fit_predict(df, t, alpha, shrink)
            frames.append(to_vorp(preds, t, adp, ecr, hist))
        out = pl.concat(frames)
        out.write_parquet(OUT_EVAL)
        print(f"wrote {OUT_EVAL} ({out.height} rows, seasons "
              f"{out['season'].min()}-{out['season'].max()})")

    report_context_coefs(df, alpha)


if __name__ == "__main__":
    main()
