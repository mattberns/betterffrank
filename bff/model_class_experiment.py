"""Model-class experiment: GBT and Random Forest residual heads vs the ridge.

TEMP SPACE — this module changes nothing in bff.model. It swaps ONLY the
residual regressor inside the walk-forward fit (anchor, implied expectation,
residual clip, shrink blending, VORP conversion all identical) and scores
each candidate on the 2012-2017 tune window through the canonical
to_vorp -> backtest.eval_season pipeline.

Hypothesis: three feature rounds show the remaining signal is in feature
INTERACTIONS a linear ridge cannot express (age x injury x position, draft
capital x vacated volume, ...). Trees capture those natively.

PRE-REGISTERED protocol (frozen before any run; do not widen after seeing
results):
  - Grids:
      GBT (HistGradientBoostingRegressor, random_state=0):
        learning_rate in {0.03, 0.1} x max_depth in {2, 3}
        x max_iter in {100, 300}                        -> 8 cells
      RF  (RandomForestRegressor, n_estimators=300, random_state=0):
        min_samples_leaf in {5, 20} x max_depth in {4, 8, None} -> 6 cells
      each crossed with shrink in {0.3, 0.5, 0.7, 1.0}
  - Selection: mean spearman_vorp over TUNE_SEASONS (2012-2017), same
    frozen procedure as bff.model.tune.
  - Gate: the best challenger must beat the ridge baseline tune mean by
    >= +0.0020 to earn ONE test look (2018-2025). Below the gate -> the
    experiment is discarded, test set untouched.
  - Determinism: fixed random_state=0; RF single-threaded is unnecessary
    (sklearn RF is seed-deterministic); verify a repeated cell reproduces.

Usage:  uv run python -m bff.model_class_experiment
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from bff.backtest import build_pool, eval_season
from bff.model import (FEATURES, FIRST_TARGET, PROC, TUNE_SEASONS,
                       build_dataset, implied_expectation, to_vorp, tune)
from bff.vorp import pool_market_ranks

SHRINKS = (0.3, 0.5, 0.7, 1.0)

GBT_GRID = [
    dict(learning_rate=lr, max_depth=d, max_iter=n)
    for lr in (0.03, 0.1) for d in (2, 3) for n in (100, 300)
]
RF_GRID = [
    dict(min_samples_leaf=msl, max_depth=d, n_estimators=300)
    for msl in (5, 20) for d in (4, 8, None)
]


def fit_predict_tree(df: pl.DataFrame, eval_yr: int, make_model, shrink: float,
                     ) -> pl.DataFrame:
    """bff.model.fit_predict with the ridge swapped for a tree regressor.
    Same anchor, implied expectation, ppg_mismatch-last contract, residual
    clip and shrink blend. No scaler: trees are scale-invariant."""
    train = df.filter(
        (pl.col("season") >= FIRST_TARGET) & (pl.col("season") < eval_yr)
    )
    test = df.filter(pl.col("season") == eval_yr)

    tr_implied = implied_expectation(train, train)
    te_implied = implied_expectation(train, test)
    resid = train["log_pts"].to_numpy() - tr_implied

    sched_tr = np.where(train["season"].to_numpy() >= 2021, 17.0, 16.0)
    sched_te = np.where(test["season"].to_numpy() >= 2021, 17.0, 16.0)
    tr_mismatch = (train["prev_ppg"].to_numpy() - np.expm1(tr_implied) / sched_tr
                   ) * train["has_prior"].to_numpy()
    te_mismatch = (test["prev_ppg"].to_numpy() - np.expm1(te_implied) / sched_te
                   ) * test["has_prior"].to_numpy()

    feats = [f for f in FEATURES if f != "ppg_mismatch"]
    Xtr = np.column_stack([train.select(feats).to_numpy(), tr_mismatch])
    Xte = np.column_stack([test.select(feats).to_numpy(), te_mismatch])

    model = make_model()
    model.fit(Xtr, np.clip(resid, -4.0, 4.0))
    score = te_implied + shrink * model.predict(Xte)
    return test.select("season", "gsis_id").with_columns(pl.Series("score", score))


def tune_class(df, name, grid, make_model_fn, adp, ecr, actuals, hist, pools):
    best, best_val = None, -np.inf
    for params in grid:
        for w in SHRINKS:
            vals = []
            for s in TUNE_SEASONS:
                preds = fit_predict_tree(df, s, lambda: make_model_fn(**params), w)
                vorp_preds = to_vorp(preds, s, adp, ecr, hist)
                vals.append(eval_season(pools[s], vorp_preds, actuals, s)["spearman_vorp"])
            m = float(np.mean(vals))
            if m > best_val:
                best_val, best = m, (params, w)
    p, w = best
    print(f"  {name} best: {p} shrink={w} -> tune mean {best_val:.4f}")
    return best_val, p, w


def main() -> None:
    df = build_dataset()
    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_market_ranks(adp, ecr, actuals)
    pools = {s: build_pool(s, adp, ecr)[0] for s in TUNE_SEASONS}

    _, _, ridge_val = tune(df, quiet=True)
    print(f"ridge baseline tune mean: {ridge_val:.4f}  (gate: >= {ridge_val + 0.0020:.4f})")

    def make_gbt(**p):
        return HistGradientBoostingRegressor(random_state=0, **p)

    def make_rf(**p):
        # n_jobs capped at 4: -1 oversubscribed the box against the sequential
        # grid (load ~18, no progress in 83 min). RF is seed-deterministic
        # regardless of n_jobs, so this changes runtime only, not results.
        return RandomForestRegressor(random_state=0, n_jobs=4, **p)

    gbt_val, gbt_p, gbt_w = tune_class(df, "GBT", GBT_GRID, make_gbt,
                                       adp, ecr, actuals, hist, pools)
    rf_val, rf_p, rf_w = tune_class(df, "RF ", RF_GRID, make_rf,
                                    adp, ecr, actuals, hist, pools)

    # determinism spot-check: repeat the best GBT cell, must reproduce
    s0 = TUNE_SEASONS[0]
    a = fit_predict_tree(df, s0, lambda: make_gbt(**gbt_p), gbt_w)["score"].to_numpy()
    b = fit_predict_tree(df, s0, lambda: make_gbt(**gbt_p), gbt_w)["score"].to_numpy()
    assert np.array_equal(a, b), "GBT nondeterministic!"

    print("\n=== verdict (tune window 2012-2017) ===")
    print(f"  ridge  {ridge_val:.4f}  (shipped)")
    print(f"  GBT    {gbt_val:.4f}  ({gbt_val - ridge_val:+.4f})")
    print(f"  RF     {rf_val:.4f}  ({rf_val - ridge_val:+.4f})")
    gate = ridge_val + 0.0020
    winners = [(v, n) for v, n in [(gbt_val, "GBT"), (rf_val, "RF")] if v >= gate]
    if winners:
        v, n = max(winners)
        print(f"  GATE PASSED by {n} ({v:.4f} >= {gate:.4f}) -> earns ONE test look")
    else:
        print(f"  GATE FAILED (need >= {gate:.4f}) -> discard, test set untouched")


if __name__ == "__main__":
    main()
