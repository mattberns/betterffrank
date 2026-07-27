"""Player-level audit dump of the model's own inputs for one season.

WHAT IT DOES. Rebuilds exactly what `bff.model --season 2026` runs -- same
`build_dataset()`, same tuned (alpha, shrink), same `fit_predict` -- and writes
the whole thing out flat with NAMES attached, one row per player:

  reports/audit_<season>.csv          every column that exists for that season
  reports/audit_<season>_columns.csv  column dictionary (group / in-model /
                                      fitted ridge coefficient / label)

Column order in the wide file: identity, market/anchor, model output, the
shipped FEATURES (raw values, in FEATURES order), the per-player ridge
contributions `c_<feature>` (log-points, already shrink-scaled), then every
remaining dataset column -- internal scratch plus the inert candidate blocks
(props, redzone, snaps, vegas, coach_scheme, ...). Nothing is dropped, so a
column missing from this file is a column the model never sees.

This is a READ-ONLY view: it writes only under reports/, never touches
data/processed, is not scored, and is not a test look.

Run: uv run python -m bff.audit [--season 2026]
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from bff.model import (
    CANDIDATE_BLOCKS, FEATURES, PROC, ROOT, build_dataset, fit_predict,
    rank_by_vorp, tune,
)
from bff.vorp import POSITIONS, build_curve, pool_market_ranks

IDENTITY = ["our_rank", "name", "position", "team", "gsis_id"]
MARKET = ["adp", "adp_rank", "ecr_rank", "mkt_val", "mkt_rank", "log_rank",
          "vs_adp", "vs_adp_log", "vs_adp_pct", "vs_adp_clip"]
OUTPUT = ["score_log_pts", "pred_vorp", "pos_rank", "delta", "anchor_rank",
          "scarcity_gain", "feature_gain"]


def _labels() -> dict[str, tuple[str, str]]:
    """(short label, description) per feature, borrowed from the site's
    FEATURE_META so the dictionary and the website say the same thing."""
    try:
        from bff.site import FEATURE_META
    except Exception:                                    # site is optional here
        return {}
    return {k: (v[1], v[2]) for k, v in FEATURE_META.items()}


def build(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = build_dataset()
    alpha, shrink, tune_mean = tune(df)
    print(f"tuned alpha={alpha:g} shrink={shrink:g} (tune mean {tune_mean:.4f})")

    preds, contrib, model, feat_order = fit_predict(
        df, season, alpha, shrink, return_contrib=True)
    if preds.height == 0:
        raise SystemExit(f"no rows for season {season} in build_dataset()")

    adp_all = pl.read_parquet(PROC / "adp.parquet")
    ecr_all = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    curves = build_curve(pool_market_ranks(adp_all, ecr_all, actuals), season)

    adp = adp_all.filter(pl.col("season") == season)
    scored = adp.join(preds.select("gsis_id", "score"), on="gsis_id", how="inner")

    # identical call sequence to bff.model.run_future_season, so our_rank /
    # pred_vorp / the delta decomposition match rankings_<season>.csv exactly
    ranked = rank_by_vorp(scored, curves, "score", "our_rank").with_columns(
        (pl.col("adp_rank") - pl.col("our_rank")).alias("delta"))
    anchor = rank_by_vorp(
        scored.with_columns((-pl.col("adp_rank").cast(pl.Float64)).alias("neg_adp")),
        curves, "neg_adp", "anchor_rank",
    ).select("gsis_id", "anchor_rank")
    ranked = ranked.join(anchor, on="gsis_id", how="left").with_columns(
        (pl.col("adp_rank") - pl.col("anchor_rank")).alias("scarcity_gain"),
        (pl.col("anchor_rank") - pl.col("our_rank")).alias("feature_gain"),
    ).select("gsis_id", "name", "our_rank", "pos_rank", "pred_vorp", "delta",
             "anchor_rank", "scarcity_gain", "feature_gain")

    wide = (
        df.filter(pl.col("season") == season)
        .join(ranked, on="gsis_id", how="left")
        .join(preds.select("gsis_id", pl.col("score").alias("score_log_pts")),
              on="gsis_id", how="left")
        .join(contrib, on="gsis_id", how="left")
        .with_columns(pl.col("adp_team").alias("team"))
    )

    contrib_cols = [f"c_{f}" for f in feat_order] + ["raw_ppg_mismatch"]
    head = [c for c in IDENTITY + MARKET + OUTPUT if c in wide.columns]
    # ppg_mismatch is derived inside fit_predict; vs_adp already sits in MARKET
    feats = [f for f in FEATURES if f in wide.columns and f not in set(head)]
    rest = [c for c in wide.columns
            if c not in set(head + feats + contrib_cols) and c != "adp_team"]
    wide = wide.select(head + feats + contrib_cols + sorted(rest)).sort(
        "our_rank", nulls_last=True)

    # dictionary: what each column is, and whether the ridge actually uses it
    coefs = dict(zip(feat_order, model.coef_)) if model is not None else {}
    candidate_of = {c: b for b, cols in CANDIDATE_BLOCKS.items() for c in cols}
    labels = _labels()
    rows = []
    for c in wide.columns:
        if c == "raw_ppg_mismatch":
            # the one in-model feature with no column in build_dataset: it is
            # derived inside fit_predict, so its raw value only exists here
            group = "feature (IN MODEL, derived in fit_predict)"
        elif c in contrib_cols:
            group = "contribution (log-points)"
        elif c in feats or c in FEATURES:
            group = ("market feature (IN MODEL)" if c in MARKET
                     else "feature (IN MODEL)")
        elif c in head:
            group = ("identity" if c in IDENTITY else
                     "market" if c in MARKET else "output")
        elif c in candidate_of:
            group = f"candidate block '{candidate_of[c]}' (NOT in model)"
        else:
            group = "internal / unused"
        base = c[2:] if c.startswith("c_") and c[2:] in coefs else c
        label, desc = labels.get(base, ("", ""))
        rows.append({
            "column": c, "group": group,
            "in_model": base in coefs,
            "ridge_coef": round(float(coefs[base]), 6) if base in coefs else None,
            "label": label, "description": desc,
        })
    return wide, pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()

    wide, dictionary = build(args.season)
    out = ROOT / "reports" / f"audit_{args.season}.csv"
    out_cols = ROOT / "reports" / f"audit_{args.season}_columns.csv"
    wide.write_csv(out, float_precision=6)
    dictionary.write_csv(out_cols)
    print(f"wrote {out} ({wide.height} rows x {wide.width} cols)")
    print(f"wrote {out_cols} ({dictionary.height} rows)")

    n_feat = dictionary.filter(pl.col("group") == "feature (IN MODEL)").height
    n_unranked = wide.filter(pl.col("our_rank").is_null()).height
    print(f"in-model features: {n_feat} raw columns "
          f"(+ ppg_mismatch derived in fit_predict) | "
          f"unranked rows (in dataset, not in the ADP-pool board): {n_unranked}")
    for pos in POSITIONS:
        sub = wide.filter(pl.col("position") == pos)
        print(f"  {pos}: {sub.height} rows")
    if n_unranked:
        with pl.Config(tbl_rows=20):
            print(wide.filter(pl.col("our_rank").is_null())
                  .select("gsis_id", "position", "adp_rank"))
    assert np.isfinite(wide["score_log_pts"].drop_nulls().to_numpy()).all()


if __name__ == "__main__":
    main()
