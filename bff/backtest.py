"""Canonical evaluation harness.

POOL(t): players in season t's adp.parquet with adp_rank<=150, non-null gsis_id,
positions QB/RB/WR/TE. If a season has no ADP rows, falls back to ecr.parquet
ecr_rank<=150 (recorded in output as pool_source='ecr').

RANKING: parquet with cols (season, gsis_id, score); higher score = better.
Rank within POOL(t) by score desc; pool players absent from the ranking go to
the bottom, ordered by adp_rank (or ecr_rank for ECR-fallback pools).

ACTUAL(t): pts_ppr from actuals.parquet; missing actuals -> 0.0.

Usage:
    uv run python -m bff.backtest <preds.parquet> --name <name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

POSITIONS = ("QB", "RB", "WR", "TE")
MEAN_SEASONS = range(2018, 2026)  # 2018-2025 inclusive (test set, S=8)

# replacement rank per position (1QB league): pool player at this actual-points
# rank within position defines replacement level for VORP
REPL_RANKS = {"QB": 12, "RB": 30, "WR": 36, "TE": 12}

METRICS = [
    "spearman", "ndcg100", "top24_hit", "top50_hit",
    "spearman_vorp", "ndcg100_vorp", "top24_hit_vorp",
    "spearman_QB", "spearman_RB", "spearman_WR", "spearman_TE",
]


def build_pool(season: int, adp: pl.DataFrame, ecr: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Return (pool_df with cols gsis_id, position, tiebreak_rank; source)."""
    a = adp.filter(
        (pl.col("season") == season)
        & (pl.col("adp_rank") <= 150)
        & pl.col("gsis_id").is_not_null()
        & pl.col("position").is_in(POSITIONS)
    )
    if a.height > 0:
        pool = a.select(
            "gsis_id", "position", pl.col("adp_rank").alias("tiebreak_rank")
        )
        src = "adp"
    else:
        e = ecr.filter(
            (pl.col("season") == season)
            & (pl.col("ecr_rank") <= 150)
            & pl.col("gsis_id").is_not_null()
            & pl.col("position").is_in(POSITIONS)
        )
        pool = e.select(
            "gsis_id", "position", pl.col("ecr_rank").alias("tiebreak_rank")
        )
        src = "ecr"
    # dedupe in case two rows mapped to the same gsis_id: keep best rank
    pool = pool.sort("tiebreak_rank").unique(subset=["gsis_id"], keep="first")
    return pool.sort("tiebreak_rank"), src


def eval_season(pool: pl.DataFrame, preds: pl.DataFrame, actuals: pl.DataFrame,
                season: int) -> dict[str, float]:
    p = preds.filter(pl.col("season") == season).select("gsis_id", "score")
    # dedupe preds per player: keep max score
    p = p.group_by("gsis_id").agg(pl.col("score").max())

    df = pool.join(p, on="gsis_id", how="left")
    act = actuals.filter(pl.col("season") == season).select(
        "gsis_id", pl.col("pts_ppr").alias("actual")
    )
    df = df.join(act, on="gsis_id", how="left").with_columns(
        pl.col("actual").fill_null(0.0)
    )
    # order: scored players by score desc, then unscored by tiebreak_rank asc
    df = df.with_columns(pl.col("score").is_null().alias("_unscored")).sort(
        ["_unscored", "score", "tiebreak_rank"], descending=[False, True, False]
    ).with_columns(pl.arange(1, pl.len() + 1).alias("pred_rank"))

    n = df.height
    pred_rank = df["pred_rank"].to_numpy().astype(float)
    actual = df["actual"].to_numpy().astype(float)

    out: dict[str, float] = {}
    out["spearman"] = float(spearmanr(-pred_rank, actual).statistic)
    gains = np.clip(actual, 0.0, None)
    out["ndcg100"] = float(
        ndcg_score(gains.reshape(1, -1), (-pred_rank).reshape(1, -1), k=100)
    )
    actual_order = np.argsort(-actual, kind="stable")
    for k, key in ((24, "top24_hit"), (50, "top50_hit")):
        true_top = set(df["gsis_id"].to_numpy()[actual_order[:k]])
        list_top = set(df.sort("pred_rank")["gsis_id"].head(k).to_list())
        out[key] = len(true_top & list_top) / min(k, n)

    # --- VORP metrics: value over positional replacement (ground truth, same season) ---
    # replacement level = actual pts_ppr of the REPL_RANKS[pos]-th best POOL player
    # at that position (last one if fewer)
    positions = df["position"].to_numpy()
    repl = {}
    for pos in POSITIONS:
        pts = np.sort(actual[positions == pos])[::-1]
        if len(pts) == 0:
            repl[pos] = 0.0
        else:
            repl[pos] = float(pts[min(REPL_RANKS[pos], len(pts)) - 1])
    vorp = actual - np.array([repl[p] for p in positions])
    out["spearman_vorp"] = float(spearmanr(-pred_rank, vorp).statistic)
    vorp_gains = vorp - vorp.min()  # shift to >=0 for ndcg
    out["ndcg100_vorp"] = float(
        ndcg_score(vorp_gains.reshape(1, -1), (-pred_rank).reshape(1, -1), k=100)
    )
    vorp_order = np.argsort(-vorp, kind="stable")
    true_top24_v = set(df["gsis_id"].to_numpy()[vorp_order[:24]])
    list_top24 = set(df.sort("pred_rank")["gsis_id"].head(24).to_list())
    out["top24_hit_vorp"] = len(true_top24_v & list_top24) / min(24, n)

    for pos in POSITIONS:
        sub = df.filter(pl.col("position") == pos)
        key = f"spearman_{pos}"
        if sub.height >= 10:
            out[key] = float(
                spearmanr(-sub["pred_rank"].to_numpy(), sub["actual"].to_numpy()).statistic
            )
        else:
            out[key] = float("nan")
    return out


def run(preds_path: str, name: str) -> pl.DataFrame:
    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    preds = pl.read_parquet(preds_path)

    actual_seasons = set(actuals["season"].unique().to_list())
    pred_seasons = sorted(set(preds["season"].unique().to_list()) & actual_seasons)
    ecr_seasons = set(ecr["season"].unique().to_list())

    rows = []
    fallback_seasons = []
    for season in pred_seasons:
        pool, src = build_pool(season, adp, ecr)
        if pool.height == 0:
            continue
        if src == "ecr":
            fallback_seasons.append(season)
        m = eval_season(pool, preds, actuals, season)
        m["pool_n"] = pool.height
        for metric, value in m.items():
            rows.append({"season": str(season), "metric": metric, "value": value})

    scores = pl.DataFrame(rows)

    def mean_block(label: str, seasons: list[int]) -> list[dict]:
        sub = scores.filter(pl.col("season").is_in([str(s) for s in seasons]))
        out = []
        for metric in METRICS:
            vals = sub.filter(pl.col("metric") == metric)["value"].drop_nans().drop_nulls()
            if len(vals):
                out.append({"season": label, "metric": metric, "value": float(vals.mean())})
        return out

    mean_seasons = [s for s in pred_seasons if s in MEAN_SEASONS]
    ecr_era = [s for s in pred_seasons if s in ecr_seasons]
    extra = mean_block("MEAN", mean_seasons) + mean_block("MEAN_ECR_ERA", ecr_era)
    scores = pl.concat([scores, pl.DataFrame(extra)]) if extra else scores

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS / f"scores_{name}.csv"
    scores.write_csv(out_csv)

    # print per-season table
    wide = (
        scores.filter(pl.col("metric").is_in(METRICS + ["pool_n"]))
        .pivot(on="metric", index="season", values="value")
    )
    print(f"\n=== {name} === (rows: {out_csv})")
    if fallback_seasons:
        print(f"ECR fallback pool used for seasons: {fallback_seasons}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=200):
        print(wide)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest a fantasy ranking.")
    ap.add_argument("preds", help="parquet with cols (season, gsis_id, score)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    run(args.preds, args.name)


if __name__ == "__main__":
    main()
