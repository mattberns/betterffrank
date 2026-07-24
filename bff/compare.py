"""Paired sign-flip permutation test between two prediction files.

Recomputes the metric per season for both files through the canonical
harness (bff.backtest.build_pool / eval_season, including the
unscored-to-bottom rule), takes per-season deltas d_s = metric(A) -
metric(B), and tests the mean delta with a paired sign-flip permutation:

- S <= 20 seasons: all 2^S sign vectors enumerated exactly (deterministic,
  identity flip included). p_one = P(mean(+/-d) >= mean(d)),
  p_two = P(|mean(+/-d)| >= |mean(d)|), both with a 1e-12 tolerance.
- S > 20 (not expected): 100,000 random flips, np.random.default_rng(0).

Power floors: one-sided 1/2^S, two-sided 2/2^S -- with S=8 (the 2018-2025
test set) the one-sided floor is 0.0039, so significance at 0.05 is
reachable (needs 8/8 wins, or 7/8 with margins).

Usage:
    uv run python -m bff.compare <preds_a.parquet> <preds_b.parquet>
        [--metric spearman_vorp] [--seasons 2021 2022 ...]

Default seasons: intersection of both files' seasons, 2018-2025, and
seasons with actuals.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from bff.backtest import PROC, build_pool, eval_season


def season_deltas(preds_a: pl.DataFrame, preds_b: pl.DataFrame,
                  seasons: list[int], metric: str) -> list[float]:
    adp = pl.read_parquet(PROC / "adp.parquet")
    ecr = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    deltas = []
    for s in seasons:
        pool, _ = build_pool(s, adp, ecr)
        m_a = eval_season(pool, preds_a, actuals, s)[metric]
        m_b = eval_season(pool, preds_b, actuals, s)[metric]
        d = m_a - m_b
        deltas.append(d)
        print(f"  {s}: A={m_a:.4f}  B={m_b:.4f}  delta={d:+.4f}")
    return deltas


def sign_flip_test(deltas: list[float]) -> tuple[float, float]:
    """Exact (S <= 20) or Monte-Carlo paired sign-flip test on the mean delta."""
    d = np.asarray(deltas, dtype=float)
    s = len(d)
    obs = d.mean()
    if s <= 20:
        n = 2 ** s
        signs = np.where(
            (np.arange(n)[:, None] >> np.arange(s)[None, :]) & 1, -1.0, 1.0
        )
        means = signs @ d / s
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(100_000, s))
        means = signs @ d / s
    p_one = float(np.mean(means >= obs - 1e-12))
    p_two = float(np.mean(np.abs(means) >= abs(obs) - 1e-12))
    return p_one, p_two


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired sign-flip permutation test.")
    ap.add_argument("preds_a", help="parquet with cols (season, gsis_id, score)")
    ap.add_argument("preds_b", help="parquet with cols (season, gsis_id, score)")
    ap.add_argument("--metric", default="spearman_vorp",
                    help="metric from bff.backtest (default and only decision "
                         "metric: spearman_vorp)")
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    args = ap.parse_args()

    preds_a = pl.read_parquet(args.preds_a)
    preds_b = pl.read_parquet(args.preds_b)
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    avail = (
        set(preds_a["season"].unique().to_list())
        & set(preds_b["season"].unique().to_list())
        & set(range(2018, 2026))
        & set(actuals["season"].unique().to_list())
    )
    seasons = sorted(set(args.seasons) & avail) if args.seasons else sorted(avail)
    assert seasons, "no common seasons to compare"

    print(f"A = {args.preds_a}\nB = {args.preds_b}\n"
          f"metric = {args.metric}, seasons = {seasons}")
    deltas = season_deltas(preds_a, preds_b, seasons, args.metric)
    p_one, p_two = sign_flip_test(deltas)
    s = len(deltas)
    print(f"\nmean delta (A - B) = {np.mean(deltas):+.4f} over S={s} seasons")
    print(f"sign-flip permutation: p_one = {p_one:.6g}, p_two = {p_two:.6g}")
    print(f"power floors at S={s}: one-sided {1 / 2**s:.6g}, "
          f"two-sided {2 / 2**s:.6g}")


if __name__ == "__main__":
    main()
