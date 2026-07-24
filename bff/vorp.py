"""Leakage-safe VORP (value over replacement) curve library.

The historical points-by-positional-rank curve is the mechanism that lets any
ordinal ranking be converted into a cross-position-comparable predicted-VORP
score. For season t everything is built only from seasons < t:

- ``pool_market_ranks`` ranks each season's POOL (backtest.build_pool) within
  position by PRESEASON MARKET rank (build_pool's tiebreak_rank: ECR when the
  season has preseason ECR, else ADP), and carries each player's actual
  pts_ppr (missing actuals -> 0). So the k-th row per position is the player
  the market DRAFTED at within-position slot k, tagged with what he then
  scored.
- ``build_curve(hist, t)``: for each prior season, average points at each
  within-position DRAFT slot, smooth (rolling mean 3), and enforce monotone
  non-increasing -> pos -> expected-points array. This is a DRAFTED-SLOT
  curve: expected realizable points of the player taken at slot k, NOT a
  finish-rank curve (the finish-rank version priced the hindsight order
  statistic and massively over-valued the noisy mid-TE/QB tails; fixed
  2026-07-24).
- ``curve_at(curve, rank)``: expected points at a within-position rank.

Predicted VORP for a scored player = CURVE(t)[pos][their within-pos rank]
minus CURVE(t)[pos][replacement rank]. The model (bff.model.to_vorp) and the
baselines (bff.model.write_baselines) all go through this same conversion, so
cross-model comparisons are fair. Imported as a library; no CLI.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from bff.backtest import POSITIONS, build_pool


def pool_market_ranks(adp: pl.DataFrame, ecr: pl.DataFrame,
                      actuals: pl.DataFrame) -> pl.DataFrame:
    """(season, position, pos_rank, pts): pool players ranked within position
    by PRESEASON MARKET rank (build_pool's tiebreak_rank), each carrying the
    actual pts_ppr they then scored, for every season with a pool and actuals.
    So pos_rank = k is the player DRAFTED at within-position slot k; build_curve
    averages pts over that slot -> drafted-slot expected-points curve."""
    seasons = sorted(set(adp["season"].unique()) | set(ecr["season"].unique()))
    act_seasons = set(actuals["season"].unique().to_list())
    frames = []
    for s in seasons:
        if s not in act_seasons:
            continue
        pool, _ = build_pool(s, adp, ecr)
        if pool.height == 0:
            continue
        df = (
            pool.join(
                actuals.filter(pl.col("season") == s).select("gsis_id", "pts_ppr"),
                on="gsis_id", how="left",
            )
            .with_columns(pl.col("pts_ppr").fill_null(0.0).alias("pts"))
            .with_columns(
                pl.col("tiebreak_rank").rank(method="ordinal")
                .over("position").alias("pos_rank"),
                pl.lit(s).cast(pl.Int64).alias("season"),
            )
            .select("season", "position", "pos_rank", "pts")
        )
        frames.append(df)
    return pl.concat(frames)


def build_curve(hist: pl.DataFrame, t: int) -> dict[str, np.ndarray]:
    """CURVE(t): pos -> array of expected pts at within-position DRAFT slot
    1..n (avg actual points of the market's slot-k pick), from seasons < t
    only. Smoothed (rolling mean 3) and monotone non-increasing."""
    prior = hist.filter(pl.col("season") < t)
    if prior.height == 0:
        return {}
    curves: dict[str, np.ndarray] = {}
    for pos in POSITIONS:
        sub = (
            prior.filter(pl.col("position") == pos)
            .group_by("pos_rank").agg(pl.col("pts").mean())
            .sort("pos_rank")
        )
        if sub.height == 0:
            continue
        # dense rank grid 1..max (fill any gaps by interpolation)
        max_r = int(sub["pos_rank"].max())
        grid = np.arange(1, max_r + 1)
        vals = np.interp(grid, sub["pos_rank"].to_numpy(), sub["pts"].to_numpy())
        # rolling mean window 3 (edges keep smaller windows)
        sm = np.array([
            vals[max(0, i - 1): i + 2].mean() for i in range(len(vals))
        ])
        # enforce monotone non-increasing
        sm = np.minimum.accumulate(sm)
        curves[pos] = sm
    return curves


def curve_at(curve: np.ndarray, rank: int) -> float:
    return float(curve[min(rank, len(curve)) - 1])
