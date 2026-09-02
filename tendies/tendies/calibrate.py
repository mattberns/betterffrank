"""Is a survival probability of 70% actually right 70% of the time?

This is the product's central claim. The page tells you "Puka Nacua: 61%
available at your next turn" and you decide whether to wait on that basis, so
the number has to mean what it says. Accuracy of the pick model is upstream
plumbing; THIS is the number the user acts on.

Method: replay each historical draft, and at a sample of picks run the forward
simulation to get P(available) for every board player, then check against what
actually happened over the intervening picks. Bin the predictions and compare
predicted to realised.

Two things make the statistics non-obvious and both are handled:

* Within one pick, the ~40 (prediction, outcome) pairs are strongly dependent —
  exactly one player is taken next. So binomial confidence intervals are wrong;
  the bootstrap resamples whole (season, pick) occasions instead.
* Survival depends heavily on the horizon (1 pick at the turn, 18 at the wrap),
  so reliability is reported stratified by horizon as well as pooled.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import predict as pred
from . import simulate as sim
from .autopick import AutoChain
from .replay import Board, DraftState, replay_season
from .train import Fitted

N_BINS = 10
WATCH_TOP = 60


def collect(
    board: Board,
    cfg,
    picks: pl.DataFrame,
    f: Fitted,
    chain: AutoChain | None,
    n_sims: int = 200,
    every: int = 5,
    seed: int = 20482930,
) -> pl.DataFrame:
    """(predicted, actual) survival pairs for one season.

    `every` subsamples pick occasions: the simulation is expensive in Python
    and neighbouring picks are nearly the same state, so every 5th pick gives
    independent-enough occasions at a fifth of the cost.
    """
    ordered = picks.sort("overall_pick")
    taken_at = {}
    for i, row in enumerate(ordered.iter_rows(named=True)):
        if row["pid"] is not None:
            taken_at[int(row["pid"])] = i + 1          # 1-based overall pick

    rows = []
    for i, (st, row, pid, cpos, iw) in enumerate(replay_season(board, cfg, ordered)):
        if i % every:
            continue
        seat = st.seat_on_clock()
        horizon = st.horizon(seat)
        if horizon <= 0:
            continue
        sv = sim.simulate(st, f, chain, seat=seat, n_sims=n_sims, seed=seed + i,
                          watch_top=WATCH_TOP)
        if not len(sv.pids):
            continue
        # Truth: was he still there when this seat picked again?
        my_next = st.next_pick_for(seat, after=len(st.picks) + 1)
        if my_next is None:
            continue
        for j, p in enumerate(sv.pids):
            gone = taken_at.get(int(p))
            survived = gone is None or gone >= my_next
            rows.append(
                {
                    "season": int(board.season),
                    "pick": int(st.pick_no),
                    "horizon": int(horizon),
                    "pid": int(p),
                    "p_pred": float(sv.p_available[j]),
                    "survived": bool(survived),
                }
            )
    return pl.DataFrame(rows)


def reliability(df: pl.DataFrame, bins: int = N_BINS) -> pl.DataFrame:
    """Equal-width bins of predicted probability against realised frequency."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(df["p_pred"].to_numpy(), edges[1:-1]), 0, bins - 1)
    return (
        df.with_columns(bin=pl.Series(idx))
        .group_by("bin")
        .agg(
            n=pl.len(),
            predicted=pl.col("p_pred").mean().round(4),
            actual=pl.col("survived").mean().round(4),
        )
        .sort("bin")
        .with_columns(gap=(pl.col("actual") - pl.col("predicted")).round(4))
    )


def metrics(df: pl.DataFrame, bins: int = N_BINS) -> dict:
    rel = reliability(df, bins)
    n = df.height
    p = df["p_pred"].to_numpy()
    y = df["survived"].to_numpy().astype(float)
    ece = float((rel["n"] / n * rel["gap"].abs()).sum())
    return {
        "n_pairs": n,
        "n_occasions": df.select(pl.struct("season", "pick").n_unique()).item(),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "brier_baseline": round(float(np.mean((y.mean() - y) ** 2)), 5),
        "ece": round(ece, 4),
        "mce": round(float(rel["gap"].abs().max()), 4),
        "mean_pred": round(float(p.mean()), 4),
        "mean_actual": round(float(y.mean()), 4),
    }


def by_horizon(df: pl.DataFrame) -> pl.DataFrame:
    """Long horizons are where a sequential model tends to over-predict
    survival, because it under-produces positional runs. Watch this table."""
    return (
        df.with_columns(
            band=pl.when(pl.col("horizon") <= 5).then(pl.lit("h1-5"))
            .when(pl.col("horizon") <= 12).then(pl.lit("h6-12"))
            .otherwise(pl.lit("h13+"))
        )
        .group_by("band")
        .agg(
            n=pl.len(),
            predicted=pl.col("p_pred").mean().round(4),
            actual=pl.col("survived").mean().round(4),
        )
        .with_columns(gap=(pl.col("actual") - pl.col("predicted")).round(4))
        .sort("band")
    )


def bootstrap_ci(df: pl.DataFrame, n_boot: int = 400, seed: int = 7) -> tuple[float, float]:
    """CI on the calibration gap, resampling whole pick occasions because the
    pairs within one occasion are dependent."""
    rng = np.random.default_rng(seed)
    occ = df.select("season", "pick").unique()
    keys = list(zip(occ["season"].to_list(), occ["pick"].to_list()))
    lookup = {k: g for k, g in df.group_by(["season", "pick"])}
    gaps = []
    for _ in range(n_boot):
        pick_idx = rng.integers(0, len(keys), len(keys))
        sub = pl.concat([lookup[keys[i]] for i in pick_idx])
        gaps.append(float(sub["survived"].mean() - sub["p_pred"].mean()))
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
