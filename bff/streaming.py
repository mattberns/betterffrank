"""Streaming replacement baselines — the derivation behind the QB and TE
entries of REPL_RANKS (`bff/backtest.py`).

Philosophy: a position's VORP replacement level should reflect how you actually
FILL that slot when you punt it in the draft.

- **QB and TE are STREAMABLE** in a 1-QB/1-TE league: a manager who drafts none
  starts the best available (undrafted) QB/TE each week off waivers. So their
  replacement is NOT the last starter's season total (VOLS = QB12 / TE12); it
  is what a streamer accumulates. This module simulates that and expresses it as
  an equivalent positional rank, which sets REPL_RANKS["QB"] and ["TE"].
- **RB and WR are NOT streamable**: you cannot stream a startable RB/WR off
  waivers, positional scarcity is real, so their replacement stays at
  roster-demand depth (RB30 / WR36) and this module does not touch them.

Streaming policy (no hindsight): 'owned' = the top N_OWNED players at the
position by preseason ADP. Each REG week, among the FREE (undrafted) players who
played, start the one with the best prior-weeks form (week 1: best ADP), and
receive his ACTUAL score. Sum over the season -> streaming total -> the
positional rank whose season total it matches.

Results (tune window 2012-2017, noisy and N_OWNED-dependent):
- QB: streaming is EFFECTIVE -> replacement ~QB6-9, well above QB12. Shipped a
  fixed **QB8** (central estimate). Deflated an over-valued QB block.
- TE: the sim's central estimate is ~TE5-6 (this form-streaming policy is
  optimistic for TE — it hoards breakout TEs a real manager couldn't keep, and
  the thin TE pool makes that worse). The TE drafted-slot curve also plateaus
  at TE10-12, so anything shallower than ~TE8 barely moves the board. Shipped a
  conservative fixed **TE8**: it nudges the model's ECR edge up on the tune
  window (+0.0483 -> +0.0570) with a negligible board change (elite TEs stay
  roughly put), deliberately NOT the aggressive TE5-6 that would under-value
  genuinely scarce elite TEs. TE scarcity is more real than QB scarcity.

A fixed value ships (not a jumpy per-season one); each was gated on the tune
window (held the model's ECR edge, kept a positive ADP edge) before a single
test look. This module is a derivation harness — it prints, writes nothing, and
is never part of the scored pipeline (cf. select_features).

    uv run python -m bff.streaming
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW_STATS = ROOT / "data" / "raw" / "stats"

TUNE_SEASONS = range(2013, 2018)   # derivation is tune-window only, never test
REG_WEEKS = 17
# streamable positions -> the N_OWNED grid to sweep (rostered players at the
# position; QB leagues carry more backups than TE, hence a deeper QB grid)
STREAM_POSITIONS = {"QB": (14, 18, 24), "TE": (12, 14, 18)}


def _weekly(season: int, pos: str) -> pl.DataFrame:
    f = (RAW_STATS / "stats_player_week_2025.parquet" if season == 2025
         else RAW_STATS / f"player_stats_{season}.parquet")
    d = pl.read_parquet(f)
    return (
        d.filter((pl.col("position") == pos) & (pl.col("season_type") == "REG")
                 & (pl.col("week") <= REG_WEEKS))
        .select(pl.col("player_id").alias("gsis_id"), "week",
                pl.col("fantasy_points_ppr").alias("pts"))
    )


def _pos_adp_rank(adp: pl.DataFrame, season: int, pos: str) -> dict[str, int]:
    q = (adp.filter((pl.col("season") == season) & (pl.col("position") == pos)
                    & pl.col("gsis_id").is_not_null())
         .sort("adp_rank").with_columns(pl.arange(1, pl.len() + 1).alias("pr")))
    return dict(zip(q["gsis_id"].to_list(), q["pr"].to_list()))


def stream_total(adp: pl.DataFrame, season: int, pos: str, n_owned: int) -> float:
    """Season points a position-punting manager accumulates by form-streaming
    the free pool (undrafted = ADP positional rank > n_owned)."""
    wk = _weekly(season, pos)
    ar = _pos_adp_rank(adp, season, pos)
    owned = [g for g, r in ar.items() if r <= n_owned]
    free = wk.filter(~pl.col("gsis_id").is_in(owned))
    total = 0.0
    for w in sorted(free["week"].unique().to_list()):
        cur = free.filter(pl.col("week") == w)
        if cur.height == 0:
            continue
        prior = (free.filter(pl.col("week") < w).group_by("gsis_id")
                 .agg(pl.col("pts").mean().alias("form")))
        cand = (cur.join(prior, on="gsis_id", how="left")
                .with_columns(pl.col("gsis_id").replace_strict(ar, default=999).alias("pr"))
                .sort(["form", "pr"], descending=[True, False], nulls_last=True))
        total += float(cand["pts"][0])
    return total


def equiv_rank(adp: pl.DataFrame, season: int, pos: str, total: float) -> int:
    """The positional rank whose season total the streaming total matches."""
    tot = (_weekly(season, pos).group_by("gsis_id").agg(pl.col("pts").sum())
           .sort("pts", descending=True)["pts"].to_numpy())
    return int(np.sum(tot >= total)) + 1


def main() -> None:
    adp = pl.read_parquet(PROC / "adp.parquet")
    for pos, grid in STREAM_POSITIONS.items():
        print(f"\n{pos} streaming baseline — equivalent {pos} replacement rank, "
              f"tune window")
        print(f"{'season':>6s} | " + "  ".join(f"N_owned={n}" for n in grid))
        agg = {n: [] for n in grid}
        for s in TUNE_SEASONS:
            line = f"{s:>6d} | "
            for n in grid:
                r = equiv_rank(adp, s, pos, stream_total(adp, s, pos, n))
                agg[n].append(r)
                line += f"   {pos}{r:>2d}   "
            print(line)
        print(" mean equiv rank:  " + "  ".join(
            f"N{n}: {pos}{np.mean(agg[n]):.1f}" for n in grid))
    print("\nShipped: REPL_RANKS QB=8, TE=8 (streaming-aware; see REPORT §1). "
          "RB=30/WR=36 stay roster-demand depth (not streamable).")


if __name__ == "__main__":
    main()
