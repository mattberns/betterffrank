"""QB streaming baseline — the derivation behind REPL_RANKS["QB"] = 8.

In a 1-QB/12-team league you do not roster a season's worth of one QB; you
STREAM — start the best available (undrafted) QB each week. So the replacement
level for QB is NOT the 12th-best QB's season total (VOLS = QB12); it is what a
streamer accumulates, which is much better. This module simulates that, on the
TUNE WINDOW ONLY (2012-2017), and expresses it as an equivalent QB rank.

Streaming policy (no hindsight): 'owned' = the top N_OWNED QBs by preseason
ADP. Each REG week, among the FREE (undrafted) QBs who played, start the one
with the best prior-weeks form (week 1: best ADP), and receive his ACTUAL
score. Sum over the season -> streaming total -> the QB rank whose season total
it matches.

The result is noisy and depends on N_OWNED (heavy-streaming N~14 -> ~QB6;
backup-rostering N~24 -> ~QB14), centering near QB8. We therefore SHIP A FIXED
QB8 (the central estimate) rather than a jumpy per-season value; QB8 was gated
on the tune window (it held the model's ECR edge and kept a positive ADP edge)
before the single test look. This module is a derivation harness — it prints,
writes nothing, and is never part of the scored pipeline (cf. select_features).

    uv run python -m bff.streaming
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW_STATS = ROOT / "data" / "raw" / "stats"

TUNE_SEASONS = range(2012, 2018)   # derivation is tune-window only, never test
N_OWNED_GRID = (14, 18, 24)
REG_WEEKS = 17


def _weekly_qb(season: int) -> pl.DataFrame:
    f = (RAW_STATS / "stats_player_week_2025.parquet" if season == 2025
         else RAW_STATS / f"player_stats_{season}.parquet")
    d = pl.read_parquet(f)
    return (
        d.filter((pl.col("position") == "QB") & (pl.col("season_type") == "REG")
                 & (pl.col("week") <= REG_WEEKS))
        .select(pl.col("player_id").alias("gsis_id"), "week",
                pl.col("fantasy_points_ppr").alias("pts"))
    )


def _qb_adp_rank(adp: pl.DataFrame, season: int) -> dict[str, int]:
    q = (adp.filter((pl.col("season") == season) & (pl.col("position") == "QB")
                    & pl.col("gsis_id").is_not_null())
         .sort("adp_rank").with_columns(pl.arange(1, pl.len() + 1).alias("qr")))
    return dict(zip(q["gsis_id"].to_list(), q["qr"].to_list()))


def stream_total(adp: pl.DataFrame, season: int, n_owned: int) -> float:
    """Season points a QB-punting manager accumulates by form-streaming the
    free pool (undrafted = ADP QB rank > n_owned)."""
    wk = _weekly_qb(season)
    ar = _qb_adp_rank(adp, season)
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
                .with_columns(pl.col("gsis_id").replace_strict(ar, default=999).alias("qr"))
                .sort(["form", "qr"], descending=[True, False], nulls_last=True))
        total += float(cand["pts"][0])
    return total


def equiv_rank(adp: pl.DataFrame, season: int, total: float) -> int:
    """The QB rank whose season total the streaming total matches."""
    tot = (_weekly_qb(season).group_by("gsis_id").agg(pl.col("pts").sum())
           .sort("pts", descending=True)["pts"].to_numpy())
    return int(np.sum(tot >= total)) + 1


def main() -> None:
    adp = pl.read_parquet(PROC / "adp.parquet")
    print("QB streaming baseline — equivalent QB replacement rank, tune window")
    print(f"{'season':>6s} | " + "  ".join(f"N_owned={n}" for n in N_OWNED_GRID))
    agg = {n: [] for n in N_OWNED_GRID}
    for s in TUNE_SEASONS:
        line = f"{s:>6d} | "
        for n in N_OWNED_GRID:
            r = equiv_rank(adp, s, stream_total(adp, s, n))
            agg[n].append(r)
            line += f"   QB{r:>2d}   "
        print(line)
    print("\n mean equiv rank:  " + "  ".join(
        f"N{n}: QB{np.mean(agg[n]):.1f}" for n in N_OWNED_GRID))
    print(" -> shipped a fixed REPL_RANKS['QB'] = 8 (central estimate; see REPORT §1)")


if __name__ == "__main__":
    main()
