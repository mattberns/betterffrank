"""Block-wise candidate-feature selection on the TUNE WINDOW ONLY (2013-2017).

Protocol (see CLAUDE.md): the 2018-2025 test set is NEVER consulted here.
For each candidate block in bff.model.CANDIDATE_BLOCKS this script re-runs
the full hyperparameter tune (frozen grid, spearman_vorp, walk-forward
2013-2017) with FEATURES + block, and reports the tune-window mean against
the baseline FEATURES-only tune. It also prints, for every candidate column,
its max |Pearson r| against any existing feature (computed on tune-window
rows only, feature-feature, no outcomes touched).

Keep rules (applied by the operator, documented in reports/REPORT.md):
  1. the PAIRED per-fold delta at the baseline's fixed (alpha, shrink) is
     positive and large relative to its paired SE (the printed t and fold
     sign count; the raw sweep delta includes hyperparameter re-selection
     and overstates the block),
  2. survives the era-shift haircut: the empirical record is that tune
     gains under ~+0.01 transfer to 2018-2025 at ~zero or negative even
     when tune-significant (inj_pos: t=+4.3, 5/5 folds, test -0.0037),
  3. no candidate column with |r| > 0.9 against an existing feature,
  4. coefficient sanity on a full-tune-history fit.

Usage:
    uv run python -m bff.select_features                  # all blocks
    uv run python -m bff.select_features --blocks redzone snaps
    uv run python -m bff.select_features --joint redzone snaps ...  # one combined run
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from bff.model import (CANDIDATE_BLOCKS, FEATURES, TUNE_SEASONS, build_dataset,
                       season_scores, tune)


def paired_delta(df: pl.DataFrame, base_scores: np.ndarray, cols: list[str],
                 alpha: float, shrink: float) -> str:
    """Paired per-fold delta of FEATURES+cols vs FEATURES at FIXED (alpha,
    shrink) — the baseline's tuned pair, so the comparison isolates the
    columns from hyperparameter re-selection (winner's curse over the grid).
    The tune-mean SE (~0.0022) is the wrong yardstick for a block gate: fold
    scores are highly correlated between nested configs, so the paired SE of
    the delta is what the evidence should be judged on. Judged on THIS, then
    haircut for era shift: every tune gain under ~+0.01 that ever went to
    test transferred at ~zero or negative (trajectory +0.0059 -> +0.0003,
    inj_pos +0.0068 -> -0.0037 at t=+4.3, 5/5 folds — tune-significant and
    still not real on 2018-2025)."""
    s = np.array(season_scores(df, alpha, shrink, TUNE_SEASONS, FEATURES + cols))
    d = s - base_scores
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else float("nan")
    return (f"  paired@fixed({alpha:g},{shrink:g}): delta {d.mean():+.4f}"
            f"  SE {se:.4f}  t {t:+.2f}  folds+ {int((d > 0).sum())}/{len(d)}"
            f"  per-fold {np.round(d, 4).tolist()}")


def correlation_report(df: pl.DataFrame, block: list[str]) -> list[str]:
    """Max |r| of each candidate column vs the existing FEATURES, tune-window
    rows only (no outcome columns involved)."""
    sub = df.filter(pl.col("season") <= TUNE_SEASONS[-1])
    base = {f: sub[f].cast(pl.Float64).to_numpy() for f in FEATURES
            if f != "ppg_mismatch"}  # ppg_mismatch is built inside fit_predict
    lines = []
    for c in block:
        x = sub[c].cast(pl.Float64).to_numpy()
        if np.std(x) < 1e-12:
            lines.append(f"    {c:24s} NEAR-CONSTANT on tune era")
            continue
        best_f, best_r = "", 0.0
        for f, y in base.items():
            if np.std(y) < 1e-12:
                continue
            r = abs(float(np.corrcoef(x, y)[0, 1]))
            if r > best_r:
                best_r, best_f = r, f
        flag = "  <-- COLLINEAR" if best_r > 0.9 else ""
        lines.append(f"    {c:24s} max|r| {best_r:.3f} vs {best_f}{flag}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune-window block selection.")
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="subset of CANDIDATE_BLOCKS keys (default: all)")
    ap.add_argument("--joint", nargs="*", default=None,
                    help="run ONE tune with these blocks combined")
    args = ap.parse_args()

    df = build_dataset()

    print("=== baseline (shipped FEATURES, n=%d) ===" % len(FEATURES))
    a0, w0, base_val = tune(df, quiet=True)
    print(f"  alpha={a0}, shrink={w0}, tune mean spearman_vorp={base_val:.4f}\n")
    base_scores = np.array(season_scores(df, a0, w0, TUNE_SEASONS, FEATURES))

    if args.joint is not None:
        cols = [c for b in args.joint for c in CANDIDATE_BLOCKS[b]
                if c not in FEATURES]
        a, w, v = tune(df, features=FEATURES + cols, quiet=True)
        print(f"=== JOINT {'+'.join(args.joint)} (+{len(cols)} cols) ===")
        print(f"  alpha={a}, shrink={w}, tune mean={v:.4f} "
              f"(delta {v - base_val:+.4f})")
        print(paired_delta(df, base_scores, cols, a0, w0))
        for line in correlation_report(df, cols):
            print(line)
        return

    keys = args.blocks or list(CANDIDATE_BLOCKS)
    results = []
    for k in keys:
        # blocks promoted into FEATURES since their listing (injury, contracts,
        # draft_capital, trend, part of trajectory) would duplicate columns
        cols = [c for c in CANDIDATE_BLOCKS[k] if c not in FEATURES]
        if not cols:
            print(f"=== block {k} — all columns already shipped, skipped ===\n")
            continue
        a, w, v = tune(df, features=FEATURES + cols, quiet=True)
        results.append((k, a, w, v, v - base_val))
        print(f"=== block {k} (+{len(cols)}: {', '.join(cols)}) ===")
        print(f"  alpha={a}, shrink={w}, tune mean={v:.4f} "
              f"(delta {v - base_val:+.4f})")
        print(paired_delta(df, base_scores, cols, a0, w0))
        for line in correlation_report(df, cols):
            print(line)
        print()

    print(f"=== summary (tune-window mean spearman_vorp, "
          f"{TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]}) ===")
    print(f"  {'baseline':16s} {base_val:.4f}")
    for k, a, w, v, d in sorted(results, key=lambda t: -t[3]):
        print(f"  {k:16s} {v:.4f}  ({d:+.4f})  alpha={a} shrink={w}")


if __name__ == "__main__":
    main()
