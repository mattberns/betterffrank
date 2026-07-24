"""Block-wise candidate-feature selection on the TUNE WINDOW ONLY (2012-2017).

Protocol (see CLAUDE.md): the 2018-2025 test set is NEVER consulted here.
For each candidate block in bff.model.CANDIDATE_BLOCKS this script re-runs
the full hyperparameter tune (frozen grid, spearman_vorp, walk-forward
2012-2017) with FEATURES + block, and reports the tune-window mean against
the baseline FEATURES-only tune. It also prints, for every candidate column,
its max |Pearson r| against any existing feature (computed on tune-window
rows only, feature-feature, no outcomes touched).

Keep rules (applied by the operator, documented in reports/REPORT.md):
  1. tune-window mean improves over baseline,
  2. no candidate column with |r| > 0.9 against an existing feature,
  3. coefficient sanity on a full-tune-history fit.

Usage:
    uv run python -m bff.select_features                  # all blocks
    uv run python -m bff.select_features --blocks redzone snaps
    uv run python -m bff.select_features --joint redzone snaps ...  # one combined run
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from bff.model import CANDIDATE_BLOCKS, FEATURES, TUNE_SEASONS, build_dataset, tune


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

    if args.joint is not None:
        cols = [c for b in args.joint for c in CANDIDATE_BLOCKS[b]]
        a, w, v = tune(df, features=FEATURES + cols, quiet=True)
        print(f"=== JOINT {'+'.join(args.joint)} (+{len(cols)} cols) ===")
        print(f"  alpha={a}, shrink={w}, tune mean={v:.4f} "
              f"(delta {v - base_val:+.4f})")
        for line in correlation_report(df, cols):
            print(line)
        return

    keys = args.blocks or list(CANDIDATE_BLOCKS)
    results = []
    for k in keys:
        cols = CANDIDATE_BLOCKS[k]
        a, w, v = tune(df, features=FEATURES + cols, quiet=True)
        results.append((k, a, w, v, v - base_val))
        print(f"=== block {k} (+{len(cols)}: {', '.join(cols)}) ===")
        print(f"  alpha={a}, shrink={w}, tune mean={v:.4f} "
              f"(delta {v - base_val:+.4f})")
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
