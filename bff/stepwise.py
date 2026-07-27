"""Forward-backward stepwise feature selection for the ridge, gated on the
anchor-only NULL. TUNE WINDOW ONLY (2013-2017); the 2018-2025 test set is
never touched here.

WHY THIS EXISTS. The shipped 53-feature set was accumulated block by block,
each block gated against the then-current set, and nobody ever asked whether
the accumulation beats doing nothing. It does not: on the tune window the
shipped model scores 0.4293 against a NULL of 0.4306. The null is the same
pipeline with the ridge stage switched off -- `score = implied_expectation
(market rank)`, i.e. fit_predict(features=[]) -- so a delta against it is
attributable to features and to nothing else. (The separate `preds_ecr`
baseline reads 0.4298; it differs only because it bottoms pool players with
no ECR row instead of falling back to their ADP, which in the tune window is
one player, Blount 2013.)

WHAT IT DOES. Greedy forward-backward search from the EMPTY set over a
76-column candidate pool, re-running the full frozen-grid tune
(alpha x shrink) at every evaluation, scoring mean spearman_vorp. Nothing is
forced in: `vs_adp` and `ppg_mismatch` are droppable like everything else.

OVERFIT CONTROL -- the point of the module. Five tune folds put the standard
error of the tune mean at ~0.0022, and a greedy max over ~76 candidates
inflates the step winner by roughly 2-2.5 se, so a single path at the standing
+0.0020 block gate would select noise and not reproduce. So the whole search
runs SIX times: five nested leave-one-fold-out runs (select on four folds,
record the fifth without ever selecting on it) plus one full-window run for
comparison. Membership comes from the MAJORITY VOTE across the five LOFO runs
(>= 3 of 5); the averaged held-out curve is the cross-check on how long the
path keeps helping.

Selection here is a tune-window activity and costs ZERO test looks. Scoring a
final list on 2018-2025 is one look and happens elsewhere, once.

Usage:
    uv run python -m bff.stepwise --check          # verify against known numbers
    uv run python -m bff.stepwise                  # full study -> reports/stepwise.json
    uv run python -m bff.stepwise --runs full      # single full-window path only
"""

from __future__ import annotations

# Single-thread the numeric stacks BEFORE polars/sklearn are imported: this
# module forks a worker per candidate, so library-level threading would
# oversubscribe the box and run slower than serial.
import os

for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from bff.model import (CANDIDATE_BLOCKS, FEATURES, TUNE_SEASONS,  # noqa: E402
                       build_dataset, scoring_context, season_scores, tune)

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "reports" / "stepwise.json"

# Candidate columns that exist in build_dataset() but cannot be SERVED on the
# 2026 board: no 2026 prop boards are posted (a market fact, not a fetch bug)
# and vegas has no usable 2025/2026 capture. Both zero-fill at serve time, so
# a selected coefficient would be applied to a column of zeros -- a silent
# train/serve skew, not a neutral default. Excluded from the pool by rule, not
# by measurement.
UNSERVABLE_2026 = ("prop_mvp", "prop_oroy", "prop_pass_yds", "prop_rush_yds",
                   "prop_rec_yds", "vegas_wins_c", "vegas_wins_delta")

EPS = 1e-6          # strict-improvement threshold for accepting a move
MAJORITY = 3        # of the 5 LOFO runs


def candidate_pool() -> list[str]:
    """The 76: everything shipped plus every built-but-unshipped candidate
    column that can actually be served in 2026.

    The unshipped ones are exactly the blocks rejected in REPORT.md section 4,
    every one of which was judged on the OLD tune surface (before the 2012-2014
    ECR backfill, with 2011 still in training). They are back in the pool on
    purpose, and here they are judged column by column rather than as
    pre-bundled blocks.
    """
    extra = sorted({c for cols in CANDIDATE_BLOCKS.values() for c in cols}
                   - set(FEATURES) - set(UNSERVABLE_2026))
    pool = list(FEATURES) + extra
    assert len(pool) == len(set(pool))
    return pool


# --- parallel evaluation -----------------------------------------------------
# SPAWN, not fork. Polars runs a rayon threadpool, and forking a parent that
# has already touched polars deadlocks the child: the first run of this module
# hung with every worker at 0% CPU. Each spawned worker therefore builds its
# own dataset and warms its own scoring context once (~2s), which is noise
# against a pool that lives for a whole search().

_DF: pl.DataFrame | None = None


def _init_worker() -> None:
    global _DF
    _DF = build_dataset()
    scoring_context()


def _evaluate(job: tuple[tuple[str, ...], tuple[int, ...]]) -> dict:
    feats, seasons = job
    return evaluate(_DF, list(feats), seasons)


def evaluate(df: pl.DataFrame, feats: list[str], seasons: tuple[int, ...]) -> dict:
    """Tune (alpha, shrink) on `seasons` for this feature set and return the
    winning grid point plus its per-season scores.

    The empty set is the null: no ridge stage, so the whole grid is one point.
    """
    if not feats:
        vals = season_scores(df, 0.0, 0.0, seasons, features=[])
        return {"features": [], "alpha": None, "shrink": None,
                "mean": float(np.mean(vals)), "per_season": vals}
    a, w, m = tune(df, features=feats, quiet=True, seasons=seasons)
    return {"features": list(feats), "alpha": a, "shrink": w, "mean": m,
            "per_season": season_scores(df, a, w, seasons, features=feats)}


def heldout(df: pl.DataFrame, feats: list[str], alpha, shrink, fold: int) -> float:
    """Score the untouched fold at the parameters chosen WITHOUT it."""
    if not feats:
        return season_scores(df, 0.0, 0.0, (fold,), features=[])[0]
    return season_scores(df, alpha, shrink, (fold,), features=feats)[0]


# --- the search --------------------------------------------------------------

def search(df: pl.DataFrame, pool: list[str], seasons: tuple[int, ...],
           holdout: int | None, workers: int, max_steps: int,
           label: str) -> dict:
    """Greedy forward-backward from the empty set.

    Each forward step evaluates every unselected candidate and adds the best
    strict improvement; each acceptance is followed by backward passes that
    try dropping each already-selected feature. Deterministic throughout: no
    RNG, ties broken on lowest pool index, and a visited-set of frozensets makes
    an add/drop cycle impossible.
    """
    t0 = time.time()
    selected: list[str] = []
    base = evaluate(df, [], seasons)
    cur = base["mean"]
    visited = {frozenset()}
    path = [{"step": 0, "move": "null", "feature": None, "k": 0,
             "mean": cur, "alpha": None, "shrink": None,
             "heldout": None if holdout is None else heldout(df, [], None, None, holdout)}]
    print(f"[{label}] null = {cur:.4f}"
          + ("" if holdout is None else f"  (held-out {holdout}: {path[0]['heldout']:.4f})"),
          flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool_proc:
        for step in range(1, max_steps + 1):
            # ---- forward: try every unselected candidate
            cands = [c for c in pool if c not in selected]
            moves = [(tuple(sorted(selected + [c])), seasons) for c in cands]
            keep = [i for i, m in enumerate(moves)
                    if frozenset(m[0]) not in visited]
            if not keep:
                break
            res = pool_proc.map(_evaluate, [moves[i] for i in keep])
            best_i = max(range(len(res)), key=lambda i: (res[i]["mean"], -keep[i]))
            best, feat = res[best_i], cands[keep[best_i]]
            if best["mean"] <= cur + EPS:
                print(f"[{label}] step {step}: no add clears (best "
                      f"{feat} {best['mean'] - cur:+.4f}) -- stop", flush=True)
                break
            selected = sorted(selected + [feat])
            cur = best["mean"]
            visited.add(frozenset(selected))
            ho = None if holdout is None else heldout(
                df, selected, best["alpha"], best["shrink"], holdout)
            path.append({"step": step, "move": "add", "feature": feat,
                         "k": len(selected), "mean": cur, "alpha": best["alpha"],
                         "shrink": best["shrink"], "heldout": ho})
            print(f"[{label}] step {step:2d}: +{feat:26s} k={len(selected):2d} "
                  f"mean={cur:.4f} (vs null {cur - base['mean']:+.4f}) "
                  f"alpha={best['alpha']:g} shrink={best['shrink']:g}"
                  + ("" if ho is None else f"  held-out {ho:.4f}"), flush=True)

            # ---- backward: try dropping each selected feature, repeat while it helps
            while len(selected) > 1:
                drops = [(tuple(x for x in selected if x != c), seasons)
                         for c in selected]
                keep_d = [i for i, m in enumerate(drops)
                          if frozenset(m[0]) not in visited]
                if not keep_d:
                    break
                res_d = pool_proc.map(_evaluate, [drops[i] for i in keep_d])
                bi = max(range(len(res_d)), key=lambda i: (res_d[i]["mean"], -keep_d[i]))
                bd, dropped = res_d[bi], selected[keep_d[bi]]
                if bd["mean"] <= cur + EPS:
                    break
                selected = [x for x in selected if x != dropped]
                cur = bd["mean"]
                visited.add(frozenset(selected))
                ho = None if holdout is None else heldout(
                    df, selected, bd["alpha"], bd["shrink"], holdout)
                path.append({"step": step, "move": "drop", "feature": dropped,
                             "k": len(selected), "mean": cur, "alpha": bd["alpha"],
                             "shrink": bd["shrink"], "heldout": ho})
                print(f"[{label}]          -{dropped:26s} k={len(selected):2d} "
                      f"mean={cur:.4f} (vs null {cur - base['mean']:+.4f})"
                      + ("" if ho is None else f"  held-out {ho:.4f}"), flush=True)

    final = evaluate(df, selected, seasons)
    out = {"label": label, "seasons": list(seasons), "holdout": holdout,
           "null_mean": base["mean"], "null_per_season": base["per_season"],
           "selected": selected, "k": len(selected), "mean": final["mean"],
           "delta_vs_null": final["mean"] - base["mean"],
           "alpha": final["alpha"], "shrink": final["shrink"],
           "per_season": final["per_season"], "path": path,
           "seconds": round(time.time() - t0, 1)}
    print(f"[{label}] DONE k={len(selected)} mean={final['mean']:.4f} "
          f"({out['delta_vs_null']:+.4f} vs null) in {out['seconds']:.0f}s\n",
          flush=True)
    return out


# --- verification ------------------------------------------------------------

def check(df: pl.DataFrame) -> None:
    """Reproduce numbers this module must agree with before it is trusted."""
    pool = candidate_pool()
    print(f"candidate pool: {len(pool)} columns "
          f"({len(FEATURES)} shipped + {len(pool) - len(FEATURES)} revived)")
    print(f"  revived: {', '.join(c for c in pool if c not in FEATURES)}")
    print(f"  excluded as unservable in 2026: {', '.join(UNSERVABLE_2026)}\n")
    assert len(pool) == 76, f"expected a 76-column pool, got {len(pool)}"

    # grid-independent regression anchors: the null has no hyperparameters, and
    # the shipped set at the ORIGINAL frozen optimum must still be 0.4293
    # whatever the grid now contains.
    null = evaluate(df, [], TUNE_SEASONS)
    print("null (anchor only)   " + "  ".join(f"{v:.4f}" for v in null["per_season"])
          + f"   mean {null['mean']:.4f}   expect 0.4306")
    legacy = season_scores(df, 300.0, 0.3, TUNE_SEASONS, features=list(FEATURES))
    print("shipped 53 @300/0.3  " + "  ".join(f"{v:.4f}" for v in legacy)
          + f"   mean {np.mean(legacy):.4f}   expect 0.4293")
    ship = evaluate(df, list(FEATURES), TUNE_SEASONS)
    print(f"shipped 53, tuned on the frozen grid: {ship['mean']:.4f} "
          f"(alpha={ship['alpha']:g} shrink={ship['shrink']:g})")
    assert abs(null["mean"] - 0.4306) < 5e-5, null["mean"]
    assert abs(np.mean(legacy) - 0.4293) < 5e-5, np.mean(legacy)

    # cross-check against the block-wise tool: this delta must match
    # `uv run python -m bff.select_features --blocks sos`
    sos = evaluate(df, list(FEATURES) + CANDIDATE_BLOCKS["sos"], TUNE_SEASONS)
    print(f"\nFEATURES + sos block  mean {sos['mean']:.4f} "
          f"(delta {sos['mean'] - ship['mean']:+.4f}) "
          f"-- must match bff.select_features --blocks sos")
    print("\nchecks passed")


# --- study -------------------------------------------------------------------

def majority_vote(runs: list[dict]) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    for r in runs:
        for f in r["selected"]:
            counts[f] = counts.get(f, 0) + 1
    keep = sorted([f for f, n in counts.items() if n >= MAJORITY])
    return keep, dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def heldout_curve(runs: list[dict]) -> list[dict]:
    """Held-out score by path position, averaged over the LOFO runs. This is
    the honest out-of-selection read: where it peaks is where the path stops
    helping and starts fitting the selection folds."""
    by_k: dict[int, list[float]] = {}
    for r in runs:
        seen = {}
        for p in r["path"]:
            if p["heldout"] is not None:
                seen[p["k"]] = p["heldout"]   # last state at this size
        for k, v in seen.items():
            by_k.setdefault(k, []).append(v)
    return [{"k": k, "mean_heldout": float(np.mean(v)), "n_runs": len(v)}
            for k, v in sorted(by_k.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stepwise selection on the tune window.")
    ap.add_argument("--check", action="store_true",
                    help="verify against known numbers and exit")
    ap.add_argument("--runs", choices=("both", "lofo", "full"), default="both")
    ap.add_argument("--pool", choices=("full", "shipped"), default="full",
                    help="'shipped' restricts the search to the current 53")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    df = build_dataset()
    scoring_context()

    if args.check:
        check(df)
        return

    pool = candidate_pool() if args.pool == "full" else list(FEATURES)
    print(f"pool={len(pool)} workers={args.workers} folds={TUNE_SEASONS}\n", flush=True)

    lofo: list[dict] = []
    if args.runs in ("both", "lofo"):
        for h in TUNE_SEASONS:
            sub = tuple(s for s in TUNE_SEASONS if s != h)
            lofo.append(search(df, pool, sub, h, args.workers,
                               args.max_steps, f"lofo-{h}"))

    full = None
    if args.runs in ("both", "full"):
        full = search(df, pool, TUNE_SEASONS, None, args.workers,
                      args.max_steps, "full")

    study = {"pool_size": len(pool), "pool": pool,
             "excluded_unservable_2026": list(UNSERVABLE_2026),
             "tune_seasons": list(TUNE_SEASONS), "majority_threshold": MAJORITY,
             "lofo_runs": lofo, "full_run": full}

    if lofo:
        keep, counts = majority_vote(lofo)
        curve = heldout_curve(lofo)
        study["selection_counts"] = counts
        study["majority_set"] = keep
        study["heldout_curve"] = curve
        # the majority set re-tuned once on all five folds
        final = evaluate(df, keep, TUNE_SEASONS)
        null = evaluate(df, [], TUNE_SEASONS)
        shipped = evaluate(df, list(FEATURES), TUNE_SEASONS)
        study["majority_eval"] = final
        study["null_eval"] = null
        study["shipped_eval"] = shipped

        print("=== selection counts (of 5 LOFO runs) ===")
        for f, n in counts.items():
            print(f"  {n}  {f}")
        print("\n=== held-out curve (mean over runs) ===")
        for row in curve:
            print(f"  k={row['k']:2d}  {row['mean_heldout']:.4f}  "
                  f"(n={row['n_runs']})")
        print(f"\n=== majority set (>= {MAJORITY}/5): {len(keep)} features ===")
        print("  " + (", ".join(keep) if keep else "(empty)"))
        print(f"\n  null        {null['mean']:.4f}")
        print(f"  shipped 53  {shipped['mean']:.4f}  ({shipped['mean'] - null['mean']:+.4f})")
        print(f"  majority    {final['mean']:.4f}  ({final['mean'] - null['mean']:+.4f})"
              f"  alpha={final['alpha']} shrink={final['shrink']}")

    Path(args.out).write_text(json.dumps(study, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
