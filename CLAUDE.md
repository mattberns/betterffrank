# betterffrank — project invariants

Goal: preseason fantasy football rankings (12-team PPR redraft) that beat
market ADP and tie/beat FantasyPros ECR, evaluated on draft value (VORP),
walk-forward with zero leakage. Protocol (2026-07-23, authoritative: `reports/REPORT.md`): tune 2012-2017 (6 folds,
spearman_vorp, frozen grid), test 2018-2025 (S=8, sign-flip floor 0.0039 —
significance at 0.05 is reachable). Current standing (51-feature model,
offset-log vs_adp, drafted-slot VORP curve, QB8 streaming replacement):
model 0.5193 / ECR 0.5139 / ADP 0.4984 mean spearman_vorp; **beats ADP
+0.0209 (7/8, p_one = 0.023 — CERTIFIED at 0.05, two-sided 0.047); edges ECR
+0.0054 (6/8, p_one = 0.164, positive but not yet significant)** — the primary
benchmark is ECR. Integrity ledger: 2018-2020 briefly served as tune folds
during the 2026-07-23 protocol work before moving to test (all selections
re-derived from scratch on 2012-2017; they reproduced exactly). Test-set
looks: FOUR on 2026-07-23, TWO on 2026-07-24 (SIX total) — count future
looks, keep them rare. The two 2026-07-24 looks were a-priori data-corrections
to the VORP conversion, each tune-window-validated before one test look:
(1) finish-rank → drafted-slot curve, (2) QB12 → QB8 streaming replacement
(derived by `bff/streaming.py`; see "The metric"). Same-day history (07-23):
ECR window widened to 2015-2025 via Wayback backfill (`bff/ecr_wayback.py`);
feature expansion 40 → 51. All current numbers and methodology live in
`reports/REPORT.md` (the ONLY report file — no versioned reports).

Run everything from the repo root with `uv run python ...`. Stack: polars +
scikit-learn. Never `git commit` or `git push` unless the user asks.

## The metric

- **The only decision metric is mean `spearman_vorp`** from `bff/backtest.py`
  (Spearman of the list vs actual season VORP), pool = the season's ADP
  top-150 (QB/RB/WR/TE, GSIS-matched).
- **Raw-points Spearman is a QB-stacking trap and is never used for
  decisions.** Elite QBs outscore every RB/WR in total points, so a raw-points
  metric rewards stacking QBs at the top; in a 1-QB league the QB1-QB12 drop
  is small and that ordering is bad draft advice. The ban is now absolute:
  the `bff/model.py` tuner scores `spearman_vorp` (as of 2026-07-23); no
  raw-points path remains anywhere in the codebase.
- **The VORP conversion curve (`bff/vorp.py`) is a DRAFTED-SLOT curve, not a
  finish-rank curve** (fixed 2026-07-24). `pool_market_ranks` ranks each prior
  season's pool within position by PRESEASON MARKET rank (`build_pool`'s
  tiebreak_rank) and carries each player's actual points; `build_curve` then
  averages points over the market draft slot → expected realizable points of
  the player taken at slot k. The old finish-rank version (rank by actual pts
  desc) priced the hindsight order statistic and over-valued the noisy
  mid-TE/QB tails (realizable value at TE8/QB8 was 7-23% of the finish number
  vs ~77% for WR), which shoved mid-round TEs/QBs 20-40 ranks up the board.
  The finish curve scored ~0.018 higher in ABSOLUTE `spearman_vorp` (matches a
  single season's top-heavy realized shape) but that was a level shift shared
  by the model and both baselines; the drafted-slot curve WIDENED the model's
  edge over ADP and held the ECR tie. Do NOT revert to finish-rank ranking to
  chase the absolute metric — it is the wrong exchange rate for a preseason
  board. Cross-position position MIX on any board is a pure function of this
  curve (the model sets only within-position order), so this is the lever for
  positional over/under-valuation, not a per-position hand fudge.
- **QB replacement is QB8, not QB12** (`REPL_RANKS` in `bff/backtest.py`, set
  2026-07-24). In a 1-QB league you stream QBs, so the real fallback beats the
  12th-best QB's season. `bff/streaming.py` simulates form-streaming the free
  pool on the tune window → effective replacement ~QB6-9 (noisy, owned-pool
  dependent); QB8 is the shipped central estimate, gated on the tune window
  (held the ECR edge, kept a positive ADP edge) before one test look. It fixed
  the QB block (Herbert QB5 fell from board #39 to #50; first QB #17 → #22) and
  certified the ADP claim. RB/WR/TE stay 30/36/12. The QB curve plateaus
  QB10-12, so only a replacement ≤ QB8 moves the board — don't set QB9/10
  expecting an effect. (Its sibling, a BEER man-games replacement, was REJECTED
  — see Rules.)

## Anchor and vs_adp

- Anchor: `mkt_val = log(ecr_rank)` when a preseason ECR snapshot exists for
  the season (2015-2025 + 2026), else `log(adp_rank)`; then ordinal re-rank per
  season → `log_rank = log(mkt_rank)`. Detection is per-row on
  `ecr_rank.is_not_null()`, not a season list.
- `vs_adp` (regression feature) = `log((ecr_rank + 10) / (adp_rank + 10))`,
  exactly 0.0 when ECR is missing (only pre-2015 seasons now). **Sign
  convention: positive = the market drafts him earlier than the experts rank
  him** (ADP rank < ECR rank). Verified cases: 2023 Jonathan Taylor, ADP 25 /
  ECR 84 → +0.99; 2026 Ja'Marr Chase, ADP 4 / ECR 1 → −0.24.
  The +10 offset is load-bearing: the raw log difference explodes
  mechanically at the top of rank data (Chase would be −1.39, a 5.4σ value
  for a 3-spot disagreement, and it pushed ECR's #1 to #17 on the 2026
  board — fixed 2026-07-23). The shipped variant is `VS_ADP_VARIANT` in
  `bff/model.py`; alternatives (`vs_adp_log/pct/clip`) stay built in
  `build_dataset` for tune-window re-tests. Chosen on the 2012-2017 tune
  window (clip 0.4515 / log 0.4513 / off 0.4512, noise-level tie);
  offset-log picked over clip on principle (smooth, no kink at the cap).

## Leakage rules

- Walk-forward: predictions for season t use only seasons < t
  outcomes/stats plus season-t PRESEASON facts (ADP, ECR, roster/draft
  context, static attributes). Never season-t outcomes.
- Season-t roster membership comes from `ctx_rosters_week1.parquet`
  (week-1 REG snapshot, CUT/RET excluded). `ctx_rosters.parquet` is a
  LAST-OBSERVED-TEAM table — in-season information for season t; it is
  used only for t-1 lookups (arrivals' prevros) and the 2026 draftee gsis
  backfill. Never use it for season-t membership (that was a leak, fixed
  2026-07-23; it inflated mean VORP Spearman by ≈ 0.003).
- Hyperparameters are tuned ONLY on walk-forward validation seasons
  2012-2017, on the frozen grid alpha ∈ {3, 10, 30, 100, 300} × shrink ∈
  {0.3, 0.5, 0.7, 1.0}, scoring mean `spearman_vorp`. Do not expand the grid
  (the VORP-tuned winner is alpha = 100, shrink = 0.3 — off the edge). Test
  seasons 2018-2025 are never touched for tuning or feature selection.
- Preseason ECR in `data/processed/ecr.parquet`: 2015-2020 from Wayback
  FantasyPros PPR-cheatsheet snapshots (`data/raw/db_fpecr_wayback.parquet`,
  built once by `bff/ecr_wayback.py` from cached HTML in
  `data/raw/wayback_ppr/`; overall displayed rank = the `ecr` value, and each
  row carries a real `fantasypros_id` → 100% top-150 gsis match), 2021-2025
  from the DynastyProcess archive snapshots, 2026 from a FantasyPros
  draft-rankings export (`data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv`).
  All raw sources are git-ignored (commercial data); the derived
  `ecr.parquet` is committed. Each season uses its LATEST preseason capture
  (same rule `bff.ecr` uses for the archive), every one before that season's
  Week 1 — verified. The 2018 snapshot is the Sep 1 capture; an accidentally
  -early Aug 4 capture (which ranked full-season holdout Le'Veon Bell #1) was
  discarded. Any future season WITHOUT ECR rows falls back to an ADP-only
  anchor with `vs_adp` = 0, and the CLI prints a warning.
- All models AND baselines go through the same VORP conversion
  (`bff.model.to_vorp` over `bff/vorp.py`'s leakage-safe drafted-slot curve;
  replacement ranks QB8/RB30/WR36/TE12). Never score a baseline on raw
  -adp/-ecr.

## Commands and stable artifact names

```
# one-time: backfill 2015-2020 ECR from Wayback (needs network; caches HTML,
# then reruns are offline). Produces data/raw/db_fpecr_wayback.parquet.
uv run python -m bff.ecr_wayback

# data rebuild (only if raw data changes)
uv run python -m bff.adp && uv run python -m bff.ecr && uv run python -m bff.actuals
uv run python -m bff.context_data && uv run python -m bff.context_features
uv run python -m bff.opportunity_features
uv run python -m bff.redzone_features      # <- data/raw/pbp/ (nflverse, 2010-2025)
uv run python -m bff.situation_features    # <- data/raw/{snaps,injuries,contracts,vegas}/
uv run python -m bff.vegas_wayback         # one-time Wayback scrape (needs network; cached after)

# QB streaming baseline derivation (tune window ONLY; justifies REPL_RANKS QB8)
uv run python -m bff.streaming

# candidate-feature selection (tune window 2012-2017 ONLY; never the test set)
uv run python -m bff.select_features [--blocks k ...] [--joint k ...]

# model
uv run python -m bff.model                       # -> data/processed/preds_model.parquet (2018-2025)
uv run python -m bff.model --baselines           # -> preds_adp.parquet, preds_ecr.parquet
uv run python -m bff.model --season 2026         # -> preds_model_2026.parquet, reports/rankings_2026.csv, reports/steals_2026.csv

# draft-strategy overlay (VONA) — derived from the 2026 board, NOT scored/tuned
uv run python -m bff.vona                        # -> reports/vona_2026.csv (+ prints snake-turn matrix)

# evaluation
uv run python -m bff.backtest data/processed/preds_model.parquet --name model
uv run python -m bff.backtest data/processed/preds_adp.parquet   --name adp
uv run python -m bff.backtest data/processed/preds_ecr.parquet   --name ecr

# significance
uv run python -m bff.compare data/processed/preds_model.parquet data/processed/preds_adp.parquet
uv run python -m bff.compare data/processed/preds_model.parquet data/processed/preds_ecr.parquet

# static site (GitHub Pages, /docs; derives numbers from the artifacts above)
uv run python -m bff.site              # render (assumes preds/scores current)
uv run python -m bff.site --refresh    # rerun model + backtests, then render
```

## Output contract — each run regenerates and replaces these in place

| Command | Overwrites |
| --- | --- |
| `bff.model` | `data/processed/preds_model.parquet` (1190 rows, 2018-2025) + prints tuned (alpha, shrink) and the coefficient report |
| `bff.model --baselines` | `data/processed/preds_adp.parquet`, `preds_ecr.parquet` |
| `bff.model --season 2026` | `data/processed/preds_model_2026.parquet` (185 rows), `reports/rankings_2026.csv`, `reports/steals_2026.csv` |
| `bff.vona` | `reports/vona_2026.csv` (150 rows) + prints the 12-seat snake-turn matrix |
| `bff.streaming` | stdout only (tune-window QB streaming equiv-rank table; derivation behind QB8) |
| `bff.backtest <preds> --name <n>` | `reports/scores_<n>.csv` |
| `bff.compare A B` | stdout only (per-season deltas + exact sign-flip p-values) |

Model and evaluation outputs (`preds_*.parquet`, `scores_*.csv`) are
deterministic and byte-stable across reruns. The data-rebuild steps are
content-stable but not byte-stable: `bff.actuals` reshuffles row order and
`bff.context_features` carries ~1e-15 float noise run-to-run (parallel
aggregation); neither propagates to any reported number.

## Keeping the docs in sync — MANDATORY after any change to model results

Whenever a change alters the model's numbers (metric values, p-values,
windows, feature count, hyperparams, the 2026 board) or the protocol:

1. `reports/REPORT.md` — hand-written, THE single report (no versioned or
   dated report files, ever). Update the headline table, per-season deltas,
   feature counts, and integrity ledger in place.
2. This file (CLAUDE.md) — the standing paragraph at the top, the window
   numbers in "Leakage rules"/"Rules", and any invariant that moved.
3. `README.md` — the headline table, the one-paragraph method (feature
   count, windows), pipeline commands, repo layout, and caveats.
4. `/docs` static site — regenerate with `uv run python -m bff.site`
   (`--refresh` if preds/scores were not just rebuilt). The generator
   derives numbers from the artifacts, so a rebuild is usually sufficient;
   if you add features, add their labels to FEATURE_META in `bff/site.py`.

The three hand-written docs must agree with each other and with the
artifacts; the site must be regenerated, never hand-edited. A results change
without this sync is an incomplete change.

## Rules

- Never evaluate or select on raw-points Spearman (no raw-points path remains).
- Never tune or feature-select on the test seasons 2018-2025; the 2012-2017
  validation window and its frozen grid are the only tuning surface.
- Report worse numbers plainly; a regression is a result, not a bug to hide
  (precedent: the 11-feature expansion shipped despite costing a hair on an
  earlier test window, because keep/drop is a tune-window decision).
- **VONA (`bff/vona.py`) is a draft-strategy OVERLAY, never part of the scored
  model.** It reads the finished 2026 board + ADP and reports value-lost-by-
  waiting per pick (the site's Draft page). It must NOT feed `REPL_RANKS`, the
  curve, `to_vorp`, the tuner, or the metric — it is presentation only, so it
  is never a test look. The value board stays WR-first (correct for PPR);
  VONA is where RB scarcity legitimately surfaces, as pick-timing. (A BEER
  man-games replacement was prototyped 2026-07-24 and REJECTED on the tune
  window: in PPR the flex resolves to WR, so it deepened WR and narrowed the
  model's edge over ADP/ECR — do not revive it expecting RB-first.)
- **The user owns ALL git.** Never stage, commit, push, branch, or otherwise
  change git state — he does every git interaction himself. Read-only git
  (status/diff/log) is fine when it serves a task. Do NOT end summaries with
  "nothing committed" / "uncommitted" notes; leave the tree as-is and say
  nothing about commit state unless asked.

## Fragility list — do not "fix" or reorder these

- `implied_expectation`'s quadratic **vertex clamp** (monotone implied value).
- Residual clip at ±4 before the ridge fit.
- **`ppg_mismatch` is always the LAST matrix column** in `fit_predict`; any
  reorder silently corrupts `feat_order` and the per-player contributions.
- The anchor's `rank(method="ordinal")` breaks ties by row order; keep
  `build_dataset`'s join sequence unchanged or pre-2021 anchors drift.
- `to_vorp` uses an **inner** join (pool players without preds are dropped)
  while `bff.backtest` bottoms unscored pool players by tiebreak rank —
  different rules on purpose; keep both.
- `to_vorp`'s season == 2026 branch scores the full 185-player ADP pool
  instead of `build_pool`.
- Backtest dedupe rules: pool keeps best tiebreak rank per gsis_id; preds
  keep max score per gsis_id.
- Three different `norm_name` variants exist across adp/ecr/crosswalk joins;
  do not unify them.
- 2025 stats are special-cased (`stats_player_week_2025.parquet` /
  `stats_player_reg_2025.parquet`); 2025 ADP came from Wayback.
- `games_missed` uses the PRIOR season's schedule length (17-game era from
  target season 2022 on). This was a v1-era off-by-one at the 2020→2021
  boundary, fixed 2026-07-23; don't regress it to the target season's length.
- Season-t roster joins in `context_features.py` use `ctx_rosters_week1`;
  `arrivals`' prevros and `draft_rows`' 2026 backfill intentionally stay on
  full-season `ctx_rosters` — do not unify the two tables.
- `build_team_qb` sorts with a gsis_id tiebreak (2012 SF: Smith and
  Kaepernick tied at 218 attempts); removing it makes qb_change flip
  run-to-run.
- StandardScaler's zero-variance fallback is load-bearing for `vs_adp`
  during tuning: the early validation folds have training windows that
  predate all ECR (fold 2012 trains on 2011 alone; ECR starts 2015), so
  `vs_adp` has zero train variance there and its coefficient is fit as 0.
  Every 2018-2025 test fold trains on ECR-bearing seasons, so `vs_adp` is
  fully active across the whole test set.
- **No stored hyperparams anywhere**: the tuner re-derives (alpha, shrink)
  deterministically on every run.
- 2026 sanity gates (≤ 2 QBs in top 15, top-3 all RB/WR, positions only
  QB/RB/WR/TE) are asserts: investigate a failure, never weaken silently.
- `build_dataset`'s candidate-block joins (incl. a `join_asof` that re-sorts
  rows) come AFTER the anchor's ordinal rank is materialized — keep it that
  way, or pre-2021 anchors drift (the rank ties break by row order).
- Rejected candidate blocks (redzone/snaps/vegas/landing_spot) stay in
  `build_dataset` as inert zero-filled columns and in `CANDIDATE_BLOCKS`;
  they are NOT in FEATURES. Re-test via `bff.select_features`, never by
  hand-editing FEATURES.
- Known-degenerate candidate data (nulls by design, zero-filled at join):
  snaps source 2012 is empty upstream (features start target 2014); vegas
  has no usable 2025/2026 capture; contracts miss the 2026 draft class
  (14 pool players). `vegas_wins` is per-season CENTERED before zero-fill
  (a missing line imputes league-average, not 0 wins).
