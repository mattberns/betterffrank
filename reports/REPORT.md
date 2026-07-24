# betterffrank: Preseason Fantasy Rankings, Scored on Draft Value (VORP)

**The model is ECR+**: a single ridge-residual model (`bff/model.py`) over a
FantasyPros-ECR market anchor, converted to predicted VORP. The anchor is the
expert consensus; the ridge learns, from history, where the experts were wrong
against actual outcomes and by how much. ADP enters only as feature material
(the expert/market disagreement signal `vs_adp`, depth ranks, expected-QB
identification), as the scoring pool definition, and as tiebreaks — never as
the anchor when ECR exists (98.7-100% of every scored season is ECR-anchored).

Every number below reproduces from the commands in the last section.

## 1. Metric

Mean Spearman correlation of the list against actual season VORP (points
above the actual QB12/RB30/WR36/TE12 in that season's ADP top-150 pool),
`spearman_vorp` from `bff/backtest.py`. Raw-points Spearman is never used
anywhere (it rewards QB stacking; no raw-points path remains in the
codebase). All lists, including both baselines, go through the same
leakage-safe VORP conversion — baselines are never scored on raw ranks.

The conversion curve (`bff/vorp.py`) is a **drafted-slot** curve: expected
actual points of the player the market DRAFTED at each within-position slot,
built from seasons < t. It replaced a finish-rank curve on 2026-07-24, which
priced each slot at the hindsight order statistic (whoever *ended up* TE3,
QB5…) and so massively over-valued the noisy mid-TE/QB tails — realizable
value at TE8/QB8 is 7-23% of the finish-curve number, vs ~77% for WR at every
slot. The finish curve scored ~0.018 higher in absolute `spearman_vorp` (it
matches the top-heavy shape of a single realized season), but that gain was a
level shift shared by the model AND both baselines; the drafted-slot curve is
the correct expected-value exchange rate for a preseason board and, as §3
shows, does not weaken the relative verdict.

## 2. Protocol

- **Tuning and feature selection: walk-forward seasons 2012-2017 only** (6
  folds; fold s trains on 2011..s-1). Frozen hyperparameter grid alpha ∈
  {3, 10, 30, 100, 300} × shrink ∈ {0.3, 0.5, 0.7, 1.0}, scored on mean
  `spearman_vorp` through the same VORP pipeline as evaluation.
  Deterministic, re-derived every run, no stored parameters.
- **Test: seasons 2018-2025 (S=8), never touched for any decision.** The
  paired sign-flip permutation test (`bff/compare.py`) is exact; at S=8 the
  one-sided significance floor is 0.0039, so p < 0.05 is reachable.
- Walk-forward leakage rules (see CLAUDE.md): predictions for season t use
  only seasons < t outcomes plus season-t preseason facts (ADP, ECR, April
  draft, offseason rosters, week-1 coaches, contracts signed ≤ t).

**Integrity ledger.** The tune/test split was redrawn on 2026-07-23 (it was
previously tune 2012-2014 / eval 2015-2025). Seasons 2018-2020 briefly
served as tuning folds during that day's intermediate work before landing in
the test set; to purge contamination, every selection (feature blocks,
vs_adp transformation) was re-derived from scratch on 2012-2017, and
reproduced identically. The test set was consulted 4 times on 2026-07-23 and
once on 2026-07-24 (5 total); future looks should stay rare. The 2026-07-24
look was the drafted-slot curve swap (§1): an a-priori data-correction to the
VORP conversion, derived and validated on the 2012-2017 tune window first
(where it preserved the model's edge over ADP), then scored once on test — not
a selection over test outcomes. Residual caveat: an operator who has seen
2018-2020 results cannot fully unsee them.

## 3. Headline (test seasons 2018-2025)

| | mean spearman_vorp |
|---|---|
| **model** | **0.4978** |
| ECR baseline | 0.4963 |
| ADP baseline | 0.4765 |

| Comparison | mean delta | seasons won | p (one-sided / two-sided) |
|---|---|---|---|
| vs **ECR** (primary benchmark) | **+0.0015** | 6 of 8 | 0.395 / 0.789 |
| vs **ADP** | **+0.0213** | 6 of 8 | 0.051 / 0.102 |

**The claim: beats ADP (at the edge of certification, p = 0.051); ties/edges
ECR.** Per-season deltas:

| season | model | vs ADP | vs ECR |
|---|---|---|---|
| 2018 | 0.4341 | −0.0249 | +0.0067 |
| 2019 | 0.4734 | +0.0247 | +0.0021 |
| 2020 | 0.4947 | −0.0071 | −0.0168 |
| 2021 | 0.5093 | +0.0227 | +0.0006 |
| 2022 | 0.5811 | +0.0496 | +0.0252 |
| 2023 | 0.5476 | +0.0244 | +0.0083 |
| 2024 | 0.4607 | +0.0083 | −0.0179 |
| 2025 | 0.4813 | +0.0727 | +0.0039 |

Note the ECR baseline itself beats ADP (0.4963 vs 0.4765): expert consensus
is a better draft signal than market ADP, which is why the model's margin
over ECR is much smaller than over ADP. The drafted-slot curve (§1) widened
the ADP margin (from +0.0183 to +0.0213, 5→6 seasons won, p 0.078→0.051) and
left the ECR tie intact (+0.0025→+0.0015); all absolute levels fell ~0.018 in
the shared level shift. At the current effect size the ADP claim plausibly
certifies within 1-2 future seasons (each concluded season adds one paired
observation).

## 4. Features (51)

Score(t, player) = market-implied expectation of log1p(season points) from
the ECR anchor (per-position quadratic in log rank, vertex-clamped) +
shrink × ridge residual. Tuned: alpha = 100, shrink = 0.3 (off the grid
edge). Feature groups:

- **Base (11)**: age curve (centered, squared, RB/QB interactions),
  ppg_mismatch (prior production vs market-implied, always the LAST matrix
  column), games_missed, rookie_pedigree, td_share_c, team_change,
  has_prior, draft_ovr_log.
- **Market (1)**: `vs_adp` = log((ecr_rank+10)/(adp_rank+10)), exactly 0.0
  when ECR is missing; positive = the market drafts him earlier than the
  experts rank him. The +10 offset is load-bearing: a raw log difference
  explodes at the top of rank data (ECR 1 / ADP 4 → −1.39, a 5.4σ value for
  a 3-spot disagreement; it pushed ECR's #1 to #17 on the 2026 board before
  the fix). Transformation chosen on the tune window (offset-log / clip /
  raw log tie within noise; percentile diff worse); offset-log ships for
  smoothness. Verified cases: 2023 Jonathan Taylor ADP 25 / ECR 84 → +0.99;
  2026 Ja'Marr Chase ADP 4 / ECR 1 → −0.24.
- **Preseason context (17)** + **position interactions (3)**: vacated
  volume, arriving veterans, April draft competition, QB change/quality,
  coach change, team priors, returning same-position competition,
  depth_rank_adp, is_rookie (see `bff/context_features.py`).
- **Opportunity (8, curated)**: t-1 weekly target/air-yards shares, share
  velocity, carry-share velocity, production-over-opportunity, TD
  efficiency vs position, boom rate (see `bff/opportunity_features.py`).
- **Expansion (11, added 2026-07-23)**: selected block-wise on the tune
  window (joint 0.4513 vs 0.4456 baseline, +0.0057):
  - *trend*: ppg_delta (t-1 minus t-2 ppg), career_missed_rate
  - *injury*: weeks on injury report (2y), soft-tissue mentions (2y),
    same-injury recurrence — from nflverse injury reports
  - *contracts*: apy_cap_pct (largest new coefficient, +0.04),
    contract_year, rookie_deal_yr — from nflverse/OTC contracts
  - *draft capital*: draft_r1, draft_r23 round buckets, rb_early_rookie

**Rejected candidate blocks** (kept building as inert zero-filled columns;
re-test only via `bff.select_features`):

| block | tune delta | why it failed |
|---|---|---|
| red-zone/goal-line usage | −0.0049 | rz_target_share r = 0.89 with opp_target_share — the opportunity block already carries it |
| Vegas preseason win totals | −0.0027 | r = 0.54 with team_fp_prior_z; backward-looking priors already price it |
| snap share | −0.0020 | r = 0.53 with has_prior; adds little over volume shares |
| rookie×vacated landing spot | hurt the joint | redundant with rookie_pedigree + vacated interactions |
| rookie_log_pick | (dropped pre-score) | r = 0.917 with rookie_pedigree |

The broad finding: most "expert" stats are already priced into the
ECR/ADP/prior-volume feature set. What survived is orthogonal information —
durability history, team financial commitment, draft-round structure.

## 5. Data

nflverse (weekly stats, rosters incl. week-1 snapshots, draft picks,
schedules/coaches, play-by-play, snap counts, injuries, contracts), ADP
exports, FantasyPros ECR: 2015-2020 via Wayback cheatsheet captures,
2021-2025 via the DynastyProcess archive, 2026 via a FantasyPros export;
preseason Vegas win totals 2010-2024 via Wayback captures of
sportsoddshistory.com. All raw commercial data git-ignored; derived
parquets committed. Fetch commands live in the module docstrings.

## 6. The 2026 board

185 players, `reports/rankings_2026.csv`; steals (ADP-rank minus our-rank ≥
24, ADP ≤ 120) with plain-language reasons in `reports/steals_2026.csv` (6
under the drafted-slot curve; the finish-rank curve manufactured 15, most of
them mid-TE/QB artifacts of the old curve's inflated tails). Sanity gates
(asserted every run): ≤ 2 QBs in top 15, top-3 all RB/WR, only QB/RB/WR/TE.
Top 5, all WR: Nacua, Smith-Njigba, Jefferson, Chase, Lamb; RBs (Bijan,
Gibbs, McCaffrey, Taylor) fill 6-9. The WR-heavy top is the drafted-slot
curve expressing itself — the finish-rank curve had overpriced early-RB
scarcity the way it overpriced mid-TE/QB. Ja'Marr Chase sits #4 (ECR 1 /
ADP 4), consistent with his expert rank; no QB before #17.

Live-2026 caveats: contracts data lacks the 2026 draft class (14 pool
players zero-filled); Vegas/snap 2026 inputs are null (both blocks rejected
anyway, columns inert).

## 7. Reproduce

```bash
uv run python -m bff.model                       # tune + preds_model.parquet (2018-2025)
uv run python -m bff.model --baselines           # preds_adp.parquet, preds_ecr.parquet
uv run python -m bff.model --season 2026         # 2026 board + steals
uv run python -m bff.backtest data/processed/preds_model.parquet --name model
uv run python -m bff.backtest data/processed/preds_adp.parquet   --name adp
uv run python -m bff.backtest data/processed/preds_ecr.parquet   --name ecr
uv run python -m bff.compare data/processed/preds_model.parquet data/processed/preds_adp.parquet
uv run python -m bff.compare data/processed/preds_model.parquet data/processed/preds_ecr.parquet
uv run python -m bff.select_features             # candidate blocks, tune window only
```

Model and evaluation outputs are deterministic and byte-stable across
reruns. This report is hand-written; update it whenever the numbers it
cites change.
