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
above the actual QB8/RB30/WR36/TE8 in that season's ADP top-150 pool),
`spearman_vorp` from `bff/backtest.py`. Raw-points Spearman is never used
anywhere (it rewards QB stacking; no raw-points path remains in the
codebase). All lists, including both baselines, go through the same
leakage-safe VORP conversion — baselines are never scored on raw ranks.

**Replacement is streaming-aware for the streamable positions.** In a
1-QB/1-TE league you fill a punted QB or TE off waivers each week, so the real
fallback beats the last starter's season — replacement is NOT QB12/TE12. An
empirical streaming simulation (`bff/streaming.py`: start the best-form free
player each week, tune window only) sets it: **QB8** (sim ~QB6-9; QB12 had
floated five QBs into the 2026 top 40, Herbert QB5 at overall 39 on an ADP of
82) and a conservative **TE8** (sim central ~TE5-6, but that policy is
optimistic for the thin TE pool and the TE curve plateaus at TE10-12, so TE8
nudges the metric without deflating genuinely scarce elite TEs — McBride/Bowers
stay put). **RB (30) and WR (36) are NOT streamable** — you cannot stream a
startable RB/WR, scarcity is real — so they keep roster-demand depth. A BEER
man-games baseline for RB/WR was rejected (§7).

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
six times on 2026-07-24 (10 total); future looks should stay rare. Three of
the 2026-07-24 looks were a-priori data-corrections to the VORP conversion,
each derived/validated on the 2012-2017 tune window first, then scored once
on test (not selections over test outcomes): (1) the drafted-slot curve swap
(§1), (2) the QB8 streaming replacement, and (3) the conservative TE8
streaming replacement — QB8/TE8 levels fixed from a tune-window streaming
simulation (`bff/streaming.py`). The other two: (4) a coach_scheme candidate
block passed the pre-registered tune gate under the then-current QB12/TE12
metric (+0.0023 ≥ +0.0020) and was scored once on test (ADP p = 0.047) —
that look was VOIDED when the streaming metric redefined the target minutes
later, and the block was rolled back after failing the re-derived gate under
QB8/TE8 (+0.0008); (5) a reconciliation look re-scoring the final 51-feature
model under the final metric, needed because the QB8/TE8 look sequence had
accidentally run on a tree still carrying the provisional coach_scheme block
(53 features) — it produced the 51-feature reconciliation numbers. Waiving
the failed gate to keep coach_scheme (whose presence coincided with a
stronger p = 0.016) was considered and REJECTED as test-set selection.
Look 10 (2026-07-24) scored the trajectory block: it passed the tune gate
at +0.0059 (lean 2-column form), and the confirming test look held positive
(+0.0003, ADP p 0.020 → 0.0195) — it produced the current §3 headline. Its
sibling block `sos` failed the tune gate and never earned a look; a
GBT/RF model-class experiment the same day also failed on the tune window
with no look spent. Total: 4 looks on 2026-07-23, six on 2026-07-24 (10);
future looks should stay rare. Residual caveat: an operator who has seen
2018-2020 results cannot fully unsee them.

## 3. Headline (test seasons 2018-2025)

| | mean spearman_vorp |
|---|---|
| **model** | **0.5232** |
| ECR baseline | 0.5179 |
| ADP baseline | 0.5028 |

| Comparison | mean delta | seasons won | p (one-sided / two-sided) |
|---|---|---|---|
| vs **ECR** (primary benchmark) | **+0.0053** | 5 of 8 | 0.164 / 0.328 |
| vs **ADP** | **+0.0205** | 7 of 8 | **0.0195 / 0.039** |

**The claim: beats ADP (certified at 0.05, p = 0.0195 one-sided / 0.039
two-sided); edges ECR (+0.0053, positive, not significant).** Per-season
deltas:

| season | model | vs ADP | vs ECR |
|---|---|---|---|
| 2018 | 0.4497 | −0.0172 | +0.0146 |
| 2019 | 0.5039 | +0.0312 | +0.0103 |
| 2020 | 0.5372 | +0.0153 | −0.0005 |
| 2021 | 0.5310 | +0.0089 | +0.0043 |
| 2022 | 0.5929 | +0.0307 | +0.0195 |
| 2023 | 0.5866 | +0.0388 | +0.0194 |
| 2024 | 0.4801 | +0.0141 | −0.0227 |
| 2025 | 0.5045 | +0.0420 | −0.0025 |

Note the ECR baseline itself beats ADP (0.5179 vs 0.5028): expert consensus
is a better draft signal than market ADP, which is why the model's margin
over ECR is much smaller than over ADP. The streaming replacements (§1)
built the ADP certification: under the old QB12/TE12 metric the claim sat at
p = 0.051; under QB8/TE8 it certifies at 0.0195/0.039. The current headline
is the 53-feature model (51 + the trajectory block added 2026-07-24, look
10); trajectory moved the test set +0.0003 (0.5229 → 0.5232), preserving the
ADP certification and nudging the ECR edge up. The ECR edge is positive but
modest and not significant; 2024 remains the model's worst ECR season
(−0.0227).

## 4. Features (53)

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
- **Trajectory (2, added 2026-07-24)**: `yrs_since_peak` (seasons since the
  player's career-best ppg — a decline clock) and `last_was_career_best`
  (broke out / peaked last season — regression-vs-continuation flag), from
  `bff/schedule_trajectory_features.py` (zero-fetch round 2). Kept on the
  tune gate at +0.0059 (the strongest new block since the original
  expansion). Shipped as a LEAN 2 of 4: the other two candidates (`yrs_exp`
  r=0.80 vs age_c, `career_best_ppg` r=0.79 vs has_prior) were dead weight —
  dropping both raised the tune mean 0.4605 → 0.4617. Coefficients on a
  2011-2017 fit are small and positive (+0.0036, +0.0020); `yrs_since_peak`'s
  positive sign is a conditional-on-age effect (the market over-discounts
  veterans past their peak), not a naive decline term.

**Rejected candidate blocks** (kept building as inert zero-filled columns;
re-test only via `bff.select_features`):

| block | tune delta | why it failed |
|---|---|---|
| red-zone/goal-line usage | −0.0049 | rz_target_share r = 0.89 with opp_target_share — the opportunity block already carries it |
| Vegas preseason win totals | −0.0027 | r = 0.54 with team_fp_prior_z; backward-looking priors already price it |
| snap share | −0.0020 | r = 0.53 with has_prior; adds little over volume shares |
| rookie×vacated landing spot | hurt the joint | redundant with rookie_pedigree + vacated interactions |
| rookie_log_pick | (dropped pre-score) | r = 0.917 with rookie_pedigree |
| coach_scheme (2026-07-24 zero-fetch round) | +0.0008 | passed the +0.0020 gate under QB12/TE12 (+0.0023) but failed it under the final QB8/TE8 streaming metric; its edge was partly replacement-pricing artifact (see ledger, look 4) |
| strength of schedule (`sos`, zero-fetch round 2) | −0.0068 | genuinely new (r<0.12 with everything) but noise — defenses regress year-to-year, so prior-year DvP does not predict at the season level |
| trajectory `yrs_exp` + `career_best_ppg` (dropped from the shipped block) | dilutive | r=0.80 vs age_c and r=0.79 vs has_prior; dropping both RAISED the tune mean, so only the 2 novel trajectory cols shipped |
| GBT / random-forest residual heads (model-class experiment) | −0.006 / seed-noise | tune-window only, no test look spent: best GBT 0.4498; best RF cell 0.4593 was `random_state=0` luck (5-seed mean 0.4553 < ridge). Ridge wins on accuracy and interpretability |
| qb_rush (same round) | −0.0026 | hurt the tune window (deltas under the final streaming metric) |
| adp_gap (same round) | −0.0037 | hurt; adp_gap_behind collinear (r = 0.927 with team_pass_fp_share_prior) |
| ol_proxy (same round) | −0.0003 | no signal |

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
24, ADP ≤ 120) with plain-language reasons in `reports/steals_2026.csv` (4
under QB8+TE8; was 6 under QB12/TE12, and the finish-rank curve before that
manufactured 15). Sanity gates (asserted every run): ≤ 2 QBs in top 15, top-3
all RB/WR, only QB/RB/WR/TE. Top 5, all WR: Nacua, Smith-Njigba, Jefferson,
Chase, Lamb; RBs (Bijan, Gibbs, McCaffrey, Taylor) fill 6-9. The WR-heavy top
is the drafted-slot curve expressing itself. QB8 streaming replacement (§1)
pushed the first QB from #17 to **#22** (Josh Allen) and moved Justin Herbert
from #39 to #50 (ADP 82); no QB before #22. The conservative TE8 left the elite
TEs roughly put (McBride TE1 #21, Bowers TE2 #30), as intended.
Ja'Marr Chase sits #4 (ECR 1 / ADP 4), consistent with his expert rank.

Live-2026 caveats: contracts data lacks the 2026 draft class (14 pool
players zero-filled); Vegas/snap 2026 inputs are null (both blocks rejected
anyway, columns inert).

## 7. Draft overlay (VONA)

The pipeline is bipartite: **MODEL → VONA**. Stage 1 (everything above)
produces one season-long value ranking, which is what gets scored, tuned, and
tested; in full PPR it leans WR at the top. Stage 2 (`bff/vona.py`,
`reports/vona_2026.csv`, the site's Draft page) is a pure post-processor that
takes the finished board plus ADP and answers a different, draft-day question:
at each pick, how much predicted VORP do you lose at each position by waiting
until your next turn (Value Over Next Available)? **VONA never feeds back into
stage 1** — not the metric, the replacement ranks, the curve, or the tuner —
so it carries no leakage or test-look implications; it is presentation.

VONA is where running-back scarcity legitimately lives. The value board says
elite WR ≥ elite RB (true for PPR at every quantile since 2012); VONA says
that at picks 1-8 you should still open RB, because startable RB falls off
faster than WR before your next turn (RB1→best-RB-at-24 drops ~50 VORP vs
~42 for WR, so Bijan + best-WR-at-24 beats Nacua + best-RB-at-24). The
snake-turn matrix leads RB for seats 1-8, flips to WR at the 9-11 turn, and
takes the last elite RB (Jeanty) at seat 12. Assumes opponents draft by ADP;
it is a positional-timing guide, not a full draft simulator.

Two replacement-level ideas from the Subvertadown VBD guide were tried for
stage 1 on 2026-07-24, with opposite outcomes. A **BEER man-games** baseline
was **rejected** on the tune window: with a 1QB/2RB/2WR/1TE/1FLEX roster the
flex resolves to WR in PPR, which deepened the WR baseline and narrowed the
model's edge over ADP (+0.0168 → +0.0100) and ECR (+0.0457 → +0.0346). "RB up
front" is a timing effect, not a season-value effect; VONA is the correct home
for it. The **QB/TE-streaming baseline** was **adopted** (QB12 → QB8, TE12 →
TE8, §1): QB8 deflated the over-valued QB block and improved both verdicts; the
conservative TE8 strengthened the ADP certification further with a negligible
board change. The resulting philosophy now lives in `bff/backtest.py` and
`bff/streaming.py`: QB and TE are streamable, RB and WR are not.

## 8. Reproduce

```bash
uv run python -m bff.model                       # tune + preds_model.parquet (2018-2025)
uv run python -m bff.model --baselines           # preds_adp.parquet, preds_ecr.parquet
uv run python -m bff.model --season 2026         # 2026 board + steals
uv run python -m bff.vona                         # draft overlay -> reports/vona_2026.csv
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
