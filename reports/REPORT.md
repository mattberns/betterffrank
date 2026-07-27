# betterffrank: Preseason Fantasy Rankings, Scored on Draft Value (VORP)

**The model is ECR+**: a single ridge-residual model (`bff/model.py`) over a
FantasyPros-ECR market anchor, converted to predicted VORP. The anchor is the
expert consensus; the ridge learns, from history, where the experts were wrong
against actual outcomes and by how much. ADP enters only as feature material
(the expert/market disagreement signal `vs_adp`, depth ranks, expected-QB
identification), as the scoring pool definition, and as tiebreaks — never as
the anchor when ECR exists. As of the 2026-07-25 rebuild (§2a, §2b)
**every row the model ever sees — training, tuning and test — is anchored on
a real preseason expert board**, with no season standing in ADP for a missing
ECR. Pool ECR coverage is 99.3-100% across 2012-2025.

That sentence cost something to be able to write. The current headline:
**the model beats ADP (one-sided) and does not beat ECR** (§3). Earlier
versions claimed
an ECR edge; they were measured on inputs that included an ADP-anchored
training season, a half-missing 2012 ADP board, and three ADP-anchored tuning
folds — each of those corrections shrank the measured edge. A fourth
correction (§2f, fabricated zero-point outcome seasons) moved it back up
some, and a fifth (§2g, the player-season feature audit: a regime-shifted
and IR-blind injury variable, a stale QB table, unresolvable teams, one ECR
duplicate, truncated career histories) took a hair back; the verdict never
changed.

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

- **Tuning and feature selection: walk-forward seasons 2013-2017 only** (5
  folds; fold s trains on 2012..s-1). It was 2012-2017 / 6 folds until
  2026-07-25, when 2011 left the training set (§2b) and fold 2012 lost its
  training window. Frozen hyperparameter grid alpha ∈
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
(+0.0003, ADP p 0.020 → 0.0195) — it produced the 53-feature configuration
that still stands in §3. Its
sibling block `sos` failed the tune gate and never earned a look; a
GBT/RF model-class experiment the same day also failed on the tune window
with no look spent.
Look 11 (2026-07-25) scored the ECR backfill described in §2a: three new
preseason FantasyPros boards (2012-2014) recovered from Wayback, which is a
data completion under the anchor rule already in the code (anchor on ECR
wherever a preseason snapshot exists), not a selection over test outcomes. It
was NOT gated on the tune metric improving — it does not improve it, it costs
−0.0050 — and that is deliberate: see §2a for why the tune number falling is
the intended consequence.
Look 12 (2026-07-25) scored the clean-data rebuild in §2b: the repaired 2012
ADP board, the draft-year-gated ECR name resolution, and the removal of 2011
from the training set. Like look 11 it was not gated on the metric; it makes
the metric substantially worse and shipped because the inputs it removes are
wrong.
Look 13 (2026-07-25) scored the extended hyperparameter grid (alpha ×1000,
shrink down to 0.10) jointly with the `inj_pos` block, and REVERTED it: tune
+0.0068 (LOFO-validated) landed as −0.0037 on test, the ECR edge went to
−0.0004 and the ADP p_one from 0.0469 to 0.0820. See §2e. Look 14
(2026-07-27) scored the actuals repair in §2f: the player-season outcomes
table had silently dropped drafted players whose nflverse position field is
null, FB, or CB, fabricating 0.0-point seasons for Trent Richardson
(2012-2014), Mike Tolbert (2012), Marcel Reece (2013) and Travis Hunter
(2025) — poisoning training targets, prev-season features AND the backtest's
ground truth. Like looks 11-12 it is a data correction, not a selection: it
was tune-window-validated first (the model, ADP and ECR all rise; the model
rises most), then scored once on test. Look 15 (2026-07-27) scored the
player-season feature audit in §2g: five more input defects (regime-shifted
and IR-blind injury features, a stale expected-QB table, unresolvable pool
teams, one ECR duplicate, truncated career histories), bundled into ONE
look, each verified against raw sources and none gated on the metric.
Total: 4 looks on 2026-07-23, six on
2026-07-24, three on 2026-07-25, two on 2026-07-27 (15); future looks should
stay rare.

Two residual caveats, both real. An operator who has seen 2018-2020 results
cannot fully unsee them. And **fifteen looks is a lot.** The test set has now
been consulted often enough that "never touched for any decision" is true of
each individual look and increasingly strained as a description of the whole
process; the ADP p-value of 0.0313 should be read with that in mind, not as a
clean pre-registered result.

### 2a. The tuning surface was ADP-anchored until 2026-07-25

Until the backfill, `data/processed/ecr.parquet` started in 2015. The tune
window was then 2012-2017, so **half the tuning folds had no ECR at all**: folds
2012-2014 anchored on `log(adp_rank)` while every test fold anchors on
`log(ecr_rank)`. `vs_adp` was worse off still — its coefficient can only be
learned where the TRAINING window carries ECR variance, which was true in
just 2 of 6 folds (train std 0.0000 for folds 2012-2015, and
StandardScaler's zero-variance fallback fits the coefficient as 0).

So the tuning surface graded the model against the easy benchmark on half its
folds. Every tune-gate decision in this report — the frozen-grid
hyperparameters, the +0.0020 block gate that rejected props / coach_scheme /
sos / the tree model classes, the `vs_adp` variant choice — was measured in a
regime where "beat the market" meant "beat ADP", while the test set asks
"beat ECR". A block that adds information over ADP but not over ECR would
have passed the gate and delivered nothing.

The backfill (`bff.ecr_wayback`, three Wayback cheatsheet captures: 2012-09-01,
2013-08-18, 2014-09-01) closes that gap. Pool coverage is 98.9% / 97.3% /
100%; `vs_adp` is live in every surviving fold (fold 2012, the one place it
stayed dead because it trained on 2011 alone, no longer exists after §2b).

**What it cost, measured on the then-current 6-fold window.** The tune-window
model score fell 0.4617 → 0.4567 (−0.0050, negative in 5 of 6 folds).
Decomposed against a run with `vs_adp` dropped, `vs_adp`'s own contribution
was unchanged (+0.0018 → +0.0020); the entire loss was the anchor swap. The
new boards are not bad data — as a *baseline* they beat ADP on that window by
+0.0088 mean, the same edge ECR shows in 2015-2017. What changed is what the
tune window measures:

| tune-window edge (2012-2017, pre-§2b) | folds measurable | before | after |
|---|---|---|---|
| model − ADP | 6 of 6 | +0.0205 | +0.0155 |
| model − ECR | 3 of 6 → 6 of 6 | +0.0220 *(2015-17 only)* | +0.0068 |

The old surface could only measure the ECR edge where ECR existed, and there
it read +0.0220 — while test delivered +0.0053. The corrected surface read
+0.0068 against ECR on all six folds, in family with the test result. The
validation window went from optimistic to calibrated, and the −0.0050 was the
cost of no longer grading on a curve. (§2b then removed 2011 and the 2012
fold with it, so the current window is 2013-2017 and these six-fold numbers
are historical; the current tune-window table is in §2b.)

The uncomfortable corollary, stated plainly: under that configuration the
model tied its own ECR anchor in 2012 (−0.0003) and trailed it in 2014
(−0.0182). That is the same failure mode as test seasons 2018 and 2024, and
it foreshadowed the §2b result where the tune-window edge vanishes entirely.

### 2b. Two more input defects, and the removal of 2011 (2026-07-25)

Chasing the ECR gap surfaced two further problems in the inputs. Both are
fixed; together with §2a they are the reason every headline number in §3
moved.

**The 2012 ADP board was half missing, and it looked fine.** FantasyFootball-
Calculator's API returns 93 players for 2012 PPR, stopping at ADP 156.9, with
ten gaps wider than 4 ADP slots scattered from pick 34 to 157; 2011 returns
188 with zero such gaps and 2013 returns 191 with two. The contemporaneous
2012-09-02 capture of `adp_ppr.php?teams=12` (three days before Week 1, so
preseason-clean) carries **189 players to ADP 172.7**. The two also disagree
underneath: the API reports `total_drafts: 303` yet no player on its board was
drafted more than 130 times, impossible for a board whose top pick goes 1.02;
the capture has Arian Foster at 251. FFC pruned its own 2012 PPR data at some
point after 2012. This mattered twice: 2012 is a scored fold that was being
graded on **91 pool players against ~150 everywhere else**, and it is training
data for every later fold. Repaired via `bff.adp`'s `LEGACY_RECOVERY` (cached
under `data/raw/adp/ppr_2012_wayback.json`, with a 150-player floor assert so a
dud capture cannot silently reintroduce the truncation). 2012 now carries 162
skill players and a 98.7% top-150 gsis match, in family with every neighbour.

**ECR name resolution lacked draft-year gating.** `bff.ecr` matched id-less
rows on (norm_name, position) with no era restriction, so suffix stripping
collided across generations: "Frank Gore" (2005) and "Frank Gore Jr." (2024)
both normalize to `frank gore`, the join saw two gsis ids, gave up, and
dropped a top-40 RB from both 2012 and 2013. `bff.adp` had always gated on
`draft_year <= season`; `bff.ecr` now does too, plus three aliases for
cheatsheet name forms (`Steve Johnson` → Stevie, `Christopher Ivory` → Chris,
`Chris "Beanie" Wells` → Beanie, the last needing `norm_name` to strip double
quotes). Verified strictly additive against the previous build: **0 ids
changed, 0 lost, 8 filled.** Pool ECR coverage is now 100% for 2012 and 99.3%
for 2013 (the one miss, LeGarrette Blount at pool rank 148, is genuinely
absent from the Aug-18 board, not mismatched).

**2011 was removed from the training set as contaminated input.**
`FIRST_TARGET` moved 2011 → 2012. 2011 has no preseason ECR and never will
(no Wayback capture of the cheatsheet before 2012-06-10), so its 160 rows
anchored on `log(adp_rank)` and carried `vs_adp = 0`.

The argument that kept 2011 until now was that a training row only supplies
(market rank, outcome), so an ADP anchor is a noisier instrument for the same
quantity. **That argument is wrong for this architecture.** The model is an
anchor plus a learned *residual*, and a residual is defined relative to its
anchor: on 2011 rows the ridge learns "how wrong was ADP", and that correction
is then applied to an ECR anchor at serve time. ADP and ECR have different
systematic biases — the entire premise of `vs_adp` carrying signal — so this
is a train/serve mismatch, not added noise. And `vs_adp = 0` on those rows
does not encode "unknown"; it asserts that experts and market agreed exactly,
for 160 players with no expert board. That is fabricated feature content, at
14% of the 2018 fold's training rows and roughly half of the earliest folds'.

**The cost, reported not hidden.** Removing 2011 empties fold 2012's training
window, so the tune window is now **2013-2017, five folds**. The tuner moves
to alpha = 300 (grid maximum) / shrink = 0.3, itself a symptom of a
smaller training set. And the tune-window edge disappears entirely:

| tune window 2013-2017 | mean spearman_vorp |
|---|---|
| model | 0.4293 |
| ECR baseline | 0.4298 |
| ADP baseline | 0.4305 |

The model is 0.0013 *behind* ADP and 0.0005 behind ECR on its own validation
window. It still shows +0.0197 over ADP on test, which is a real tension and
is left standing rather than explained away. (*These numbers were measured on
the corrupted actuals table found two days later — §2f. After the §2f repair
the tune window briefly read model 0.4392 / ECR 0.4356 / ADP 0.4336; the §2g
feature audit put it back to 0.4354 — the tension stands: the model does not
lead its anchors on its own window.*)

A metric that improves when fabricated rows are added is not evidence those
rows belong; it is evidence the edge was partly resting on them. Restoring
2011 would raise the tune mean to 0.4446 and manufacture a +0.0140 tune-window
ADP edge out of an input we know to be wrong. Do not do it.

### 2c. Pre-registration for look 12 (written BEFORE the test look)

Challenger: an 8-feature ridge selected by cluster-guided forward selection on
the corrected tune window (2013-2017, all ECR-anchored):

    age_qb, ol_stuff_rate_prior, opp_ts_slope, ppg_delta, qb_rush_ypg,
    rb_early_rookie, team_pass_fp_share_prior, yrs_since_peak

at alpha=300, shrink=0.3. Selection rule stated in advance: the last admission
before the greedy trajectory's first zero-delta step. Tune-window evidence:
0.4490 vs ECR 0.4298 (+0.0192, 5/5 folds, paired t=4.40).

Context for why a rebuild was needed: on the corrected window the shipped
53-feature model scores 0.4293 against an ECR baseline of 0.4298 -- it is
NEUTRAL-TO-WORSE than the benchmark, and worse than the zero-feature anchor
(0.4306). Note the anchor-only null and the ECR baseline are the same board:
they score identically in 3 of 5 folds and the whole +0.0008 gap is one missing
pool player in 2013.

Prediction being tested: the 8-feature challenger beats ECR on 2018-2025 by
more than the shipped 53 does. Whatever the number, it is recorded here.
The tune-window figures above are optimistically biased (selected on that
window); the test look is the unbiased read.

### 2d. Look 12 RESULT: the stepwise challenger FAILED

Run 2026-07-25, exactly as pre-registered in §2c. Test seasons 2018-2025.

| | mean spearman_vorp | vs ECR | folds won |
|---|---|---|---|
| shipped 53 | 0.5237 | +0.0032 | 4/8 |
| ECR baseline | 0.5205 | -- | -- |
| **challenger (8 feat)** | **0.5192** | **-0.0013** | **2/8** |
| ADP baseline | 0.5040 | -- | -- |

**The challenger is WORSE than ECR on the test set.** It scored +0.0192 over ECR
on the tune window winning 5 of 5 folds (paired t = 4.40) and delivered -0.0013
winning 2 of 8. That is a swing of **-0.0205** between the window it was selected
on and the window it was tested on.

That number is the headline finding, not the challenger. **The selection bias in
this setup is roughly 0.02 spearman_vorp** -- larger than any real effect claimed
anywhere in this report. A 5-fold tune window with a measured per-evaluation noise
sd of 0.0020 cannot support feature selection over an 81-column pool; the search
reliably finds sets that fit the window and do not generalise. The shadow-calibrated
gated walk, which admitted NOTHING, was right, and the greedy trajectory peak was
the artifact.

Secondary reading, also unflattering: the shipped 53 beats ECR by +0.0032 on 4 of 8
seasons (paired t = 0.78). That is not a demonstrated edge over the expert consensus
-- it is a coin flip. Both models still beat ADP (+0.0197 and +0.0152, 7/8 each).

Total test looks after look 12: 12.

### 2e. Look 13: extended grid + inj_pos block, REVERTED (2026-07-25)

The tuned optimum sits ON the grid boundary at (alpha 300, shrink 0.3) — the
tuner asking for less model than the grid can express. The grid was extended
to a strict superset (alpha up to 1000, shrink down to 0.10) jointly with a
position-conditional injury block (`inj_recurrence_qb`,
`inj_weeks_listed_l2y_qb`; found via the QB-split residual-correlation table,
not a metric sweep). Tune-window evidence: 0.4293 → 0.4361 (+0.0068,
LOFO-validated), beating the ECR baseline by +0.0064 on 4 of 5 folds — the
first tune-window ECR win the project has had.

It did not transfer. One test look: mean spearman_vorp 0.5237 → 0.5200
(−0.0037); the ECR edge +0.0032 → −0.0004 (4/8, p_one 0.543); the ADP edge
+0.0197 → +0.0160, p_one 0.0469 → 0.0820. Reverted in full: the grid stays
frozen at {3,10,30,100,300} × {0.3,0.5,0.7,1.0} and the two columns stay as
the inert `inj_pos` CANDIDATE_BLOCK.

The lesson mirrors §2d at hyperparameter scale: the 5-fold tune mean has a
standard error of ~0.0022, so it cannot resolve effects of the size this
project chases. Do not spend a test look on a tune-window gain under ~0.01.
(A follow-up stepwise audit, `bff/stepwise.py` — forward-backward from the
empty set with leave-one-fold-out control, `reports/stepwise.json`, zero test
looks — makes the same point from the other direction: its held-out curve
degrades monotonically from k=0, so out of selection no selected feature set
beats the anchor-only null on this window.)

Total test looks: 13.

### 2f. Look 14: the actuals table silently dropped drafted players (2026-07-27)

Found while investigating why tune fold 2015 responded so strongly to
de-weighting bust rows (the residual-cap and drop-injury-season experiments,
both tune-window only). The answer was not a modeling insight; it was a bug.

**The defect.** `bff/actuals.py` kept only weekly rows whose nflverse
`position` ∈ {QB, RB, WR, TE}. That field is not trustworthy for fantasy
purposes: it is **null** for some ids (Trent Richardson, all three of his
seasons — an upstream metadata gap), and it uses roster positions the fantasy
boards don't (**FB** for Mike Tolbert and Marcel Reece, **CB** for two-way
Travis Hunter 2025). Those player-seasons vanished from `actuals.parquet`, and
a missing actuals row reads as **0.0 points** everywhere downstream. The full
audit (every ADP-board player × every season 2010-2025, same-id raw-weekly
comparison plus a name-based cross-id sweep) found exactly four affected
players, eight corrupt board-rows:

| player | seasons | real PPR recorded as 0.0 | where it bit |
|---|---|---|---|
| Trent Richardson | 2012, 2013, 2014 | 254.7 / 144.9 / 117.8 | training targets; scored 2013/2014 pool truth; his 2013 row (ADP **10**) also read `has_prior = 0` — a top-10 pick coded as a rookie |
| Mike Tolbert | 2012 | 114.1 | training target |
| Marcel Reece | 2013 | 111.8 | training target; scored 2013 pool truth; prev-season features |
| Travis Hunter | 2025 | 63.8 | scored 2025 **test** pool truth; 2026 board prev-season features (`has_prior = 0` on the live board) |

Three channels, stated explicitly: fabricated mega-busts in the training
target (the model learned that the 2013 #10 pick scored zero); corrupted
prev-season features (`prev_ppg = 0`, `has_prior = 0`, `games_missed` wrong);
and corrupted **ground truth** — a pool player missing from actuals is scored
as if he scored nothing, so folds 2013, 2014 and 2025 graded every list
(model, ADP, ECR) against a false outcome. 162 player-seasons were missing in
total (mostly FB seasons of sometime-board players); all channels by which
any of them reach the model or the metric are the eight rows above.

**The fix.** Rows whose raw position fails the filter now fall back to the
player's ADP/ECR-board position (`board_positions` in `bff/actuals.py`), so a
player the fantasy market drafts is always scored. `check_board_coverage`
asserts the invariant that would have caught this years of data ago: **every
ADP-board player with nonzero raw weekly points must appear in actuals for
that season.** Downstream feature tables (`context_features`,
`schedule_trajectory_features`) rebuilt from the corrected actuals.

**Tune-window validation (before the test look).** Model 0.4293 → **0.4392**;
ADP baseline 0.4305 → 0.4336; ECR baseline 0.4298 → 0.4356. All three rise
because the eval truth improved; the model rises most because it alone also
sheds the training poison. The tuner moves shrink 0.3 → 0.5 (alpha stays at
the 300 grid maximum): with the fabricated busts gone, the tune window
supports more model. The anchor-only null is 0.4366, so the model is +0.0026
over doing-nothing on its own window, where §2b had it *behind* the null.

**Why the earlier bust-tail findings must be re-read.** The residual-cap
sweep and the drop-injury-season experiments both "improved" the tune window
mostly by de-weighting these fabricated rows: on corrected data, dropping
points<50 training rows at fixed hyperparameters is worth +0.0001 (mixed
across folds). The loss-vs-metric mismatch argument (§ methodology audit)
survives as theory; its measurable payoff on this window was mostly the bug.

**Test (look 14), the current headline in §3:** model 0.5250 / ECR 0.5190 /
ADP 0.5045. vs ADP +0.0205, 6/8, p_one 0.0195 (two-sided 0.0391 — the ADP
claim clears both tests for the first time since the clean-data rebuild). vs
ECR +0.0060 at 5/8, p_one 0.117 — still not an edge. Note the direction: this
is the first data correction that moved the measured edges UP. The 07-25
corrections removed *flattering* inputs; this one removed fabricated
*outcomes* that had been degrading both the training signal and the grading.

### 2g. Look 15: the player-season feature audit (2026-07-27)

A column-by-column audit of every feature parquet (per-season null/zero
rates, means, ranges, adjacent-season discontinuities, verified against raw
sources before anything was called a defect) found five input defects. All
five are data corrections under the standing rule — none was gated on the
metric — and they shipped as ONE bundle with one test look. Several suspects
were checked and cleared as real (Guerendo's 0.0-point 2025 is a genuine
special-teams-only season, play-by-play confirmed; Shaheed's 18 games in
2025 is a midseason trade that dodged both byes; the `opp_fp_oe_pg` level
shift at 2019 is real league scoring drift over deliberately fixed weights).

**(1) The injury features were doubly broken.** `inj_weeks_listed_l2y`
counted any non-null `report_status`. The NFL abolished the "Probable"
designation after 2015 (~2,500 listings/season), and 2016+ files carry
~3,000 practice-only rows/season with null status (pre-2016: ~200), so tune
targets measured ~2x what test targets measured (pool mean 6.2 vs 3.1) —
the tune window was graded on a different variable than the test window.
Independently, players on injured reserve drop OFF the weekly report, so
the most severe injuries produced the LOWEST counts: Christian McCaffrey
missed 13 games in 2024 but had 3 listed weeks — less than a nagging
hamstring. Fix: count only Questionable/Doubtful/Out weeks (all three
features, recurrence included), unioned with reserve-list weeks (roster
status RES/PUP; INA excluded as 2020+-only, PUP folded into RES pre-2016).
The repaired series is level (3.5-5.2 tune, 4.3-6.1 test; the mild 2022+
rise is the real league-wide increase in IR usage). CMC's target-2026 count:
4 → 15.

**(2) `ctx_team_qb.parquet` was stale — built Jul 23 from the broken 2012
ADP board,** never rebuilt after the Jul 25 board repair. 18 of 32 teams had
no expected QB in 2012, so the whole 2012 tune fold had `qb_change` /
`qb_quality_delta` / `qb_rookie` constant at 0 and `qb_expected_missing`
firing on 44% of rows; 2012 rows also train every test fold. Rebuilt: 7 of
32 missing (in family with other seasons), all four features live.

**(3) 17 pool rows (2011-2015, mostly BUF) had a null team** — FFC listed
them FA at capture (C.J. Spiller 2013 at ADP **6**, Watkins 2014/2015, Fred
Jackson x4), and `build_pool` had no fallback, so every team-keyed feature
zero-filled. Fix: fall back to the week-1 roster team (the sanctioned
season-t membership table). All 17 resolve; `team_missing` now 0 everywhere.

**(4) One ECR duplicate:** 2019 "Mike Davis" the WR name-resolved onto RB
Mike Davis's gsis_id (rank 409 vs the RB's 169; both outside the pool, so
impact ~0). Fixed by id-level dedup keeping the best rank — NOT by
position-gating the name fallback, which legitimately rescues cross-position
listings (Cordarrelle Patterson WR/RB at ECR 48, Jordan Matthews WR/TE at
30). `bff.ecr` now asserts (season, gsis_id) uniqueness. Verified surgical:
exactly one row changed.

**(5) The trajectory features truncated careers at source-2010:** tune-fold
players had 2-7 visible seasons vs 8-15 in test, so `yrs_since_peak` pool
mean ramped 0.33 (2012) → 1.5 (2025) purely from window truncation. Fix:
career history now spans 1999-2025 via raw `player_stats_{1999..2009}`
(actuals.parquet deliberately stays 2010+ — it feeds training targets and
backtest truth, and pre-2010 seasons are never scored). The ramp is gone
(1.13-1.5, flat); Steven Jackson 2013 now correctly peaks in 2006.

**Tune-window validation (before the test look).** Model 0.4392 → **0.4354**
(−0.0038, ~1.7 SE); ADP 0.4336 and ECR 0.4356 baselines unchanged (no board
or outcome moved); anchor-only null 0.4366. The tuner returns to alpha=300,
shrink=0.3 (was 0.5 after §2f). The small §2f tune edge over the anchor is
gone again (−0.0012 vs the null): part of what looked like signal was fit to
the corrupted injury variable and the truncated career window. LOFO on the
repaired surface, for the record: dropping the inj trio would gain +0.0016,
dropping trajectory would lose −0.0018 — both under one SE (~0.0022), so per
the §2e lesson neither justifies a FEATURES change.

**Test (look 15), the current headline in §3:** model 0.5245 / ECR 0.5190 /
ADP 0.5045. vs ADP **+0.0201, 6/8, p_one = 0.0313 — but two-sided 0.0625,
so the ADP claim no longer clears the two-sided test** (§2f briefly had
0.0391). vs ECR +0.0056 at 5/8, p_one 0.117 — unchanged story. The
correction moved the model −0.0005 on test while both baselines held still;
per-season model numbers shuffled more (2018 0.4485 → 0.4376, 2023 0.5765 →
0.5876) than the mean suggests.

Total test looks: 15.

## 3. Headline (test seasons 2018-2025)

| | mean spearman_vorp |
|---|---|
| **model** | **0.5245** |
| ECR baseline | 0.5190 |
| ADP baseline | 0.5045 |

| Comparison | mean delta | seasons won | p (one-sided / two-sided) |
|---|---|---|---|
| vs **ECR** (primary benchmark) | +0.0056 | **5 of 8** | 0.117 / 0.234 |
| vs **ADP** | **+0.0201** | 6 of 8 | 0.0313 / 0.0625 |

**The claim, as of the 2026-07-27 feature audit (§2g): the model beats ADP
one-sided (+0.0201, p = 0.0313) but the two-sided test reads 0.0625 and no
longer clears 0.05. It does NOT beat ECR: +0.0056 at 5 of 8, p = 0.117.**
ECR is the primary benchmark, so the correct summary of this model is still:
*it beats the drafting market and matches the expert consensus* — with the
ADP claim now resting on the one-sided test alone. Per-season deltas:

| season | model | vs ADP | vs ECR |
|---|---|---|---|
| 2018 | 0.4376 | −0.0250 | +0.0067 |
| 2019 | 0.5259 | +0.0408 | +0.0259 |
| 2020 | 0.5366 | +0.0242 | +0.0161 |
| 2021 | 0.5202 | −0.0014 | −0.0026 |
| 2022 | 0.5706 | +0.0025 | +0.0005 |
| 2023 | 0.5876 | +0.0384 | +0.0118 |
| 2024 | 0.4976 | +0.0320 | −0.0080 |
| 2025 | 0.5200 | +0.0490 | −0.0058 |

Note the ECR baseline itself nearly matches the model (0.5190 vs 0.5245) and
comfortably beats ADP (0.5045). Expert consensus is a strong draft signal;
most of the model's apparent skill is inherited from its anchor (§3a).

**How the headline has moved with the data corrections.** The 2026-07-25
corrections (§2a, §2b) removed flattering inputs and shrank every edge: ECR
+0.0053 → +0.0050 → +0.0032, ADP +0.0205 (p = 0.0195) → +0.0201 (0.0352) →
+0.0197 (0.0469). The 2026-07-27 actuals repair (§2f) removed fabricated
*outcomes* and partially reversed that (ECR +0.0060, ADP +0.0205 at 0.0195 /
0.0391 two-sided). The same-day feature audit (§2g) took a hair back: ECR
+0.0056, ADP +0.0201 at 0.0313 / 0.0625. None of these moves came from
changing the model; all came from fixing inputs. Quote only the current row;
the older figures were measured against inputs now known to be wrong. And
after fifteen test looks, none of these p-values should be read as a clean
pre-registered result.

## 3a. Anchor quality: ECR vs ADP, and how much of the win is inherited

The model sits on an ECR anchor, so it inherits whatever edge ECR has. Scored
on its own through the identical VORP conversion, the expert board is **not**
uniformly better than the market — it depends on the era, and the split only
became measurable after §2a/§2b.

| | ADP | ECR | ECR − ADP | ECR wins |
|---|---|---|---|---|
| tune 2013-2017 | 0.4336 | 0.4356 | **+0.0020** | 3 of 5 |
| test 2018-2025 | 0.5045 | 0.5190 | **+0.0145** | 7 of 8 |

Per season on test: −0.0317 (2018), +0.0149, +0.0080, +0.0012, +0.0020,
+0.0266, +0.0400, **+0.0548** (2025). The margin widens materially over time.
Per season on tune: +0.0010, +0.0017, −0.0064 (2015), −0.0014, +0.0152.

(Numbers re-measured after the §2f actuals repair. One earlier soft reading
did not survive it: the pre-repair table had 2013 as ECR's largest ADP loss
anywhere, −0.0151, and blamed the mid-August 2013 snapshot's staleness. On
corrected ground truth 2013 reads **+0.0010** — the "stale board" deficit was
mostly Trent Richardson's and Marcel Reece's fabricated zero-point seasons.
The one large ADP win left standing is test 2018.)

The hard reading, and it belongs in any honest summary of this project: the
model beats ADP on test by +0.0201, but ECR *alone* beats ADP by +0.0145 over
the same seasons. The ridge contributes the remaining **+0.0056**. Roughly
seven tenths of the margin over the market is the expert list, not the model.

After the §2g feature audit the model's own contribution over its anchor is
**−0.0002 on tune and +0.0056 on test** (it was +0.0036 / +0.0060 after §2f;
part of the brief tune edge was fit to the corrupted injury variable and the
truncated career window). What differs between windows is mostly the anchor's
own value (+0.0020 in the tune years, +0.0145 in the test years), which the
model inherits wherever it stands — plus a test-window ridge contribution
that the tune window does not corroborate.

## 4. Features (53)

Score(t, player) = market-implied expectation of log1p(season points) from
the ECR anchor (per-position quadratic in log rank, vertex-clamped) +
shrink × ridge residual. Tuned: alpha = 300, shrink = 0.3 (§2f briefly moved
shrink to 0.5; the §2g feature audit moved it back — with the corrupted
injury variable and truncated career window repaired, the tune window again
supports less model). Alpha still sits ON
the grid maximum, which remains a warning sign rather than a result: the
tuner is regularizing as hard as it is allowed to, consistent with the
smaller post-§2b training set. The grid stays frozen regardless. Feature
groups:

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
  - *injury*: weeks injured (2y: Questionable/Doubtful/Out listings unioned
    with reserve-list RES/PUP weeks — see §2g for why both rules are
    load-bearing), soft-tissue mentions (2y), same-injury recurrence — from
    nflverse injury reports + weekly rosters
  - *contracts*: apy_cap_pct (largest new coefficient, +0.04),
    contract_year, rookie_deal_yr — from nflverse/OTC contracts
  - *draft capital*: draft_r1, draft_r23 round buckets, rb_early_rookie
- **Trajectory (2, added 2026-07-24)**: `yrs_since_peak` (seasons since the
  player's career-best ppg — a decline clock) and `last_was_career_best`
  (broke out / peaked last season — regression-vs-continuation flag), from
  `bff/schedule_trajectory_features.py`. Career history spans source seasons
  1999-2025 as of §2g (was 2010+, which truncated tune-fold careers). Kept
  on the tune gate at +0.0059 (the strongest new block since the original
  expansion). Shipped as a LEAN 2 of 4: the other two candidates (`yrs_exp`
  r=0.80 vs age_c, `career_best_ppg` r=0.79 vs has_prior) were dead weight —
  dropping both raised the tune mean 0.4605 → 0.4617 (both measured on the
  pre-backfill tune window; see §2a). Coefficients on a
  2011-2017 fit are small and positive (+0.0036, +0.0020); `yrs_since_peak`'s
  positive sign is a conditional-on-age effect (the market over-discounts
  veterans past their peak), not a naive decline term.

**Rejected candidate blocks** (kept building as inert zero-filled columns;
re-test only via `bff.select_features`).

**Every tune delta in this table is HISTORICAL and none of it is current.**
All of it was measured on the pre-2026-07-25 window: six folds starting at
2012, a half-ADP-anchored tuning surface (§2a), a truncated 2012 ADP board and
a 2011 training season (§2b), against a 0.4617 baseline that is now 0.4293 on
a different fold set. Two things invalidate the numbers rather than merely
shifting them: a block judged against an ADP anchor in folds 2012-2014 was
being asked "do you beat the market?" when the test set asks "do you beat the
experts?", and the tuner's operating point has moved from alpha 100 to the
grid maximum. Re-deriving every gate below on the current window costs no test
look and is now the **highest-value open item in the repo** — more so than
before, because §2b leaves the model with no measurable tune-window edge, and
a rejected block might not deserve its rejection.

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
| preseason player props (`props` / `props_dense` / `props_lean`, 2026-07-24) | −0.0052 / −0.0024 / −0.0022 | NOT collinear (max \|r\| 0.20–0.61, so the information is genuinely new) and dense in the scored pool (72–97% of the ADP top-150 carries a quote, 86–100% of the top 50) — it simply does not predict the residual. Harm is monotone in how many prop columns are added. Restricting to the folds where props are ACTIVE (2015–2017, dropping the inert 2012 fold and the thin 2013–14 ones) makes it WORSE, not better (lean −0.0053, all5 −0.0029), so the ragged early coverage is not an excuse. Verdict re-derived after a parser fix that (a) closed an OUTCOME LEAK — 8 boards list the season's winner with `N/A` odds, and the old regex paired that winner's NAME with the next row's PRICE, fabricating e.g. "Josh Gordon +275" for 2013 rec_yds, 4 of the 8 inside the tune window — and (b) recovered 74 dropped rows incl. every apostrophe name (Ja'Marr Chase et al.) and some board-topping favourites; deltas moved by <0.001 because the loss was almost entirely in 2016-2025, not the tune window. Data kept and curated (`data/processed/season_props.parquet`, 5233 quotes); columns inert. Cross-capture reconciliation over 365 independent Wayback captures: 91 boards agree, 8 differ (6 cosmetic punctuation/encoding, 1 an earlier in-preseason snapshot correctly passed over, 1 a source revision to a 2013 board), 10 single-sourced and therefore unverifiable (`reports/props_reconcile.json`) |

The broad finding: most "expert" stats are already priced into the
ECR/ADP/prior-volume feature set. What survived is orthogonal information —
durability history, team financial commitment, draft-round structure.

## 5. Data

nflverse (weekly stats 1999-2025 — 2010+ feeds actuals/training, 1999-2009
career-history only per §2g; rosters incl. week-1 snapshots and the weekly
reserve-list statuses feeding the injury features; draft picks,
schedules/coaches, play-by-play, snap counts, injuries, contracts), ADP
exports (FantasyFootballCalculator API, except 2012 and 2025 which come from
Wayback captures — see §2b for why 2012 must not use the API),
FantasyPros ECR: 2012-2020 via Wayback cheatsheet captures,
2021-2025 via the DynastyProcess archive, 2026 via a FantasyPros export;
preseason Vegas win totals 2010-2024 via Wayback captures of
sportsoddshistory.com. All raw commercial data git-ignored; derived
parquets committed. Fetch commands live in the module docstrings.

## 6. The 2026 board

185 players, `reports/rankings_2026.csv`; steals (ADP-rank minus our-rank ≥
24, ADP ≤ 120) with plain-language reasons in `reports/steals_2026.csv`
(**4** after the §2g feature audit — Lamar Jackson, Maye, Hurts, Caleb
Williams, all QBs; the §2f board had Maye/Hurts/Caleb/Herbert; it was 3
after the 2026-07-25 rebuild, and the finish-rank curve manufactured 15
before any of it). Sanity gates (asserted every run): ≤ 2 QBs in top
15, top-3 all RB/WR, only QB/RB/WR/TE. Top 5: Jefferson, Nacua,
Smith-Njigba, Chase, CeeDee Lamb; Bijan Robinson, Gibbs, Taylor and
McCaffrey fill 6-9 (CMC holds #9 despite his repaired injury count of 15
listed weeks — the coefficient is small). The WR-heavy top is the
drafted-slot curve expressing itself. QB8
streaming replacement (§1) keeps the first QB at **#27** (Josh Allen), with
none in the top 15. Travis Hunter sits #148 (ADP 163) — before §2f the board
scored him as a rookie with no 2025 season. Ja'Marr Chase sits #4
(ECR 1 / ADP 4), consistent with his expert rank.

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
