# betterffrank: Preseason Fantasy Rankings, Scored on Draft Value (VORP)

**Shipped list = `opp_residual` (v3): ridge residual over a frozen ECR-primary market anchor, with preseason roster/context features plus prior-year weekly opportunity features, re-ranked by predicted VORP.**

- **vs ADP (2015-2025, 11 seasons): mean VORP Spearman 0.4931 vs 0.4678 for VORP-ordered ADP; wins 9 of 11 seasons** (losses: 2018 by 0.004, 2021 by 0.013). Exact sign-flip permutation p = 0.0039 one-sided (0.0078 two-sided). This edge is real and is the project's defensible claim.
- **vs FantasyPros ECR (its era, 2021-2025): mean VORP Spearman 0.5353 vs 0.5298 (+0.0055), winning 3 of 5 seasons. This is a statistical tie, and the headline must say so.** The raw permutation p is 0.31 one-sided; adjusted for the 14 model variants this project scored on the same 5 seasons, the selection-adjusted p is 0.75 (max-statistic joint sign-flip test). The pre-stated v3 bar (4-5 season wins or mean edge >= +0.02) is not met. **The model ties ECR; it does not beat it.**
- vs v2 (`context_residual`, 0.4875 / 0.5348): v3 beats v2 in 8 of 11 seasons vs ADP (mean +0.0056) and is flat in the ECR era (+0.0005), flipping 2025 from an ECR loss to a win.

2026 rankings: `reports/rankings_2026.csv` (185 players, FFC ADP snapshot July 2026; score = predicted VORP). Steals: `reports/steals_2026.csv` (13 players, reasons read off the model's own feature contributions, opportunity/velocity drivers tagged). 2026 opportunity features (including late-season velocity) were computed from the 2025 weekly file and were available for 161 of 185 pool players; the 24 without prior NFL weekly data (mostly rookies) are handled by the `has_prior_weekly` flag.

## The three-round arc (read this first)

**Stage 1 — the raw-points trap.** The first version of this project claimed a huge edge (+0.17 Spearman vs ADP) by scoring rankings against raw total PPR points. That metric rewards stacking QBs at the top (the old list had Josh Allen #1 overall): elite QBs outscore all RBs/WRs in total points, but in a 1QB league you start one QB and the drop from QB1 to QB12 is small relative to RB1 to RB30. Raw-points Spearman measured "who predicts total points", not draft value. On that metric our raw model lists scored 0.5495 vs ADP's 0.3763 — and on draft value those same lists fell *below* ADP.

**Stage 2 — VORP correction (v1).** All lists are re-scored against actual VORP (points above the actual QB12/RB30/WR36/TE12 in that season's pool), and every model list is re-ranked by *predicted* VORP: within-position order is kept, cross-position interleaving comes from a leakage-safe historical points-by-positional-rank curve (seasons < t only, smoothed, monotone; `bff/vorp.py`). The same conversion is applied to the ADP and ECR baselines, so the comparison is fair. Result: v1 `market_residual` beat ADP 10 of 11 seasons (0.4863 vs 0.4678) but lost to VORP-ordered ECR in the ECR era (0.5233 vs 0.5298, 2 of 5 seasons).

**Stage 3 — context (v2).** Two changes: (a) the market anchor re-blended ECR-primary (0.7 log ECR + 0.3 log ADP when ECR exists; ADP-only pre-2021 and for 2026), and (b) 20 preseason context features — vacated volume from departures, arriving veterans, draft-capital competition, QB changes and expected-QB quality, coaching changes, team offensive environment — plus three position interactions. Same walk-forward, leak-free protocol; hyperparameters tuned only on 2012-2014. Result: 0.4875 vs ADP (8-3), 0.5348 vs ECR (3-2) — first parity with ECR, not a win.

**Stage 4 — opportunity (v3, shipped).** 39 prior-year weekly *opportunity* features (`bff/opportunity_features.py`): usage levels (target/air-yards/carry share, per-game volume), opportunity-production divergence (fantasy points over expected, TD rate vs position on real opportunity, YPT/RACR/EPA), late-season **velocity** (OLS slope, last-6-minus-mean, last-4-vs-first-4 for target share, WOPR, carry share, targets, attempts), stability (target-share stdev, boom rate), and QB volume. The anchor stayed frozen at v2's. Three architectures were tested; the winner (`opp_residual`) adds a 9-feature curated opportunity subset to v2's ridge with group shrinkage, tuned only on 2012-2014. Result: 0.4931 vs ADP (9-2) — the project's best — and 0.5353 vs ECR (3-2, still a tie).

## Data sources

| Source | File(s) | Use |
|---|---|---|
| nflverse weekly player stats 2010-2024 | `data/raw/stats/player_stats_{2010..2024}.parquet` | Actual PPR outcomes; lagged production features; TD shares; team volume; v3 opportunity/velocity features |
| nflverse 2025 weekly + season aggregate (REG) | `data/raw/stats/stats_player_week_2025.parquet`, `stats_player_reg_2025.parquet` | 2025 actuals; prior-year and opportunity features for 2026 scoring |
| FantasyFootballCalculator preseason PPR ADP (12-team) | `data/raw/adp/ppr_{2010..2026}.json` | Market baseline, model feature, evaluation pool (2025 via wayback snapshot) |
| DynastyProcess archive of FantasyPros ECR (redraft-overall) | `data/raw/db_fpecr.parquet` | Second market signal, 2021-2025 preseason snapshots (Sep 5-10 each year); no 2026 feed |
| DynastyProcess player-ID crosswalk | `data/raw/db_playerids.csv` | GSIS/FantasyPros ID joins, birthdate, draft pedigree (its current-team column is never used for historical rosters) |
| nflverse draft picks, rosters, schedules | `data/raw/context/` → `data/processed/ctx_*.parquet` | v2 context: April draft classes, offseason/season rosters, week-1 coaches, expected QBs |

Name matching for joins: lowercase, strip punctuation and Jr/Sr/II-V suffixes; join on (normalized name, position) with an unambiguous name-only fallback.

## Methodology

- **Pool.** For each season t, POOL(t) = players in that season's preseason ADP with `adp_rank <= 150`, positions QB/RB/WR/TE, matched to a GSIS ID (~149-150/season). All rankings are evaluated on the identical pool.
- **Target and metrics.** Actual season VORP (total PPR points minus same-season positional replacement, QB12/RB30/WR36/TE12 within the pool). Primary metric: Spearman vs actual VORP; also top-24 VORP hit rate and NDCG@100 on VORP gains. Means over 2015-2025; ECR-era means over 2021-2025.
- **Walk-forward, leak-free.** For each eval season t: training targets only from seasons < t; features only from seasons < t stats/outcomes plus season-t *preseason* facts (ADP, ECR, April draft picks, offseason roster assignment, week-1 coaches) and static attributes. Hyperparameters tuned only on pre-2015 walk-forward seasons (2012-2014). The VORP curve for season t uses seasons < t only.
- **v3 winner architecture (`opp_residual`).** Score = per-position market-implied expectation of log1p(points) (quadratic in log blended market rank; anchor frozen at v2's 70/30 ECR/ADP, ADP-only pre-2021 and 2026) + 0.3 × ridge-predicted residual (alpha=30, standardized features). Features: v2's full set (v1 + 20 context + 3 interactions) + a curated 9-feature opportunity block (`opp_target_share`, `opp_air_yards_share`, `opp_ts_slope`, `opp_ts_l4f4`, `opp_cs_l6_delta`, `opp_fp_oe_pg`, `opp_td_per_opp_vs_pos`, `opp_boom_rate`, `has_prior_weekly`) with group scale g=1.0. The subset (from 5 predeclared candidates), alpha, g, and shrink were tuned only on 2012-2014 (winner 0.5498 pre-2015 vs 0.5462 for no-opp under the same protocol). Opportunity nulls filled 0.0 with missingness flags; exact-collinear columns (`opp_ppg`, `opp_wopr`, raw `opp_ypt`/`opp_td_per_opp`) dropped a priori. Ordinal scores converted to predicted VORP through the historical curve.
- **v3 alternatives.** `opp_gbm`: v2's `context_gbm` architecture (seed-bagged LightGBM residual on VORP over the same frozen anchor, per-season fitted stacking shrink) + all 40 opportunity features; tuned {head, monotone, feature_fraction} on 2012-2014. `opp_ablation`: v2 ridge + opportunity *levels* only (B) and levels + *velocity* (C) — the clean marginal test of what opportunity data adds linearly.
- **Prior rounds.** v2 `context_residual` (previously shipped): same architecture without the opp block, alpha=300. v2 `context_gbm`: GBM residual head. v2 ablations A (v1 + ECR anchor) / B (A + context).

## Results (all VORP Spearman)

### Summary means, all rounds

| Round | List | 2015-2025 (11 szn) | 2021-2025 (ECR era) |
|---|---|---|---|
| market | ADP, VORP-ordered | 0.4678 | 0.5006 |
| market | ECR, VORP-ordered | — | 0.5298 |
| v1 | `market_residual` | 0.4863 | 0.5233 |
| v2 | ablation A (v1 + ECR anchor) | 0.4866 | 0.5241 |
| v2 | ablation B (A + context feats) | 0.4892 | 0.5254 |
| v2 | `context_residual` (prev. shipped) | 0.4875 | 0.5348 |
| v2 | `context_gbm` | 0.4888 | 0.5342 |
| v3 | opp ablation A (= v2 rewired, verified) | 0.4875 | 0.5348 |
| v3 | opp ablation B (+ opp levels) | 0.4856 | 0.5324 |
| v3 | opp ablation C (+ levels + velocity) | 0.4849 | 0.5323 |
| **v3** | **`opp_residual` (shipped)** | **0.4931** | **0.5353** |
| v3 | `opp_gbm` | 0.4890 | 0.5325 |

### Per-season, key lists

| Season | ADP-V | ECR-V | v1 mkt_resid | v2 ctx_resid | v3 opp_resid | v3 opp_gbm |
|---|---|---|---|---|---|---|
| 2015 | 0.3678 | — | **0.3909** | 0.3815 | 0.3876 | 0.3672 |
| 2016 | 0.3374 | — | 0.3396 | 0.3292 | **0.3455** | 0.3297 |
| 2017 | 0.4552 | — | 0.4777 | 0.4745 | 0.4821 | **0.4878** |
| 2018 | 0.4644 | — | 0.4662 | 0.4474 | 0.4601 | **0.4965** |
| 2019 | 0.5096 | — | 0.5473 | 0.5465 | 0.5517 | **0.5553** |
| 2020 | 0.5087 | — | 0.5104 | 0.5097 | **0.5210** | 0.4799 |
| 2021 | 0.5336 | **0.5401** | 0.5253 | 0.5327 | 0.5206 | 0.5298 |
| 2022 | 0.5522 | 0.5693 | 0.5756 | 0.5885 | 0.6000 | **0.6156** |
| 2023 | 0.5276 | 0.5580 | 0.5672 | **0.5674** | 0.5605 | 0.5512 |
| 2024 | 0.4768 | **0.5154** | 0.4958 | 0.5148 | 0.5119 | 0.5015 |
| 2025 | 0.4126 | 0.4660 | 0.4528 | 0.4707 | **0.4835** | 0.4643 |
| **MEAN** | 0.4678 | 0.5298* | 0.4863 | 0.4875 | **0.4931** | 0.4890 |
| **ECR era** | 0.5006 | 0.5298 | 0.5233 | 0.5348 | **0.5353** | 0.5325 |

*ECR mean over 2021-2025 only. Bold = best list that season.

**Win records vs ADP-VORP (11 seasons):** v1 10-1; v2 `context_residual` 8-3; **v3 `opp_residual` 9-2** (losses 2018 -0.004, 2021 -0.013; one-sided exact permutation p = 0.0039). v3 also beats v2 head-to-head 8 of 11, mean +0.0056.
**Win records vs ECR-VORP (2021-2025):** v1 2-3; v2 `context_residual` 3-2 (+0.0050); **v3 `opp_residual` 3-2 (+0.0055)** — wins 2022 +0.031, 2023 +0.003, 2025 +0.018; losses 2021 -0.020, 2024 -0.004. `opp_gbm` 1-4 (+0.0027); opp ablation B 4-1 but only +0.0026 mean. None meets the pre-stated v3 bar (4-5/5 wins or mean edge >= +0.02).

### What the opportunity/velocity ablation says

The clean marginal test (same ridge architecture, same frozen anchor, per-variant tuning on 2012-2014):

- **Levels (B - A): mean -0.0020 (2015-2025) / -0.0025 (ECR era); B > A in only 6 of 11 seasons; per-season deltas span -0.016 to +0.014.** Prior-year usage levels add nothing linearly — v1's `prev_ppg`/`ppg_mismatch` already carries the t-1 production signal and levels are collinear with it.
- **Velocity on top (C - B): mean -0.0006 / -0.0001; C > B in 5 of 11 seasons; deltas span -0.020 to +0.021.** Late-season role velocity, added linearly, is noise — no consistent sign, no season pattern.
- What *did* move the needle is the curated mix in `opp_residual` — divergence (TD rate on real opportunity, points over expected) and stability (boom rate) terms alongside two velocity terms — worth +0.0056 over v2 on 2015-2025. Even that is modest, and the ablation says the levels/velocity tiers per se are not where it comes from.
- Inside the GBM (`opp_gbm`), opportunity features absorb ~32% of gain importance but displace v1/context gain rather than adding rank skill (+0.0002 vs `context_gbm` over 2015-2025; market anchor still ~36% of gain, log_rank alone 28%).

**Velocity verdict: prior-year usage velocity did not add measurable ranking skill in any architecture tested.** One nuance: ridge splits the near-collinear velocity terms into opposite-signed pairs (diagnostic fit: `opp_wopr_slope` +0.19 vs `opp_ts_slope` -0.13), so a joint late-season-WOPR-ramp direction exists in the fit, but its out-of-sample contribution is too small to see against 11 seasons of noise.

## Audit summary (v3 round + cumulative fragility)

An independent audit re-ran all v3 models end-to-end from raw weekly data (the v2 audit's findings, including the not-credibly-a-priori anchor flag and the immaterial roster-timing sensitivity, stand and carry forward). Findings:

- **Code-clean, all three v3 models.** Every `opp_*` feature for season t aggregates t-1 REG weekly rows only; velocity windows sort strictly within t-1; 2026 rows use only the 2025 weekly file. Rebuilt feature parquet byte-identical. Training capped at seasons < t everywhere; tuning hard-coded to 2012-2014 with v2's frozen alpha/shrink grids; anchor imported unchanged from v2; no stored-params file where an eval-informed value could hide. All claimed scores reproduced to 0.00e+00 (preds byte-identical; `opp_gbm` to 6e-14 LightGBM float noise).
- **vs ADP survives scrutiny:** 9/11, mean +0.0253, one-sided exact permutation p = 0.0039, sign test p = 0.033; qualitatively robust even under a 14-variant max-statistic view, since nearly all variants beat ADP.
- **vs ECR does not — the key finding.** Across the whole project, 14 variants were scored on the same 5 ECR-era seasons. Under a max-statistic joint sign-flip test (flips applied jointly across variants, preserving their correlation), the best observed edge (+0.0055, `opp_residual`) has **selection-adjusted p = 0.75**. The edge is fully consistent with picking the best of 14 correlated tries on 5 seasons. The ECR-era comparison is exhausted; further mining of these 5 seasons cannot produce a defensible claim.
- **Disclosures.** (1) Exactly two v3 configurations ever touched eval seasons: an initial tuner variant with an expanded alpha grid was scored once (0.4827/0.5309), then the grid was reverted to v2's frozen protocol — treat the grid-cap choice as partially eval-informed; the final config within the grid was selected purely on 2012-2014. (2) The "curated" opportunity subset is labeled a-priori but was authored after v2's eval results were known; designer degrees of freedom exist upstream of the pre-2015 selection. (3) Permutation p-values are quoted one-sided unless noted (two-sided: 0.0078 vs ADP, 0.625 vs ECR). (4) The pre-2015 tuning landscape is nearly flat (~0.001 separations), so the subset choice is weakly identified; shrink=0.3 gives the residual limited influence by construction. (5) FP-expected weights (1.5/tgt, 0.07/air yd, 0.6/carry) are code constants never optimized; they feed only 2 of 39 features.

## Which features mattered

### Opportunity (v3 winner, standardized ridge coefficients; full-history fit, residual log-points per 1 SD)

- **`opp_td_per_opp_vs_pos` +0.052 — the strongest opportunity signal:** TD rate relative to position on real opportunity carries information the market anchor does not fully price.
- **`opp_boom_rate` -0.059:** spike-week-driven seasons fade relative to their market price; steady producers hold value.
- **`opp_fp_oe_pg` +0.019:** production over opportunity-expected persists slightly — it is *not* mean-reverting, contrary to the regression-to-volume hypothesis.
- **`opp_target_share` +0.024, `opp_air_yards_share` -0.020:** small level effects after the anchor.
- **Velocity (`opp_ts_slope` +0.042, `opp_ts_l4f4` -0.046, `opp_cs_l6_delta` +0.023):** the slope/l4f4 pair is near-collinear and ridge splits it with opposite signs — individual signs unstable; joint contribution real but modest.
- **`has_prior_weekly` -0.267** nearly cancels v1's `has_prior` +0.289 (net ~0).
- Diagnostic all-39 fit: raw prior volume (`opp_games` -0.13, `opp_carries_pg` -0.07, `opp_targets_pg` -0.05) is an *anti-signal* once the anchor prices it; the dominant positive velocity direction is `opp_wopr_slope` +0.19.

### Context (v2 findings, unchanged)

`arriving_vet_usage` (-0.066) and `draft_competition` (-0.038) remain the clearest context penalties; vacated carries help RBs and vacated receiving PPR production helps generally; `is_rookie` (+0.070) offsets most of the v1 rookie penalty once pedigree and competition are controlled; `qb_quality_delta` mildly positive. Dead weight: `coach_change`, `qb_change`, `qb_rookie`, `depth_rank_adp`, `team_missing`. The non-context workhorses remain the market anchor itself, `ppg_mismatch`, `games_missed`, and `td_share_c`.

## 2026 rankings (VORP-based)

`opp_residual` trained on target seasons 2011-2025, scored on the 2026 FFC ADP pool (185 matched players; no 2026 ECR exists, so the frozen anchor's ADP-only path applies), converted to predicted VORP with the 2010-2025 positional curve. Output: `reports/rankings_2026.csv` (`our_rank, player, position, team, adp, adp_rank, delta = adp_rank - our_rank, score` = predicted season points above a replacement starter at the same position, comparable across positions). 2026 opportunity/velocity features were computed from the 2025 weekly file and were available for 161 of 185 pool players; the 24 without prior NFL weekly data (mostly rookies) run on the flagged-missing path.

Sanity gates pass: 1 QB in the top 15 (Josh Allen, #15; gate allows up to 2), no kickers/DST in the pool, top 3 all elite RB/WR.

### Top 25

| # | Player | Pos | Team | ADP rank | Delta | Pred VORP |
|---|---|---|---|---|---|---|
| 1 | De'Von Achane | RB | MIA | 9 | +8 | 229.4 |
| 2 | Christian McCaffrey | RB | SF | 5 | +3 | 216.4 |
| 3 | Jaxon Smith-Njigba | WR | SEA | 6 | +3 | 198.9 |
| 4 | Bijan Robinson | RB | ATL | 1 | -3 | 188.7 |
| 5 | Puka Nacua | WR | LAR | 3 | -2 | 187.9 |
| 6 | Jahmyr Gibbs | RB | DET | 2 | -4 | 171.4 |
| 7 | Ja'Marr Chase | WR | CIN | 4 | -3 | 166.8 |
| 8 | Ashton Jeanty | RB | LV | 12 | +4 | 155.1 |
| 9 | Justin Jefferson | WR | MIN | 10 | +1 | 153.5 |
| 10 | Trey McBride | TE | ARI | 32 | +22 | 148.9 |
| 11 | Amon-Ra St. Brown | WR | DET | 7 | -4 | 143.3 |
| 12 | Jonathan Taylor | RB | IND | 8 | -4 | 139.8 |
| 13 | CeeDee Lamb | WR | DAL | 11 | -2 | 134.1 |
| 14 | Brock Bowers | TE | LV | 37 | +23 | 133.7 |
| 15 | Josh Allen | QB | BUF | 29 | +14 | 130.7 |
| 16 | Chase Brown | RB | CIN | 16 | 0 | 127.2 |
| 17 | A.J. Brown | WR | NE | 15 | -2 | 125.3 |
| 18 | Lamar Jackson | QB | BAL | 56 | +38 | 120.8 |
| 19 | Drake London | WR | ATL | 13 | -6 | 116.8 |
| 20 | James Cook III | RB | BUF | 14 | -6 | 116.1 |
| 21 | George Pickens | WR | DAL | 19 | -2 | 110.3 |
| 22 | Derrick Henry | RB | BAL | 17 | -5 | 108.4 |
| 23 | Tyler Warren | TE | IND | 54 | +31 | 106.0 |
| 24 | Zay Flowers | WR | BAL | 25 | +1 | 104.9 |
| 25 | Saquon Barkley | RB | PHI | 20 | -5 | 102.9 |

Differences from the v2 list are small and concentrated where opportunity features speak: Gibbs and Jeanty move up (role velocity/efficiency), Jonathan Taylor and Amon-Ra St. Brown drop slightly, and Tyler Warren replaces Colston Loveland as the third TE steal.

### Steals (our rank ≥ 24 spots ahead of ADP, within ADP top 120)

`reports/steals_2026.csv` — 13 players. Reasons are read off the model itself: the top positive ridge feature contributions ([context] and [opp]/[opp velocity] drivers tagged), plus the delta decomposed into positional-scarcity repricing (ranking the ADP anchor alone through the same VORP curve) and the feature residual. Most of the delta is still the curve repricing mid-round QB/TE starter slots above where ADP drafts them; the feature residual adds the player-specific part (largest: Jalen Hurts +15 ranks from features, Tyler Warren +10). Opportunity contributions now appear directly in the reasons (TD efficiency on real opportunity, week-to-week stability); no 2026 steal is driven primarily by a velocity term.

| # | Player | Pos | ADP rank | Delta | Why (model contributions) |
|---|---|---|---|---|---|
| 18 | Lamar Jackson | QB | 56 | +38 | no veteran arrivals competing [context]; no positional draft capital [context]; QB2 slot: +30 scarcity, +8 features |
| 23 | Tyler Warren | TE | 54 | +31 | near-full season (durability); no arriving competition [context]; TE3 slot: +21 scarcity, +10 features |
| 26 | Dak Prescott | QB | 60 | +34 | near-full season; no arriving competition [context]; QB3 slot: +25 scarcity, +9 features |
| 35 | Drake Maye | QB | 65 | +30 | near-full season; outproduced market rank (+6.7 PPR pts/g); QB4 slot: +23 scarcity, +7 features |
| 40 | Harold Fannin Jr. | TE | 69 | +29 | team offensive environment [context]; steady week-to-week, not spike-dependent [opp]; TE5 slot: +29 scarcity |
| 47 | Tucker Kraft | TE | 76 | +29 | TD-efficient on real opportunity [opp]; outproduced market rank (+7.1 PPR pts/g); TE6 slot: +22 scarcity, +7 features |
| 50 | Justin Herbert | QB | 82 | +32 | no arriving competition [context]; no positional draft capital [context]; QB6 slot: +27 scarcity, +5 features |
| 54 | Kyle Pitts Sr. | TE | 87 | +33 | near-full season; steady week-to-week [opp]; TE7 slot: +26 scarcity, +7 features |
| 60 | Trevor Lawrence | QB | 89 | +29 | TD-efficient on real opportunity [opp]; outproduced market rank (+7.6 PPR pts/g); QB8 slot: +29 scarcity |
| 66 | Jalen Hurts | QB | 97 | +31 | TD-efficient on real opportunity [opp]; no arriving competition [context]; QB9 slot: +16 scarcity, +15 features |
| 68 | Travis Kelce | TE | 98 | +30 | near-full season; no arriving competition [context]; TE9 slot: +30 scarcity |
| 75 | George Kittle | TE | 111 | +36 | TD-efficient on real opportunity [opp]; outproduced market rank (+8.3 PPR pts/g); improved expected QB play [context]; TE10 slot: +36 scarcity |
| 80 | Jake Ferguson | TE | 116 | +36 | near-full season; no arriving competition [context]; TE11 slot: +36 scarcity |

## Limitations

- **The ECR-era result is parity, not a win.** +0.0055 over 5 seasons, one-sided p = 0.31, selection-adjusted p = 0.75 across the project's 14 variants; the edge comes almost entirely from 2022 and 2025, and 2021/2024 remain losses. Nothing in this project beats ECR once selection is accounted for. The defensible claim: ties ECR, beats ADP.
- **Opportunity data added little.** The clean ablation shows levels and velocity are noise in the linear architecture; the winner's +0.0056 over v2 rides on a weakly-identified curated-subset choice (pre-2015 tuning separations ~0.001), and its composition postdates v2's eval results.
- **2018 and 2021 remain ADP losses**; v1's 10-1 record is still unmatched even as means improved (v3 is 9-2).
- **Replacement levels fixed** at QB12/RB30/WR36/TE12 (12-team 1QB). 2QB/superflex or TE-premium leagues change the curve and the QB/TE placements materially.
- **Curve-based cross-position placement.** Gap sizes between positions come from historical points-at-rank norms, not player-specific projections; the steals list is QB/TE-dominated for exactly this reason.
- **Context-roster proxy.** Historical vacated/arriving features use full-season rosters (quantified as immaterial), and 2026 uses a cleaner offseason snapshot — a mild train/apply shift.
- **No 2026 ECR feed** — the 2026 list runs on the ADP-only anchor, the configuration with the weaker (ADP-era) evidence base; 24 of 185 pool players (mostly rookies) have no prior weekly data and rely on the missingness-flag path.
- **FFC-only ADP (12-team PPR), PPR-only scoring, short ECR era** (5 seasons, snapshots 1-2 days after the Thursday opener); name-match misses drop from the pool.

## Reproduce

```bash
cd betterffrank            # run every command from the repo root
uv sync                    # Python >=3.13; installs locked deps
# data prep
uv run python -m bff.adp && uv run python -m bff.ecr && uv run python -m bff.actuals
uv run python -m bff.context_data && uv run python -m bff.context_features
uv run python -m bff.opportunity_features
# v3 winner (walk-forward preds 2015-2025, then 2026)
uv run python -m bff.models.opp_residual && uv run python -m bff.models.opp_residual --season 2026
uv run python -m bff.backtest data/processed/preds_opp_residual.parquet --name opp_residual
# 2026 rankings + steals (consistency-checked against preds_opp_residual_2026.parquet)
uv run python -m bff.models.rank_2026_v3
```

> The exploratory alternatives discussed above (`opp_gbm`, ablations, the v1/v2
> GBM/ensemble/blend variants) were pruned from the shipped repo to keep the
> canonical pipeline clean; they remain in the project's history.

> Rebuilding `data/processed/ecr.parquet` requires `data/raw/db_fpecr.parquet`
> (FantasyPros ECR via DynastyProcess), which is not committed — see the
> README's Data section. `ecr.parquet` itself is committed, so every step after
> `bff.ecr` reproduces without it.

2026 rankings/steals are generated by `bff/models/rank_2026_v3.py` from the same walk-forward fit (per-player ridge contributions + anchor-only VORP decomposition); the script asserts its predicted VORP matches `data/processed/preds_opp_residual_2026.parquet` before writing.
