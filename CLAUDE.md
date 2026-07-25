# betterffrank — project invariants

Goal: preseason fantasy football rankings (12-team PPR redraft) that beat
market ADP and tie/beat FantasyPros ECR, evaluated on draft value (VORP),
walk-forward with zero leakage. Protocol (2026-07-23, authoritative: `reports/REPORT.md`): tune 2012-2017 (6 folds,
spearman_vorp, frozen grid), test 2018-2025 (S=8, sign-flip floor 0.0039 —
significance at 0.05 is reachable). Current standing (53-feature model,
offset-log vs_adp, drafted-slot VORP curve, QB8+TE8 streaming replacement):
model 0.5232 / ECR 0.5179 / ADP 0.5028 mean spearman_vorp; **beats ADP
+0.0205 (7/8, p_one = 0.0195 — CERTIFIED at 0.05, two-sided 0.039); edges ECR
+0.0053 (5/8, p_one = 0.164, positive but not significant)** — the primary
benchmark is ECR. Integrity ledger: 2018-2020 briefly served as tune folds
during the 2026-07-23 protocol work before moving to test (all selections
re-derived from scratch on 2012-2017; they reproduced exactly). Test-set
looks: FOUR on 2026-07-23, SIX on 2026-07-24 (TEN total) — count future
looks, keep them rare. Three 07-24 looks were a-priori data-corrections to
the VORP conversion, each tune-window-validated before one test look:
(1) finish-rank → drafted-slot curve, (2) QB12 → QB8 and (3) TE12 → TE8
streaming replacements (derived by `bff/streaming.py`; see "The metric").
Look 4: a coach_scheme block passed the tune gate under QB12/TE12 (+0.0023),
scored once on test, then VOIDED when the streaming metric landed; it failed
the re-derived gate (+0.0008 < +0.0020) and was rolled back (waiving the
gate after seeing its test number was rejected as test-set selection). Look
5: 51-feature reconciliation under the final metric. Look 10: the trajectory
block (yrs_since_peak, last_was_career_best) — passed the tune gate at
+0.0059 (lean 2-col), test-confirmed +0.0003, → the current 53-feature
standing; its sibling `sos` and a GBT/RF model-class experiment both failed
on the tune window with no look spent. Same-day history (07-23): ECR window
widened to 2015-2025 via Wayback backfill (`bff/ecr_wayback.py`); feature
expansion 40 → 51 → 53. The
zero-fetch candidate round 1 (07-24, `bff/candidate_features.py`) shipped
NOTHING (coach_scheme/qb_rush/ol_proxy/adp_gap all inert CANDIDATE_BLOCKS);
round 2 (07-24, `bff/schedule_trajectory_features.py`) shipped the trajectory
block (2 cols) and rejected `sos`. The FETCHED round (07-24,
`bff/props_wayback.py` + `bff/props.py`: 5233 preseason player-prop quotes,
2012-2025) shipped NOTHING — all three prop variants hurt the tune window
(−0.0022 to −0.0052), no test look spent; see "Player props". All current
numbers and methodology live in
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
- **Replacement is STREAMING-AWARE for QB and TE; roster-demand for RB/WR**
  (`REPL_RANKS` in `bff/backtest.py`, set 2026-07-24). Philosophy: QB and TE
  are streamable in a 1-QB/1-TE league (you fill a punted slot off waivers each
  week), so replacement beats the last starter's season → QB8, TE8 (not
  QB12/TE12). RB and WR are NOT streamable (scarcity is real) → stay at
  roster-demand depth RB30/WR36. `bff/streaming.py` derives QB/TE by
  form-streaming the free pool on the tune window (QB sim ~QB6-9; TE sim central
  ~TE5-6 but optimistic for the thin TE pool). Each was gated on the tune window
  (held the ECR edge, kept a positive ADP edge) before one test look. QB8 fixed
  the over-valued QB block (Herbert QB5 #39 → #50; first QB #17 → #22) and
  certified the ADP claim; **TE8 is deliberately conservative** — the TE curve
  plateaus TE10-12 so TE8 barely moves the board (McBride #21, Bowers #30
  stay), nudging the metric without deflating scarce elite TEs (TE5-6 would).
  Only a replacement ≤ QB8 / ≤ TE8 moves the board past the plateau — don't set
  QB9/TE9 expecting an effect. (The sibling BEER man-games replacement for
  RB/WR was REJECTED — see Rules.)

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

## Player props (TESTED, REJECTED — data kept, columns inert)

Built 2026-07-24 by `bff/props_wayback.py` + `bff/props.py`. Preseason
player-level futures boards from sportsoddshistory's award / stat-leader pages
(Wayback; the covers.com mirror is the same publisher). **Nine markets** — mvp,
oroy, comeback, pass_yds, pass_td, rush_yds, rush_td, rec_yds, rec_td — as
5233 attested quotes over 2012-2025, 99.5% matched to gsis ids.

- **TESTED AND REJECTED on the tune window 2026-07-24; ZERO test looks spent.**
  All three variants hurt: `props` (5 markets) −0.0052, `props_dense` (mvp +
  rush_yds + rec_yds) −0.0024, `props_lean` (rush_yds + rec_yds) −0.0022
  against the 0.4617 baseline (re-derived after the 2026-07-24 parser fix
  below; the pre-fix numbers were −0.0051/−0.0033/−0.0026, same verdict). Harm is MONOTONE in the number of prop columns.
  The columns stay in `build_dataset` and `CANDIDATE_BLOCKS` as inert
  zero-filled candidates and are NOT in `FEATURES` (same status as
  redzone/snaps/vegas). Verified inert: `preds_model.parquet` is byte-identical
  before and after the join. Re-test via `bff.select_features`, never by
  hand-editing FEATURES.
- Why it failed is NOT the usual "already priced" story, and that is worth
  remembering before anyone retries it. Max |r| against existing features is
  only 0.20-0.61 (prop_rush_yds 0.20), so the information IS new; and reach is
  wide — 72-97% of the scored ADP top-150 pool carries a quote, 86-100% of the
  top 50. It is new, it is dense, and it still does not predict the residual.
- The "ragged early folds" defence was tested and DIED. Fold 2012 contributes
  exactly +0.0000 (train window predates all props → zero variance → coef fit
  as 0, confirming the `vs_adp` mechanism in the fragility list). Restricting
  the mean to the folds where props are genuinely active (2015-2017) makes the
  deltas WORSE, not better: lean −0.0053, dense −0.0021, all5 −0.0029 against a
  0.4216 sub-window baseline. Do not reopen this on a coverage argument.
- Leakage: every accepted page carries the literal header "As of <date> -
  prior to the start of the season", and `_validate` REJECTS any capture
  without it, so a page's content is a permanent preseason artifact and capture
  timing is irrelevant (same argument as `bff/vegas_wayback.py`). The `Result`
  column is a season outcome and is never parsed. `bff.props.check_leakage`
  asserts every board's as-of date ≤ that season's Week 1 kickoff date: all
  fourteen land **exactly on opener day** (the site dates the board the morning
  of the Thursday opener), the tightest margin of any input in the repo — hence
  the executable check.
- Coverage, after the Save-Page-Now backfill (below). **Complete 2012-2025
  (6 tune + 8 test seasons): mvp, oroy, pass_yds, rush_yds, rec_yds.**
  comeback runs 2014-2025 (4 tune). The three TD markets **start in 2017** — one
  tune season, so they are effectively untunable on this protocol even though
  their test coverage is 8/8; do not try to promote them.
- **2026 IS NOT AVAILABLE, and that is a market fact, not a fetch bug** (checked
  hard 2026-07-24). The mirror's 2026 pages exist and render — breadcrumb,
  season selector, dead-heat boilerplate — with an EMPTY table; the leader
  markets are not posted yet. RotoWire's DraftKings-fed leader pages
  independently say "There are no odds available right now" for
  rushing/receiving/passing leader and all three TD-leader variants. What IS
  live for 2026 is a DIFFERENT market: season yardage OVER/UNDER totals
  (rotowire.com/betting/nfl/{rush,rec,pass}-yards-odds.php, multi-book,
  JS-rendered so it needs a browser), plus a thin 11-name editorial MVP table.
  Those O/U totals have NO 2012-2025 history here, so they cannot extend this
  series and cannot be tuned. **This is now MOOT for the decision: 2026
  availability was never the binding constraint, the signal was.** The tune
  gate is scored on 2012-2017 and does not involve 2026 at all, so a posted
  2026 board would not change the verdict by one digit. Do NOT re-open this
  block when the 2026 lines appear. (Had it passed, 2026 would then have
  mattered a lot: with no 2026 boards every prop column zero-fills on the live
  board, so the model would apply coefficients learned on real quotes to an
  all-zero column — a silent train/serve skew, not a neutral default. Any
  future market-price block must be checked for live-season availability
  BEFORE it is tuned, not after.)
- Backfill route when a board has no capture on either host: Wayback's Save
  Page Now fetches server-side, so it reaches the live mirror this environment
  cannot resolve (both odds hosts are DNS-blocked here; `web.archive.org`,
  `rotowire.com` and `fantasypros.com` are not). Ten boards were recovered this
  way on 2026-07-24 — mvp 2021-2025, rec_yds 2024, pass_yds/oroy/comeback 2025,
  rush_td 2024 — every one validated for the preseason attestation before being
  cached. It is `--save-missing`, OPT-IN, because it writes to a third-party
  public archive; a normal run must never trigger it.
- `ROW_RE`'s `&(?:amp;)?` is load-bearing: mirror pages come back
  Wayback-rewritten with escaped ampersands, and a plain `&y=` silently parses
  them to ZERO rows, which then look like empty shells and get rejected. That
  bug is why mvp/rec_yds/etc. first appeared to be missing from the mirror.
- Board depth swings 13 → 123 players, so raw implied probability is not
  comparable across seasons. `novig_prob` (implied ÷ board total) is the
  comparable column; `board_rank` / `board_n` keep the raw shape. The wide
  table zero-fills absent players and carries `has_prop` / `prop_n_markets` /
  `prop_market_seasons` so a consumer can tell an unpriced PLAYER from a
  missing SOURCE — the distinction the vegas block never had.
- gsis match rate 99.5% (5209/5233). `bff.props.season_identity` resolves names
  against that season's ADP/ECR board FIRST, then `ctx_rosters_week1` — both
  preseason-safe, no outcome table. That season scoping is what splits Alex
  Smith QB from Alex Smith TE and the ARI David Johnson from the PIT one; a
  crosswalk-only match left 4% unresolved. The 23 leftovers are defenders/OL
  and never-NFL college names off the oroy/comeback boards. One non-player
  entry exists (2017 rec_td "FIELD", 11.9% of that board): flagged `is_field`,
  kept in the no-vig denominator, never given an id.
- **An OUTCOME LEAK lived in the first build; found 2026-07-24 and fixed.**
  Eight boards carry a row for a player with `N/A` odds and `** WINNER **` in
  Result — the season's winner, who was never on the preseason board. The old
  document-wide regex matched that name, skipped the `N/A`, and took the NEXT
  row's price, FABRICATING quotes: "Josh Gordon +275" for 2013 rec_yds (the real
  +275 favourite was Calvin Johnson), Ben Roethlisberger 2014 pass_yds, Kareem
  Hunt 2017 rush_yds, Carson Wentz 2017 pass_td, Ryan Tannehill 2019 comeback,
  Geno Smith 2022 comeback, Jamaal Williams 2022 rush_td, Joe Flacco 2023
  comeback. That is season-t OUTCOME written into a preseason feature at the top
  of the board, 4 of the 8 inside the tune window. Rows with no priced cell now
  yield nothing. Note the direction: leakage would FLATTER the block, so the
  reject is stronger post-fix, not weaker.
- **Parser bug found and fixed 2026-07-24 (after the first reject), worth
  knowing because of HOW it hid.** A single document-wide regex silently lost 74
  rows across 53 files, and the losses were biased toward the most informative
  rows: (1) the name character class excluded `'`, so EVERY apostrophe player
  truncated at the quote and vanished — the dataset contained ZERO Ja'Marr
  Chase, De'Von Achane or Le'Veon Bell quotes, i.e. part of the elite tier was
  missing; (2) a stray `nfl-award-player` link above the table let the lazy
  `.*?` consume the first data row's odds cell, eating the top line of the board
  (the favourite/winner). Parsing is now ROW-SCOPED (`iter_rows`, one `<tr>` at
  a time), which makes both failure modes structurally impossible. The
  regression test is cheap and should be kept in mind for any future scrape:
  **count player links and count parsed rows; they must be equal.** Nearly all
  the loss was in 2016-2025 (the tune window gained only 2 rows), which is why
  the verdict did not move — but that was luck, not design.
- 2014 is genuinely the thinnest season (mvp 20 names, pass_yds 13) and that is
  NOT a parse artifact: all six 2014 boards were re-fetched from the live mirror
  and match the cached row counts exactly. The book simply priced fewer names
  that year; the 13-name passing board is the 13 obvious starting QBs.
- **The SOURCE duplicates a player on at least one board.** 2015 comeback lists
  "Micahel Crabtree" +6600 and "Michael Crabtree" +10000 — one player, two
  spellings, two prices, a site data-entry error (confirmed in the raw HTML).
  `bff/props_wayback.py` cannot see it (the strings differ); the `_ALIASES`
  mapping is what exposes it, by resolving both to one gsis_id.
  `bff.props.dedupe_by_player` keeps the shortest price and runs BEFORE
  novig_prob/board_n, so the phantom is also removed from the board denominator
  (2015 comeback board_n 51 → 50) rather than diluting every other player's
  share. `bff.props.check_join` asserts this as an identity — distinct
  (season, market, gsis_id) triples must equal non-zero wide cells — which is
  the check the curation step previously lacked entirely. The retained row still
  DISPLAYS the site's misspelling; gsis_id is the key, the name is cosmetic.
- **Cross-capture reconciliation (2026-07-24, `bff/props_reconcile.py`,
  results in `reports/props_reconcile.json`).** Every board re-parsed from up to
  4 timestamp-diverse INDEPENDENT Wayback captures (365 captures, read-only
  replay, no Save Page Now) and required to yield an identical quote set:
  **91 AGREE / 8 DISAGREE / 10 SINGLE-SOURCED.**
  - 6 of the 8 disagreements are COSMETIC, identical odds: the source changed
    punctuation between captures ("AJ Green" ↔ "A.J. Green", "TY Hilton" ↔
    "T.Y. Hilton", "Odell Beckham Jr" ↔ "Jr.") or served curly vs straight
    apostrophes. `norm_name` absorbs all of it.
  - 2017 oroy: the comparator is dated **May 3, 2017** — an EARLY preseason
    board (16 names, different field). Not an error: the page gets updated
    within a preseason, and newest-first correctly selected the September
    board. Verified corpus-wide: all 14 seasons carry exactly ONE as_of date
    and every one is September, i.e. no board is an early snapshot.
  - 2013 oroy: three OLD captures (2015/2020/2022) show 19 names, the NEWEST
    capture and the shipped board show 20 (Zach Ertz +2000). The source revised
    a historical board after 2022; newest-first takes the revision. Documented
    rather than "fixed" — preferring the site's most-corrected version is the
    same rule `bff.ecr` uses.
  - The 10 SINGLE-SOURCED boards are exactly the 10 recovered via Save Page Now
    (mvp 2021-2025, rec_yds 2024, rush_td 2024, comeback/oroy/pass_yds 2025).
    By construction they had no other capture, so **this method cannot verify
    them**; their only provenance is the live mirror at fetch time. Do not
    describe the corpus as fully reconciled.
- `norm_name` here is a FOURTH variant (adp/ecr/crosswalk have their own) and
  also strips parentheticals ("Josh Allen (BUF)"). Do not unify them.

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
  replacement ranks QB8/RB30/WR36/TE8). Never score a baseline on raw
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
uv run python -m bff.candidate_features    # coach/qb-rush/ol/adp-gap candidates (all rejected, inert)
uv run python -m bff.schedule_trajectory_features  # <- sched (games.csv) + trajectory (SHIPPED)

# preseason player-prop boards (MVP / yardage+TD leaders / OROY / comeback).
# Fetch is a one-time Wayback scrape (needs network; caches HTML, then offline);
# curation is offline and byte-stable. NOT WIRED INTO THE MODEL -- see "Player
# props" below. --save-missing additionally asks Wayback to archive live pages
# that were never captured (writes to archive.org; opt-in).
uv run python -m bff.props_wayback         # -> data/raw/props/season_props.parquet
uv run python -m bff.props                 # -> data/processed/{season_props,props_features}.parquet

# QB/TE streaming baseline derivation (tune window ONLY; justifies REPL_RANKS QB8/TE8)
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
| `bff.streaming` | stdout only (tune-window QB streaming equiv-rank table; derivation behind QB8/TE8) |
| `bff.props_wayback` | `data/raw/props/season_props.parquet` (5234 quotes, 2012-2025; 5233 after curation dedupe) + cached HTML (git-ignored) |
| `bff.props` | `data/processed/season_props.parquet` (long, 5233 rows) and `props_features.parquet` (wide, 2697 rows keyed season × gsis_id) |
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
- Rejected candidate blocks (redzone/snaps/vegas/landing_spot, and the
  player-prop blocks props/props_dense/props_lean) stay in
  `build_dataset` as inert zero-filled columns and in `CANDIDATE_BLOCKS`;
  they are NOT in FEATURES. Re-test via `bff.select_features`, never by
  hand-editing FEATURES.
- Known-degenerate candidate data (nulls by design, zero-filled at join):
  snaps source 2012 is empty upstream (features start target 2014); vegas
  has no usable 2025/2026 capture; contracts miss the 2026 draft class
  (14 pool players). `vegas_wins` is per-season CENTERED before zero-fill
  (a missing line imputes league-average, not 0 wins).
