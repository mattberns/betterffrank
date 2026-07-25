# betterffrank

Preseason fantasy football draft rankings (12-team PPR redraft) that start
from the experts' list and correct it. Every claim below is backtested over
eight NFL seasons the model never saw during development, and graded on
**draft value**, not raw points.

The headline (test seasons 2018–2025, never used for tuning or feature
selection):

| Metric (mean VORP Spearman, walk-forward) | Model | Baseline | Verdict |
| --- | --- | --- | --- |
| vs **ECR** (2018–2025) | 0.5237 | 0.5205 | **matches ECR**, 4 of 8 seasons, p = 0.227 one-sided (no demonstrated edge) |
| vs **ADP** (2018–2025) | **0.5237** | 0.5040 | **beats ADP**, 7 of 8 seasons, p = 0.0469 one-sided (0.094 two-sided) |

The defensible claim is deliberately narrow: **the model beats the drafting
market and matches the expert consensus.** ECR is the primary benchmark, and
against it the model wins 4 of 8 seasons, which is a coin flip. The model is
built *on top of* the experts' list, so it wins only where it corrects them
well, and on this evidence it does not reliably do so.

Earlier versions of this README claimed an edge over ECR. That claim did not
survive contact with clean data: on 2026-07-25 three input defects were fixed
(a 2011 training season with no expert board, a half-missing 2012 ADP board,
and three tuning folds that ranked off ADP instead of ECR), and the measured
ECR edge shrank at every step. Full methodology, per-season tables, and every
caveat: [`reports/REPORT.md`](reports/REPORT.md).

## How it works, in plain English

The model does not build rankings from scratch. It makes small, evidence-based
corrections to the FantasyPros expert consensus (ECR), then converts everything
into draft value. Five steps:

1. **Start from the experts.** Each player's starting point is his preseason
   expert-consensus rank. Every season the model touches (2012–2026) has a
   real ECR board; no season falls back to ADP.
2. **Turn a rank into expected points.** A rank alone says nothing about
   points, so history fills the gap: for each position, a curve fit on prior
   seasons answers "how many points does the WR ranked 5th usually score?
   The WR ranked 30th?" Every player now carries an expected point total.
3. **Predict where the experts are wrong.** This is the model proper: a
   regression on 53 facts knowable before the season (age, how much a player
   outplayed his draft price last year, injury history, vacated targets on
   his new team, rookie draft capital, the gap between his expert rank and
   his market ADP, and so on) predicts the *gap* between what players
   actually scored and what step 2 said they should score. Its only job is
   learning the patterns in when the experts miss high or miss low.
4. **Nudge, don't overrule.** The final projection is the expert-implied
   expectation plus a *fraction* of the predicted correction. The experts
   are good; the model adjusts at the margin.
5. **Convert points into draft value.** Raw points mislead across positions:
   the 10th QB outscores the 10th TE but is worth less in a 1-QB league,
   because a decent QB is always available. So each projection becomes
   **VORP** — points above the replacement you could actually get at that
   position (QB8/RB30/WR36/TE8; QB and TE are streaming-aware, since a
   punted QB/TE is refilled off waivers weekly). That conversion is what
   merges four positions into one draft board.

## How it's scored: champion vs challenger

Each test season (2018–2025) is a round of the same contest. Three boards
enter — the model, the expert consensus (ECR), and the market (ADP) — and
each is graded the same way: rank-correlate the preseason ordering against
players' *actual* end-of-season value over replacement. Best correlation wins
the year. No board sees the season before predicting it, and the model's
knobs were frozen on 2013–2017 before any test season was scored.

Over eight rounds the challenger beats ADP 7 of 8 (average 0.5237 vs 0.5040;
p = 0.0469 one-sided, which clears the usual 0.05 bar by a hair and does not
clear the stricter two-sided version at 0.094). Against ECR it wins 4 of 8
(0.5237 vs 0.5205); that is a tie, and it should be described as a tie. Note
ECR itself beats ADP comfortably, so most of what looks like the model's skill
is inherited from the expert list it starts from.

## Are the experts actually better than the market?

The model is built on top of ECR, so it inherits whatever edge ECR has. That
makes one question worth asking on its own, with no model involved: graded on
realized VORP, is the expert consensus a better preseason board than ADP? The
answer depends sharply on the era, and it was invisible until the 2026-07-25
data cleanup made both windows measurable.

**Tuning years — a dead heat.**

| Season | ADP | ECR | ECR − ADP | Better |
| --- | --- | --- | --- | --- |
| 2013 | 0.5563 | 0.5412 | −0.0151 | ADP |
| 2014 | 0.4134 | 0.4165 | +0.0031 | ECR |
| 2015 | 0.3958 | 0.3906 | −0.0052 | ADP |
| 2016 | 0.3448 | 0.3453 | +0.0005 | ECR |
| 2017 | 0.4423 | 0.4552 | +0.0129 | ECR |
| **mean** | **0.4305** | **0.4298** | **−0.0008** | ECR 3/5 |

The mean favours ADP by 0.0008, which is noise; ECR still wins the majority of
seasons. One caveat on 2013, the largest single ADP win in the table: its ECR
snapshot is the Aug 18 capture, the last one Wayback archived that preseason
and roughly two weeks staler than the early-September boards every other
season uses. A staler board is a plausible reason for it to underperform, so
that row is weaker evidence than the rest.

**Test years — the experts pull clearly ahead, and the gap widens.**

| Season | ADP | ECR | ECR − ADP | Better |
| --- | --- | --- | --- | --- |
| 2018 | 0.4641 | 0.4346 | −0.0296 | ADP |
| 2019 | 0.4764 | 0.4958 | +0.0194 | ECR |
| 2020 | 0.5186 | 0.5283 | +0.0097 | ECR |
| 2021 | 0.5180 | 0.5257 | +0.0076 | ECR |
| 2022 | 0.5644 | 0.5751 | +0.0108 | ECR |
| 2023 | 0.5550 | 0.5728 | +0.0178 | ECR |
| 2024 | 0.4667 | 0.5074 | +0.0407 | ECR |
| 2025 | 0.4689 | 0.5243 | +0.0554 | ECR |
| **mean** | **0.5040** | **0.5205** | **+0.0165** | ECR 7/8 |

ECR wins 7 of 8 by a mean of +0.0165, and the margin grows over time (+0.008
in 2021 to +0.055 in 2025). Whether that is FantasyPros improving, the ADP
sample thinning, or both, this data cannot settle.

**What it implies about the model, and it is not flattering.** On the test set
the model beats ADP by +0.0197. ECR on its own, with no model attached,
already beats ADP by **+0.0165** over the same seasons. The ridge contributes
the remaining **+0.0032**. So roughly five sixths of the margin over the market
is the expert list this project starts from, not the corrections it applies.

It also resolves a puzzle in the numbers above. The model shows no edge on its
own tuning window (0.4293 vs ADP 0.4305) yet +0.0197 on test. That is not a
contradiction: in the tuning years the anchor is worth nothing over ADP, so a
model built on that anchor has nothing inherited to win with. In the test years
the anchor is worth +0.0165 before the model does anything.

## Why draft value, not raw points

An earlier version of this project claimed a huge edge (+0.17 Spearman vs
ADP) by grading against raw total PPR points. That grade rewards stacking
quarterbacks at the top: elite QBs outscore every RB and WR in total points,
but in a one-QB league you start one, and the drop-off to the next decent QB
is small. A points grade measures "who predicts point totals"; a draft needs
"who finds value." So everything here — the model *and* both baselines — is
graded on VORP, through the identical conversion. Nobody gets a private
exchange rate; the comparison is apples to apples by construction.

## Method in one paragraph (technical)

For each season *t*, players are anchored to the market — `log(ecr_rank)`,
which exists for every season the model uses (2012–2025 + 2026), with a
`log(adp_rank)` fallback that no longer fires — ordinally re-ranked per
season. A per-position quadratic fit on seasons < *t*
converts the anchor into an implied expectation of log points; a ridge
regression on 53 preseason features predicts the residual (position enters
only through the per-position anchor fit and the VORP curve, never as a
regression feature), and the shrunken sum is re-ranked into predicted **VORP**
through a leakage-safe drafted-slot curve (`bff/vorp.py`, seasons < *t* only,
replacement QB8/RB30/WR36/TE8): the expected actual points of the player the
market drafted at each within-position slot — not a hindsight finish-rank
curve, which over-values the noisy mid-TE/QB tails. The ADP and ECR baselines
go through the same curve. Hyperparameters and all feature selection are
tuned only on walk-forward seasons 2013–2017 (frozen grid, VORP Spearman);
the test set is 2018–2025 and is never touched for decisions.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
git clone <this-repo> betterffrank
cd betterffrank            # run every command from the repo root
uv sync                    # create the venv, install locked deps

# Verify the shipped result without rebuilding anything (uses committed data):
uv run python -m bff.backtest data/processed/preds_model.parquet --name model
```

## Pipeline

```bash
# data rebuild (only if raw data changes)
uv run python -m bff.adp && uv run python -m bff.ecr && uv run python -m bff.actuals
uv run python -m bff.context_data && uv run python -m bff.context_features
uv run python -m bff.opportunity_features
uv run python -m bff.redzone_features      # <- data/raw/pbp/ (nflverse)
uv run python -m bff.situation_features    # <- data/raw/{snaps,injuries,contracts,vegas}/

# candidate-feature selection (tune window only; never the test set)
uv run python -m bff.select_features

# model
uv run python -m bff.model                       # -> data/processed/preds_model.parquet (2018-2025)
uv run python -m bff.model --baselines           # -> preds_adp.parquet, preds_ecr.parquet
uv run python -m bff.model --season 2026         # -> preds_model_2026.parquet, reports/rankings_2026.csv, reports/steals_2026.csv

# draft overlay (VONA) — post-processes the 2026 board; not part of scoring
uv run python -m bff.vona                        # -> reports/vona_2026.csv

# QB/TE streaming baseline derivation (tune window only; justifies REPL_RANKS QB8/TE8)
uv run python -m bff.streaming                   # stdout only

# evaluation
uv run python -m bff.backtest data/processed/preds_model.parquet --name model
uv run python -m bff.backtest data/processed/preds_adp.parquet   --name adp
uv run python -m bff.backtest data/processed/preds_ecr.parquet   --name ecr

# significance
uv run python -m bff.compare data/processed/preds_model.parquet data/processed/preds_adp.parquet
uv run python -m bff.compare data/processed/preds_model.parquet data/processed/preds_ecr.parquet

# static site (GitHub Pages, /docs)
uv run python -m bff.site                        # render from current artifacts
uv run python -m bff.site --refresh              # rerun model + backtests first
```

## Artifacts

| File | Produced by |
| --- | --- |
| `data/processed/preds_model.parquet` | `uv run python -m bff.model` (2018–2025, score = predicted VORP) |
| `data/processed/preds_model_2026.parquet` | `uv run python -m bff.model --season 2026` (185 rows) |
| `data/processed/preds_adp.parquet`, `preds_ecr.parquet` | `uv run python -m bff.model --baselines` |
| `reports/scores_model.csv`, `scores_adp.csv`, `scores_ecr.csv` | `uv run python -m bff.backtest <preds> --name <n>` |
| `reports/rankings_2026.csv`, `reports/steals_2026.csv` | `uv run python -m bff.model --season 2026` |
| `reports/vona_2026.csv` | `uv run python -m bff.vona` (draft-timing overlay; 150 picks) |
| `reports/REPORT.md` | written by hand; every number maps to a command above |
| `data/processed/season_props.parquet`, `props_features.parquet` | `uv run python -m bff.props` (preseason player-prop boards, 2012–2025; **data only, not an input to the model**) |

## Repo layout

```
bff/                      the pipeline (run as python -m bff.<module>)
  adp.py                  build ADP table            (FantasyFootballCalculator)
  ecr.py                  build ECR table            (FantasyPros via DynastyProcess + Wayback)
  ecr_wayback.py          one-time: backfill 2012-2020 ECR from Wayback FantasyPros cheatsheets
  actuals.py              build player-season actual PPR points   (nflverse)
  context_data.py         fetch/shape roster, draft, team context (nflverse)
  context_features.py     preseason context features
  opportunity_features.py prior-year weekly opportunity features
  redzone_features.py     prior-year red-zone/goal-line usage (from pbp; candidate block)
  situation_features.py   snap share, injury history, contracts, Vegas (candidate blocks)
  vegas_wayback.py        one-time: preseason win totals from Wayback sportsoddshistory
  props_wayback.py        one-time: preseason player-prop boards from Wayback sportsoddshistory
  props.py                curate those boards -> gsis-keyed tables (data only; not in the model)
  select_features.py      candidate-block selection harness (tune window only)
  vorp.py                 leakage-safe VORP curve library
  streaming.py            QB/TE streaming-baseline derivation (tune window; justifies QB8/TE8 replacement)
  backtest.py             evaluation harness (the metrics table)
  compare.py              paired sign-flip permutation test between two preds files
  model.py                THE model: dataset, tune, fit, VORP conversion, baselines, 2026 deliverables
  vona.py                 draft-strategy overlay (VONA): post-processes the 2026 board, not scored
  site.py                 static-site generator -> /docs (GitHub Pages)
data/
  raw/                    source data (see Data and attribution)
  processed/              built parquets (committed so results reproduce)
reports/                  REPORT.md (the only report), scores tables, 2026 rankings + steals
docs/                     generated static site (bff.site)
```

## Data notes

- **ECR coverage**: 2012–2020 from Wayback FantasyPros PPR-cheatsheet snapshots
  (one preseason capture per season, extracted once by `bff/ecr_wayback.py`;
  all captures verified pre–Week 1), 2021–2025 from DynastyProcess preseason
  snapshots (Sep 5–10 each year), 2026 from a FantasyPros draft-rankings export
  (`data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv`), so the 2026 list runs on
  the full ECR anchor with `vs_adp` live. All six new seasons match the gsis
  crosswalk at 150/150 of the top 150. A future season without ECR rows falls
  back to an ADP-only anchor, and the CLI prints a warning when that happens.
- **Preseason player props** (`bff/props_wayback.py` → `bff/props.py`): nine
  markets (MVP, OROY, comeback, passing/rushing/receiving yards and TD leaders)
  from Wayback captures of sportsoddshistory's award and stat-leader pages,
  5,234 quotes across 2012–2025, 99.5% matched to gsis ids. Only pages that
  carry the site's literal "prior to the start of the season" header are
  accepted, and a check asserts every board predates its season's Week 1
  kickoff. Five markets (MVP, OROY, and the passing/rushing/receiving yards
  leaders) are complete across all fourteen seasons; the three TD-leader markets
  only start in 2017. **Tested on the 2012-2017 tune window and
  rejected**: every variant hurt (−0.0022 to −0.0052), monotone in how many prop
  columns were added, so the columns stay inert and are not features. No test
  look was spent. Notably the failure is not redundancy — max correlation with
  an existing feature is only 0.20–0.61 and 72–97% of the scored pool carries a
  quote — the market's leader prices just don't predict where the experts are
  wrong. The curated data is kept for the record.
- **Season-t roster membership** (vacated volume, arriving vets, returning
  competition) uses each team's **week-1 REG roster** from the nflverse
  weekly-rosters release (`data/raw/context/roster_weekly_*.parquet`), not the
  seasonal roster files — those keep only each player's last-observed team,
  which is in-season information (a midseason trade would leak into "preseason"
  features). Statuses CUT/RET are excluded; IR/PUP stay members.
- **2025 stats** ship as special nflverse files (`stats_player_week_2025.parquet`,
  `stats_player_reg_2025.parquet`), not the older `player_stats_*` layout.
- **2025 ADP** was recovered from a Wayback Machine snapshot of the
  FantasyFootballCalculator API.

**Rebuilding `data/processed/ecr.parquet`** needs raw files that are **not
committed** (FantasyPros' terms restrict redistribution of their commercial
rankings): the 38 MB ECR scrape archive `data/raw/db_fpecr.parquet` (obtain
from the [DynastyProcess data repo](https://github.com/dynastyprocess/data)),
the 2026 export `data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv` (download
from FantasyPros' draft rankings page), and the 2012–2020 Wayback backfill
`data/raw/db_fpecr_wayback.parquet` (produced once by
`uv run python -m bff.ecr_wayback`, which fetches and caches the archived
cheatsheet HTML under `data/raw/wayback_ppr/`). Then run
`uv run python -m bff.ecr`. The committed `ecr.parquet` contains only the
derived preseason snapshots, so every step after `bff.ecr` reproduces without
any raw file.

## Data and attribution

| Source | Files | Terms |
| --- | --- | --- |
| [nflverse](https://github.com/nflverse) | weekly stats, rosters, draft picks, schedules, play-by-play, snap counts, injuries, contracts (OTC) | Open data, CC-BY — attribution |
| [DynastyProcess](https://github.com/dynastyprocess/data) | `db_playerids.csv` (player ID crosswalk) | Open data — attribution |
| [FantasyFootballCalculator](https://fantasyfootballcalculator.com/) | `data/raw/adp/ppr_*.json` (12-team PPR ADP) | Public draft-aggregate data |
| [FantasyPros](https://www.fantasypros.com/) | ECR (2012–2020 via Wayback cheatsheet snapshots, 2021–2025 via DynastyProcess archive) + 2026 draft-rankings export | **Commercial — not redistributed here** |
| [sportsoddshistory.com](https://www.sportsoddshistory.com/) (via Wayback) | preseason Vegas win totals 2010–2024 (candidate feature; not in the shipped model) | Scrape cached locally — not redistributed |

## Caveats

See the integrity ledger and caveats in [`reports/REPORT.md`](reports/REPORT.md).
In short: the ADP win (+0.0197 over 2018–2025) holds one-sided at p = 0.0469
and does **not** clear two-sided (0.094); there is **no demonstrated ECR
edge** (+0.0032, 4 of 8 seasons, p = 0.227); seasons 2018–2020 briefly served
as tuning folds during the 2026-07-23 protocol work before moving to the test
set (all selections were re-derived from scratch and reproduced exactly); and
the 2012–2020 ECR comes from a different capture pipeline (Wayback cheatsheet
HTML) than 2021–2025 (DynastyProcess archive), with 2012–2013 predating
FantasyPros' inline player ids so those seasons resolve by name.

Three more that matter more than the above, all from the 2026-07-25 cleanup:

**The model has no edge on its own validation window.** On 2013–2017 it scores
0.4293 against ADP 0.4305 and ECR 0.4298. It is very slightly behind both. The
+0.0197 it shows against ADP on the 2018–2025 test set is therefore in tension
with its own tuning evidence, and that tension is unresolved.

**The test set has been consulted twelve times.** Each look was individually
defensible, but twelve is enough that a p-value of 0.0469 should not be read
as a clean pre-registered result.

**Every rejected-feature verdict in `reports/REPORT.md` §4 is stale.** All of
them were decided on the old six-fold, partly-ADP-anchored window with a
truncated 2012 board. Re-deriving them costs no test look and is the top open
item.

## License

Code: [MIT](LICENSE). Data: third-party terms as above. Not affiliated with the
NFL, FantasyPros, FantasyFootballCalculator, DynastyProcess, or nflverse.
