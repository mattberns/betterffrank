# betterffrank

Backtested preseason fantasy football rankings (12-team PPR redraft) that aim to
beat the two public baselines — market ADP and FantasyPros expert consensus
(ECR) — evaluated honestly on **value over replacement**, not raw points.

The headline, reproduced from committed data (test seasons 2018–2025, never
used for tuning or feature selection):

| Metric (mean VORP Spearman, walk-forward) | Model | Baseline | Verdict |
| --- | --- | --- | --- |
| vs **ECR** (2018–2025) | **0.5233** | 0.5179 | **edges ECR**, 7 of 8 seasons, p = 0.152 one-sided (positive, not yet significant) |
| vs **ADP** (2018–2025) | **0.5233** | 0.5028 | **beats ADP**, 7 of 8 seasons, p = 0.016 one-sided (certified; 0.031 two-sided) |

The defensible claim is deliberately narrow: **the model matches the expert
consensus and beats the drafting market.** ECR is the primary benchmark — the
model is architecturally ECR+ (the expert consensus is the anchor; features
predict where it historically missed). The full methodology, per-season
tables, and every caveat are in [`reports/REPORT.md`](reports/REPORT.md).

## Method in one paragraph

For each season *t*, players are anchored to the market — `log(ecr_rank)` when
a preseason ECR snapshot exists (2015–2025 + 2026), `log(adp_rank)` otherwise —
ordinally re-ranked per season. A per-position quadratic fit on seasons < *t*
converts the anchor into an implied expectation of log points; a ridge
regression on 51 preseason features (age curve, production-vs-price mismatch,
durability and injury history, rookie pedigree and draft-round capital, TD
regression, the ECR-vs-ADP gap, roster/draft context, prior-year
opportunity/usage, contract commitment) predicts the residual, and the shrunken
sum is re-ranked into predicted **VORP** through a leakage-safe drafted-slot
curve (`bff/vorp.py`, seasons < *t* only, replacement QB8/RB30/WR36/TE8):
the expected actual points of the player the market drafted at each
within-position slot — not a hindsight finish-rank curve, which over-values
the noisy mid-TE/QB tails. The ADP and ECR baselines go through the same curve, so
every comparison is apples to apples. Hyperparameters and all
feature selection are tuned only on walk-forward seasons 2012–2017 (frozen
grid, VORP Spearman); the test set is 2018–2025 and is never touched for
decisions.

## Why VORP Spearman, not points

An earlier version of this project claimed a huge edge (+0.17 Spearman vs ADP)
by scoring against raw total PPR points. That metric rewards stacking
quarterbacks at the top: elite QBs outscore all RBs/WRs in total points, but in
a one-QB league you start one and the drop from QB1 to QB12 is small. Raw-points
Spearman measures "who predicts total points," not draft value — so the whole
project is evaluated on **VORP** (points above the actual QB8 / RB30 / WR36 /
TE8 in that season's pool), and every list, including the ADP and ECR
baselines, is re-ranked through the same leakage-safe historical
points-by-positional-rank curve.

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

## Repo layout

```
bff/                      the pipeline (run as python -m bff.<module>)
  adp.py                  build ADP table            (FantasyFootballCalculator)
  ecr.py                  build ECR table            (FantasyPros via DynastyProcess + Wayback)
  ecr_wayback.py          one-time: backfill 2015-2020 ECR from Wayback FantasyPros cheatsheets
  actuals.py              build player-season actual PPR points   (nflverse)
  context_data.py         fetch/shape roster, draft, team context (nflverse)
  context_features.py     preseason context features
  opportunity_features.py prior-year weekly opportunity features
  redzone_features.py     prior-year red-zone/goal-line usage (from pbp; candidate block)
  situation_features.py   snap share, injury history, contracts, Vegas (candidate blocks)
  vegas_wayback.py        one-time: preseason win totals from Wayback sportsoddshistory
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

- **ECR coverage**: 2015–2020 from Wayback FantasyPros PPR-cheatsheet snapshots
  (one preseason capture per season, extracted once by `bff/ecr_wayback.py`;
  all captures verified pre–Week 1), 2021–2025 from DynastyProcess preseason
  snapshots (Sep 5–10 each year), 2026 from a FantasyPros draft-rankings export
  (`data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv`), so the 2026 list runs on
  the full ECR anchor with `vs_adp` live. All six new seasons match the gsis
  crosswalk at 150/150 of the top 150. A future season without ECR rows falls
  back to an ADP-only anchor, and the CLI prints a warning when that happens.
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
from FantasyPros' draft rankings page), and the 2015–2020 Wayback backfill
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
| [FantasyPros](https://www.fantasypros.com/) | ECR (2015–2020 via Wayback cheatsheet snapshots, 2021–2025 via DynastyProcess archive) + 2026 draft-rankings export | **Commercial — not redistributed here** |
| [sportsoddshistory.com](https://www.sportsoddshistory.com/) (via Wayback) | preseason Vegas win totals 2010–2024 (candidate feature; not in the shipped model) | Scrape cached locally — not redistributed |

## Caveats

See the integrity ledger and caveats in [`reports/REPORT.md`](reports/REPORT.md).
In short: the ADP win (+0.0206 over 2018–2025) is certified at p = 0.016
one-sided (0.031 two-sided); the ECR edge (+0.0054) is positive but not yet
significant (p = 0.152); seasons 2018–2020
briefly served as tuning folds during the 2026-07-23 protocol work before
moving to the test set (all selections were re-derived from scratch on
2012–2017 and reproduced exactly); the 2015–2020 ECR comes from a different
capture pipeline (Wayback cheatsheet HTML) than 2021–2025 (DynastyProcess
archive); and the 2012 ADP source is unusually thin (~91 pool players), which
lands inside the tuning window.

## License

Code: [MIT](LICENSE). Data: third-party terms as above. Not affiliated with the
NFL, FantasyPros, FantasyFootballCalculator, DynastyProcess, or nflverse.
