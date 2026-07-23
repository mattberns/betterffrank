# betterffrank

Backtested preseason fantasy football rankings (12-team PPR redraft) that aim to
beat the two public baselines — market ADP and FantasyPros expert consensus
(ECR) — evaluated honestly on **value over replacement**, not raw points.

The headline, reproduced from committed data:

| Metric (2015–2025, walk-forward) | Model (`opp_residual`) | Baseline | Verdict |
| --- | --- | --- | --- |
| Mean VORP Spearman vs **ADP** | **0.4931** | 0.4678 | **beats ADP**, 9 of 11 seasons, permutation p = 0.004 |
| Mean VORP Spearman vs **ECR** (2021–2025) | 0.5353 | 0.5298 | **ties** (selection-adjusted p = 0.75) |

The defensible claim is deliberately narrow: **it beats ADP and ties expert
consensus.** Nothing here reliably beats ECR once model-selection is accounted
for. The full methodology, per-season tables, and every caveat are in
[`reports/REPORT.md`](reports/REPORT.md).

## Why VORP Spearman, not points

An earlier version of this project claimed a huge edge (+0.17 Spearman vs ADP)
by scoring against raw total PPR points. That metric rewards stacking
quarterbacks at the top: elite QBs outscore all RBs/WRs in total points, but in
a one-QB league you start one and the drop from QB1 to QB12 is small. Raw-points
Spearman measures "who predicts total points," not draft value — so the whole
project is evaluated on **VORP** (points above the actual QB12 / RB30 / WR36 /
TE12 in that season's pool), and every list, including the ADP and ECR
baselines, is re-ranked through the same leakage-safe historical
points-by-positional-rank curve. Apples to apples.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
git clone <this-repo> betterffrank
cd betterffrank            # run every command from the repo root
uv sync                    # create the venv, install locked deps

# Verify the shipped result without rebuilding anything (uses committed data):
uv run python -m bff.backtest data/processed/preds_opp_residual.parquet --name opp_residual
```

That prints the per-season and mean VORP Spearman table above. To rebuild the
full pipeline from raw data, see [Reproduce](#reproduce).

## Repo layout

```
bff/                      the pipeline (run as python -m bff.<module>)
  adp.py                  build ADP table            (FantasyFootballCalculator)
  ecr.py                  build ECR table            (FantasyPros via DynastyProcess)
  actuals.py              build player-season actual PPR points   (nflverse)
  context_data.py         fetch/shape roster, draft, team context (nflverse)
  context_features.py     preseason context features
  opportunity_features.py prior-year weekly opportunity features
  vorp.py                 leakage-safe VORP curve library
  backtest.py             evaluation harness (the metrics table)
  models/
    market_residual.py    v1 — market anchor + residual        (load-bearing)
    context_residual.py   v2 — + preseason context features    (load-bearing)
    opp_residual.py       v3 — + opportunity features  ← SHIPPED WINNER
    rank_2026_v3.py       final 2026 rankings + steals deliverable
data/
  raw/                    source data (see Data and attribution)
  processed/              built parquets (committed so results reproduce)
reports/
  REPORT.md               full methodology, results, and caveats
  rankings_2026.csv       2026 rankings (predicted VORP)
  steals_2026.csv         2026 values vs ADP, with model reasons
  scores_opp_residual.csv backtest scores for the shipped model
```

`market_residual.py` and `context_residual.py` are older model versions but are
**not dead** — the shipped `opp_residual` imports directly from both.

## Reproduce

The pipeline is leakage-safe: predictions for season *t* use only seasons < *t*,
plus season-*t* preseason facts (ADP, ECR, draft picks, opening-day rosters).
Hyperparameters are tuned only on walk-forward seasons 2012–2014; evaluation is
2015–2025.

```bash
# data prep  (run from repo root; these read/write data/)
uv run python -m bff.adp && uv run python -m bff.ecr && uv run python -m bff.actuals
uv run python -m bff.context_data && uv run python -m bff.context_features
uv run python -m bff.opportunity_features
# v3 winner: walk-forward eval preds (2015-2025), then 2026
uv run python -m bff.models.opp_residual && uv run python -m bff.models.opp_residual --season 2026
uv run python -m bff.backtest data/processed/preds_opp_residual.parquet --name opp_residual
# 2026 rankings + steals (asserts consistency with preds_opp_residual_2026.parquet)
uv run python -m bff.models.rank_2026_v3
```

**Note on `bff.ecr`:** rebuilding `data/processed/ecr.parquet` needs
`data/raw/db_fpecr.parquet`, which is **not committed** (see below).
`ecr.parquet` itself *is* committed, so every step after `bff.ecr` reproduces
without it.

## Data and attribution

This project stands on open and third-party data. Please respect each source's
terms.

| Source | Files | Terms |
| --- | --- | --- |
| [nflverse](https://github.com/nflverse) | weekly stats, rosters, draft picks, schedules | Open data, CC-BY — attribution |
| [DynastyProcess](https://github.com/dynastyprocess/data) | `db_playerids.csv` (player ID crosswalk) | Open data — attribution |
| [FantasyFootballCalculator](https://fantasyfootballcalculator.com/) | `data/raw/adp/ppr_*.json` (12-team PPR ADP) | Public draft-aggregate data |
| [FantasyPros](https://www.fantasypros.com/) | ECR + ADP (via DynastyProcess archive) | **Commercial — not redistributed here** |

**FantasyPros data is intentionally excluded from this repo.** The 38 MB ECR
scrape archive (`data/raw/db_fpecr.parquet`) and a FantasyPros ADP export are
git-ignored because FantasyPros' terms restrict redistribution of their
commercial rankings. To rebuild `ecr.parquet` yourself, obtain the ECR archive
from the [DynastyProcess data repo](https://github.com/dynastyprocess/data),
place it at `data/raw/db_fpecr.parquet`, and run `uv run python -m bff.ecr`. The
committed `data/processed/ecr.parquet` contains only the derived preseason
snapshot used for evaluation.

## Caveats

The project is honest about its limits; see the "Limitations" and "Disclosures"
sections of [`reports/REPORT.md`](reports/REPORT.md). In short: the ECR-era
result is parity, not a win; the 2026 list runs on an ADP-only anchor (no 2026
ECR feed); the curated feature subset carries designer degrees of freedom; and
the 2012 ADP source is unusually thin (~91 pool players vs ~150), which lands
inside the tuning window.

## License

Code: [MIT](LICENSE). Data: third-party terms as above. Not affiliated with the
NFL, FantasyPros, FantasyFootballCalculator, DynastyProcess, or nflverse.
