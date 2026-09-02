"""Build the canonical preseason ADP table from FantasyPros boards.

Output: data/processed/adp.parquet with columns
  season, name, norm_name, position, team, adp, adp_rank, times_drafted, gsis_id

**Source swapped 2026-08-31: FantasyFootballCalculator -> FantasyPros.** The
scraper is `bff/fp_adp.py` (Playwright, signed-in); this module is the assembler
that turns its cached boards into the table the whole pipeline reads. The old
FFC builder is preserved, runnable, at `bff/adp_ffc.py` and writes to the
non-canonical `adp_ffc.parquet` so the FFC-era numbers stay reproducible.

`ADP_FORMAT` selects which FantasyPros board is canonical. It is **ppr** — the
scored pipeline is full PPR end to end (`pts_ppr` targets, PPR ECR anchor), and
half-PPR ADP does not exist before 2018 at either publisher, so a half-PPR
anchor cannot cover the 2013-2017 tune window at all. The half board is built
by `bff.fp_adp --build --format half` and read by nothing.

Why the swap (full detail in the `bff/adp_ffc.py` docstring and CLAUDE.md):
FFC prunes its stored boards at the TOP, not the tail. Its 2022 half-PPR board
has no Austin Ekeler (PPR ADP rank 3) and no FFC endpoint has him; 2022 loses 9
of the PPR top-120 and 2025 loses 5. FantasyPros carries them all, goes back to
2012 in full PPR, and resolves 100% of every season's top 150 to a gsis id
(FFC: 99.5%, with six permanently unresolvable "Mike Williams" rows).

Three consequences of the swap, none of them cosmetic:

  1. **The scored pool changes.** The metric's pool is the season's ADP top-150,
     so a new ADP source redefines the evaluation set: across 2012-2026, 130
     players leave the pool and 147 enter, 4-16 per season. Numbers from before
     2026-08-31 are not comparable to numbers after it, on the baseline's
     ordering AND on pool membership.
  2. **`team` is absent on many FantasyPros rows** (1271 of the 2249 PPR
     top-150 rows have no team element at all — the source simply omits it on
     historical pages; zero parse failures). `context_features.build_pool`
     already falls back to the week-1 roster for FA/null teams, so the season-t
     team still resolves, but that fallback now carries thousands of rows
     instead of 17. Watch `team_missing` (BOTH sources failed) after any
     rebuild — it must stay 0.
  3. **`times_drafted` does not exist at FantasyPros.** FFC used it as the sort
     tiebreak behind `adp`; the deterministic tiebreak is now `name`, and the
     column is kept, all-null, because the output schema is documented. Nothing
     outside the ADP builders ever read it (verified by grep at swap time).

Identity resolution and per-board provenance live in `bff.fp_adp` (id-first on
the FantasyPros player id, then the name ladder `bff.ecr` uses). Provenance
worth knowing: FantasyPros' ADP is a CONSENSUS across sites, and the number of
contributing sites THICKENS over time — the PPR board shows 0-2 populated site
columns in 2012-2017 and 3-5 in 2018-2026. That was checked for era contamination
against the FFC board and does NOT show up as era-structured divergence (tune
2013-2017 agrees with FFC at spearman 0.9796, test 2018-2025 at 0.9665 — the
tune window agrees BETTER), but it is a real difference in what the early
numbers are made of. 2012 and 2014 carry an AVG with no populated site column
at all, so their consensus rests on sources the page does not disclose.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from bff.fp_adp import build_table, join_gsis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "adp.parquet"

# Which FantasyPros board is canonical. See the docstring: ppr, because the
# targets and the ECR anchor are full PPR and half-PPR has no pre-2018 history.
ADP_FORMAT = "ppr"

# Board depth cap. FantasyPros publishes 250-950 skill players per season where
# FFC published 146-236, and `bff.model.build_dataset` trains on the WHOLE ADP
# universe (only the SCORED pool is capped, at adp_rank <= 150 in
# `backtest.build_pool`). Uncapped, the source swap would also triple the
# training population with fringe players who never made a week-1 roster
# (Terrell Owens 2012, Chris Redman 2013), and their near-zero outcomes would
# be a change in the FIT bundled invisibly into a change of SOURCE.
#
# 250 is chosen to span the retired board's depth (146-236) with a margin over
# the 150-deep pool, so the training population stays the size it always was.
# It was picked from the FFC depth table BEFORE any model was run and is NOT
# tuned. "Does deeper training data help?" is a separate question and a
# tune-window one -- never decide it on the test seasons.
BOARD_DEPTH = 250

KEEP_POS = ("QB", "RB", "WR", "TE")

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """Kept for callers that imported it from here. The canonical board now
    normalizes with `bff.ecr.norm_name` (same publisher as the boards), which is
    what `bff.fp_adp` uses; this remains the FFC-era definition and is still
    used by `bff/adp_ffc.py`. CLAUDE.md's "three norm_name variants" note still
    holds -- do not unify them."""
    s = name.lower()
    s = re.sub(r"[.'’-]", "", s)
    parts = [p for p in s.split() if p]
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def build_adp_table() -> pl.DataFrame:
    """FantasyPros cached boards -> the documented adp.parquet schema."""
    adp = join_gsis(build_table(ADP_FORMAT))
    # rank first, then truncate, so adp_rank keeps its full-board meaning
    adp = adp.filter(pl.col("adp_rank") <= BOARD_DEPTH)
    adp = adp.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("times_drafted")  # see docstring (3)
    )
    return adp.select(
        "season", "name", "norm_name", "position", "team",
        "adp", "adp_rank", "times_drafted", "gsis_id",
    ).sort(["season", "adp_rank"])


def main() -> None:
    adp = build_adp_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    adp.write_parquet(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({adp.height} rows) "
          f"from FantasyPros '{ADP_FORMAT}' boards")

    counts = adp.group_by("season").len().sort("season")
    print("rows per season:")
    for s, n in counts.iter_rows():
        print(f"  {s}: {n}")

    top = adp.filter(pl.col("adp_rank") <= 150)
    match = (
        top.group_by("season")
        .agg(pl.len().alias("n"),
             pl.col("gsis_id").is_not_null().sum().alias("matched"))
        .sort("season")
        .with_columns((pl.col("matched") / pl.col("n")).round(4).alias("rate"))
    )
    print("gsis match rate (adp_rank<=150):")
    for s, n, m, r in match.iter_rows():
        print(f"  {s}: {m}/{n} = {r:.1%}")
    tot = top.select(
        pl.col("gsis_id").is_not_null().sum().alias("m"), pl.len().alias("n")
    ).row(0)
    print(f"  overall: {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.1%}")
    miss = top.filter(pl.col("team").is_null()).height
    print(f"team absent on {miss}/{top.height} top-150 rows "
          f"(build_pool falls back to the week-1 roster; check team_missing)")


if __name__ == "__main__":
    main()
