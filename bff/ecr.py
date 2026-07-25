"""Build preseason FantasyPros ECR table.

Sources: DynastyProcess archive (db_fpecr.parquet) for 2020-2025 snapshots,
a Wayback backfill of the FantasyPros PPR cheatsheet for 2012-2020
(db_fpecr_wayback.parquet, produced once by bff.ecr_wayback; overall rank = ECR),
plus a direct FantasyPros draft-rankings export for 2026
(FantasyPros_2026_Draft_ALL_Rankings.csv; overall RK = ECR, no fantasypros_id,
so 2026 rows map to gsis_id via the name+position fallbacks).

The 2012 and 2013 cheatsheet pages predate FantasyPros' inline player ids, so
those seasons also lean on the name fallbacks. That costs a few top-150 rows
per season to crosswalk ambiguity (two Steve Smith WRs, two Mike Williams WRs,
Frank Gore after suffix stripping) -- they resolve to null rather than to the
wrong player, and the ADP board cannot resolve them either, so they are absent
from the scored pool anyway. Pool coverage is 98.9% (2012) / 97.3% (2013) /
100% (2014+).
"""
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_ECR = ROOT / "data" / "raw" / "db_fpecr.parquet"
RAW_WAYBACK = ROOT / "data" / "raw" / "db_fpecr_wayback.parquet"
RAW_FP_2026 = ROOT / "data" / "raw" / "FantasyPros_2026_Draft_ALL_Rankings.csv"
RAW_IDS = ROOT / "data" / "raw" / "db_playerids.csv"
OUT = ROOT / "data" / "processed" / "ecr.parquet"

SEASONS = range(2020, 2026)
POSITIONS = ["QB", "RB", "WR", "TE"]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# FantasyPros nicknames -> crosswalk (db_playerids) names, post-norm_name.
# The 2012-2013 entries come from the id-less Wayback cheatsheets, where the
# site wrote a player's given or nickname form rather than the one the
# crosswalk uses.
_ALIASES = {
    "kenny gainwell": "kenneth gainwell",
    "chig okonkwo": "chigoziem okonkwo",
    "steve johnson": "stevie johnson",        # 2012, 2013
    "christopher ivory": "chris ivory",       # 2013
    "chris beanie wells": "beanie wells",     # 2012, quoted nickname
}


def norm_name(name: str) -> str:
    s = name.lower()
    # quotes matter here and nowhere else: the 2012 cheatsheet writes
    # 'Chris "Beanie" Wells'
    s = re.sub(r"[.'\-\"]", "", s)
    parts = [p for p in s.split() if p not in _SUFFIXES]
    out = " ".join(parts)
    return _ALIASES.get(out, out)


def load_fp_2026() -> pl.DataFrame | None:
    """FantasyPros 2026 draft-rankings export -> archive-schema rows (season 2026).
    Overall RK is the ECR value; the export has no fantasypros_id column."""
    if not RAW_FP_2026.exists():
        return None
    return (
        pl.read_csv(RAW_FP_2026, null_values=["-"], infer_schema_length=None)
        .select(
            pl.col("PLAYER NAME").alias("player"),
            pl.col("POS").alias("pos"),
            pl.col("TEAM").alias("team"),
            pl.col("RK").cast(pl.Float64).alias("ecr"),
            pl.lit(None, dtype=pl.Utf8).alias("id"),
            pl.lit(2026).alias("season"),
        )
    )


def build() -> pl.DataFrame:
    df = (
        pl.read_parquet(RAW_ECR)
        .filter(pl.col("page_type") == "redraft-overall")
        .with_columns(pl.col("scrape_date").str.to_date().alias("scrape_dt"))
    )

    frames = []
    snapshots = {}
    for yr in SEASONS:
        window = df.filter(
            pl.col("scrape_dt").is_between(pl.date(yr, 7, 1), pl.date(yr, 9, 10))
        )
        if window.height == 0:
            snapshots[yr] = None
            continue
        snap = window["scrape_dt"].max()
        snapshots[yr] = snap
        frames.append(
            window.filter(pl.col("scrape_dt") == snap)
            .select("player", "pos", "team", pl.col("ecr").cast(pl.Float64),
                    pl.col("id").cast(pl.Utf8))
            .with_columns(pl.lit(yr).alias("season"))
        )

    # Wayback backfill for 2015-2020 (archive schema; carries fantasypros_id)
    wayback_seasons = []
    if RAW_WAYBACK.exists():
        wb = pl.read_parquet(RAW_WAYBACK).select(
            "player", "pos", "team",
            pl.col("ecr").cast(pl.Float64),
            pl.col("id").cast(pl.Utf8),
            pl.col("season").cast(pl.Int32),
        )
        frames.append(wb)
        wayback_seasons = sorted(wb["season"].unique().to_list())

    fp26 = load_fp_2026()
    if fp26 is not None:
        frames.append(fp26)

    ecr = (
        pl.concat(frames)
        .with_columns(pl.col("pos").str.replace(r"\d+$", "").alias("position"))
        .filter(pl.col("position").is_in(POSITIONS))
        .with_columns(pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name"))
        # deterministic tie-break: player name orders rows with identical ecr,
        # so rank("ordinal") (which breaks ties by row position) is reproducible
        .sort(["season", "ecr", "player"])
        .with_columns(pl.col("ecr").rank("ordinal").over("season").cast(pl.Int32).alias("ecr_rank"))
        .select(
            "season", "player", "norm_name", "position", "team", "ecr", "ecr_rank",
            pl.col("id").alias("fantasypros_id"),
        )
    )

    ids = pl.read_csv(RAW_IDS, null_values=["NA"]).with_columns(
        pl.col("fantasypros_id").cast(pl.Utf8),
        pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name"),
    )

    # 1) join on fantasypros_id
    by_fp = ids.filter(pl.col("fantasypros_id").is_not_null()).select(
        "fantasypros_id", pl.col("gsis_id").alias("gsis_fp")
    ).unique(subset=["fantasypros_id"], keep="first")
    ecr = ecr.join(by_fp, on="fantasypros_id", how="left")

    # 2) fallback: (norm_name, position) restricted to players who already
    # existed that season. Suffix stripping collides across generations --
    # "Frank Gore" (2005) and "Frank Gore Jr." (2024) both normalize to
    # "frank gore", so the ungated join below sees two gsis ids and gives up,
    # dropping a top-40 RB from both 2012 and 2013. bff.adp has always gated
    # this way; bff.ecr did not until 2026-07-25. draft_year null = unknown,
    # always eligible.
    by_np_era = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .join(ecr.select("norm_name", "position", "season").unique(),
              on=["norm_name", "position"], how="inner")
        .filter(pl.col("draft_year").is_null()
                | (pl.col("draft_year") <= pl.col("season")))
        .group_by("norm_name", "position", "season")
        .agg(pl.col("gsis_id").n_unique().alias("n"),
             pl.col("gsis_id").first().alias("gsis_era"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "season", "gsis_era")
    )
    ecr = ecr.join(by_np_era, on=["norm_name", "position", "season"], how="left")

    # 3) fallback: (norm_name, position), only unambiguous gsis within the pair
    by_np = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .group_by("norm_name", "position")
        .agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first().alias("gsis_np"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "gsis_np")
    )
    ecr = ecr.join(by_np, on=["norm_name", "position"], how="left")

    # 4) fallback: norm_name alone when unambiguous
    by_n = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .group_by("norm_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first().alias("gsis_n"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "gsis_n")
    )
    ecr = ecr.join(by_n, on="norm_name", how="left")

    # id join is authoritative; then era-gated name+position (most precise of
    # the name routes), then the two ungated ones
    ecr = ecr.with_columns(
        pl.coalesce("gsis_fp", "gsis_era", "gsis_np", "gsis_n").alias("gsis_id")
    ).drop("gsis_fp", "gsis_era", "gsis_np", "gsis_n")

    ecr.write_parquet(OUT)

    for yr in SEASONS:
        if snapshots[yr] is None:
            print(f"{yr}: no snapshot in Jul 1 - Sep 10 window; dropped")
            continue
        sub = ecr.filter(pl.col("season") == yr)
        top = sub.filter(pl.col("ecr_rank") <= 150)
        rate = top["gsis_id"].is_not_null().sum() / top.height
        print(
            f"{yr}: snapshot={snapshots[yr]} rows={sub.height} "
            f"gsis_match_top150={top['gsis_id'].is_not_null().sum()}/{top.height} ({rate:.1%})"
        )
    for yr in wayback_seasons:
        sub = ecr.filter(pl.col("season") == yr)
        top = sub.filter(pl.col("ecr_rank") <= 150)
        rate = top["gsis_id"].is_not_null().sum() / top.height
        print(
            f"{yr}: source={RAW_WAYBACK.name} rows={sub.height} "
            f"gsis_match_top150={top['gsis_id'].is_not_null().sum()}/{top.height} ({rate:.1%})"
        )
    if fp26 is not None:
        sub = ecr.filter(pl.col("season") == 2026)
        top = sub.filter(pl.col("ecr_rank") <= 150)
        rate = top["gsis_id"].is_not_null().sum() / top.height
        print(
            f"2026: source={RAW_FP_2026.name} rows={sub.height} "
            f"gsis_match_top150={top['gsis_id'].is_not_null().sum()}/{top.height} ({rate:.1%})"
        )
    return ecr


if __name__ == "__main__":
    build()
