"""Build preseason FantasyPros ECR table from DynastyProcess archive."""
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_ECR = ROOT / "data" / "raw" / "db_fpecr.parquet"
RAW_IDS = ROOT / "data" / "raw" / "db_playerids.csv"
OUT = ROOT / "data" / "processed" / "ecr.parquet"

SEASONS = range(2020, 2026)
POSITIONS = ["QB", "RB", "WR", "TE"]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[.'\-]", "", s)
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


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
        frames.append(window.filter(pl.col("scrape_dt") == snap).with_columns(pl.lit(yr).alias("season")))

    ecr = (
        pl.concat(frames)
        .with_columns(pl.col("pos").str.replace(r"\d+$", "").alias("position"))
        .filter(pl.col("position").is_in(POSITIONS))
        .with_columns(pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8).alias("norm_name"))
        .sort(["season", "ecr"])
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

    # 2) fallback: (norm_name, position), only unambiguous gsis within the pair
    by_np = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .group_by("norm_name", "position")
        .agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first().alias("gsis_np"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "position", "gsis_np")
    )
    ecr = ecr.join(by_np, on=["norm_name", "position"], how="left")

    # 3) fallback: norm_name alone when unambiguous
    by_n = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .group_by("norm_name")
        .agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first().alias("gsis_n"))
        .filter(pl.col("n") == 1)
        .select("norm_name", "gsis_n")
    )
    ecr = ecr.join(by_n, on="norm_name", how="left")

    ecr = ecr.with_columns(
        pl.coalesce("gsis_fp", "gsis_np", "gsis_n").alias("gsis_id")
    ).drop("gsis_fp", "gsis_np", "gsis_n")

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
    return ecr


if __name__ == "__main__":
    build()
