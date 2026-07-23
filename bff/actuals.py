"""Build player-season actuals table from nflverse weekly stats (2010-2024) + 2025 season aggregate."""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "stats"
OUT = ROOT / "data" / "processed" / "actuals.parquet"
POS = ["QB", "RB", "WR", "TE"]


def build_weekly() -> pl.DataFrame:
    frames = []
    for season in range(2010, 2025):
        df = pl.read_parquet(RAW / f"player_stats_{season}.parquet")
        df = df.filter(
            (pl.col("season_type") == "REG") & pl.col("position").is_in(POS)
        )
        agg = (
            df.sort("week")
            .group_by(["player_id", "season"])
            .agg(
                position=pl.col("position").last(),
                team=pl.col("recent_team").last(),
                games=pl.col("week").n_unique(),
                pts_std=pl.col("fantasy_points").sum(),
                pts_ppr=pl.col("fantasy_points_ppr").sum(),
            )
        )
        frames.append(agg)
    return pl.concat(frames)


def build_2025() -> pl.DataFrame:
    df = pl.read_parquet(RAW / "stats_player_reg_2025.parquet")
    return (
        df.filter(pl.col("position").is_in(POS))
        .select(
            player_id="player_id",
            season="season",
            position="position",
            team="recent_team",
            games="games",
            pts_std="fantasy_points",
            pts_ppr="fantasy_points_ppr",
        )
    )


def main() -> None:
    df = pl.concat(
        [build_weekly(), build_2025()],
        how="vertical_relaxed",
    ).cast({"season": pl.Int32, "games": pl.Int32})
    df = df.rename({"player_id": "gsis_id"}).with_columns(
        pts_half=(pl.col("pts_std") + pl.col("pts_ppr")) / 2,
        ppg_ppr=pl.col("pts_ppr") / pl.col("games"),
    )
    df = df.with_columns(
        rank_overall_ppr=pl.col("pts_ppr")
        .rank(method="dense", descending=True)
        .over("season"),
        rank_pos_ppr=pl.col("pts_ppr")
        .rank(method="dense", descending=True)
        .over(["season", "position"]),
    ).sort(["season", "rank_overall_ppr"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)

    print(df.group_by("season").len().sort("season"))
    print("\n2023 top 5 overall PPR:")
    print(
        df.filter(pl.col("season") == 2023)
        .head(5)
        .select(["gsis_id", "position", "team", "games", "pts_ppr", "rank_overall_ppr"])
    )


if __name__ == "__main__":
    main()
