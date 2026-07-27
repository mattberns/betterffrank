"""Build player-season actuals table from nflverse weekly stats (2010-2024) + 2025 season aggregate.

Position filter with a BOARD FALLBACK (2026-07-27). nflverse's weekly position
field is not trustworthy for fantasy purposes: it is null for some ids (Trent
Richardson 2012-2014), and it uses roster positions the fantasy boards don't
(FB for Mike Tolbert / Marcel Reece, CB for two-way Travis Hunter 2025). A
plain `position.is_in(POS)` filter silently deleted those player-seasons,
which fabricated 0.0-point seasons for drafted players -- poisoning training
targets, prev-season features AND the backtest's ground truth (a scored-pool
player missing from actuals reads as 0 points). Rows whose raw position fails
the filter now fall back to the player's ADP/ECR-board position, so a player
the fantasy market drafts is always scored. `check_board_coverage` asserts the
invariant that would have caught this: every ADP-board player with nonzero raw
weekly points must appear in actuals for that season.

Requires data/processed/{adp,ecr}.parquet (built by bff.adp / bff.ecr, which
precede this step in the rebuild pipeline).
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "stats"
PROC = ROOT / "data" / "processed"
OUT = PROC / "actuals.parquet"
POS = ["QB", "RB", "WR", "TE"]


def board_positions() -> pl.DataFrame:
    """Fantasy position per gsis_id from the ADP/ECR boards (most frequent
    across all board appearances; alphabetical tiebreak for determinism)."""
    boards = pl.concat([
        pl.read_parquet(PROC / "adp.parquet").select("gsis_id", "position"),
        pl.read_parquet(PROC / "ecr.parquet").select("gsis_id", "position"),
    ]).drop_nulls().filter(pl.col("position").is_in(POS))
    return (
        boards.group_by("gsis_id", "position").len()
        .sort(["gsis_id", "len", "position"], descending=[False, True, False])
        .unique(subset=["gsis_id"], keep="first")
        .select("gsis_id", pl.col("position").alias("board_pos"))
    )


def _apply_position(df: pl.DataFrame, bpos: pl.DataFrame) -> pl.DataFrame:
    """Keep raw position when it is a fantasy position; otherwise fall back to
    the player's board position; drop rows with neither."""
    return (
        df.join(bpos, left_on="player_id", right_on="gsis_id", how="left")
        .with_columns(
            pl.when(pl.col("position").is_in(POS))
            .then(pl.col("position"))
            .otherwise(pl.col("board_pos"))
            .alias("position")
        )
        .drop("board_pos")
        .filter(pl.col("position").is_in(POS))
    )


def build_weekly(bpos: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for season in range(2010, 2025):
        df = pl.read_parquet(RAW / f"player_stats_{season}.parquet")
        df = df.filter(pl.col("season_type") == "REG")
        agg = (
            df.sort("week")
            .group_by(["player_id", "season"])
            .agg(
                position=pl.col("position").drop_nulls().last(),
                team=pl.col("recent_team").last(),
                games=pl.col("week").n_unique(),
                pts_std=pl.col("fantasy_points").sum(),
                pts_ppr=pl.col("fantasy_points_ppr").sum(),
            )
        )
        frames.append(_apply_position(agg, bpos))
    return pl.concat(frames)


def build_2025(bpos: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet(RAW / "stats_player_reg_2025.parquet")
    df = df.select(
        player_id="player_id",
        season="season",
        position="position",
        team="recent_team",
        games="games",
        pts_std="fantasy_points",
        pts_ppr="fantasy_points_ppr",
    )
    return _apply_position(df, bpos)


def check_board_coverage(actuals: pl.DataFrame) -> None:
    """Every ADP-board player with nonzero raw weekly PPR must be in actuals.
    This is the invariant whose absence hid the Trent Richardson / FB / CB
    fabricated-zero seasons until 2026-07-27."""
    frames = []
    for season in range(2010, 2025):
        wk = pl.read_parquet(RAW / f"player_stats_{season}.parquet")
        frames.append(
            wk.filter(pl.col("season_type") == "REG")
            .group_by("player_id", "season")
            .agg(pl.col("fantasy_points_ppr").sum().alias("raw_ppr"))
        )
    r25 = pl.read_parquet(RAW / "stats_player_reg_2025.parquet")
    frames.append(
        r25.group_by("player_id", "season")
        .agg(pl.col("fantasy_points_ppr").sum().alias("raw_ppr"))
    )
    raw = pl.concat(frames).rename({"player_id": "gsis_id"}).cast({"season": pl.Int32})
    adp = pl.read_parquet(PROC / "adp.parquet")
    missing = (
        adp.filter(pl.col("gsis_id").is_not_null())
        .select("season", "gsis_id", "name")
        .cast({"season": pl.Int32})
        .join(raw, on=["season", "gsis_id"], how="inner")
        .filter(pl.col("raw_ppr") > 0)
        .join(
            actuals.select("season", "gsis_id").with_columns(ok=pl.lit(True)),
            on=["season", "gsis_id"], how="left",
        )
        .filter(pl.col("ok").is_null())
    )
    assert missing.height == 0, (
        f"{missing.height} ADP-board player-seasons with raw points are missing "
        f"from actuals:\n{missing.select('season', 'name', 'raw_ppr').head(20)}"
    )


def main() -> None:
    bpos = board_positions()
    df = pl.concat(
        [build_weekly(bpos), build_2025(bpos)],
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

    check_board_coverage(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)

    print(df.group_by("season").len().sort("season"))
    print("\n2023 top 5 overall PPR:")
    print(
        df.filter(pl.col("season") == 2023)
        .head(5)
        .select(["gsis_id", "position", "team", "games", "pts_ppr", "rank_overall_ppr"])
    )
    adp = pl.read_parquet(PROC / "adp.parquet")
    hunter = adp.filter(pl.col("name") == "Travis Hunter")["gsis_id"].drop_nulls()
    hunter_id = hunter[0] if hunter.len() else ""
    print("\nboard-fallback spot checks (fabricated zeros until 2026-07-27):")
    print(
        df.filter(
            (pl.col("gsis_id") == "00-0029675") & pl.col("season").is_in([2012, 2013, 2014])
            | (pl.col("gsis_id") == "00-0026069") & (pl.col("season") == 2012)
            | (pl.col("gsis_id") == "00-0026393") & (pl.col("season") == 2013)
            | (pl.col("gsis_id") == hunter_id) & (pl.col("season") == 2025)
        ).select("season", "gsis_id", "position", "games", "pts_ppr")
    )


if __name__ == "__main__":
    main()
