"""Context data for v2 features: draft picks, rosters, week-1 coaches, team volume, team QBs.

Raw inputs (fetched from nflverse releases; see fetch commands in docstring of `main`):
    data/raw/context/draft_picks.parquet   (nflverse draft_picks release; drafts 1980-2026)
    data/raw/context/roster_YYYY.parquet   (nflverse rosters release; 2010-2026)
    data/raw/context/games.csv             (nflverse schedules release; 1999-2026, incl. coaches)

Normalized outputs (all team codes canonicalized via `norm_team`):
    data/processed/ctx_draft_picks.parquet
        season, round, pick, team, gsis_id, pfr_player_id, name, position, category
        Drafts 2010-2026. 2026 gsis_ids from nflverse are placeholders
        (e.g. 'MEN516487'); any gsis_id not matching '00-00#####' is nulled,
        so match 2026 rookies on (name, position, team).
    data/processed/ctx_rosters.parquet
        season, team, gsis_id, position, name, status
        Seasons 2010-2026, one row per (season, team, gsis_id) (dedup keeps
        first). 2026 is the offseason roster snapshot (preseason-known).
    data/processed/ctx_week1_coaches.parquet
        season, team, week1_coach
        Seasons 2010-2026, 32 teams/season (30/31 before 2026 expansion?  no:
        always full league; verified in main). Week-1 REG coach only, which is
        preseason-known; mid-season firings never enter this table.
    data/processed/ctx_team_volume.parquet
        season, team, gsis_id, name, position, targets, carries, rec_fp_ppr,
        fp_ppr, team_targets, team_carries, team_rec_fp_ppr, team_fp_ppr
        REG-season only. 2010-2024 from weekly stats; 2025 from the
        season-level REG file (single recent_team per player there).
        rec_fp_ppr = receptions + 0.1*receiving_yards + 6*receiving_tds.
    data/processed/ctx_team_qb.parquet
        season, team, primary_qb_gsis, primary_qb_name, primary_qb_att,
        expected_qb_gsis, expected_qb_name, expected_qb_adp
        primary_qb = QB with most REG pass attempts that season (use only for
        t-1 lookups; it is in-season information for season t).
        expected_qb = QB on that team with the best (lowest) ADP in that
        season's ADP pool (preseason-known; safe for season t). Seasons
        2010-2025 for primary, 2010-2026 for expected.

Leakage notes: ctx_draft_picks, ctx_rosters, ctx_week1_coaches, and the
expected_qb columns are preseason facts for season t. ctx_team_volume and
primary_qb are season-t outcomes; only join them as t-1 (or earlier) features.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CTX = RAW / "context"
PROC = ROOT / "data" / "processed"

# Canonical team codes = the 32 codes used by data/processed/adp.parquet
# (LAR for Rams, LAC for Chargers, LV for Raiders, WAS, JAX, ...).
# Franchise moves/renames map to the modern franchise code:
#   Rams: STL/SL/LA -> LAR   Chargers: SD/SDG -> LAC   Raiders: OAK/LVR -> LV
# Source-specific spellings (PFR: GNB/KAN/NOR/NWE/SFO/TAM/LVR/SDG;
# GSIS rosters: ARZ/BLT/CLV/HST/SL) map to canonical.
TEAM_MAP = {
    # Rams
    "STL": "LAR", "SL": "LAR", "LA": "LAR",
    # Chargers
    "SD": "LAC", "SDG": "LAC",
    # Raiders
    "OAK": "LV", "LVR": "LV",
    # PFR spellings
    "GNB": "GB", "KAN": "KC", "NOR": "NO", "NWE": "NE",
    "SFO": "SF", "TAM": "TB",
    # GSIS roster spellings
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    # Misc variants
    "JAC": "JAX", "WSH": "WAS", "LVE": "LV",
}

GSIS_RE = re.compile(r"^00-00\d{5}$")


def norm_team(col: str = "team") -> pl.Expr:
    """Expression that maps any source team code to the canonical code."""
    return pl.col(col).replace(TEAM_MAP)


def _valid_gsis(col: str = "gsis_id") -> pl.Expr:
    """Null out non-standard gsis_ids (nflverse stamps placeholders like
    'MEN516487' on 2026 draftees before real GSIS ids exist)."""
    return (
        pl.when(pl.col(col).str.contains(r"^00-00\d{5}$"))
        .then(pl.col(col))
        .otherwise(None)
        .alias(col)
    )


def build_draft_picks() -> pl.DataFrame:
    dp = pl.read_parquet(CTX / "draft_picks.parquet")
    out = (
        dp.filter(pl.col("season").is_between(2010, 2026))
        .select(
            "season", "round", "pick",
            norm_team("team").alias("team"),
            _valid_gsis("gsis_id"),
            "pfr_player_id",
            pl.col("pfr_player_name").alias("name"),
            "position", "category",
        )
        .sort("season", "pick")
    )
    out.write_parquet(PROC / "ctx_draft_picks.parquet")
    return out


def build_rosters() -> pl.DataFrame:
    frames = []
    for y in range(2010, 2027):
        r = pl.read_parquet(CTX / f"roster_{y}.parquet")
        frames.append(
            r.select(
                pl.lit(y, dtype=pl.Int32).alias("season"),
                norm_team("team").alias("team"),
                "gsis_id",
                "position",
                pl.col("full_name").alias("name"),
                "status",
            )
        )
    out = (
        pl.concat(frames)
        .filter(pl.col("gsis_id").is_not_null())
        .unique(subset=["season", "team", "gsis_id"], keep="first")
        .sort("season", "team")
    )
    out.write_parquet(PROC / "ctx_rosters.parquet")
    return out


def build_week1_coaches() -> pl.DataFrame:
    g = pl.read_csv(CTX / "games.csv", infer_schema_length=10000)
    reg = g.filter(
        (pl.col("season").is_between(2010, 2026)) & (pl.col("game_type") == "REG")
    )
    away = reg.select(
        "season", "week",
        norm_team("away_team").alias("team"),
        pl.col("away_coach").alias("week1_coach"),
    )
    home = reg.select(
        "season", "week",
        norm_team("home_team").alias("team"),
        pl.col("home_coach").alias("week1_coach"),
    )
    # Coach of each team's FIRST regular-season game (week 1 for everyone
    # except postponements, e.g. 2017 MIA/TB moved to week 11 by Hurricane
    # Irma; the season-opening coach is still the preseason-known one).
    out = (
        pl.concat([away, home])
        .sort("week")
        .group_by("season", "team", maintain_order=True)
        .agg(pl.col("week1_coach").first())
        .sort("season", "team")
    )
    out.write_parquet(PROC / "ctx_week1_coaches.parquet")
    return out


def _volume_2010_2024() -> pl.DataFrame:
    frames = []
    for y in range(2010, 2025):
        w = pl.read_parquet(RAW / "stats" / f"player_stats_{y}.parquet")
        frames.append(
            w.filter(pl.col("season_type") == "REG")
            .group_by("season", "recent_team", "player_id")
            .agg(
                pl.col("player_display_name").first().alias("name"),
                pl.col("position").first(),
                pl.col("targets").sum(),
                pl.col("carries").sum(),
                (
                    pl.col("receptions").sum()
                    + 0.1 * pl.col("receiving_yards").sum()
                    + 6 * pl.col("receiving_tds").sum()
                ).alias("rec_fp_ppr"),
                pl.col("fantasy_points_ppr").sum().alias("fp_ppr"),
            )
        )
    return pl.concat(frames)


def _volume_2025() -> pl.DataFrame:
    s = pl.read_parquet(RAW / "stats" / "stats_player_reg_2025.parquet")
    return s.select(
        pl.col("season"),
        pl.col("recent_team"),
        pl.col("player_id"),
        pl.col("player_display_name").alias("name"),
        pl.col("position"),
        pl.col("targets"),
        pl.col("carries"),
        (
            pl.col("receptions") + 0.1 * pl.col("receiving_yards") + 6 * pl.col("receiving_tds")
        ).alias("rec_fp_ppr"),
        pl.col("fantasy_points_ppr").alias("fp_ppr"),
    )


def build_team_volume() -> pl.DataFrame:
    vol = pl.concat([_volume_2010_2024(), _volume_2025()], how="vertical_relaxed")
    vol = vol.with_columns(norm_team("recent_team").alias("team")).drop("recent_team")
    vol = vol.rename({"player_id": "gsis_id"})
    out = (
        vol.with_columns(
            pl.col("targets").sum().over("season", "team").alias("team_targets"),
            pl.col("carries").sum().over("season", "team").alias("team_carries"),
            pl.col("rec_fp_ppr").sum().over("season", "team").alias("team_rec_fp_ppr"),
            pl.col("fp_ppr").sum().over("season", "team").alias("team_fp_ppr"),
        )
        .select(
            "season", "team", "gsis_id", "name", "position",
            "targets", "carries", "rec_fp_ppr", "fp_ppr",
            "team_targets", "team_carries", "team_rec_fp_ppr", "team_fp_ppr",
        )
        .sort("season", "team", pl.col("fp_ppr"), descending=[False, False, True])
    )
    out.write_parquet(PROC / "ctx_team_volume.parquet")
    return out


def build_team_qb() -> pl.DataFrame:
    # Primary QB: most REG pass attempts, per (season, team). In-season info.
    frames = []
    for y in range(2010, 2025):
        w = pl.read_parquet(RAW / "stats" / f"player_stats_{y}.parquet")
        frames.append(
            w.filter((pl.col("season_type") == "REG") & (pl.col("position") == "QB"))
            .group_by("season", "recent_team", "player_id")
            .agg(
                pl.col("player_display_name").first().alias("name"),
                pl.col("attempts").sum(),
            )
        )
    s25 = pl.read_parquet(RAW / "stats" / "stats_player_reg_2025.parquet")
    frames.append(
        s25.filter(pl.col("position") == "QB").select(
            "season", "recent_team", "player_id",
            pl.col("player_display_name").alias("name"),
            "attempts",
        )
    )
    qb = (
        pl.concat(frames, how="vertical_relaxed")
        .with_columns(norm_team("recent_team").alias("team"))
        .sort("attempts", descending=True)
        .group_by("season", "team", maintain_order=True)
        .agg(
            pl.col("player_id").first().alias("primary_qb_gsis"),
            pl.col("name").first().alias("primary_qb_name"),
            pl.col("attempts").first().alias("primary_qb_att"),
        )
    )

    # Expected QB: best-ADP QB on the team in that season's preseason ADP pool.
    adp = pl.read_parquet(PROC / "adp.parquet")
    exp = (
        adp.filter(
            (pl.col("position") == "QB")
            & pl.col("team").is_not_null()
            & (pl.col("team") != "FA")
        )
        .with_columns(norm_team("team").alias("team"))
        .sort("adp")
        .group_by("season", "team", maintain_order=True)
        .agg(
            pl.col("gsis_id").first().alias("expected_qb_gsis"),
            pl.col("name").first().alias("expected_qb_name"),
            pl.col("adp").first().alias("expected_qb_adp"),
        )
    )

    out = qb.join(exp, on=["season", "team"], how="full", coalesce=True).sort(
        "season", "team"
    )
    out.write_parquet(PROC / "ctx_team_qb.parquet")
    return out


def main() -> None:
    """Fetch commands used to populate data/raw/context/ (July 2026):

    curl -sL -o data/raw/context/draft_picks.parquet \\
      https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.parquet
    for y in $(seq 2010 2026); do curl -sL -o data/raw/context/roster_$y.parquet \\
      https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_$y.parquet; done
    curl -sL -o data/raw/context/games.csv \\
      https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
    """
    canon = set(
        "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LAC LAR "
        "LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()
    )
    for label, df, team_col in [
        ("draft_picks", build_draft_picks(), "team"),
        ("rosters", build_rosters(), "team"),
        ("week1_coaches", build_week1_coaches(), "team"),
        ("team_volume", build_team_volume(), "team"),
        ("team_qb", build_team_qb(), "team"),
    ]:
        bad = set(df[team_col].drop_nulls().unique().to_list()) - canon
        assert not bad, f"{label}: non-canonical team codes {bad}"
        seasons = df["season"]
        print(
            f"{label}: {df.height} rows, seasons {seasons.min()}-{seasons.max()}, "
            f"teams/season(last)={df.filter(pl.col('season') == seasons.max())[team_col].n_unique()}"
        )


if __name__ == "__main__":
    main()
