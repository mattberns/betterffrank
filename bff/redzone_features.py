"""Red-zone / high-leverage opportunity features, keyed (season, gsis_id), seasons 2011-2026.

One row per season-t ADP-pool player (QB/RB/WR/TE from data/processed/adp.parquet).
Every feature for target season t is computed from season t-1 REGULAR-SEASON
play-by-play ONLY (data/raw/pbp/play_by_play_{t-1}.parquet, t-1 = 2010..2025).
Never season t. Leakage-safe by construction: t-1 pbp outcomes are known before
season-t drafts. Players join the pbp via receiver_player_id / rusher_player_id,
which are GSIS ids.

Play definitions (all on t-1 REG plays):
  target = pass_attempt == 1 with a non-null receiver_player_id (incompletes count)
  carry  = rush_attempt == 1 with a non-null rusher_player_id
  team   = posteam
  yardline buckets for expected-TD: 1-5, 6-10, 11-20, 21+ (from yardline_100)

FEATURES (prefix rz_, all from t-1 REG pbp):
  rz_target_share      player inside-20 (yardline_100 <= 20) targets / team inside-20
                       targets. Numerator and denominator are both taken on the
                       player's PRIMARY team (the posteam with his most total
                       opportunities); traded players are measured on that team.
                       Null when the team denominator < 10.
  rz_gl_carry_share    player goal-line (yardline_100 <= 5) carries / team goal-line
                       carries, on the primary team. Null when team denom < 10.
  rz_ez_target_share   player end-zone targets / team end-zone targets, on the primary
                       team. An end-zone target is a pass thrown to/past the goal line
                       (yardline_100 <= air_yards, air_yards non-null). Null when team
                       denom < 10.
  rz_opp_pg            (inside-20 targets + inside-10 carries) / games. Games = distinct
                       game_id in which the player had ANY target or carry anywhere on
                       the field. Uses opportunities on ALL teams (not a share).
  rz_td_minus_expected player actual (receiving + rushing) TDs minus expected TDs.
                       Expected = sum over the player's targets/carries of the
                       LEAGUE-WIDE t-1 TD rate for that opportunity type (target vs
                       carry) in that yardline_100 bucket. League rates are computed
                       from the SAME t-1 pbp season only (no cross-season data).
                       Positive = scored more than located opportunity implies
                       (regression-DOWN candidate); negative = TD shortfall on real
                       opportunity (buy signal). Null below 20 total opportunities.

Nulls stay null in the parquet (the model zero-fills at join time, same as the opp
features). Rows for pool players with no t-1 pbp opportunities are all-null rz_*.

pbp fetch (nflverse, one file per season 2010..2025):
  for y in $(seq 2010 2025); do \
    curl -L -o data/raw/pbp/play_by_play_${y}.parquet \
      https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_${y}.parquet; \
  done

Build: uv run python -m bff.redzone_features
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PBP = ROOT / "data" / "raw" / "pbp"

SEASONS = range(2011, 2027)  # target seasons t; pbp source is t-1 = 2010..2025
FANTASY_POS = ["QB", "RB", "WR", "TE"]

# Only these columns are read from each ~19MB pbp file.
PBP_COLS = [
    "season_type", "game_id", "posteam", "yardline_100", "air_yards",
    "pass_attempt", "rush_attempt", "complete_pass",
    "receiver_player_id", "rusher_player_id", "pass_touchdown", "rush_touchdown",
]

MIN_TEAM_DENOM = 10   # share denominators below this -> null
MIN_TD_OPPS = 20      # rz_td_minus_expected needs this many total opportunities


def _bucket() -> pl.Expr:
    """yardline_100 -> distance bucket; null when yardline_100 is null."""
    y = pl.col("yardline_100")
    return (
        pl.when(y <= 5).then(pl.lit("1-5"))
        .when(y <= 10).then(pl.lit("6-10"))
        .when(y <= 20).then(pl.lit("11-20"))
        .when(y.is_not_null()).then(pl.lit("21+"))
        .otherwise(None)
        .alias("bucket")
    )


def process_season(source: int) -> pl.DataFrame:
    """Feature row per player from ONE source (t-1) pbp season; season -> source+1."""
    pbp = (
        pl.read_parquet(PBP / f"play_by_play_{source}.parquet", columns=PBP_COLS)
        .filter(pl.col("season_type") == "REG")
        .with_columns(_bucket())
    )

    # Targets: pass attempts with a receiver (incompletes included).
    targets = pbp.filter(
        (pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null()
    ).with_columns(
        (pl.col("yardline_100") <= 20).alias("t_in20"),
        (pl.col("air_yards").is_not_null() & (pl.col("yardline_100") <= pl.col("air_yards")))
        .alias("t_ez"),
    )
    # Carries: rush attempts with a rusher.
    carries = pbp.filter(
        (pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null()
    ).with_columns(
        (pl.col("yardline_100") <= 5).alias("c_in5"),
        (pl.col("yardline_100") <= 10).alias("c_in10"),
    )

    # League-wide TD rate per opportunity type per bucket (this season only).
    tgt_rate = (
        targets.filter(pl.col("bucket").is_not_null())
        .group_by("bucket").agg(pl.col("pass_touchdown").mean().alias("rate"))
    )
    car_rate = (
        carries.filter(pl.col("bucket").is_not_null())
        .group_by("bucket").agg(pl.col("rush_touchdown").mean().alias("rate"))
    )
    targets = targets.join(tgt_rate, on="bucket", how="left").with_columns(
        pl.col("rate").fill_null(0.0).alias("exp_td")
    )
    carries = carries.join(car_rate, on="bucket", how="left").with_columns(
        pl.col("rate").fill_null(0.0).alias("exp_td")
    )

    # Primary team = posteam with the player's most total opportunities
    # (targets + carries). Tiebreak on posteam for determinism.
    opps_pt = pl.concat([
        targets.select(pl.col("receiver_player_id").alias("pid"), "posteam"),
        carries.select(pl.col("rusher_player_id").alias("pid"), "posteam"),
    ]).group_by("pid", "posteam").len()
    primary = (
        opps_pt.sort(["len", "posteam"], descending=[True, False])
        .group_by("pid", maintain_order=True).first()
        .select("pid", pl.col("posteam").alias("primary_team"))
    )

    targets = targets.join(primary, left_on="receiver_player_id", right_on="pid", how="left")
    carries = carries.join(primary, left_on="rusher_player_id", right_on="pid", how="left")

    # Team denominators, per posteam.
    team_tgt = targets.group_by("posteam").agg(
        pl.col("t_in20").sum().alias("team_in20"),
        pl.col("t_ez").sum().alias("team_ez"),
    )
    team_car = carries.group_by("posteam").agg(pl.col("c_in5").sum().alias("team_in5"))

    # Player receiving aggregates.
    rec = targets.group_by(pl.col("receiver_player_id").alias("pid")).agg(
        pl.col("t_in20").sum().alias("rec_in20_all"),
        (pl.col("t_in20") & (pl.col("posteam") == pl.col("primary_team"))).sum().alias("rec_in20_pt"),
        (pl.col("t_ez") & (pl.col("posteam") == pl.col("primary_team"))).sum().alias("rec_ez_pt"),
        pl.col("pass_touchdown").sum().alias("rec_tds"),
        pl.col("exp_td").sum().alias("rec_exp"),
        pl.len().alias("rec_opps"),
    )
    # Player rushing aggregates.
    rush = carries.group_by(pl.col("rusher_player_id").alias("pid")).agg(
        pl.col("c_in10").sum().alias("rush_in10_all"),
        (pl.col("c_in5") & (pl.col("posteam") == pl.col("primary_team"))).sum().alias("rush_in5_pt"),
        pl.col("rush_touchdown").sum().alias("rush_tds"),
        pl.col("exp_td").sum().alias("rush_exp"),
        pl.len().alias("rush_opps"),
    )

    # Games with any target or carry anywhere on the field.
    games = pl.concat([
        targets.select(pl.col("receiver_player_id").alias("pid"), "game_id"),
        carries.select(pl.col("rusher_player_id").alias("pid"), "game_id"),
    ]).unique().group_by("pid").len().rename({"len": "rz_games"})

    # Merge everything on pid.
    player = (
        rec.join(rush, on="pid", how="full", coalesce=True)
        .join(games, on="pid", how="left")
        .join(primary, on="pid", how="left")
    )
    count_cols = [
        "rec_in20_all", "rec_in20_pt", "rec_ez_pt", "rec_tds", "rec_exp", "rec_opps",
        "rush_in10_all", "rush_in5_pt", "rush_tds", "rush_exp", "rush_opps",
    ]
    player = player.with_columns([pl.col(c).fill_null(0.0) for c in count_cols])

    # Attach primary-team denominators.
    player = (
        player.join(team_tgt.rename({"posteam": "primary_team"}), on="primary_team", how="left")
        .join(team_car.rename({"posteam": "primary_team"}), on="primary_team", how="left")
    )

    share = lambda num, den: pl.when(pl.col(den) >= MIN_TEAM_DENOM).then(
        pl.col(num) / pl.col(den)
    ).otherwise(None)

    out = player.with_columns(
        share("rec_in20_pt", "team_in20").alias("rz_target_share"),
        share("rush_in5_pt", "team_in5").alias("rz_gl_carry_share"),
        share("rec_ez_pt", "team_ez").alias("rz_ez_target_share"),
        pl.when(pl.col("rz_games") > 0)
        .then((pl.col("rec_in20_all") + pl.col("rush_in10_all")) / pl.col("rz_games"))
        .otherwise(None).alias("rz_opp_pg"),
        pl.when((pl.col("rec_opps") + pl.col("rush_opps")) >= MIN_TD_OPPS)
        .then((pl.col("rec_tds") + pl.col("rush_tds")) - (pl.col("rec_exp") + pl.col("rush_exp")))
        .otherwise(None).alias("rz_td_minus_expected"),
    ).select(
        pl.lit(source + 1, dtype=pl.Int64).alias("season"),
        pl.col("pid").alias("gsis_id"),
        "rz_target_share", "rz_gl_carry_share", "rz_ez_target_share",
        "rz_opp_pg", "rz_td_minus_expected",
    )
    return out


RZ_FEATS = [
    "rz_target_share", "rz_gl_carry_share", "rz_ez_target_share",
    "rz_opp_pg", "rz_td_minus_expected",
]


def build_pool() -> pl.DataFrame:
    adp = pl.read_parquet(PROC / "adp.parquet")
    pool = adp.filter(
        pl.col("season").is_between(min(SEASONS), max(SEASONS))
        & pl.col("position").is_in(FANTASY_POS)
        & pl.col("gsis_id").is_not_null()
    ).select("season", "gsis_id", "name", "position")
    assert pool.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    return pool


def build() -> pl.DataFrame:
    feats = pl.concat([process_season(s) for s in range(2010, 2026)])
    # A player can appear on multiple pbp seasons but each maps to a distinct
    # target season; still guard uniqueness of the feature key.
    assert feats.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    pool = build_pool()
    out = pool.join(feats, on=["season", "gsis_id"], how="left")
    assert out.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    assert out["season"].min() >= 2011 and out["season"].max() <= 2026
    return out.sort("season", "gsis_id")


def coverage_report(out: pl.DataFrame) -> None:
    for label, df in [
        ("2015-2025", out.filter(pl.col("season").is_between(2015, 2025))),
        ("2026", out.filter(pl.col("season") == 2026)),
    ]:
        print(f"\n== coverage {label} (n={df.height}) ==")
        for c in RZ_FEATS:
            nn = 1.0 - df[c].null_count() / max(df.height, 1)
            mean = df[c].cast(pl.Float64).mean()
            print(f"  {c:22s} non-null {nn:6.1%}  mean {mean if mean is not None else float('nan'):9.4f}")


def sanity(out: pl.DataFrame) -> None:
    cases = [
        (2026, "Derrick Henry", RZ_FEATS),        # 2025 BAL goal-line back -> high gl_carry_share
        (2026, "Ja'Marr Chase", RZ_FEATS),         # WR1 -> high target_share, real ez targets
        (2026, "Amon-Ra", RZ_FEATS),               # high-target slot WR -> modest ez share
    ]
    for season, sub, cols in cases:
        row = out.filter((pl.col("season") == season) & pl.col("name").str.contains(sub))
        print(f"\n{season} {sub}:")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(row.select(["name", "position"] + cols))


def main() -> None:
    out = build()
    out.write_parquet(PROC / "redzone_features.parquet")
    print(f"wrote {out.height} rows, seasons {out['season'].min()}-{out['season'].max()}, "
          f"{len(RZ_FEATS)} rz_ features")
    coverage_report(out)
    sanity(out)


if __name__ == "__main__":
    main()
