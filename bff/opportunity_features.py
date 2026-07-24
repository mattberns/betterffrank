"""Weekly-opportunity features, keyed (season, gsis_id), seasons 2011-2026.

One row per season-t ADP-pool player (QB/RB/WR/TE from data/processed/adp.parquet).
Every feature for target season t is computed from season t-1 REGULAR-SEASON weekly
rows ONLY (data/raw/stats/player_stats_{t-1}.parquet for t-1 <= 2024,
stats_player_week_2025.parquet for t-1 = 2025). Never season t. Leakage-safe by
construction: t-1 weekly outcomes are known before season-t drafts.

Shares are recomputed from raw counts against team weekly totals derived from the
same file (consistent across schema vintages; the 2025 file renames recent_team ->
team and drops the precomputed share columns' guarantees).

"Games played" = weeks where the player has a stat line in the weekly file.

FEATURES (prefix opp_, all from t-1 REG weeks):
  Levels
    opp_games            games with a stat line
    opp_target_share     mean weekly targets / team targets
    opp_air_yards_share  mean weekly receiving_air_yards / team air yards
    opp_wopr             mean weekly 1.5*target_share + 0.7*air_yards_share
    opp_carry_share      mean weekly carries / team carries
    opp_targets_pg       total targets / games
    opp_carries_pg       total carries / games
  Opportunity-production divergence (each ratio null below a minimum-volume
  guard, so near-zero denominators cannot fabricate extreme values:
  racr needs >=100 air yards; per-target ratios >=10 targets; per-carry >=10
  carries; td_per_opp >=10 opportunities; per-attempt >=30 attempts)
    opp_racr             sum receiving_yards / sum receiving_air_yards
    opp_ypt              sum receiving_yards / sum targets
    opp_ypt_vs_pos       opp_ypt minus same-position aggregate ypt in season t-1
    opp_td_per_opp       (rec TDs + rush TDs) / (targets + carries)
    opp_td_per_opp_vs_pos  minus same-position aggregate (negative = TD shortfall
                           on real opportunity = buy signal)
    opp_epa_per_target   sum receiving_epa / sum targets
    opp_epa_per_carry    sum rushing_epa / sum carries
    opp_ppg              mean weekly fantasy_points_ppr
    opp_fp_exp_pg        opportunity-implied PPR/game with FIXED public-style
                         weights: 1.5*targets + 0.07*receiving_air_yards +
                         0.6*carries (documented constant, not fit to eval years)
    opp_fp_oe_pg         opp_ppg - opp_fp_exp_pg (production over opportunity)
  Velocity (for target_share=ts, wopr, carry_share=cs, weekly targets=tpg,
            weekly pass attempts=att; null + opp_short_season=1 when games < 6)
    opp_{m}_slope        OLS slope of the weekly value on week number
    opp_{m}_l6_delta     mean of last 6 games minus season mean
    opp_{m}_l4f4         mean of last 4 games minus mean of first 4 games
                         (windows overlap when 6 <= games < 8; accepted)
  Stability
    opp_ts_std           weekly stdev of target share
    opp_boom_rate        share of games with >= 20 PPR points
  QB passing-volume analogues (near-zero for non-QBs)
    opp_attempts_pg      pass attempts / game
    opp_pass_air_pg      passing_air_yards / game
    opp_epa_per_att      sum passing_epa / sum attempts
  Flags
    has_prior_weekly     1 if any t-1 REG weekly rows exist (0 = rookie/DNP:
                         all opp_* null, nothing fabricated)
    opp_short_season     1 if 0 < games < 6 (velocity features null)
    opp_has_velocity     1 if games >= 6

Build: uv run python -m bff.opportunity_features
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "stats"

SEASONS = range(2011, 2027)  # target seasons t; weekly source is t-1 = 2010..2025
FANTASY_POS = ["QB", "RB", "WR", "TE"]

STAT_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_epa",
    "carries", "rushing_tds", "rushing_epa",
    "attempts", "passing_air_yards", "passing_epa",
    "fantasy_points_ppr",
]

# Fixed opportunity-implied PPR weights (public-style constants; never tuned).
W_TGT, W_AIR, W_CAR = 1.5, 0.07, 0.6

MIN_GAMES_VELOCITY = 6

# Minimum-volume guards for divergence ratios (noise control, not tuned).
MIN_AIR, MIN_TGT, MIN_CAR, MIN_OPP, MIN_ATT = 100, 10, 10, 10, 30


def load_weekly() -> pl.DataFrame:
    """Season t-1 REG weekly rows, 2010-2025, harmonized schema."""
    frames = []
    for yr in range(2010, 2025):
        df = pl.read_parquet(RAW / f"player_stats_{yr}.parquet").rename(
            {"recent_team": "team"}
        )
        frames.append(_std(df))
    frames.append(_std(pl.read_parquet(RAW / "stats_player_week_2025.parquet")))
    wk = pl.concat(frames).filter(pl.col("season_type") == "REG")
    return wk.drop("season_type")


def _std(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "season_type",
        pl.col("player_id").alias("gsis_id"),
        "position",
        "team",
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in STAT_COLS],
    )


def weekly_with_shares(wk: pl.DataFrame) -> pl.DataFrame:
    team = wk.group_by("season", "week", "team").agg(
        pl.col("targets").sum().alias("team_targets"),
        pl.col("carries").sum().alias("team_carries"),
        pl.col("receiving_air_yards").sum().alias("team_air"),
    )
    w = wk.join(team, on=["season", "week", "team"], how="left")
    safe = lambda num, den: pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)
    return w.with_columns(
        safe("targets", "team_targets").alias("ts"),
        safe("receiving_air_yards", "team_air").alias("ays"),
        safe("carries", "team_carries").alias("cs"),
    ).with_columns(
        (1.5 * pl.col("ts") + 0.7 * pl.col("ays")).alias("wopr_w"),
        (W_TGT * pl.col("targets") + W_AIR * pl.col("receiving_air_yards")
         + W_CAR * pl.col("carries")).alias("fp_exp"),
    )


def _slope(m: str) -> pl.Expr:
    wkc = pl.col("week")
    x = pl.col(m)
    cov = ((wkc - wkc.mean()) * (x - x.mean())).mean()
    var = ((wkc - wkc.mean()) ** 2).mean()
    return (cov / var).alias(f"opp_{m}_slope")


def _l6(m: str) -> pl.Expr:
    return (pl.col(m).sort_by("week").tail(6).mean() - pl.col(m).mean()).alias(
        f"opp_{m}_l6_delta"
    )


def _l4f4(m: str) -> pl.Expr:
    s = pl.col(m).sort_by("week")
    return (s.tail(4).mean() - s.head(4).mean()).alias(f"opp_{m}_l4f4")


VELOCITY_METRICS = {"ts": "ts", "wopr": "wopr_w", "cs": "cs", "tpg": "targets", "att": "attempts"}


def aggregate(w: pl.DataFrame) -> pl.DataFrame:
    """Per (t-1 season, gsis_id) feature row; season is then shifted to t."""
    vel_exprs = []
    for name, col in VELOCITY_METRICS.items():
        # aliasing uses the underlying col name; re-alias to the short metric name
        vel_exprs += [
            _slope(col).alias(f"opp_{name}_slope"),
            _l6(col).alias(f"opp_{name}_l6_delta"),
            _l4f4(col).alias(f"opp_{name}_l4f4"),
        ]
    ratio = lambda num, den, mn: pl.when(pl.col(den).sum() >= mn).then(
        pl.col(num).sum() / pl.col(den).sum()
    ).otherwise(None)

    agg = w.group_by("season", "gsis_id").agg(
        pl.len().alias("opp_games"),
        pl.col("position").mode().first().alias("position_wk"),
        # levels
        pl.col("ts").mean().alias("opp_target_share"),
        pl.col("ays").mean().alias("opp_air_yards_share"),
        pl.col("wopr_w").mean().alias("opp_wopr"),
        pl.col("cs").mean().alias("opp_carry_share"),
        (pl.col("targets").sum() / pl.len()).alias("opp_targets_pg"),
        (pl.col("carries").sum() / pl.len()).alias("opp_carries_pg"),
        # divergence (volume guards keep near-zero denominators from exploding)
        ratio("receiving_yards", "receiving_air_yards", MIN_AIR).alias("opp_racr"),
        ratio("receiving_yards", "targets", MIN_TGT).alias("opp_ypt"),
        pl.when((pl.col("targets").sum() + pl.col("carries").sum()) >= MIN_OPP)
        .then((pl.col("receiving_tds").sum() + pl.col("rushing_tds").sum())
              / (pl.col("targets").sum() + pl.col("carries").sum()))
        .otherwise(None).alias("opp_td_per_opp"),
        ratio("receiving_epa", "targets", MIN_TGT).alias("opp_epa_per_target"),
        ratio("rushing_epa", "carries", MIN_CAR).alias("opp_epa_per_carry"),
        pl.col("fantasy_points_ppr").mean().alias("opp_ppg"),
        pl.col("fp_exp").mean().alias("opp_fp_exp_pg"),
        # stability
        pl.col("ts").std().alias("opp_ts_std"),
        (pl.col("fantasy_points_ppr") >= 20.0).mean().alias("opp_boom_rate"),
        # QB analogues
        (pl.col("attempts").sum() / pl.len()).alias("opp_attempts_pg"),
        (pl.col("passing_air_yards").sum() / pl.len()).alias("opp_pass_air_pg"),
        ratio("passing_epa", "attempts", MIN_ATT).alias("opp_epa_per_att"),
        # velocity (masked below when games < MIN_GAMES_VELOCITY)
        *vel_exprs,
        # for position baselines
        pl.col("targets").sum().alias("_tgt_sum"),
        pl.col("carries").sum().alias("_car_sum"),
        pl.col("receiving_yards").sum().alias("_recyd_sum"),
        (pl.col("receiving_tds").sum() + pl.col("rushing_tds").sum()).alias("_td_sum"),
    )

    agg = agg.with_columns(
        (pl.col("opp_ppg") - pl.col("opp_fp_exp_pg")).alias("opp_fp_oe_pg"),
        (pl.col("opp_games") >= MIN_GAMES_VELOCITY).cast(pl.Int8).alias("opp_has_velocity"),
        ((pl.col("opp_games") > 0) & (pl.col("opp_games") < MIN_GAMES_VELOCITY))
        .cast(pl.Int8).alias("opp_short_season"),
    )
    vel_cols = [f"opp_{n}_{s}" for n in VELOCITY_METRICS for s in ("slope", "l6_delta", "l4f4")]
    agg = agg.with_columns(
        [pl.when(pl.col("opp_has_velocity") == 1).then(pl.col(c)).otherwise(None).alias(c)
         for c in vel_cols]
    )

    # position aggregate baselines within the same t-1 season (all weekly players,
    # volume-weighted aggregate ratio -> robust to small samples)
    pos = agg.group_by("season", "position_wk").agg(
        (pl.col("_recyd_sum").sum() / pl.col("_tgt_sum").sum()).alias("pos_ypt"),
        (pl.col("_td_sum").sum() / (pl.col("_tgt_sum").sum() + pl.col("_car_sum").sum()))
        .alias("pos_td_per_opp"),
    )
    agg = agg.join(pos, on=["season", "position_wk"], how="left").with_columns(
        (pl.col("opp_ypt") - pl.col("pos_ypt")).alias("opp_ypt_vs_pos"),
        (pl.col("opp_td_per_opp") - pl.col("pos_td_per_opp")).alias("opp_td_per_opp_vs_pos"),
    ).drop("_tgt_sum", "_car_sum", "_recyd_sum", "_td_sum", "pos_ypt", "pos_td_per_opp",
           "position_wk")
    return agg


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
    wk = load_weekly()
    w = weekly_with_shares(wk)
    feats = aggregate(w).with_columns((pl.col("season") + 1).alias("season"))
    pool = build_pool()
    out = pool.join(feats, on=["season", "gsis_id"], how="left").with_columns(
        pl.col("opp_games").is_not_null().cast(pl.Int8).alias("has_prior_weekly"),
        pl.col("opp_short_season").fill_null(0),
        pl.col("opp_has_velocity").fill_null(0),
    )
    assert out.group_by("season", "gsis_id").len().filter(pl.col("len") > 1).height == 0
    assert out["season"].min() >= 2011 and out["season"].max() <= 2026
    return out.sort("season", "gsis_id")


def coverage_report(out: pl.DataFrame) -> None:
    feats = [c for c in out.columns if c.startswith("opp_") or c == "has_prior_weekly"]
    for label, df in [
        ("2015-2025", out.filter(pl.col("season").is_between(2015, 2025))),
        ("2026", out.filter(pl.col("season") == 2026)),
        ("2026 non-rookie", out.filter((pl.col("season") == 2026) & (pl.col("has_prior_weekly") == 1))),
    ]:
        print(f"\n== coverage {label} (n={df.height}) ==")
        for c in feats:
            nn = 1.0 - df[c].null_count() / max(df.height, 1)
            mean = df[c].cast(pl.Float64).mean()
            print(f"  {c:26s} non-null {nn:6.1%}  mean {mean if mean is not None else float('nan'):9.4f}")


def sanity(out: pl.DataFrame) -> None:
    cases = [
        (2024, "Puka Nacua", ["opp_games", "opp_target_share", "opp_wopr", "opp_targets_pg",
                              "opp_ppg", "opp_fp_oe_pg", "opp_ts_slope", "opp_ts_l6_delta"]),
        (2022, "Amon-Ra", ["opp_games", "opp_target_share", "opp_ts_slope",
                           "opp_ts_l6_delta", "opp_ts_l4f4", "opp_boom_rate"]),
        (2024, "Marvin Harrison", ["has_prior_weekly", "opp_games", "opp_target_share",
                                   "opp_ts_slope"]),
        (2026, "Puka Nacua", ["has_prior_weekly", "opp_games", "opp_target_share",
                              "opp_wopr", "opp_ts_slope", "opp_ts_l6_delta"]),
    ]
    for season, sub, cols in cases:
        row = out.filter((pl.col("season") == season) & pl.col("name").str.contains(sub))
        print(f"\n{season} {sub}:")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(row.select(["name", "position"] + cols))


def main() -> None:
    out = build()
    out.write_parquet(PROC / "opportunity_features.parquet")
    print(f"wrote {out.height} rows, seasons {out['season'].min()}-{out['season'].max()}, "
          f"{len([c for c in out.columns if c.startswith('opp_')])} opp_ features")
    coverage_report(out)
    sanity(out)


if __name__ == "__main__":
    main()
