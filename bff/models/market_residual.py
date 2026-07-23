"""market_residual: start from the market, correct its known biases.

Score(t, player) = market-implied expectation of log1p(season PPR pts)
                 + w * ridge-predicted residual.

Market = preseason ADP rank, blended 50/50 (in log-rank space) with preseason
ECR rank when ECR exists for that season (2021+). Implied expectation is a
per-position quadratic in log(market rank), fit walk-forward on seasons < t.
Residual model is a small ridge on: age curve, prior-year ppg vs ADP-implied
ppg mismatch, games missed last year, rookie flag + pedigree, prior-year TD
points share (TD-regression proxy), team change flag.

Leakage: for target season t we use only seasons < t stats, season t's own
ADP/ECR, and static attributes (birthdate, draft pedigree).

Run:  uv run python -m bff.models.market_residual
Eval: uv run python -m bff.backtest data/processed/preds_market_residual.parquet --name market_residual
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW_STATS = ROOT / "data" / "raw" / "stats"
CROSSWALK = ROOT / "data" / "raw" / "db_playerids.csv"
OUT = PROC / "preds_market_residual.parquet"

POSITIONS = ("QB", "RB", "WR", "TE")
EVAL_SEASONS = range(2015, 2026)
FIRST_TARGET = 2011  # first season with a prior-year of stats

FEATURES = [
    "age_c", "age_c2", "age_rb", "age_qb",
    "ppg_mismatch", "games_missed", "rookie", "rookie_pedigree",
    "td_share_c", "team_change", "has_prior", "draft_ovr_log",
]


def load_td_shares() -> pl.DataFrame:
    """Prev-season TD points share per (season, gsis_id) from weekly data."""
    frames = []
    for yr in range(2010, 2025):
        df = pl.read_parquet(RAW_STATS / f"player_stats_{yr}.parquet").filter(
            pl.col("season_type") == "REG"
        )
        frames.append(
            df.group_by("player_id", "season").agg(
                pl.col("passing_tds").sum().alias("pass_td"),
                pl.col("rushing_tds").sum().alias("rush_td"),
                pl.col("receiving_tds").sum().alias("rec_td"),
                pl.col("fantasy_points_ppr").sum().alias("pts"),
            )
        )
    # 2025 exists only as a season-level aggregate (REG only) with the same
    # TD/pts columns; used solely as prior-year info for season-2026 scoring.
    s25 = RAW_STATS / "stats_player_reg_2025.parquet"
    if s25.exists():
        frames.append(
            pl.read_parquet(s25).select(
                "player_id", "season",
                pl.col("passing_tds").alias("pass_td"),
                pl.col("rushing_tds").alias("rush_td"),
                pl.col("receiving_tds").alias("rec_td"),
                pl.col("fantasy_points_ppr").alias("pts"),
            ).with_columns(pl.col("season").cast(pl.Int32))
        )
    frames = [f.with_columns(pl.col("season").cast(pl.Int32)) for f in frames]
    td = pl.concat(frames).with_columns(
        (
            (4.0 * pl.col("pass_td") + 6.0 * (pl.col("rush_td") + pl.col("rec_td")))
            / pl.col("pts").clip(lower_bound=1.0)
        ).alias("td_share")
    )
    return td.select(
        pl.col("player_id").alias("gsis_id"), "season",
        pl.col("td_share").clip(0.0, 1.0),
    )


def build_dataset() -> pl.DataFrame:
    adp = pl.read_parquet(PROC / "adp.parquet").filter(
        pl.col("gsis_id").is_not_null() & pl.col("position").is_in(POSITIONS)
    )
    ecr = pl.read_parquet(PROC / "ecr.parquet").filter(
        pl.col("gsis_id").is_not_null()
    ).select("season", "gsis_id", pl.col("ecr_rank").cast(pl.Int64))
    ecr = ecr.sort("ecr_rank").unique(subset=["season", "gsis_id"], keep="first")
    actuals = pl.read_parquet(PROC / "actuals.parquet").select(
        "gsis_id", pl.col("season").cast(pl.Int64), "team", "games",
        "pts_ppr", "ppg_ppr",
    )
    td = load_td_shares().with_columns(pl.col("season").cast(pl.Int64))
    xw = pl.read_csv(CROSSWALK, null_values=["NA"]).filter(
        pl.col("gsis_id").is_not_null()
    ).select("gsis_id", "birthdate", "draft_year", "draft_ovr").unique(
        subset=["gsis_id"], keep="first"
    )

    df = adp.select(
        "season", "gsis_id", "position", pl.col("team").alias("adp_team"),
        "adp", "adp_rank",
    )
    # blended market rank: mean of log(adp_rank) and log(ecr_rank) when ECR exists
    df = df.join(ecr.with_columns(pl.col("season").cast(pl.Int64)),
                 on=["season", "gsis_id"], how="left")
    df = df.with_columns(
        pl.when(pl.col("ecr_rank").is_not_null())
        .then(0.5 * pl.col("adp_rank").log() + 0.5 * pl.col("ecr_rank").cast(pl.Float64).log())
        .otherwise(pl.col("adp_rank").cast(pl.Float64).log())
        .alias("mkt_val")
    ).with_columns(
        pl.col("mkt_val").rank(method="ordinal").over("season").alias("mkt_rank")
    ).with_columns(pl.col("mkt_rank").cast(pl.Float64).log().alias("log_rank"))

    # current-season outcome (target; used only for seasons < eval season)
    df = df.join(
        actuals.select("gsis_id", "season", pl.col("pts_ppr").alias("tgt_pts")),
        on=["season", "gsis_id"], how="left",
    ).with_columns(pl.col("tgt_pts").fill_null(0.0))

    # prior-season stats
    prev = actuals.select(
        "gsis_id", (pl.col("season") + 1).alias("season"),
        pl.col("team").alias("prev_team"), pl.col("games").alias("prev_games"),
        pl.col("ppg_ppr").alias("prev_ppg"),
    )
    df = df.join(prev, on=["season", "gsis_id"], how="left")
    df = df.join(
        td.select("gsis_id", (pl.col("season") + 1).alias("season"),
                  pl.col("td_share").alias("prev_td_share")),
        on=["season", "gsis_id"], how="left",
    )

    # static attributes
    df = df.join(xw, on="gsis_id", how="left")
    df = df.with_columns(
        (
            (pl.date(pl.col("season"), 9, 1) - pl.col("birthdate").str.to_date(strict=False))
            .dt.total_days() / 365.25
        ).alias("age")
    )

    sched = pl.when(pl.col("season") >= 2021).then(17).otherwise(16)
    df = df.with_columns(
        pl.col("age").fill_null(pl.col("age").median().over("position")),
        pl.col("prev_games").fill_null(0).cast(pl.Float64),
        pl.col("prev_ppg").fill_null(0.0),
        pl.col("prev_td_share").fill_null(0.0),
        (pl.col("draft_year") == pl.col("season")).fill_null(False).alias("is_rk_draft"),
        pl.col("prev_games").is_not_null().alias("_dummy"),
    ).with_columns(
        pl.col("age").fill_null(26.0),
        (pl.col("prev_ppg") > 0).cast(pl.Float64).alias("has_prior"),
    ).with_columns(
        (pl.col("is_rk_draft") | (pl.col("has_prior") == 0.0))
        .cast(pl.Float64).alias("rookie"),
    ).with_columns(
        (pl.col("age") - 26.0).alias("age_c"),
        ((pl.col("age") - 26.0) ** 2).alias("age_c2"),
        ((pl.col("age") - 26.0) * (pl.col("position") == "RB").cast(pl.Float64)).alias("age_rb"),
        ((pl.col("age") - 26.0) * (pl.col("position") == "QB").cast(pl.Float64)).alias("age_qb"),
        (sched - pl.col("prev_games")).clip(0, 17)
        .mul(pl.col("has_prior")).alias("games_missed"),
        pl.col("draft_ovr").fill_null(270).cast(pl.Float64).log().alias("draft_ovr_log"),
        (pl.col("rookie") * (270.0 - pl.col("draft_ovr").fill_null(270).cast(pl.Float64)).clip(0, 270) / 270.0)
        .alias("rookie_pedigree"),
        (
            (pl.col("adp_team") != pl.col("prev_team")).fill_null(False)
            & (pl.col("has_prior") == 1.0)
        ).cast(pl.Float64).alias("team_change"),
        (pl.col("prev_td_share") - pl.col("prev_td_share").mean().over("position", "season"))
        .alias("td_share_c"),
        pl.col("tgt_pts").log1p().alias("log_pts"),
    )
    return df


def implied_expectation(train: pl.DataFrame, score_df: pl.DataFrame) -> np.ndarray:
    """Per-position OLS of log1p(pts) on [1, log_rank, log_rank^2] from train,
    applied to score_df."""
    out = np.zeros(score_df.height)
    for pos in POSITIONS:
        tr = train.filter(pl.col("position") == pos)
        X = np.column_stack([
            np.ones(tr.height),
            tr["log_rank"].to_numpy(),
            tr["log_rank"].to_numpy() ** 2,
        ])
        beta, *_ = np.linalg.lstsq(X, tr["log_pts"].to_numpy(), rcond=None)
        mask = (score_df["position"] == pos).to_numpy()
        lr = score_df["log_rank"].to_numpy()[mask]
        # clamp to the decreasing branch of the quadratic so implied value is
        # monotone non-increasing in market rank within position
        if abs(beta[2]) > 1e-12:
            vertex = -beta[1] / (2.0 * beta[2])
            lr = np.maximum(lr, vertex) if beta[2] < 0 else np.minimum(lr, vertex)
        out[mask] = beta[0] + beta[1] * lr + beta[2] * lr**2
    return out


def fit_predict(df: pl.DataFrame, eval_season: int, ridge_alpha: float,
                shrink: float) -> pl.DataFrame:
    train = df.filter(
        (pl.col("season") >= FIRST_TARGET) & (pl.col("season") < eval_season)
    )
    test = df.filter(pl.col("season") == eval_season)
    if test.height == 0:
        return pl.DataFrame(schema={"season": pl.Int64, "gsis_id": pl.Utf8,
                                    "score": pl.Float64})

    # implied expectation on both sets (fit on train only)
    tr_implied = implied_expectation(train, train)
    te_implied = implied_expectation(train, test)
    resid = train["log_pts"].to_numpy() - tr_implied

    # ppg mismatch needs implied ppg -> add as feature computed here
    sched_tr = np.where(train["season"].to_numpy() >= 2021, 17.0, 16.0)
    sched_te = np.where(test["season"].to_numpy() >= 2021, 17.0, 16.0)
    tr_mismatch = train["prev_ppg"].to_numpy() - np.expm1(tr_implied) / sched_tr
    te_mismatch = test["prev_ppg"].to_numpy() - np.expm1(te_implied) / sched_te
    tr_mismatch *= train["has_prior"].to_numpy()
    te_mismatch *= test["has_prior"].to_numpy()

    feats = [f for f in FEATURES if f != "ppg_mismatch"]
    Xtr = np.column_stack([train.select(feats).to_numpy(), tr_mismatch])
    Xte = np.column_stack([test.select(feats).to_numpy(), te_mismatch])

    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=ridge_alpha)
    model.fit(scaler.transform(Xtr), np.clip(resid, -4.0, 4.0))
    pred_resid = model.predict(scaler.transform(Xte))

    score = te_implied + shrink * pred_resid
    return test.select("season", "gsis_id").with_columns(
        pl.Series("score", score)
    )


def tune(df: pl.DataFrame) -> tuple[float, float]:
    """Pick (ridge_alpha, shrink) on pre-eval walk-forward seasons 2012-2014."""
    from scipy.stats import spearmanr
    actuals = pl.read_parquet(PROC / "actuals.parquet").select(
        "gsis_id", pl.col("season").cast(pl.Int64), "pts_ppr"
    )
    best, best_val = (10.0, 0.5), -np.inf
    for alpha in (3.0, 10.0, 30.0, 100.0):
        for w in (0.3, 0.5, 0.7, 1.0):
            vals = []
            for s in (2012, 2013, 2014):
                preds = fit_predict(df, s, alpha, w)
                pool = df.filter((pl.col("season") == s) & (pl.col("adp_rank") <= 150))
                j = pool.select("gsis_id").join(preds, on="gsis_id").join(
                    actuals.filter(pl.col("season") == s), on="gsis_id", how="left"
                ).with_columns(pl.col("pts_ppr").fill_null(0.0))
                vals.append(spearmanr(j["score"].to_numpy(), j["pts_ppr"].to_numpy()).statistic)
            m = float(np.mean(vals))
            if m > best_val:
                best_val, best = m, (alpha, w)
    print(f"tuned: alpha={best[0]}, shrink={best[1]}, pre2015 spearman={best_val:.4f}")
    return best


def main() -> None:
    df = build_dataset()
    alpha, shrink = tune(df)
    frames = []
    for t in EVAL_SEASONS:
        frames.append(fit_predict(df, t, alpha, shrink))
    preds = pl.concat(frames)
    preds.write_parquet(OUT)
    print(f"wrote {OUT} ({preds.height} rows, seasons "
          f"{preds['season'].min()}-{preds['season'].max()})")


if __name__ == "__main__":
    main()
