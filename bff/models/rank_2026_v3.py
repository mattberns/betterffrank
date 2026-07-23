"""2026 rankings + steals from the v3 winner (opp_residual).

Pipeline:
1. opp_residual trained walk-forward with eval_season=2026 (target seasons
   2011-2025), scored on the 2026 FFC ADP pool (185 matched players). No 2026
   ECR exists, so the frozen anchor's documented missing-ECR path applies
   (market rank = ADP rank alone). Hyperparameters come from the frozen
   2012-2014 tune (alpha=30, opp_scale=1.0, shrink=0.3, subset=curated).
2. Scores -> predicted VORP via the leakage-safe historical curve
   (bff.vorp, seasons <= 2025 only). Consistency-checked against
   data/processed/preds_opp_residual_2026.parquet.
3. Steals: our_rank beats adp_rank by >= 24 within adp_rank <= 120. Reasons
   read off the model: top positive ridge contributions (v1 / context / opp
   drivers tagged) plus an anchor-only VORP decomposition that separates the
   positional-scarcity repricing from the feature residual.

Usage: uv run python -m bff.models.rank_2026_v3
Output: reports/rankings_2026.csv, reports/steals_2026.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bff.backtest import POSITIONS, REPL_RANKS
from bff.models.market_residual import FIRST_TARGET, PROC, implied_expectation
from bff.models.context_residual import ALL_FEATURES as V2_FEATURES
from bff.models.opp_residual import OPP_SUBSETS, OPP_VELOCITY, build_dataset_v3, tune
from bff.vorp import build_curve, curve_at, pool_actual_ranks

ROOT = Path(__file__).resolve().parents[2]
OUT_RANK = ROOT / "reports" / "rankings_2026.csv"
OUT_STEALS = ROOT / "reports" / "steals_2026.csv"

STEAL_MIN_DELTA = 24
STEAL_MAX_ADP_RANK = 120


def fit_predict_contrib(df: pl.DataFrame, eval_season: int, ridge_alpha: float,
                        opp_scale: float, shrink: float, opp_feats: list[str],
                        ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """opp_residual.fit_predict, plus per-player ridge feature contributions
    (shrink * coef * scaled z-feature, in log-points units) for the test season."""
    train = df.filter(
        (pl.col("season") >= FIRST_TARGET) & (pl.col("season") < eval_season)
    )
    test = df.filter(pl.col("season") == eval_season)

    tr_implied = implied_expectation(train, train)
    te_implied = implied_expectation(train, test)
    resid = train["log_pts"].to_numpy() - tr_implied

    sched_tr = np.where(train["season"].to_numpy() >= 2021, 17.0, 16.0)
    sched_te = np.where(test["season"].to_numpy() >= 2021, 17.0, 16.0)
    tr_mismatch = train["prev_ppg"].to_numpy() - np.expm1(tr_implied) / sched_tr
    te_mismatch = test["prev_ppg"].to_numpy() - np.expm1(te_implied) / sched_te
    tr_mismatch *= train["has_prior"].to_numpy()
    te_mismatch *= test["has_prior"].to_numpy()

    feats = [f for f in V2_FEATURES if f != "ppg_mismatch"] + opp_feats
    Xtr = np.column_stack([train.select(feats).to_numpy(), tr_mismatch])
    Xte = np.column_stack([test.select(feats).to_numpy(), te_mismatch])
    feat_order = feats + ["ppg_mismatch"]

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    opp_mask = np.array([f in opp_feats for f in feat_order])
    Xtr[:, opp_mask] *= opp_scale
    Xte[:, opp_mask] *= opp_scale

    model = Ridge(alpha=ridge_alpha)
    model.fit(Xtr, np.clip(resid, -4.0, 4.0))
    pred_resid = model.predict(Xte)

    score = te_implied + shrink * pred_resid
    preds = test.select("season", "gsis_id").with_columns(pl.Series("score", score))

    contrib = shrink * Xte * model.coef_[None, :]
    cdf = test.select("gsis_id").with_columns(
        [pl.Series(f"c_{f}", contrib[:, i]) for i, f in enumerate(feat_order)]
        + [pl.Series("raw_ppg_mismatch", te_mismatch)]
    )
    return preds, cdf


def rank_by_vorp(scored: pl.DataFrame, curves: dict, score_col: str,
                 rank_name: str) -> pl.DataFrame:
    """Per-position order by score_col -> curve VORP -> overall rank."""
    out = []
    for pos in POSITIONS:
        sub = scored.filter(pl.col("position") == pos).sort(
            [score_col, "adp_rank"], descending=[True, False]
        ).with_columns(pl.arange(1, pl.len() + 1).alias("pos_rank"))
        c = curves[pos]
        repl_pts = curve_at(c, REPL_RANKS[pos])
        vorp = np.array([curve_at(c, r) - repl_pts for r in range(1, sub.height + 1)])
        out.append(sub.with_columns(pl.Series("pred_vorp", vorp)))
    return (
        pl.concat(out)
        .sort(["pred_vorp", "adp_rank"], descending=[True, False])
        .with_columns(pl.arange(1, pl.len() + 1).alias(rank_name))
    )


def reason_string(row: dict) -> str:
    """Plain-language 'why' from the biggest positive ridge contributions,
    checked against raw feature values so the phrase matches the player."""
    parts: list[tuple[float, str]] = []

    # --- v1 drivers ---
    age_c = sum(row[f"c_{f}"] for f in ("age_c", "age_c2", "age_rb", "age_qb"))
    if age_c > 0.01:
        side = "young for the position" if row["age_c"] < 0 else "age profile"
        parts.append((age_c, f"favorable age curve ({side})"))

    c = row["c_ppg_mismatch"]
    if c > 0.01 and row["raw_ppg_mismatch"] > 0:
        parts.append((c, "outproduced his market rank last season "
                         f"(+{row['raw_ppg_mismatch']:.1f} PPR pts/game vs market-implied)"))

    c = row["c_games_missed"]
    if c > 0.01 and row["games_missed"] <= 2:
        parts.append((c, "played a near-full season last year (durability)"))

    c = row["c_rookie"] + row["c_rookie_pedigree"] + row.get("c_is_rookie", 0.0)
    if c > 0.01 and row["rookie"] > 0:
        parts.append((c, "rookie with early-draft pedigree"))

    c = row["c_td_share_c"]
    if c > 0.01 and row["td_share_c"] < 0:
        parts.append((c, "scoring was not TD-dependent last year (low regression risk)"))

    # --- context drivers ---
    c = row["c_arriving_vet_usage"]
    if c > 0.01 and row["arriving_vet_usage"] <= 0.02:
        parts.append((c, "no veteran arrivals competing for his role [context]"))

    c = row["c_draft_competition"]
    if c > 0.01 and row["draft_competition"] <= 0.0:
        parts.append((c, "no April draft capital spent at his position [context]"))

    c = row["c_qb_quality_delta"] + row["c_qb_delta_wrte"]
    if c > 0.01 and row["qb_quality_delta"] > 0:
        parts.append((c, "improved expected QB play [context]"))

    c = (row["c_vacated_target_share"] + row["c_vacated_carry_share"]
         + row["c_vacated_rec_fp_share"] + row["c_vacated_tgt_wrte"]
         + row["c_vacated_carry_rb"])
    if c > 0.01 and (row["vacated_target_share"] > 0.05
                     or row["vacated_carry_share"] > 0.05):
        parts.append((c, "vacated volume from departures [context]"))

    c = (row["c_team_fp_prior"] + row["c_team_fp_prior_z"]
         + row["c_team_pass_fp_share_prior"] + row["c_team_pass_rate_prior"])
    if c > 0.01:
        parts.append((c, "team offensive environment [context]"))

    # --- opportunity drivers (curated subset) ---
    c = row["c_opp_target_share"]
    if c > 0.008 and row["opp_target_share"] >= 0.18:
        parts.append((c, f"{row['opp_target_share']*100:.0f}% target share "
                         "last season [opp]"))

    c_vel = (row["c_opp_ts_slope"] + row["c_opp_ts_l4f4"])
    if c_vel > 0.008 and row["opp_ts_l4f4"] > 0.01:
        parts.append((c_vel, "target share "
                             f"{row['opp_ts_l4f4']*100:+.0f} pts over final 4 games "
                             "vs first 4 [opp velocity]"))
    elif c_vel > 0.008 and row["opp_ts_slope"] > 0:
        parts.append((c_vel, "target share trending up through last season "
                             "[opp velocity]"))

    c = row["c_opp_cs_l6_delta"]
    if c > 0.008 and row["opp_cs_l6_delta"] > 0.01:
        parts.append((c, "carry share rising over the final 6 games [opp velocity]"))

    # fp-over-expected is structural for QBs (opportunity model excludes pass
    # attempts), so only cite it as a player-specific reason for RB/WR/TE
    c = row["c_opp_fp_oe_pg"]
    if c > 0.008 and row["opp_fp_oe_pg"] > 0 and row["position"] != "QB":
        parts.append((c, "produced over his opportunity volume "
                         f"(+{row['opp_fp_oe_pg']:.1f} PPR pts/game) [opp]"))

    c = row["c_opp_td_per_opp_vs_pos"]
    if c > 0.008 and row["opp_td_per_opp_vs_pos"] > 0:
        parts.append((c, "TD-efficient on real opportunity last season [opp]"))
    elif c < -0.008 and row["opp_td_per_opp_vs_pos"] < 0:
        # negative contribution but flagged for transparency when volume is real
        pass

    c = row["c_opp_boom_rate"]
    if c > 0.008 and 0 < row["opp_boom_rate"] < 0.25:
        parts.append((c, "steady week-to-week, not spike-week dependent [opp]"))

    parts.sort(key=lambda t: -t[0])
    why = "; ".join(p for _, p in parts[:3])
    scarcity = (f"{row['position']}{row['pos_rank']} slot priced above ADP "
                f"(+{row['scarcity_gain']} ranks from positional scarcity, "
                f"{row['feature_gain']:+d} from features)")
    return f"{why}; {scarcity}" if why else scarcity


def main() -> None:
    df = build_dataset_v3()
    alpha, g, shrink, subset = tune(df)  # 2012-2014 only; deterministic
    opp_feats = OPP_SUBSETS[subset]
    preds, contrib = fit_predict_contrib(df, 2026, alpha, g, shrink, opp_feats)

    adp_all = pl.read_parquet(PROC / "adp.parquet")
    ecr_all = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_actual_ranks(adp_all, ecr_all, actuals)
    curves = build_curve(hist, 2026)
    assert all(pos in curves for pos in POSITIONS)

    adp = adp_all.filter(pl.col("season") == 2026)
    scored = adp.join(preds.select("gsis_id", "score"), on="gsis_id", how="inner")

    ranked = rank_by_vorp(scored, curves, "score", "our_rank").with_columns(
        (pl.col("adp_rank") - pl.col("our_rank")).alias("delta")
    )

    # consistency check vs the audited 2026 preds parquet
    shipped = pl.read_parquet(PROC / "preds_opp_residual_2026.parquet")
    chk = ranked.join(shipped.select("gsis_id", pl.col("score").alias("v")),
                      on="gsis_id", how="inner")
    max_diff = (chk["pred_vorp"] - chk["v"]).abs().max()
    assert max_diff < 1e-6, f"pred VORP mismatch vs shipped parquet: {max_diff}"

    # anchor-only ranking (2026 anchor = ADP alone) through the same curve,
    # to decompose delta into scarcity repricing vs feature residual
    anchor = rank_by_vorp(
        scored.with_columns((-pl.col("adp_rank").cast(pl.Float64)).alias("neg_adp")),
        curves, "neg_adp", "anchor_rank",
    ).select("gsis_id", "anchor_rank")
    ranked = ranked.join(anchor, on="gsis_id", how="left").with_columns(
        (pl.col("adp_rank") - pl.col("anchor_rank")).alias("scarcity_gain"),
        (pl.col("anchor_rank") - pl.col("our_rank")).alias("feature_gain"),
    )

    out = ranked.select(
        "our_rank", pl.col("name").alias("player"), "position", "team",
        "adp", "adp_rank", "delta", pl.col("pred_vorp").round(4).alias("score"),
    )
    out.write_csv(OUT_RANK)
    print(f"wrote {OUT_RANK} ({out.height} rows)")

    # sanity gates
    n_qb_top15 = out.head(15).filter(pl.col("position") == "QB").height
    first_qb = out.filter(pl.col("position") == "QB")["our_rank"].min()
    top3_pos = out.head(3)["position"].to_list()
    bad_pos = out.filter(~pl.col("position").is_in(POSITIONS)).height
    print(f"gates: QBs in top 15 = {n_qb_top15} (<=2), first QB #{first_qb}, "
          f"top3 = {top3_pos}, non-QB/RB/WR/TE rows = {bad_pos}")
    assert n_qb_top15 <= 2 and bad_pos == 0
    assert all(p in ("RB", "WR") for p in top3_pos), top3_pos

    # steals
    raw_cols = ["gsis_id", "age_c", "games_missed", "rookie", "td_share_c",
                "team_change", "arriving_vet_usage", "draft_competition",
                "qb_quality_delta", "vacated_target_share", "vacated_carry_share",
                "opp_target_share", "opp_ts_slope", "opp_ts_l4f4",
                "opp_cs_l6_delta", "opp_fp_oe_pg", "opp_td_per_opp_vs_pos",
                "opp_boom_rate"]
    feat_raw = df.filter(pl.col("season") == 2026).select(raw_cols)
    steals = (
        ranked.filter(
            (pl.col("adp_rank") <= STEAL_MAX_ADP_RANK)
            & (pl.col("delta") >= STEAL_MIN_DELTA)
        )
        .join(contrib, on="gsis_id", how="left")
        .join(feat_raw, on="gsis_id", how="left")
    )
    reasons = [reason_string(r) for r in steals.iter_rows(named=True)]
    steals = steals.with_columns(pl.Series("reason", reasons)).select(
        "our_rank", pl.col("name").alias("player"), "position", "team", "adp",
        "adp_rank", "delta", pl.col("pred_vorp").round(4).alias("score"), "reason",
    ).sort("our_rank")
    steals.write_csv(OUT_STEALS)
    print(f"wrote {OUT_STEALS} ({steals.height} rows)")

    with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=200):
        print(out.head(25))
        print(steals)


if __name__ == "__main__":
    main()
