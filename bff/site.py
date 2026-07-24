"""Regenerate the /docs GitHub Pages site from the shipped model.

Self-contained static site (no CDN, no web fonts, no libraries). Presentation
is a terse statistical report: serif body, monospace numerics, booktabs tables,
numbered sections, no hero/marketing. Interactivity (rankings filter/sort/
expand, per-player feature attribution, coefficient bars) is retained.

Every page is one .html file with an inline window.BFF_DATA blob (only the
slice it needs) and inline vanilla JS. The payload reuses the model's own
walk-forward 2026 fit (bff.model.fit_predict with return_contrib=True) plus its
reason_string, so every number reproduces from the model. It changes no model
output.

Run:
  uv run python -m bff.site              # render (assumes reports current)
  uv run python -m bff.site --refresh    # rerun model + backtests, then render
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from bff.backtest import POSITIONS, build_pool, eval_season
from bff.compare import sign_flip_test
from bff.model import (
    FEATURES,
    build_curve,
    build_dataset,
    fit_predict,
    pool_market_ranks,
    rank_by_vorp,
    reason_string,
    tune,
)
from bff.vona import MATRIX_ROUNDS, per_pick_table, turn_matrix

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SEASON = 2026

# ---------------------------------------------------------------------------
# 1. DATA PAYLOAD
# ---------------------------------------------------------------------------

# group: base | market | context | interaction | opportunity | trend | injury
#        | draft | contract
FEATURE_META = {
    "age_c":        ("base", "Age vs 26", "Player age centered at 26; the spine of the aging curve."),
    "age_c2":       ("base", "Age curvature", "Squared age term; lets value fall off faster at the extremes."),
    "age_rb":       ("base", "Age x RB", "Age effect specific to running backs, who decline earliest."),
    "age_qb":       ("base", "Age x QB", "Age effect specific to quarterbacks, who age most gracefully."),
    "ppg_mismatch": ("base", "PPG vs market-implied", "Last year's PPR points/game minus what the market rank implies. Positive = outscored draft cost."),
    "games_missed": ("base", "Games missed (prior yr)", "Games absent last season vs a full schedule."),
    "rookie_pedigree": ("base", "Rookie draft pedigree", "For rookies: how early they were taken in the NFL draft."),
    "td_share_c":   ("base", "TD-share (regression proxy)", "Share of last year's points from TDs, centered by position. High = more regression risk."),
    "team_change":  ("base", "Changed NFL team", "Returning veteran on a new team this season."),
    "has_prior":    ("base", "Has prior-season stats", "Player has prior NFL production to learn from."),
    "draft_ovr_log": ("base", "Draft slot (log)", "Log of overall NFL draft pick; a static pedigree prior."),
    "vs_adp":       ("market", "ECR vs ADP gap", "log(ECR rank) - log(ADP rank). Positive = the market drafts him earlier than the experts rank him."),
    "vacated_target_share": ("context", "Vacated target share", "Team targets left behind by departed teammates."),
    "vacated_carry_share":  ("context", "Vacated carry share", "Team carries left behind by departed teammates."),
    "arriving_vet_usage":   ("context", "Arriving vets' usage", "Prior-year usage of veterans who just joined and compete for his role."),
    "draft_competition":    ("context", "Rookie draft competition", "April draft capital the team spent at his position."),
    "qb_change":    ("context", "QB change", "Projected starting QB changed."),
    "qb_quality_delta": ("context", "QB quality delta", "Expected change in QB play vs last year."),
    "qb_rookie":    ("context", "Rookie QB", "Projected starter is a rookie."),
    "qb_expected_missing": ("context", "QB expected missing", "Expected starter-QB games lost (known preseason)."),
    "coach_change": ("context", "Coaching change", "Team changed head coach / play-caller."),
    "team_fp_prior": ("context", "Team fantasy pts (prior)", "Team fantasy offense last year."),
    "team_fp_prior_z": ("context", "Team fantasy pts (z)", "Team offense, standardized across the league."),
    "team_pass_fp_share_prior": ("context", "Team pass-offense share", "Share of team fantasy offense through the air."),
    "team_pass_rate_prior": ("context", "Team pass rate (prior)", "How pass-heavy the team was last season."),
    "returning_target_competition": ("context", "Returning target competition", "Returning teammates competing for targets."),
    "returning_carry_competition": ("context", "Returning carry competition", "Returning teammates competing for carries."),
    "depth_rank_adp": ("context", "Depth rank by ADP", "ADP rank among teammates at the same position."),
    "is_rookie":    ("context", "Is rookie", "The player's rookie season."),
    "qb_delta_wrte": ("interaction", "QB delta x WR/TE", "QB-quality change applied to pass-catchers only."),
    "vacated_tgt_wrte": ("interaction", "Vacated targets x WR/TE", "Vacated target share applied to pass-catchers only."),
    "vacated_carry_rb": ("interaction", "Vacated carries x RB", "Vacated carry share applied to running backs only."),
    "opp_target_share": ("opportunity", "Target share (last yr)", "Share of team targets last season."),
    "opp_air_yards_share": ("opportunity", "Air-yards share", "Share of team downfield volume aimed at him."),
    "opp_ts_slope": ("opportunity", "Target-share trend", "Within-season slope of target share."),
    "opp_ts_l4f4":  ("opportunity", "Target share: last 4 vs first 4", "Change in target share, first four to last four games."),
    "opp_cs_l6_delta": ("opportunity", "Carry-share trend (last 6)", "Change in carry share over the final six games."),
    "opp_fp_oe_pg": ("opportunity", "Points over expected/game", "Fantasy points above what opportunity volume predicted."),
    "opp_td_per_opp_vs_pos": ("opportunity", "TD rate vs position", "Touchdowns per opportunity vs positional baseline."),
    "opp_boom_rate": ("opportunity", "Boom-week rate", "Share of spike-week outliers; high = volatile."),
    # 2026-07-23 feature expansion
    "ppg_delta": ("trend", "PPG change (yr/yr)", "Change in PPR points/game vs the prior year."),
    "career_missed_rate": ("trend", "Career missed-game rate", "Share of career games missed; long-run durability."),
    "inj_weeks_listed_l2y": ("injury", "Weeks on injury report (2y)", "Weeks listed on the injury report over two seasons."),
    "inj_soft_tissue_l2y": ("injury", "Soft-tissue injuries (2y)", "Soft-tissue injuries (hamstring, groin, calf) over two seasons."),
    "inj_recurrence": ("injury", "Injury recurrence", "Same injury recurring; re-injury risk."),
    "draft_r1": ("draft", "First-round pick", "Taken in the first round of the NFL draft."),
    "draft_r23": ("draft", "Second/third-round pick", "Taken in round two or three of the NFL draft."),
    "rb_early_rookie": ("draft", "Early-drafted rookie RB", "Running back taken early as a rookie."),
    "apy_cap_pct": ("contract", "Contract size (% of cap)", "Average annual value as a share of the salary cap."),
    "contract_year": ("contract", "Contract year", "Playing in a contract (walk) year."),
    "rookie_deal_yr": ("contract", "On rookie contract", "Still on the initial rookie contract."),
}

GROUP_LABELS = {
    "base": "Base", "market": "Market signal", "context": "Preseason context",
    "interaction": "Position interactions", "opportunity": "Opportunity (prior-year usage)",
    "trend": "Recent trend", "injury": "Injury history", "draft": "Draft capital",
    "contract": "Contract status", "other": "Other",
}
GROUP_ORDER = ["base", "market", "context", "interaction", "opportunity",
               "trend", "injury", "draft", "contract", "other"]

# On/off indicators whose OFF-state ridge contribution is a standardization
# artifact, not a real driver -> only surface when the raw feature is ON.
INDICATOR_GATE = {
    "is_rookie": "is_rookie", "rookie_pedigree": "rookie", "has_prior": "has_prior",
    "team_change": "team_change", "qb_change": "qb_change", "qb_rookie": "qb_rookie",
    "coach_change": "coach_change", "qb_expected_missing": "qb_expected_missing",
    "draft_r1": "draft_r1", "draft_r23": "draft_r23", "rb_early_rookie": "rb_early_rookie",
    "contract_year": "contract_year", "rookie_deal_yr": "rookie_deal_yr",
    "inj_recurrence": "inj_recurrence",
}


def _meta(feat):
    m = FEATURE_META.get(feat)
    if m is not None:
        return m
    return ("other", feat.replace("_", " ").capitalize(), "")


def _verdict(delta, p_one, name):
    """One-word claim from the sign/size of the delta. A gap under ~0.005 mean
    Spearman is a tie (metric noise dwarfs it)."""
    if delta is None:
        return f"vs {name}"
    if delta > 0.005:
        return (f"beats {name}" if (p_one is not None and p_one < 0.05)
                else f"edges {name}")
    if delta < -0.005:
        return f"trails {name}"
    return f"ties {name}"


def _build_players(df, ranked, contrib):
    """Full board with reason strings and per-player feature attribution
    (each contribution carries the player's raw feature value)."""
    want = (set(FEATURES) | {"rookie"}) & set(df.columns)
    feat_raw = df.filter(pl.col("season") == SEASON).select(
        ["gsis_id"] + sorted(want - {"gsis_id"})
    )
    full = ranked.join(contrib, on="gsis_id", how="left").join(
        feat_raw, on="gsis_id", how="left"
    )
    c_cols = [c for c in full.columns if c.startswith("c_")]

    players = []
    for row in full.sort("our_rank").iter_rows(named=True):
        reason = reason_string(row)
        contribs = []
        for c in c_cols:
            feat = c[2:]
            if row[c] is None:
                continue
            gate = INDICATOR_GATE.get(feat)
            if gate is not None:
                gv = row.get(gate)
                if gv is None or gv <= 0:   # indicator OFF -> skip artifact
                    continue
            group, label, _blurb = _meta(feat)
            raw = row.get("raw_ppg_mismatch") if feat == "ppg_mismatch" else row.get(feat)
            contribs.append({"feat": feat, "label": label, "group": group,
                             "value": round(row[c], 4),
                             "raw": round(float(raw), 3) if raw is not None else None})
        contribs.sort(key=lambda d: -abs(d["value"]))
        has_ecr = row["ecr_ord"] is not None and row["delta"] is not None
        players.append({
            "rank": row["our_rank"], "name": row["name"], "pos": row["position"],
            "team": row["team"], "ecr": row["ecr_ord"], "pos_rank": row["pos_rank"],
            "delta": row["delta"], "vorp": round(float(row["pred_vorp"]), 1),
            "scarcity_gain": row["scarcity_gain"], "feature_gain": row["feature_gain"],
            "is_steal": bool(has_ecr and row["ecr_ord"] <= 120 and row["delta"] >= 24),
            "reason": reason.replace("priced above ADP", "priced above ECR").replace("+-", "-"),
            "contribs": contribs[:8],
        })
    return players


def _read_scores(name):
    return pl.read_csv(REPORTS / f"scores_{name}.csv",
                       schema_overrides={"season": pl.Utf8})


def _val(dfin, season, metric):
    r = dfin.filter((pl.col("season") == str(season)) & (pl.col("metric") == metric))
    return round(float(r["value"][0]), 4) if r.height else None


def build_payload() -> dict:
    df = build_dataset()
    alpha, shrink, _ = tune(df)   # tune() returns (alpha, shrink, tuning_score)

    preds, contrib, _, _ = fit_predict(df, SEASON, alpha, shrink, return_contrib=True)

    adp_all = pl.read_parquet(PROC / "adp.parquet")
    ecr_all = pl.read_parquet(PROC / "ecr.parquet")
    actuals = pl.read_parquet(PROC / "actuals.parquet")
    hist = pool_market_ranks(adp_all, ecr_all, actuals)
    curves = build_curve(hist, SEASON)

    adp = adp_all.filter(pl.col("season") == SEASON)
    ecr_sel = (
        ecr_all.filter(pl.col("season") == SEASON)
        .select("gsis_id", pl.col("ecr_rank").cast(pl.Int64))
        .sort("ecr_rank").unique(subset=["gsis_id"], keep="first")
    )
    scored = (
        adp.join(preds.select("gsis_id", "score"), on="gsis_id", how="inner")
        .join(ecr_sel, on="gsis_id", how="left")
        .with_columns(
            pl.when(pl.col("ecr_rank").is_not_null())
            .then(pl.col("ecr_rank").rank(method="ordinal"))
            .otherwise(None).cast(pl.Int64).alias("ecr_ord")
        )
        .with_columns(pl.coalesce([pl.col("ecr_ord"), pl.col("adp_rank")]).alias("mkt_rank"))
    )
    ranked = rank_by_vorp(scored, curves, "score", "our_rank").with_columns(
        (pl.col("ecr_ord") - pl.col("our_rank")).alias("delta")
    )
    anchor = rank_by_vorp(
        scored.with_columns((-pl.col("mkt_rank").cast(pl.Float64)).alias("neg_mkt")),
        curves, "neg_mkt", "anchor_rank",
    ).select("gsis_id", "anchor_rank")
    ranked = ranked.join(anchor, on="gsis_id", how="left").with_columns(
        (pl.col("mkt_rank") - pl.col("anchor_rank")).alias("scarcity_gain"),
        (pl.col("anchor_rank") - pl.col("our_rank")).alias("feature_gain"),
    )

    players = _build_players(df, ranked, contrib)

    # VONA draft overlay — computed from THIS payload's board (not the artifact)
    # so the site's Draft page can never drift from its own Rankings page.
    vona_board = ranked.select(
        "gsis_id", "name", "position", "adp_rank",
        pl.col("pred_vorp").alias("vorp"),
    )
    vona_matrix = turn_matrix(vona_board).to_dicts()
    vona_table = per_pick_table(vona_board).to_dicts()

    _, _, coef_model, coef_order = fit_predict(df, SEASON, alpha, 1.0, return_contrib=True)
    coefs = dict(zip(coef_order, coef_model.coef_))
    features = []
    for f in coef_order:
        group, label, blurb = _meta(f)
        features.append({"name": f, "label": label, "group": group,
                         "group_label": GROUP_LABELS.get(group, group.title()),
                         "blurb": blurb, "coef": round(float(coefs[f]), 4)})
    features.sort(key=lambda d: -abs(d["coef"]))

    sm, sa, se = _read_scores("model"), _read_scores("adp"), _read_scores("ecr")
    yrs = sorted(int(x) for x in
                 sm.filter(pl.col("metric") == "spearman_vorp")["season"].to_list()
                 if x.isdigit())
    seasons = [{"season": yr, "model": _val(sm, yr, "spearman_vorp"),
                "ecr": _val(se, yr, "spearman_vorp")} for yr in yrs]

    adp_pq = adp_all
    ecr_pq = ecr_all
    act_pq = actuals
    preds_model = pl.read_parquet(PROC / "preds_model.parquet")

    def compare_to(base_name):
        base = pl.read_parquet(PROC / f"preds_{base_name}.parquet")
        avail = sorted(set(preds_model["season"].unique().to_list())
                       & set(base["season"].unique().to_list())
                       & set(act_pq["season"].unique().to_list()))
        ma, mb = [], []
        for s in avail:
            pool, _ = build_pool(s, adp_pq, ecr_pq)
            ma.append(eval_season(pool, preds_model, act_pq, s)["spearman_vorp"])
            mb.append(eval_season(pool, base, act_pq, s)["spearman_vorp"])
        deltas = [a - b for a, b in zip(ma, mb)]
        p_one, _ = sign_flip_test(deltas) if deltas else (None, None)
        model_mean = round(sum(ma) / len(ma), 4) if ma else None
        base_mean = round(sum(mb) / len(mb), 4) if mb else None
        delta = round(model_mean - base_mean, 4) if ma else None
        return {"model": model_mean, "baseline": base_mean, "delta": delta,
                "n_seasons": len(avail),
                "span": f"{avail[0]}-{avail[-1]}" if avail else "",
                "wins": sum(1 for d in deltas if d > 0),
                "p_one": round(p_one, 4) if p_one is not None else None,
                "power_floor": round(1.0 / 2 ** len(avail), 4) if avail else None,
                "good": bool(delta is not None and delta >= -0.005),
                "verdict": _verdict(delta, p_one, base_name.upper())}

    headline = {"vs_ecr": compare_to("ecr"), "vs_adp": compare_to("adp")}
    ve = headline["vs_ecr"]

    floor_note = ("so p-values are descriptive only"
                  if (ve["power_floor"] or 1.0) > 0.05
                  else "so significance at 0.05 is reachable")
    limitations = [
        f"Test set is {ve['n_seasons']} seasons ({ve['span']}); the sign-flip "
        f"one-sided power floor is {ve['power_floor']:g}, {floor_note}.",
        "Hyperparameters are tuned on an earlier walk-forward window; fit to the "
        f"current anchor configuration is untested by construction (alpha={int(alpha)}).",
        "Some rookies run on partly or fully null-filled opportunity features.",
        "Replacement levels fixed for 12-team 1-QB PPR; other formats shift the QB/TE placements.",
    ]

    return {
        "meta": {"season": SEASON, "format": "12-team PPR redraft",
                 "alpha": alpha, "shrink": shrink, "n_players": len(players),
                 "n_features": len(features)},
        "headline": headline,
        "secondary": {"raw_spearman": _val(sm, "MEAN", "spearman"),
                      "ndcg100_vorp": _val(sm, "MEAN", "ndcg100_vorp"),
                      "top24_hit_vorp": _val(sm, "MEAN", "top24_hit_vorp"),
                      "top50_hit": _val(sm, "MEAN", "top50_hit")},
        "seasons": seasons,
        "features": features,
        "players": players,
        "vona": {"matrix": vona_matrix, "table": vona_table, "rounds": MATRIX_ROUNDS},
        "method": [
            {"step": "Pool", "text": "Each season's ADP top-150 at QB/RB/WR/TE, GSIS-matched (~145-150 players)."},
            {"step": "Start from the experts", "text": "Every player's anchor is the expert-consensus rank: log(ECR) when a preseason snapshot exists, else log(ADP); ordinally re-ranked per season."},
            {"step": "Rank to points", "text": "A per-position quadratic, fit walk-forward on prior seasons and clamped monotone, converts the anchor rank into expected log season points: what a player at that rank historically scores."},
            {"step": "Correct the experts", "text": f"A ridge regression over {len(features)} standardized preseason features predicts the gap between actual and anchor-implied points (residual clipped +/-4); a shrunken fraction of that correction is added back. Position is never a feature; it enters only via the per-position anchor and the VORP curve."},
            {"step": "Points to draft value", "text": "Projections map through a drafted-slot points curve (prior seasons only) to value over replacement: QB8/RB30/WR36/TE8, streaming-aware at QB/TE. The ECR and ADP baselines go through the identical conversion."},
            {"step": "No peeking", "text": "Season t uses only seasons < t outcomes plus season-t preseason facts. No leakage."},
            {"step": "Tuning", "text": "Ridge strength and shrink are chosen on the 2012-2017 window (frozen grid, re-derived each run); the test seasons are never touched for decisions."},
        ],
        "limitations": limitations,
    }


# ---------------------------------------------------------------------------
# 2. STATIC ASSETS (academic report style)
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#ffffff; --text:#1a1a1a; --muted:#5f5f5f; --faint:#8a8a8a;
  --rule:#dcdcdc; --rule-strong:#222; --link:#234a6b;
  --pos:#2c6e49; --neg:#9b2226; --hi:#f6f6f3;
  --serif:Georgia,Cambria,"Times New Roman",Times,serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --wrap:840px;
}
html{-webkit-text-size-adjust:100%}
body{font-family:var(--serif);font-size:16px;line-height:1.55;color:var(--text);background:var(--bg)}
.wrap{max-width:var(--wrap);margin:0 auto;padding:0 24px}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
code,.mono{font-family:var(--mono);font-size:.85em}
sup{font-size:.68em;line-height:0}

.masthead{border-bottom:1px solid var(--rule);position:sticky;top:0;background:var(--bg);z-index:20}
.masthead-inner{display:flex;align-items:baseline;gap:20px;padding:11px 0;flex-wrap:wrap}
.rubric{font-family:var(--mono);font-size:13px;font-weight:600}
.rubric a{color:var(--text)}
nav.doc{margin-left:auto;display:flex;gap:18px;font-family:var(--sans);font-size:13px}
nav.doc a{color:var(--muted)}
nav.doc a.active{color:var(--text);text-decoration:underline;text-underline-offset:3px}

.titleblock{padding:30px 0 10px;border-bottom:1px solid var(--rule-strong);margin-bottom:22px}
h1.title{font-size:26px;font-weight:700;letter-spacing:-.01em;line-height:1.18}
.subtitle{font-size:15px;color:var(--muted);margin-top:6px}
.docmeta{font-family:var(--mono);font-size:12px;color:var(--faint);margin-top:9px;line-height:1.5}

main{padding:6px 0 30px}
section{margin:26px 0}
body{counter-reset:sec}
h2{font-size:17px;font-weight:700;margin-bottom:9px;padding-bottom:4px;border-bottom:1px solid var(--rule)}
h2.numbered::before{counter-increment:sec;content:counter(sec) ". ";color:var(--faint)}
h3{font-size:14px;font-weight:700;margin:16px 0 5px}
p{margin-bottom:11px}
p.note{font-size:14px;color:var(--muted)}
.small{font-size:13px}
ul,ol{margin:0 0 12px 20px}
li{margin-bottom:5px}
ol.method li b{font-weight:700}

.cap{font-size:12.5px;color:var(--muted);margin:2px 0 5px}
.cap b{color:var(--text);font-weight:700}

table.tbl{width:100%;border-collapse:collapse;font-size:14px;margin:2px 0}
table.tbl th{font-family:var(--sans);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);text-align:left;padding:5px 10px;border-bottom:1px solid var(--rule-strong)}
table.tbl th.num{text-align:right}
table.tbl td{padding:4px 10px;border-bottom:1px solid var(--rule)}
table.tbl tbody tr:last-child td{border-bottom:1px solid var(--rule-strong)}
table.tbl .num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
table.tbl tr.summary td{font-weight:700}
.dpos{color:var(--pos)}.dneg{color:var(--neg)}.dflat{color:var(--faint)}

/* rankings controls */
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:baseline;margin:12px 0 6px;font-family:var(--sans);font-size:13px}
.controls input[type=search],.controls select{font-family:var(--mono);font-size:13px;padding:4px 7px;border:1px solid var(--rule);background:var(--bg);color:var(--text);border-radius:0}
.controls input[type=search]{min-width:190px}
.posfilter button{font-family:var(--mono);font-size:13px;background:none;border:none;color:var(--muted);cursor:pointer;padding:3px 5px}
.posfilter button:hover{color:var(--text)}
.posfilter button.active{color:var(--text);font-weight:700;text-decoration:underline;text-underline-offset:3px}
.toggle{display:inline-flex;align-items:center;gap:6px;color:var(--muted);cursor:pointer}
.count{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--faint)}

.tablewrap{overflow-x:auto}
table.data{width:100%;border-collapse:collapse;font-size:14px;min-width:560px}
table.data th{font-family:var(--sans);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);text-align:left;padding:6px 10px;border-bottom:1px solid var(--rule-strong);white-space:nowrap}
table.data th.num{text-align:right}
table.data td{padding:5px 10px;border-bottom:1px solid var(--rule);vertical-align:baseline}
table.data .num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
tr.row{cursor:pointer}
tr.row:hover{background:var(--hi)}
tr.row.open{background:var(--hi)}
.rk{font-family:var(--mono);color:var(--faint)}
.pname{font-weight:700}
.tpos{font-family:var(--mono);font-size:12px;color:var(--muted)}
.chev{font-family:var(--mono);color:var(--faint);font-size:11px;margin-right:2px}
.steal{color:var(--neg);font-family:var(--mono)}

.detail td{background:var(--hi);padding:0}
.detail-inner{padding:12px 12px 16px}
.reason{font-size:14px;margin-bottom:9px}
.decomp{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-bottom:12px}
.decomp b{color:var(--text)}
.attr-cap{font-family:var(--sans);font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:5px}

/* diverging bar (attribution + coefficients) */
.dbar-head{display:grid;grid-template-columns:194px 56px 1fr 54px;gap:10px;font-family:var(--sans);font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--faint);padding-bottom:3px}
.dbar-head span:nth-child(1),.dbar-head span:nth-child(2),.dbar-head span:nth-child(4){text-align:right}
.dbar-row{display:grid;grid-template-columns:194px 56px 1fr 54px;gap:10px;align-items:center;padding:2px 0}
.dbar-label{font-size:13px;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dbar-raw{font-family:var(--mono);font-size:12px;text-align:right;color:var(--muted)}
.dbar-track{position:relative;height:9px;background:#ececec}
.dbar-center{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--rule-strong)}
.dbar-fill{position:absolute;top:0;bottom:0}
.dbar-fill.pos{left:50%;background:var(--pos)}
.dbar-fill.neg{right:50%;background:var(--neg)}
.dbar-val{font-family:var(--mono);font-size:12px;text-align:right}
.dbar-val.pos{color:var(--pos)}.dbar-val.neg{color:var(--neg)}

/* features page */
.fgroup{margin:14px 0}
.fgroup h3{border-bottom:1px solid var(--rule);padding-bottom:3px;color:var(--muted);font-family:var(--sans);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.frow{padding:4px 0}
.frow .fhead{display:grid;grid-template-columns:210px 1fr 54px;gap:10px;align-items:center}
.flabel{font-size:13px;text-align:right}
.fblurb{font-size:12px;color:var(--faint);grid-column:1/-1;text-align:left;margin:1px 0 0 0}

/* VONA draft page */
.vona-mtx{min-width:640px;font-size:13px}
.vona-mtx th:first-child,.vona-mtx td:first-child{text-align:right;font-family:var(--mono);color:var(--faint)}
.vona-cell{white-space:nowrap;font-size:12.5px}
.vona-cell b{font-family:var(--mono);font-size:11px;margin-right:4px}
.pos-qb{background:#f1edf7}.pos-rb{background:#eaf2ec}.pos-wr{background:#e9f0f6}.pos-te{background:#f7f1e8}
.vona-cell.pos-qb b{color:#5b3f86}.vona-cell.pos-rb b{color:var(--pos)}
.vona-cell.pos-wr b{color:var(--link)}.vona-cell.pos-te b{color:#9a6a1c}
.vona-picks{min-width:640px}
.vona-picks td{vertical-align:top}
.vona-picks .vname{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:150px}
.vbar-t{height:5px;background:#ececec;margin:3px 0 1px}
.vbar-f{height:5px;background:var(--muted)}
.vp-rb .vbar-f{background:var(--pos)}
.vp-wr .vbar-f{background:var(--link)}
.vp-te .vbar-f{background:#9a6a1c}
.vp-qb .vbar-f{background:#5b3f86}
.vcost{font-family:var(--mono);font-size:11px;color:var(--muted)}
.vona-depth{margin:10px 0 4px}
.vona-depth button{font-family:var(--mono);font-size:13px;background:none;border:none;color:var(--muted);cursor:pointer;padding:3px 5px}
.vona-depth button:hover{color:var(--text)}
.vona-depth button.active{color:var(--text);font-weight:700;text-decoration:underline;text-underline-offset:3px}

.foot{border-top:1px solid var(--rule);margin-top:36px;padding:14px 0 44px;font-family:var(--mono);font-size:11.5px;color:var(--faint)}

@media(max-width:640px){
  .dbar-row,.dbar-head{grid-template-columns:108px 44px 1fr 46px}
  .frow .fhead{grid-template-columns:120px 1fr 46px}
  .dbar-label,.flabel{font-size:12px}
  h1.title{font-size:22px}
}
"""

JS_COMMON = """
const D = window.BFF_DATA;
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function f3(x){return (x==null)?'--':x.toFixed(3);}
function f4(x){return (x==null)?'--':x.toFixed(4);}
function sgn(x,d){d=d==null?3:d;return (x>=0?'+':'')+x.toFixed(d);}
function fmtRaw(x){if(x==null)return '';const a=Math.abs(x);return a>=100?x.toFixed(0):(a>=10?x.toFixed(1):x.toFixed(2));}
function dcls(x){return x==null?'dflat':(x>0?'dpos':(x<0?'dneg':'dflat'));}
// diverging green/red horizontal bar: label | raw value | bar | signed impact
function dbar(label,value,maxAbs,raw){
  const pct=maxAbs>0?Math.min(50,Math.abs(value)/maxAbs*50):0;
  const cls=value>=0?'pos':'neg';
  return '<div class="dbar-row"><div class="dbar-label" title="'+esc(label)+'">'+esc(label)+'</div>'+
    '<div class="dbar-raw">'+fmtRaw(raw)+'</div>'+
    '<div class="dbar-track"><div class="dbar-center"></div>'+
    '<div class="dbar-fill '+cls+'" style="width:'+pct.toFixed(1)+'%"></div></div>'+
    '<div class="dbar-val '+cls+'">'+sgn(value,3)+'</div></div>';
}
"""

JS_INDEX = """
(function(){
  const v=D.headline.vs_ecr;
  const rows=D.seasons.filter(r=>r.model!=null&&r.ecr!=null).map(r=>{
    const dl=r.model-r.ecr;
    return '<tr><td>'+r.season+'</td><td class="num">'+f4(r.model)+'</td><td class="num">'+f4(r.ecr)+'</td><td class="num '+dcls(dl)+'">'+sgn(dl,4)+'</td></tr>';
  }).join('');
  const meanrow='<tr class="summary"><td>mean, '+v.span+'</td><td class="num">'+f4(v.model)+'</td><td class="num">'+f4(v.baseline)+'</td><td class="num '+dcls(v.delta)+'">'+sgn(v.delta,4)+'</td></tr>';
  document.getElementById('results').innerHTML=
    '<table class="tbl"><thead><tr><th>Season</th><th class="num">Model</th><th class="num">ECR</th><th class="num">Delta</th></tr></thead><tbody>'+rows+meanrow+'</tbody></table>';
  const pstr=(v.p_one==null)?'n/a':v.p_one;
  document.getElementById('results-note').innerHTML='Model '+esc(v.verdict)+': mean '+f3(v.model)+' vs '+f3(v.baseline)+' (Delta '+sgn(v.delta,3)+'), one-sided sign-flip p = '+pstr+', n = '+v.n_seasons+'. Power floor '+v.power_floor+(v.power_floor>0.05?'; p-values descriptive only.':'.');

  const s=D.secondary;
  const sec=[['nDCG@100',s.ndcg100_vorp],['Top-24 hit rate',s.top24_hit_vorp],['Top-50 hit rate',s.top50_hit],['Raw-points Spearman',s.raw_spearman]];
  document.getElementById('secondary').innerHTML='<table class="tbl"><tbody>'+
    sec.map(c=>'<tr><td>'+c[0]+'</td><td class="num">'+f4(c[1])+'</td></tr>').join('')+'</tbody></table>';

  document.getElementById('method').innerHTML=D.method.map(m=>'<li><b>'+esc(m.step)+'.</b> '+esc(m.text)+'</li>').join('');
})();
"""

JS_RANKINGS = """
(function(){
  const tbody=document.getElementById('board');
  const countEl=document.getElementById('count');
  const state={q:'',pos:'ALL',sort:'rank',steals:false,open:new Set()};
  const nrank=x=>x==null?1e9:x;
  const nd=x=>x==null?-1e9:x;

  function detail(p){
    const maxAbs=Math.max.apply(null,[0.0001].concat(p.contribs.map(c=>Math.abs(c.value))));
    const bars=p.contribs.map(c=>dbar(c.label,c.value,maxAbs,c.raw)).join('');
    const decomp=(p.delta==null)
      ? '<div class="decomp">not in the FantasyPros expert consensus &mdash; no ECR comparison.</div>'
      : '<div class="decomp">vs ECR '+sgn(p.delta,0)+' spots = scarcity '+sgn(p.scarcity_gain,0)+' + features '+sgn(p.feature_gain,0)+'.</div>';
    return '<div class="detail-inner"><div class="reason">'+esc(p.reason)+'</div>'+decomp+
      '<div class="attr-cap">Feature attribution &mdash; player value and overall impact (log-points)</div>'+
      '<div class="dbar-head"><span>feature</span><span>value</span><span></span><span>impact</span></div>'+
      bars+'</div>';
  }

  function render(){
    let rows=D.players.filter(p=>{
      if(state.pos!=='ALL'&&p.pos!==state.pos)return false;
      if(state.steals&&!p.is_steal)return false;
      if(state.q){const q=state.q.toLowerCase();if(!(p.name.toLowerCase().includes(q)||p.team.toLowerCase().includes(q)))return false;}
      return true;
    });
    const s=state.sort;
    rows.sort((a,b)=> s==='ecr'?nrank(a.ecr)-nrank(b.ecr) : s==='vorp'?b.vorp-a.vorp : s==='movers'?nd(b.delta)-nd(a.delta) : a.rank-b.rank);
    countEl.textContent=rows.length+' / '+D.players.length;
    let html='';
    rows.forEach(p=>{
      const o=state.open.has(p.rank);
      html+='<tr class="row'+(o?' open':'')+'" data-rk="'+p.rank+'">'+
        '<td class="rk num">'+p.rank+'</td>'+
        '<td class="pname"><span class="chev">'+(o?'-':'+')+'</span>'+esc(p.name)+(p.is_steal?' <sup class="steal">s</sup>':'')+'</td>'+
        '<td class="tpos">'+p.pos+'</td>'+
        '<td>'+esc(p.team)+'</td>'+
        '<td class="num">'+(p.ecr==null?'&mdash;':p.ecr)+'</td>'+
        '<td class="num '+dcls(p.delta)+'">'+(p.delta==null?'&mdash;':sgn(p.delta,0))+'</td>'+
        '<td class="num">'+p.vorp.toFixed(1)+'</td></tr>';
      if(o)html+='<tr class="detail"><td colspan="7">'+detail(p)+'</td></tr>';
    });
    tbody.innerHTML=html||'<tr><td colspan="7" style="padding:16px;color:var(--faint)">no rows</td></tr>';
  }

  tbody.addEventListener('click',e=>{const tr=e.target.closest('tr.row');if(!tr)return;const rk=+tr.dataset.rk;state.open.has(rk)?state.open.delete(rk):state.open.add(rk);render();});
  document.getElementById('search').addEventListener('input',e=>{state.q=e.target.value;render();});
  document.getElementById('sort').addEventListener('change',e=>{state.sort=e.target.value;render();});
  document.getElementById('steals').addEventListener('change',e=>{state.steals=e.target.checked;render();});
  document.querySelectorAll('.posfilter button').forEach(c=>c.addEventListener('click',()=>{
    document.querySelectorAll('.posfilter button').forEach(x=>x.classList.remove('active'));
    c.classList.add('active');state.pos=c.dataset.pos;render();
  }));
  render();
})();
"""

JS_FEATURES = """
(function(){
  const order=['base','market','context','interaction','opportunity','trend','injury','draft','contract','other'];
  const byGroup={}; D.features.forEach(f=>{(byGroup[f.group]=byGroup[f.group]||[]).push(f);});
  Object.keys(byGroup).forEach(g=>{if(!order.includes(g))order.push(g);});
  const maxAbs=Math.max.apply(null,D.features.map(f=>Math.abs(f.coef)));
  let html='';
  order.forEach(g=>{
    const feats=byGroup[g]; if(!feats)return;
    feats.sort((a,b)=>Math.abs(b.coef)-Math.abs(a.coef));
    html+='<div class="fgroup"><h3>'+esc(feats[0].group_label)+' ('+feats.length+')</h3>';
    feats.forEach(f=>{
      const cls=f.coef>=0?'pos':'neg';
      const pct=Math.min(50,Math.abs(f.coef)/maxAbs*50);
      html+='<div class="frow"><div class="fhead">'+
        '<div class="flabel">'+esc(f.label)+'</div>'+
        '<div class="dbar-track"><div class="dbar-center"></div><div class="dbar-fill '+cls+'" style="width:'+pct.toFixed(1)+'%"></div></div>'+
        '<div class="dbar-val '+cls+'">'+sgn(f.coef,3)+'</div>'+
        '<div class="fblurb">'+esc(f.blurb)+'</div></div></div>';
    });
    html+='</div>';
  });
  document.getElementById('features-root').innerHTML=html;
})();
"""

JS_METHOD = """
(function(){
  document.getElementById('method-steps').innerHTML=D.method.map(m=>'<li><b>'+esc(m.step)+'.</b> '+esc(m.text)+'</li>').join('');
  document.getElementById('limitations').innerHTML=D.limitations.map(x=>'<li>'+esc(x)+'</li>').join('');
})();
"""

JS_VONA = """
(function(){
  const V=D.vona, R=V.rounds, POS=['QB','RB','WR','TE'];
  const pc=p=>({QB:'pos-qb',RB:'pos-rb',WR:'pos-wr',TE:'pos-te'}[p]||'');
  const bc=p=>({QB:'vp-qb',RB:'vp-rb',WR:'vp-wr',TE:'vp-te'}[p]||'');

  // turn matrix
  let mh='<tr><th>Slot</th>';
  for(let r=1;r<=R;r++)mh+='<th>Round '+r+'</th>';
  mh+='</tr>';
  const mrows=V.matrix.map(row=>{
    let tds='<td class="num">'+row.slot+'</td>';
    for(let r=1;r<=R;r++){
      const cell=row['round'+r]||'--';
      const ix=cell.indexOf(': ');
      if(ix<0){tds+='<td>&mdash;</td>';continue;}
      const p=cell.slice(0,ix), nm=cell.slice(ix+2);
      tds+='<td class="vona-cell '+pc(p)+'"><b>'+esc(p)+'</b> '+esc(nm)+'</td>';
    }
    return '<tr>'+tds+'</tr>';
  }).join('');
  document.getElementById('vona-matrix').innerHTML=
    '<table class="data vona-mtx"><thead>'+mh+'</thead><tbody>'+mrows+'</tbody></table>';

  // per-pick wait-cost table
  const maxV=Math.max.apply(null,V.table.flatMap(r=>POS.map(p=>r['vona24_'+p]||0)))||1;
  function cell(r,p){
    const v=r['vona24_'+p]||0, nm=r['best_'+p]||'';
    const w=Math.min(100,v/maxV*100);
    return '<td class="'+bc(p)+'"><div class="vname">'+(nm?esc(nm):'&mdash;')+'</div>'+
      '<div class="vbar-t"><div class="vbar-f" style="width:'+w.toFixed(0)+'%"></div></div>'+
      '<div class="vcost">'+(v>0?v.toFixed(0):'')+'</div></td>';
  }
  function render(rng){
    const rows=V.table.filter(r=>rng?r.pick<=rng:true);
    let html='';
    rows.forEach(r=>{html+='<tr><td class="num">'+r.pick+'</td>'+POS.map(p=>cell(r,p)).join('')+'</tr>';});
    document.getElementById('vona-picks').innerHTML=html;
  }
  document.querySelectorAll('.vona-depth button').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('.vona-depth button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');render(+b.dataset.n||0);
  }));
  render(24);
})();
"""

# ---------------------------------------------------------------------------
# 3. PAGE RENDERERS
# ---------------------------------------------------------------------------

NAV = [("index.html", "Results"), ("rankings.html", "Rankings"),
       ("vona.html", "Draft"),
       ("features.html", "Features"), ("methodology.html", "Methods")]


def _header(active: str) -> str:
    links = "".join(
        f'<a href="{href}" class="{"active" if href == active else ""}">{label}</a>'
        for href, label in NAV
    )
    return ('<header class="masthead"><div class="wrap masthead-inner">'
            '<span class="rubric"><a href="index.html">betterffrank</a></span>'
            f'<nav class="doc">{links}</nav></div></header>')


def _footer(meta: dict) -> str:
    return ('<footer class="foot"><div class="wrap">'
            f'betterffrank &middot; {meta["season"]} preseason &middot; {meta["format"]} '
            '&middot; walk-forward VORP &middot; generated by <code>bff.site</code>'
            '</div></footer>')


def _page(title: str, active: str, body: str, data_slice: dict, page_js: str) -> str:
    blob = "window.BFF_DATA=" + json.dumps(data_slice, separators=(",", ":")) + ";"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="icon" href="data:image/svg+xml,'
        '%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22%3E'
        '%3Crect width=%2232%22 height=%2232%22 fill=%22%231a1a1a%22/%3E'
        '%3Ctext x=%2216%22 y=%2223%22 font-size=%2219%22 font-family=%22Georgia,serif%22 '
        'fill=%22%23fff%22 text-anchor=%22middle%22%3Eb%3C/text%3E%3C/svg%3E">\n'
        f'<title>{title}</title>\n'
        f'<style>{CSS}</style>\n</head>\n<body>\n'
        + _header(active)
        + '<main>' + body + '</main>'
        + _footer(data_slice["meta"])
        + '<script>' + blob + '</script>\n'
        + '<script>' + JS_COMMON + page_js + '</script>\n'
        + '</body>\n</html>\n'
    )


def render_index(p: dict) -> str:
    m = p["meta"]
    ve, va = p["headline"]["vs_ecr"], p["headline"]["vs_adp"]
    body = (
        '<div class="wrap">'
        '<div class="titleblock">'
        '<h1 class="title">betterffrank</h1>'
        f'<p class="subtitle">{m["season"]} preseason rankings &middot; {m["format"]} &middot; walk-forward VORP evaluation</p>'
        f'<p class="docmeta">ridge residual over an ECR anchor &middot; alpha = {int(m["alpha"])}, shrink = {m["shrink"]} '
        f'&middot; {m["n_features"]} features &middot; {m["n_players"]} players '
        '&middot; metric: Spearman(list, realized VORP)</p>'
        '</div>'

        '<section><h2 class="numbered">The idea</h2>'
        '<p>These rankings start from the FantasyPros expert consensus and correct it. '
        'The experts are good, so the model never builds a board from scratch; instead, a regression '
        f'fit on past seasons learns the patterns in when the experts miss &mdash; age, last season&rsquo;s '
        'production against draft price, injury history, vacated opportunity, rookie pedigree &mdash; and '
        'nudges each player&rsquo;s projection up or down from the expert baseline. Projected points then '
        'convert to <b>value over replacement</b> (VORP): points above the best player you could get off '
        'waivers at that position. That conversion, not raw points, is what merges QB, RB, WR, and TE '
        'into one draft board; raw points would just stack quarterbacks at the top.</p>'
        '<p>Scoring is champion versus challenger. For each test season '
        f'({ve["span"].replace("-", "&ndash;")}), three boards are fixed before Week 1 &mdash; this model, '
        'the expert consensus (ECR), and the market (ADP) &mdash; and graded after the season by how well '
        'their preseason order matched players&rsquo; realized value over replacement. Same grade, same '
        'VORP conversion, and no board sees the season it predicts; the model&rsquo;s settings were frozen '
        'on 2012&ndash;2017 before any test season was scored. The record: '
        f'<b>beat ADP in {va["wins"]} of {va["n_seasons"]} seasons</b> '
        f'(mean {va["model"]:.4f} vs {va["baseline"]:.4f}, one-sided p = {va["p_one"]:g}) and '
        f'<b>edged ECR in {ve["wins"]} of {ve["n_seasons"]}</b> '
        f'(mean {ve["model"]:.4f} vs {ve["baseline"]:.4f}, p = {ve["p_one"]:g}). '
        'The ECR margin is thin; read it as matching the experts, likely a touch better. '
        'The experts themselves beat the market, which is why they are the harder benchmark.</p></section>'

        '<section><h2 class="numbered">Results</h2>'
        '<p class="cap"><b>Table 1.</b> Model vs FantasyPros ECR, per season. '
        'Spearman of the ranked list against realized VORP; Delta = model &minus; ECR.</p>'
        '<div id="results"></div>'
        '<p class="note small" id="results-note" style="margin-top:8px"></p></section>'

        '<section><h2 class="numbered">Secondary metrics</h2>'
        '<p class="cap"><b>Table 2.</b> Model, mean over the test seasons.</p>'
        '<div id="secondary"></div>'
        '<p class="note small" style="margin-top:8px">Raw-points Spearman is reported for completeness only; '
        'it is not a decision metric (it rewards QB-stacking).</p></section>'

        '<section><h2 class="numbered">Metric and method</h2>'
        '<p class="note">VORP = value over replacement (QB8 / RB30 / WR36 / TE8). '
        'Predictions for season <i>t</i> use only seasons &lt; <i>t</i> outcomes plus season-<i>t</i> '
        'preseason facts (ECR, ADP, rosters, draft): walk-forward, no leakage.</p>'
        '<ol class="method" id="method"></ol>'
        '<p class="note small">Detail: <a href="methodology.html">methods</a>. '
        'Board: <a href="rankings.html">rankings</a>. '
        'Coefficients: <a href="features.html">features</a>.</p></section>'
        '</div>'
    )
    data = {"meta": p["meta"], "headline": p["headline"],
            "secondary": p["secondary"], "seasons": p["seasons"], "method": p["method"]}
    return _page("betterffrank — results", "index.html", body, data, JS_INDEX)


def render_rankings(p: dict) -> str:
    m = p["meta"]
    body = (
        '<div class="wrap">'
        '<div class="titleblock"><h1 class="title">Rankings</h1>'
        f'<p class="subtitle">{m["season"]} preseason &middot; {m["format"]} &middot; ordered by predicted VORP</p>'
        f'<p class="docmeta">{m["n_players"]} players &middot; Delta = ECR rank &minus; model rank '
        '&middot; click a row for per-player attribution</p></div>'

        '<div class="controls">'
        '<input type="search" id="search" placeholder="filter: name or team" aria-label="filter">'
        '<span class="posfilter">'
        '<button data-pos="ALL" class="active">all</button>'
        '<button data-pos="QB">QB</button><button data-pos="RB">RB</button>'
        '<button data-pos="WR">WR</button><button data-pos="TE">TE</button></span>'
        '<select id="sort" aria-label="sort">'
        '<option value="rank">sort: model rank</option>'
        '<option value="ecr">sort: ECR</option>'
        '<option value="vorp">sort: VORP</option>'
        '<option value="movers">sort: Delta</option></select>'
        '<label class="toggle"><input type="checkbox" id="steals"> steals only</label>'
        '<span class="count" id="count"></span></div>'

        '<div class="tablewrap"><table class="data"><thead><tr>'
        '<th class="num">#</th><th>Player</th><th>Pos</th><th>Team</th>'
        '<th class="num">ECR</th><th class="num">Delta</th><th class="num">VORP</th></tr></thead>'
        '<tbody id="board"></tbody></table></div>'
        '<p class="note small" style="margin-top:8px">Delta &gt; 0: ranked above the expert consensus. '
        '<sup class="steal">s</sup> = steal (ECR &le; 120, Delta &ge; 24). '
        'Players outside the ECR consensus show &mdash;.</p>'
        '</div>'
    )
    data = {"meta": p["meta"], "players": p["players"]}
    return _page("betterffrank — rankings", "rankings.html", body, data, JS_RANKINGS)


def render_vona(p: dict) -> str:
    m = p["meta"]
    body = (
        '<div class="wrap">'
        '<div class="titleblock"><h1 class="title">Draft</h1>'
        f'<p class="subtitle">{m["season"]} &middot; {m["format"]} &middot; VONA &mdash; value lost by waiting</p>'
        '<p class="docmeta">the board ranks season-long value; this page times '
        'positions inside <i>your</i> draft &middot; scarcity, not value</p></div>'

        '<section><p class="note">The <a href="rankings.html">rankings</a> answer '
        '"who is worth the most this season" &mdash; and in full PPR that leans WR at the top. '
        'VONA (Value Over Next Available) answers a different question: at each pick, how much '
        'value do you lose at each position by waiting until your next turn? That is where '
        'running-back scarcity actually lives &mdash; startable RB dries up faster than WR at '
        'the top of a draft, so the timing guide leads RB early even though the value board does not. '
        'Assumes opponents draft by ADP; a positional-timing guide, not a full simulator.</p></section>'

        '<section><h2 class="numbered">Your draft, by seat</h2>'
        '<p class="cap"><b>Table 1.</b> Greedy VONA pick for each of the 12 snake seats, rounds 1&ndash;'
        f'{p["vona"]["rounds"]}. At every turn: take the position whose best-available player falls the most before your next pick.</p>'
        '<div class="tablewrap"><div id="vona-matrix"></div></div></section>'

        '<section><h2 class="numbered">Wait-cost by pick</h2>'
        '<p class="cap"><b>Table 2.</b> Best available at each position and its VONA cost '
        '(predicted VORP lost if you wait ~2 rounds / 24 picks). A per-position diagnostic &mdash; '
        'Table 1 is the actual call, since it also weighs which position you can best backfill '
        'at your next pick.</p>'
        '<div class="vona-depth controls">through pick: '
        '<button data-n="24" class="active">24</button><button data-n="48">48</button>'
        '<button data-n="0">all</button></div>'
        '<div class="tablewrap"><table class="data vona-picks"><thead><tr>'
        '<th class="num">#</th><th>QB</th><th>RB</th><th>WR</th><th>TE</th></tr></thead>'
        '<tbody id="vona-picks"></tbody></table></div>'
        '<p class="note small" style="margin-top:8px">VONA is non-negative by construction '
        '(waiting never gains value). Names are the best <i>remaining</i> player at that position '
        'by our VORP, which can differ from ADP order.</p></section>'
        '</div>'
    )
    data = {"meta": p["meta"], "vona": p["vona"]}
    return _page("betterffrank — draft", "vona.html", body, data, JS_VONA)


def render_features(p: dict) -> str:
    m = p["meta"]
    body = (
        '<div class="wrap">'
        '<div class="titleblock"><h1 class="title">Features</h1>'
        f'<p class="subtitle">{m["n_features"]} standardized ridge coefficients (full-history fit)</p>'
        '<p class="docmeta">bar = signed standardized weight on the residual '
        '&middot; grouped by block, sorted by |coef| within block</p></div>'
        '<p class="note small">Standardized coefficients are comparable across features. '
        'ECR-vs-ADP gap is near zero: when the market and the experts disagree, the model barely takes a side.</p>'
        '<div id="features-root"></div>'
        '</div>'
    )
    data = {"meta": p["meta"], "features": p["features"]}
    return _page("betterffrank — features", "features.html", body, data, JS_FEATURES)


def render_methodology(p: dict) -> str:
    ve = p["headline"]["vs_ecr"]
    pstr = "n/a" if ve["p_one"] is None else str(ve["p_one"])
    body = (
        '<div class="wrap">'
        '<div class="titleblock"><h1 class="title">Methods</h1>'
        '<p class="subtitle">pipeline, leakage controls, tuning, limitations</p></div>'

        '<section><h2 class="numbered">Result</h2>'
        f'<p class="note">On the {ve["span"]} test set the model {ve["verdict"]}: mean '
        f'{ve["model"]:.3f} vs {ve["baseline"]:.3f} (Delta {ve["delta"]:+.3f}); '
        f'one-sided sign-flip p = {pstr}; n = {ve["n_seasons"]} seasons.</p></section>'

        '<section><h2 class="numbered">Pipeline</h2>'
        '<ol class="method" id="method-steps"></ol></section>'

        '<section><h2 class="numbered">Leakage controls</h2>'
        '<ul class="small">'
        '<li>Walk-forward: season <i>t</i> trains on seasons &lt; <i>t</i> and uses only season-<i>t</i> '
        'preseason facts (ADP, ECR, April draft, week-1 rosters and coaches). Never season-<i>t</i> outcomes.</li>'
        '<li>Roster membership from a week-1 snapshot, not a last-observed-team table '
        '(the latter leaks in-season trades; worth about 0.003 mean Spearman).</li>'
        '<li>The VORP curve is built from prior seasons only, and every baseline goes through the '
        'same conversion; no baseline is scored on raw rank.</li></ul></section>'

        '<section><h2 class="numbered">Tuning</h2>'
        '<p class="note small">Ridge strength and residual shrink are chosen on an earlier walk-forward '
        'window, on a frozen grid, re-derived deterministically each run (no stored parameters). '
        f'Evaluation seasons are never touched for tuning or feature selection. Current: alpha = {int(p["meta"]["alpha"])}, '
        f'shrink = {p["meta"]["shrink"]}.</p></section>'

        '<section><h2 class="numbered">Limitations</h2>'
        '<ul class="small" id="limitations"></ul></section>'
        '</div>'
    )
    data = {"meta": p["meta"], "method": p["method"], "limitations": p["limitations"]}
    return _page("betterffrank — methods", "methodology.html", body, data, JS_METHOD)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def _refresh() -> None:
    steps = [
        [sys.executable, "-m", "bff.model"],
        [sys.executable, "-m", "bff.model", "--baselines"],
        [sys.executable, "-m", "bff.model", "--season", "2026"],
        [sys.executable, "-m", "bff.vona"],
        [sys.executable, "-m", "bff.backtest", str(PROC / "preds_model.parquet"), "--name", "model"],
        [sys.executable, "-m", "bff.backtest", str(PROC / "preds_adp.parquet"), "--name", "adp"],
        [sys.executable, "-m", "bff.backtest", str(PROC / "preds_ecr.parquet"), "--name", "ecr"],
    ]
    for cmd in steps:
        print("+ " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the /docs static site.")
    ap.add_argument("--refresh", action="store_true",
                    help="rerun model + baselines + 2026 + backtests before rendering")
    args = ap.parse_args()

    if args.refresh:
        _refresh()

    payload = build_payload()
    DOCS.mkdir(exist_ok=True)
    pages = {
        "index.html": render_index(payload),
        "rankings.html": render_rankings(payload),
        "vona.html": render_vona(payload),
        "features.html": render_features(payload),
        "methodology.html": render_methodology(payload),
    }
    for name, html in pages.items():
        (DOCS / name).write_text(html, encoding="utf-8")
        print(f"wrote {DOCS / name} ({len(html):,} bytes)")
    (DOCS / "data.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {DOCS / 'data.json'} "
          f"({payload['meta']['n_players']} players, {payload['meta']['n_features']} features)")


if __name__ == "__main__":
    main()
