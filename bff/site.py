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
    TUNE_SEASONS,
    build_curve,
    build_dataset,
    fit_predict,
    pool_market_ranks,
    rank_by_vorp,
    reason_string,
    to_vorp,
    tune,
)

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


def _read_audit() -> dict:
    """The methodology audit's cached numbers (index.html is an article about
    them). Not derivable from preds/scores, so it has its own artifact."""
    path = REPORTS / "methodology_audit.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nThe front-page article is built from the "
            "methodology audit. Generate it first:\n"
            "  uv run python -m bff.methodology_audit")
    return json.loads(path.read_text(encoding="utf-8"))


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

    # --- Anchor quality: is the expert list actually better than the market? --
    # Scored the same way as everything else (baseline ranks -> to_vorp ->
    # eval_season). Test seasons come from the shipped scores_*.csv; tune
    # seasons are recomputed here because no artifact carries them.
    def _baseline_preds(season, which):
        if which == "adp":
            return adp_all.filter(
                (pl.col("season") == season) & pl.col("gsis_id").is_not_null()
                & pl.col("position").is_in(POSITIONS)
            ).select(pl.col("season").cast(pl.Int64), "gsis_id",
                     (-pl.col("adp_rank").cast(pl.Float64)).alias("score"))
        return (
            ecr_all.filter(
                (pl.col("season") == season) & pl.col("gsis_id").is_not_null()
                & pl.col("position").is_in(POSITIONS)
            )
            .sort("ecr_rank").unique(subset=["gsis_id"], keep="first")
            .select(pl.col("season").cast(pl.Int64), "gsis_id",
                    (-pl.col("ecr_rank").cast(pl.Float64)).alias("score"))
        )

    def _tune_baseline(season, which):
        pool, _ = build_pool(season, adp_all, ecr_all)
        preds_b = to_vorp(_baseline_preds(season, which), season,
                          adp_all, ecr_all, hist)
        return eval_season(pool, preds_b, actuals, season)["spearman_vorp"]

    def _anchor_rows(years, from_artifacts):
        rows = []
        for s in years:
            if from_artifacts:
                a, e = _val(sa, s, "spearman_vorp"), _val(se, s, "spearman_vorp")
            else:
                a = round(_tune_baseline(s, "adp"), 4)
                e = round(_tune_baseline(s, "ecr"), 4)
            rows.append({"season": s, "adp": a, "ecr": e,
                         "delta": round(e - a, 4)})
        return rows

    def _anchor_block(rows, label):
        d = [r["delta"] for r in rows]
        return {
            "label": label, "rows": rows,
            "adp_mean": round(sum(r["adp"] for r in rows) / len(rows), 4),
            "ecr_mean": round(sum(r["ecr"] for r in rows) / len(rows), 4),
            "mean": round(sum(d) / len(d), 4),
            "wins": sum(1 for x in d if x > 0), "n": len(d),
        }

    anchor_tune = _anchor_block(
        _anchor_rows(list(TUNE_SEASONS), False),
        f"Tune window {TUNE_SEASONS[0]}–{TUNE_SEASONS[-1]}")
    anchor_test = _anchor_block(_anchor_rows(yrs, True),
                                f"Test window {yrs[0]}–{yrs[-1]}")
    # How the test-set ADP win splits: anchor contribution vs model contribution
    anchor_split = {
        "total": headline["vs_adp"]["delta"],
        "from_anchor": anchor_test["mean"],
        "from_model": headline["vs_ecr"]["delta"],
    }

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
        "audit": _read_audit(),
        "headline": headline,
        "anchor": {"tune": anchor_tune, "test": anchor_test,
                   "split": anchor_split},
        "secondary": {"raw_spearman": _val(sm, "MEAN", "spearman"),
                      "ndcg100_vorp": _val(sm, "MEAN", "ndcg100_vorp"),
                      "top24_hit_vorp": _val(sm, "MEAN", "top24_hit_vorp"),
                      "top50_hit": _val(sm, "MEAN", "top50_hit")},
        "seasons": seasons,
        "features": features,
        "players": players,
        # DraftSIM — same board as the Rankings page, so the simulator can
        # never drift from it. Presentation only.
        "draftsim": {
            "players": ranked.sort("our_rank").select(
                pl.col("gsis_id").alias("id"), "name",
                pl.col("position").alias("pos"), "team",
                pl.col("our_rank").alias("rank"),
                pl.col("adp_rank").alias("adp"),
                pl.col("ecr_ord").alias("ecr"),
                pl.col("pred_vorp").round(2).alias("vorp"),
            ).to_dicts(),
            "league": {"teams": 12, "rounds": 16,
                       "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
                       "flex": ["RB", "WR", "TE"], "caps": {"QB": 2, "TE": 2}},
        },
        "method": [
            {"step": "Pool", "text": "Each season's ADP top-150 at QB/RB/WR/TE, GSIS-matched (~145-150 players)."},
            {"step": "Start from the experts", "text": "Every player's anchor is the expert-consensus rank: log(ECR) when a preseason snapshot exists, else log(ADP); ordinally re-ranked per season."},
            {"step": "Rank to points", "text": "A per-position quadratic, fit walk-forward on prior seasons and clamped monotone, converts the anchor rank into expected log season points: what a player at that rank historically scores."},
            {"step": "Correct the experts", "text": f"A ridge regression over {len(features)} standardized preseason features predicts the gap between actual and anchor-implied points (residual clipped +/-4); a shrunken fraction of that correction is added back. Position is never a feature; it enters only via the per-position anchor and the VORP curve."},
            {"step": "Points to draft value", "text": "Projections map through a drafted-slot points curve (prior seasons only) to value over replacement: QB8/RB30/WR36/TE8, streaming-aware at QB/TE. The ECR and ADP baselines go through the identical conversion."},
            {"step": "No peeking", "text": "Season t uses only seasons < t outcomes plus season-t preseason facts. No leakage."},
            {"step": "Tuning", "text": f"Ridge strength and shrink are chosen on the {TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]} window (frozen grid, re-derived each run); the test seasons are never touched for decisions."},
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

/* position tints (draftsim rec cards) */
.pos-qb{background:#f1edf7}.pos-rb{background:#eaf2ec}.pos-wr{background:#e9f0f6}.pos-te{background:#f7f1e8}

.foot{border-top:1px solid var(--rule);margin-top:36px;padding:14px 0 44px;font-family:var(--mono);font-size:11.5px;color:var(--faint)}

@media(max-width:640px){
  .dbar-row,.dbar-head{grid-template-columns:108px 44px 1fr 46px}
  .frow .fhead{grid-template-columns:120px 1fr 46px}
  .dbar-label,.flabel{font-size:12px}
  h1.title{font-size:22px}
}
"""

# Appended only on draftsim.html so the four other pages stay byte-identical.
# One-screen dashboard: fixed-viewport grid on desktop (each panel scrolls
# internally, no page scroll), sticky recs + flowing stack on mobile.
CSS_DRAFTSIM = """
:root{--mh:47px}
main:has(.sim-app){padding:0}
.sim-app{
  max-width:1280px;margin:0 auto;padding:10px 14px;
  display:grid;gap:12px;
  grid-template-columns:180px 1fr 300px;
  grid-template-rows:auto auto minmax(0,1fr) auto;
  height:calc(100dvh - var(--mh));
}
.sim-app .status{grid-column:1/-1;grid-row:1}
.p-roster{grid-column:1;grid-row:2/4}
.p-board{grid-column:2;grid-row:2/4}
.p-recs{grid-column:3;grid-row:2}
.p-drafted{grid-column:3;grid-row:3}
.sim-details{grid-column:1/-1;grid-row:4}

/* panel chrome */
.panel{border:1px solid var(--rule);display:flex;flex-direction:column;min-height:0;background:var(--bg)}
.panel-hd{font-family:var(--sans);font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  padding:6px 10px;border-bottom:1px solid var(--rule-strong);display:flex;align-items:center;gap:8px;
  position:sticky;top:0;background:var(--bg);z-index:2;flex:0 0 auto}
.panel-hd .hd-n{margin-left:auto;font-family:var(--mono);color:var(--faint);text-transform:none;letter-spacing:0}
.panel-bd{overflow-y:auto;min-height:0;flex:1 1 auto}

/* status bar */
.status{border:1px solid var(--rule);padding:8px 12px;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
.status select{font-family:var(--mono);font-size:13px;padding:3px 6px;border:1px solid var(--rule);background:var(--bg);color:var(--text);border-radius:0}
.status button{font-family:var(--mono);font-size:12px;padding:3px 9px;border:1px solid var(--rule);background:var(--bg);color:var(--text);cursor:pointer}
.status button:hover{border-color:var(--rule-strong)}
.status button:disabled{color:var(--faint);cursor:default;border-color:var(--rule)}
.sim-status{font-family:var(--mono);font-size:12.5px;color:var(--muted);flex:1 1 220px;min-width:180px}
.sim-status b{color:var(--text)}
.sim-status .you{color:var(--neg);font-weight:700}
.sim-details{grid-column:1/-1;font-size:12.5px;color:var(--muted)}
.sim-details summary{cursor:pointer;font-family:var(--sans);font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--faint)}
.sim-details p{margin:8px 0 0}

/* recommendations — compact div list */
.sim-mode{font-family:var(--sans);font-size:10px;text-transform:none;letter-spacing:0;color:var(--faint)}
.rec{padding:6px 10px;border-bottom:1px solid var(--rule)}
.rec:last-child{border-bottom:none}
.rec-top{display:flex;align-items:baseline;gap:6px}
.rec-rk{font-family:var(--mono);font-size:11px;color:var(--faint)}
.rec-nm{font-weight:700;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rec-pt{font-family:var(--mono);font-size:11px;color:var(--muted)}
.rec-urg{margin-left:auto;font-family:var(--mono);font-size:13px;font-weight:700}
.rec-urg.hi{color:var(--neg)}
.rec-sub{font-size:11px;color:var(--faint);margin-top:1px;display:flex;gap:8px;flex-wrap:wrap}
.rec-why{color:var(--muted)}
.pos-qb .rec-pt{color:#5b3f86}.pos-rb .rec-pt{color:var(--pos)}
.pos-wr .rec-pt{color:var(--link)}.pos-te .rec-pt{color:#9a6a1c}

/* roster rail — single column */
.sim-roster{font-size:13px}
.sim-slot{display:flex;gap:8px;align-items:baseline;border-bottom:1px solid var(--rule);padding:4px 10px}
.sim-slot .sl{font-family:var(--mono);font-size:10px;color:var(--faint);min-width:34px}
.sim-slot .sp{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12.5px}
.sim-slot .sv{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:auto}
.sim-slot.empty .sp{color:var(--faint)}
.sim-slot.bnch .sl{color:var(--faint)}

/* drafted — compact names */
.sim-drafted{font-size:12px;line-height:1.5;padding:8px 10px;color:var(--muted)}
.sim-drafted .dr{white-space:nowrap}
.sim-drafted .dr.mine{color:var(--neg);font-weight:700}
.sim-drafted .dr .drk{font-family:var(--mono);font-size:10px;color:var(--faint)}

/* board */
.sim-controls{padding:6px 10px;border-bottom:1px solid var(--rule);display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-family:var(--sans);font-size:12px;flex:0 0 auto}
.sim-controls input[type=search]{font-family:var(--mono);font-size:12px;padding:3px 6px;border:1px solid var(--rule);background:var(--bg);color:var(--text);border-radius:0;min-width:120px;flex:1 1 120px}
.sim-controls .posfilter button{font-family:var(--mono);font-size:12px;background:none;border:none;color:var(--muted);cursor:pointer;padding:2px 4px}
.sim-controls .posfilter button.active{color:var(--text);font-weight:700;text-decoration:underline;text-underline-offset:3px}
.sim-controls .toggle{display:inline-flex;align-items:center;gap:4px;color:var(--muted);cursor:pointer}
table.sim-board{width:100%;border-collapse:collapse;font-size:13px}
table.sim-board th{font-family:var(--sans);font-weight:600;font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule-strong);position:sticky;top:0;background:var(--bg);z-index:1;white-space:nowrap}
table.sim-board th.num{text-align:right}
table.sim-board th.sortable{cursor:pointer;user-select:none}
table.sim-board th.sortable:hover{color:var(--text)}
table.sim-board th.sorted{color:var(--text)}
table.sim-board th .arr{font-size:8px;margin-left:2px}
table.sim-board td{padding:3px 8px;border-bottom:1px solid var(--rule)}
table.sim-board .num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
table.sim-board td.chk{width:26px;text-align:center;padding-left:10px}
table.sim-board input[type=checkbox]{cursor:pointer}
.sim-board .pname{font-weight:700}
.sim-board .tpos{font-family:var(--mono);font-size:11px;color:var(--muted)}
tr.drafted td{color:var(--faint)}
tr.drafted .pname{text-decoration:line-through;font-weight:400}
tr.mine td{background:var(--hi)}

@media(max-width:760px){
  .sim-app{display:flex;flex-direction:column;height:auto;max-width:none;padding:8px 10px;gap:10px}
  .status{order:0;position:sticky;top:var(--mh);z-index:6}
  .p-recs{order:1;position:sticky;top:var(--st,124px);z-index:5}
  .p-roster{order:2}
  .p-drafted{order:3}
  .p-board{order:4}
  .sim-details{order:5}
  .panel-bd{max-height:56vh}
  .p-recs .panel-bd,.p-roster .panel-bd{max-height:none}
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

JS_DRAFTSIM = """
(function(){
  const S=D.draftsim, L=S.league, P=S.players;      // P sorted by ADP asc; L is mutated by league controls
  const POS=['QB','RB','WR','TE'];
  const byId={}; P.forEach(p=>byId[p.id]=p);
  const TEAM_OPTS=[8,10,12], FLEX_OPTS=[1,2];
  let N_PICKS=L.teams*L.rounds;
  const KEY='bff-draftsim-'+D.meta.season;

  // ---- state: league (teams/flex) + seat + ordered pick list (id or null=off-board) ----
  let state={teams:L.teams,flex:L.starters.FLEX,seat:1,picks:[]};
  try{
    const s=JSON.parse(localStorage.getItem(KEY)||'');
    if(s&&Array.isArray(s.picks)){
      if(TEAM_OPTS.includes(s.teams))state.teams=s.teams;
      if(FLEX_OPTS.includes(s.flex))state.flex=s.flex;
      if(s.seat>=1&&s.seat<=state.teams)state.seat=s.seat;
      state.picks=s.picks.filter(id=>id===null||byId[id]);
    }
  }catch(e){}
  function save(){try{localStorage.setItem(KEY,JSON.stringify(state));}catch(e){}}

  // Apply the league selection into L (which every function reads live).
  function applyLeague(){
    L.teams=state.teams; L.starters.FLEX=state.flex;
    N_PICKS=L.teams*L.rounds;
    if(state.seat>state.teams)state.seat=state.teams;
  }
  applyLeague();

  // ---- snake arithmetic (mirrors bff/vona.py snake_picks) ----
  function owner(p){const r=Math.ceil(p/L.teams);const i=(p-1)%L.teams;return r%2===1?i+1:L.teams-i;}
  function nextMyPick(from){for(let p=from;p<=N_PICKS;p++)if(owner(p)===state.seat)return p;return null;}

  // ---- lineup engine ----
  // Fill dedicated slots with the top players per position; FLEX = best
  // leftovers among flex positions. Greedy is exact for this slot structure.
  function assign(roster){
    const by={QB:[],RB:[],WR:[],TE:[]};
    roster.forEach(p=>{if(by[p.pos])by[p.pos].push(p);});
    POS.forEach(ps=>by[ps].sort((a,b)=>b.vorp-a.vorp));
    const starters={},left=[];let val=0;
    POS.forEach(ps=>{
      const n=L.starters[ps]||0;
      starters[ps]=by[ps].slice(0,n);
      starters[ps].forEach(p=>{val+=p.vorp;});
      if(L.flex.includes(ps))by[ps].slice(n).forEach(p=>left.push(p));
    });
    left.sort((a,b)=>b.vorp-a.vorp);
    const flex=left.slice(0,L.starters.FLEX||0);
    flex.forEach(p=>{val+=p.vorp;});
    const used=new Set();
    POS.forEach(ps=>starters[ps].forEach(p=>used.add(p.id)));
    flex.forEach(p=>used.add(p.id));
    return {starters:starters,flex:flex,bench:roster.filter(p=>!used.has(p.id)),value:val};
  }
  const lineupValue=r=>assign(r).value;
  const marginal=(r,p)=>lineupValue(r.concat([p]))-lineupValue(r);

  function myRoster(){
    const out=[];
    state.picks.forEach((id,i)=>{if(id!==null&&owner(i+1)===state.seat)out.push(byId[id]);});
    return out;
  }

  // ---- recommendations for YOUR next pick ----
  // Availability now = unchecked players; at a future pick n (from current
  // pick c), remove the top (n - c) unchecked players by ADP — the live-board
  // generalization of vona.py's adp_rank >= nxt rule.
  function recommend(){
    const c=state.picks.length+1;
    const drafted=new Set(state.picks.filter(x=>x!==null));
    const avail=P.filter(p=>!drafted.has(p.id));          // ADP order
    const mine=nextMyPick(c);
    if(mine===null||c>N_PICKS)return {done:true};
    const following=nextMyPick(mine+1)||mine+L.teams;     // last round: +12, as vona.py
    const availAtMine=avail.slice(Math.min(avail.length,mine-c));
    const availAtFollowing=avail.slice(Math.min(avail.length,following-c));
    const roster=myRoster(), asn=assign(roster);

    const waitBest={QB:0,RB:0,WR:0,TE:0};
    availAtFollowing.forEach(p=>{const m=marginal(roster,p);if(m>waitBest[p.pos])waitBest[p.pos]=m;});

    let cands=availAtMine.map(p=>{
      const m=marginal(roster,p);
      return {p:p,m:m,wait:waitBest[p.pos],urg:m-waitBest[p.pos]};
    });
    // Bench mode = every starter slot (incl. FLEX) is filled. Gating on a small
    // max-marginal instead would flip early when the best remaining TE/QB has
    // near-zero VORP, and then rank a do-nothing 2nd QB above an unfilled slot.
    const filled=POS.reduce((n,ps)=>n+asn.starters[ps].length,0)+asn.flex.length;
    const total=POS.reduce((n,ps)=>n+(L.starters[ps]||0),0)+(L.starters.FLEX||0);
    const bench=filled>=total;
    if(bench){
      const cnt={};roster.forEach(p=>cnt[p.pos]=(cnt[p.pos]||0)+1);
      cands=cands.filter(x=>{const cap=L.caps[x.p.pos];return cap==null||(cnt[x.p.pos]||0)<cap;});
      cands.sort((a,b)=>b.p.vorp-a.p.vorp);
    }else{
      cands.sort((a,b)=>(b.urg-a.urg)||(b.m-a.m));
    }
    return {done:false,pick:mine,current:c,cands:cands.slice(0,5),bench:bench,
            roster:roster,asn:asn,waitBest:waitBest};
  }

  function reasonFor(x,asn,roster,bench){
    if(bench)return 'starters full — best value';
    const p=x.p,cap=L.starters[p.pos]||0;
    const cnt=roster.filter(q=>q.pos===p.pos).length;
    if(cnt<cap)return 'fills '+p.pos+(cap>1?String(cnt+1):'');
    if(L.flex.includes(p.pos)&&asn.flex.length<(L.starters.FLEX||0))return 'fills FLEX';
    if(x.m>0.05)return 'raises lineup +'+x.m.toFixed(1);
    return 'bench depth';
  }

  // ---- board view state ---- (default: our own rank, ascending)
  const view={q:'',pos:'ALL',hide:false,sortKey:'rank',sortDir:1};

  // ---- rendering ----
  function render(){
    const c=state.picks.length+1, done=c>N_PICKS;
    const r=Math.min(L.rounds,Math.ceil(Math.min(c,N_PICKS)/L.teams));
    const onClock=done?null:owner(c);
    const mine=nextMyPick(c);

    document.getElementById('sim-teams').value=String(state.teams);
    document.getElementById('sim-flex').value=String(state.flex);
    document.getElementById('sim-seat').value=String(state.seat);
    document.getElementById('sim-undo').disabled=state.picks.length===0;

    let st;
    if(done){st='<b>Draft complete</b> — '+state.picks.length+' picks recorded.';}
    else{
      st='Pick <b>'+c+'</b> (round '+r+') — on the clock: '+
        (onClock===state.seat?'<span class="you">YOU (seat '+state.seat+')</span>':'team '+onClock)+'.';
      st+=mine===null?' You have no picks left.':(mine===c?'':' Your next pick: <b>#'+mine+'</b> (in '+(mine-c)+' picks).');
    }
    document.getElementById('sim-status').innerHTML=st;

    // recommendations — compact div list
    const R=recommend();
    const recEl=document.getElementById('sim-recs'), modeEl=document.getElementById('sim-mode');
    const pcls=p=>({QB:'pos-qb',RB:'pos-rb',WR:'pos-wr',TE:'pos-te'}[p]||'');
    if(R.done||!R.cands||R.cands.length===0){
      modeEl.textContent='';
      recEl.innerHTML='<div class="rec" style="color:var(--faint)">'+(R.done?'draft complete':'no candidates')+'</div>';
    }else{
      modeEl.textContent=R.bench
        ?'bench mode — best value, caps 2 QB / 2 TE'
        :'for your pick #'+R.pick+(R.pick===R.current?' (now)':' (projected)');
      recEl.innerHTML=R.cands.map((x,i)=>{
        // headline number: urgency in fill-mode, raw VORP in bench-mode
        const head=R.bench?x.p.vorp.toFixed(1):x.urg.toFixed(1);
        const sub=R.bench
          ? 'VORP '+x.p.vorp.toFixed(1)
          : 'adds '+x.m.toFixed(1)+' &middot; if wait '+x.wait.toFixed(1);
        return '<div class="rec '+pcls(x.p.pos)+'">'+
          '<div class="rec-top"><span class="rec-rk">'+(i+1)+'</span>'+
          '<span class="rec-nm">'+esc(x.p.name)+'</span>'+
          '<span class="rec-pt">'+x.p.pos+'&middot;'+esc(x.p.team)+'</span>'+
          '<span class="rec-urg'+(!R.bench&&x.urg>0.05?' hi':'')+'">'+head+'</span></div>'+
          '<div class="rec-sub"><span>'+sub+'</span>'+
          '<span class="rec-why">'+esc(reasonFor(x,R.asn,R.roster,R.bench))+'</span></div></div>';
      }).join('');
    }

    // roster
    const roster=myRoster(), asn=assign(roster);
    const slots=[];
    POS.forEach(ps=>{const n=L.starters[ps]||0;for(let i=0;i<n;i++)
      slots.push([ps+(n>1?String(i+1):''),asn.starters[ps][i]||null,false]);});
    for(let i=0;i<(L.starters.FLEX||0);i++)slots.push(['FLEX',asn.flex[i]||null,false]);
    asn.bench.forEach(p=>slots.push(['BN',p,true]));
    document.getElementById('sim-roster').innerHTML=slots.map(s=>{
      const p=s[1];
      return '<div class="sim-slot'+(p?'':' empty')+(s[2]?' bnch':'')+'"><span class="sl">'+s[0]+'</span>'+
        '<span class="sp">'+(p?esc(p.name)+' <span class="tpos">'+p.pos+'</span>':'&mdash;')+'</span>'+
        (p?'<span class="sv">'+p.vorp.toFixed(1)+'</span>':'')+'</div>';
    }).join('')+(roster.length===0?'<div class="sim-slot empty"><span class="sp">no picks yet</span></div>':'');

    // drafted — compact names, newest first, my picks highlighted
    const drEl=document.getElementById('sim-drafted');
    if(drEl){
      const items=[];
      for(let i=state.picks.length-1;i>=0;i--){
        const id=state.picks[i]; if(id===null)continue;
        const pl=byId[id], pk=i+1, isMine=owner(pk)===state.seat;
        items.push('<span class="dr'+(isMine?' mine':'')+'"><span class="drk">'+pk+'</span> '+esc(pl.name)+'</span>');
      }
      drEl.innerHTML=items.length?items.join(', '):'<span style="color:var(--faint)">none yet</span>';
      const hd=document.getElementById('sim-drafted-n'); if(hd)hd.textContent=items.length;
    }

    // board — per-player VONA (urgency at your next pick); blank for drafted/no-context
    const pickOf={};state.picks.forEach((id,i)=>{if(id!==null)pickOf[id]=i+1;});
    const vonaOf=p=>{
      if(R.done||!R.waitBest||pickOf[p.id])return null;
      return Math.max(0, marginal(R.roster,p)-(R.waitBest[p.pos]||0));
    };
    const vona={}; P.forEach(p=>{vona[p.id]=vonaOf(p);});

    let rows=P.filter(p=>{
      if(view.pos!=='ALL'&&p.pos!==view.pos)return false;
      if(view.hide&&pickOf[p.id])return false;
      if(view.q){const q=view.q.toLowerCase();
        if(!(p.name.toLowerCase().includes(q)||p.team.toLowerCase().includes(q)))return false;}
      return true;
    });
    const sv=(p,k)=> k==='name'?p.name : k==='vona'?vona[p.id] : p[k];
    rows=rows.slice().sort((a,b)=>{
      let va=sv(a,view.sortKey), vb=sv(b,view.sortKey);
      const na=va==null, nb=vb==null;           // nulls (missing ECR, blank VONA) always last
      if(na&&nb)return 0; if(na)return 1; if(nb)return -1;
      if(typeof va==='string')return va.localeCompare(vb)*view.sortDir;
      return (va-vb)*view.sortDir;
    });
    // header sort indicator
    document.querySelectorAll('.sim-board th.sortable').forEach(th=>{
      const on=th.dataset.sort===view.sortKey;
      th.classList.toggle('sorted',on);
      th.querySelector('.arr')?.remove();
      if(on){const s=document.createElement('span');s.className='arr';s.textContent=view.sortDir>0?'\\u25B2':'\\u25BC';th.appendChild(s);}
    });
    document.getElementById('sim-count').textContent=rows.length+' / '+P.length;
    document.getElementById('sim-board').innerHTML=rows.map(p=>{
      const pk=pickOf[p.id], isMine=pk&&owner(pk)===state.seat, v=vona[p.id];
      return '<tr class="'+(pk?'drafted':'')+(isMine?' mine':'')+'">'+
        '<td class="chk"><input type="checkbox" data-id="'+esc(p.id)+'"'+(pk?' checked disabled':'')+(done?' disabled':'')+'></td>'+
        '<td class="num rk">'+p.rank+'</td>'+
        '<td class="pname">'+esc(p.name)+'</td>'+
        '<td class="tpos">'+p.pos+'</td><td>'+esc(p.team)+'</td>'+
        '<td class="num">'+p.adp+'</td>'+
        '<td class="num">'+(p.ecr==null?'&mdash;':p.ecr)+'</td>'+
        '<td class="num">'+p.vorp.toFixed(1)+'</td>'+
        '<td class="num">'+(v==null?'':v.toFixed(1))+'</td></tr>';
    }).join('')||'<tr><td colspan="9" style="padding:14px;color:var(--faint)">no rows</td></tr>';
  }

  // seat dropdown depends on team count; rebuild it when teams change
  function fillSeats(){
    const sel=document.getElementById('sim-seat');
    sel.innerHTML='';
    for(let i=1;i<=L.teams;i++){const o=document.createElement('option');o.value=String(i);o.textContent='seat '+i;sel.appendChild(o);}
    sel.value=String(state.seat);
  }

  // ---- events ----
  // Teams changes who owns each pick, so it invalidates a draft in progress -> reset.
  document.getElementById('sim-teams').addEventListener('change',e=>{
    const v=+e.target.value;
    if(state.picks.length && !confirm('Change to '+v+' teams? This resets the draft.')){
      e.target.value=String(state.teams); return;
    }
    state.teams=v; state.picks=[]; applyLeague(); fillSeats(); save(); render();
  });
  // FLEX only re-interprets the lineup; existing picks stay valid.
  document.getElementById('sim-flex').addEventListener('change',e=>{
    state.flex=+e.target.value; applyLeague(); save(); render();
  });
  document.getElementById('sim-seat').addEventListener('change',e=>{state.seat=+e.target.value;save();render();});
  document.getElementById('sim-undo').addEventListener('click',()=>{state.picks.pop();save();render();});
  document.getElementById('sim-skip').addEventListener('click',()=>{
    if(state.picks.length<N_PICKS){state.picks.push(null);save();render();}});
  document.getElementById('sim-reset').addEventListener('click',()=>{
    if(confirm('Reset the draft board?')){state.picks=[];save();render();}});
  document.getElementById('sim-board').addEventListener('change',e=>{
    const id=e.target&&e.target.dataset&&e.target.dataset.id;
    if(!id||!byId[id])return;
    if(state.picks.length<N_PICKS&&!state.picks.includes(id)){state.picks.push(id);save();render();}
  });
  document.getElementById('sim-search').addEventListener('input',e=>{view.q=e.target.value;render();});
  document.getElementById('sim-hide').addEventListener('change',e=>{view.hide=e.target.checked;render();});
  // sortable columns: same key flips direction; new key gets a sensible default
  // (rank/adp/ecr/name ascending = best first; vorp/vona descending = highest first)
  document.querySelectorAll('.sim-board th.sortable').forEach(th=>th.addEventListener('click',()=>{
    const k=th.dataset.sort;
    if(view.sortKey===k){view.sortDir*=-1;}
    else{view.sortKey=k; view.sortDir=(k==='vorp'||k==='vona')?-1:1;}
    render();
  }));
  document.querySelectorAll('#sim-posfilter button').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('#sim-posfilter button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');view.pos=b.dataset.pos;render();
  }));

  fillSeats();

  // mobile: pin recs directly under the (wrapping) sticky status bar — measure
  // its real height so the offset is exact regardless of how the bar wraps.
  function setStickyOffset(){
    const mh=parseInt(getComputedStyle(document.documentElement).getPropertyValue('--mh'))||47;
    const st=document.querySelector('.status');
    document.documentElement.style.setProperty('--st',(mh+(st?st.offsetHeight:0)+8)+'px');
  }
  setStickyOffset();
  window.addEventListener('resize',setStickyOffset);
  render();
})();
"""

# ---------------------------------------------------------------------------
# 3. PAGE RENDERERS
# ---------------------------------------------------------------------------

NAV = [("index.html", "Article"), ("rankings.html", "Rankings"),
       ("features.html", "Features"), ("methodology.html", "Methods"),
       ("draftsim.html", "Draft (beta)")]


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


def _page(title: str, active: str, body: str, data_slice: dict, page_js: str,
          extra_css: str = "", show_footer: bool = True) -> str:
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
        f'<style>{CSS}{extra_css}</style>\n</head>\n<body>\n'
        + _header(active)
        + '<main>' + body + '</main>'
        + (_footer(data_slice["meta"]) if show_footer else "")
        + '<script>' + blob + '</script>\n'
        + '<script>' + JS_COMMON + page_js + '</script>\n'
        + '</body>\n</html>\n'
    )


def _anchor_table(b: dict) -> str:
    """Table of ADP vs ECR baseline quality for one window.

    Uses `tbl` and the dpos/dneg delta colours, the two class names the CSS
    actually defines. It previously emitted `booktabs` and `pos`/`neg`, neither
    of which has a rule, so both anchor tables rendered unstyled with black
    deltas."""
    rows = "".join(
        f'<tr><td class="num">{r["season"]}</td>'
        f'<td class="num">{r["adp"]:.4f}</td>'
        f'<td class="num">{r["ecr"]:.4f}</td>'
        f'<td class="num {"dpos" if r["delta"] > 0 else "dneg"}">{r["delta"]:+.4f}</td>'
        f'<td>{"ECR" if r["delta"] > 0 else "ADP"}</td></tr>'
        for r in b["rows"]
    )
    return (
        '<div class="tablewrap"><table class="tbl">'
        '<thead><tr><th>Season</th><th class="num">ADP</th><th class="num">ECR</th>'
        '<th class="num">ECR &minus; ADP</th><th>Better</th></tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'<tfoot><tr class="summary"><td><b>mean</b></td>'
        f'<td class="num">{b["adp_mean"]:.4f}</td>'
        f'<td class="num">{b["ecr_mean"]:.4f}</td>'
        f'<td class="num {"dpos" if b["mean"] > 0 else "dneg"}">{b["mean"]:+.4f}</td>'
        f'<td>ECR {b["wins"]}/{b["n"]}</td></tr></tfoot>'
        '</table></div>'
    )


def render_index(p: dict) -> str:
    """The front page is an ARTICLE about the methodology audit, not a results
    dashboard. Every number is interpolated from the payload or from
    reports/methodology_audit.json (see _read_audit); nothing is typed by hand,
    so the page cannot drift from the artifacts."""
    m, au = p["meta"], p["audit"]
    ve, va = p["headline"]["vs_ecr"], p["headline"]["vs_adp"]
    at, ax, sp = p["anchor"]["tune"], p["anchor"]["test"], p["anchor"]["split"]
    rel, red, oos = au["reliability"], au["redundancy"], au["oos_residual"]
    geo, bu, pw = au["geometry"], au["bust"], au["power"]
    perm, cs, plv = au["permutation"], au["clip_sweep"], au["player_level"]["all"]
    win = au["window"]
    pe, pa = pw["vs_ecr"], pw["vs_adp"]
    tightest = cs["rows"][-1]
    fixed = cs["tightest_at_shipped_hyperparams"]
    perm_means = ", ".join(f'{r["mean"]:.4f}' for r in perm["runs"])
    sweep = " &rarr; ".join(f'{r["mean"]:.4f}' for r in cs["rows"])
    clips = ", ".join(f'{r["clip"]:g}' for r in cs["rows"])
    top24 = p["secondary"]["top24_hit_vorp"]

    body = (
        '<div class="wrap">'
        '<div class="titleblock">'
        '<h1 class="title">It&rsquo;s really hard to beat the ECR - 7/27</h1>'
        f'<p class="subtitle">What I learned trying to out-rank FantasyPros with '
        f'{win["n_features"]} features and {geo["n_rows"]:,} player-seasons: '
        'the experts are already very good, and eight seasons is not enough to '
        'prove otherwise.</p>'
        f'<p class="docmeta">{m["format"]} &middot; test seasons '
        f'{ve["span"].replace("-", "&ndash;")} &middot; graded on realized value '
        'over replacement &middot; every number on this page reproduces from '
        '<code>bff.methodology_audit</code></p>'
        '</div>'

        '<section><p>I built a preseason draft-ranking model, backtested it '
        'properly against the expert consensus, and it does not beat the expert '
        'consensus. That part is mildly disappointing and completely ordinary. '
        'What turned out to be interesting is <i>why</i>, and the fact that the '
        'usual way of checking (backtest a few seasons, look at the average, '
        'declare victory) cannot actually tell the difference between a real '
        'edge and a lucky one. Below is the whole thing, including the '
        'experiment that convinced me to stop.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">How the model works</h2>'
        '<p>It does not build a board from scratch. It starts from the '
        'FantasyPros expert consensus rank (ECR) and tries to correct it, '
        'which is a much easier job than ranking 150 players from nothing.</p>'
        '<p>Three steps. <b>One:</b> turn each player&rsquo;s expert rank into '
        'expected points, using history (what does the 5th-ranked WR usually '
        'score? the 30th?). <b>Two:</b> predict the <i>miss</i>. This is the '
        'model proper, and the input is '
        f'{win["n_features"]} things knowable in August: age and the shape of '
        'the aging curve at each position, how much a player outscored or '
        'undershot his draft price last year, games missed, injury history, '
        'targets and carries vacated by teammates who left, rookie draft '
        'capital, contract year, whether his quarterback changed, and so on. '
        '<b>Three:</b> convert points into value over replacement, because raw '
        'points lie across positions; the 10th-best QB outscores the 10th-best '
        'TE but is worth less, since a decent QB is always available.</p>'
        '<p>Step two is a <b>ridge regression</b>, and the choice matters for '
        f'everything that follows. With {geo["p"]} inputs and only '
        f'{geo["n_rows"]:,} training rows (really only {geo["n_players"]} '
        'distinct players, since a player shows up in several seasons), an '
        'ordinary regression would happily memorize noise: it would &ldquo;'
        'discover&rdquo; that 27-year-old tight ends on new teams bust, because '
        'three of them did. Ridge is the standard defence. It fits the '
        'regression while paying a penalty for large coefficients, so every '
        'effect gets pulled toward zero unless the data insists. One dial '
        '(<code>alpha</code>) sets how hard it pulls. This model then adds a '
        f'second dial: only {int(win["shrink"] * 100)}% of whatever correction '
        'survives is actually applied to the expert baseline. Both dials are '
        'chosen on an early block of seasons and then frozen before the test '
        'seasons are ever scored.</p>'
        f'<p class="note small">Where the dials landed is itself a result: '
        f'<code>alpha = {int(win["alpha"])}</code> is the strongest setting on '
        f'the grid and <code>shrink = {win["shrink"]}</code> the weakest. Of '
        f'{geo["p"]} inputs, that leaves about '
        f'{geo["effective_dof_x_shrink"]:g} inputs&rsquo; worth of real fitting. '
        'The search is turning the model down as far as it is allowed to.</p>'
        '</section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">The scoreboard</h2>'
        '<p>Champion versus challenger, one round per season. Three boards are '
        'fixed before Week 1 (this model, the expert consensus, and the market&rsquo;s '
        'average draft position), then graded after the season on how well the '
        'preseason order matched what players actually returned in value over '
        'replacement. Same grade for everyone, no board sees the season it '
        'predicts.</p>'
        '<p class="cap"><b>Table 1.</b> Model vs FantasyPros ECR, per season. '
        'Higher is better; 1.0 would be a perfect ordering, 0.0 a coin flip.</p>'
        '<div id="results"></div>'
        '<p class="note small" id="results-note" style="margin-top:8px"></p>'
        f'<p style="margin-top:14px">So: <b>beat the market in {va["wins"]} of '
        f'{va["n_seasons"]} seasons</b> ({va["model"]:.4f} vs {va["baseline"]:.4f}), '
        f'and <b>{ve["wins"]} of {ve["n_seasons"]} against the experts</b> '
        f'({ve["model"]:.4f} vs {ve["baseline"]:.4f}). Beating ADP is real. '
        'Against ECR it is a coin flip, and the right word for a coin flip is '
        '&ldquo;tied&rdquo;.</p>'
        f'<p>For scale, in football terms: this board puts about '
        f'{top24:.0%} of the eventual top-24 finishers in its own top 24. So does '
        'the expert list. Preseason ranking is a genuinely hard forecasting '
        'problem for everyone involved.</p></section>'

        # ------------------------------------------------------------------
        '<section id="anchor"><h2 class="numbered">Are the experts actually '
        'better than the market?</h2>'
        '<p>The model is built on top of ECR, so it inherits whatever edge ECR '
        'has. That makes one question worth asking on its own, with no model '
        'involved: graded on realized value, is the expert consensus a better '
        'preseason board than where people actually draft? The answer depends '
        'sharply on the era.</p>'
        f'<p class="cap"><b>Table 2.</b> {at["label"]}. Baseline boards only, '
        'no model, both through the identical value conversion.</p>'
        + _anchor_table(at) +
        f'<p class="note small" style="margin-top:8px">A dead heat. ECR wins '
        f'{at["wins"]} of {at["n"]} seasons and the mean gap is {at["mean"]:+.4f}, '
        'which is noise. One caveat on the earliest season, the largest single '
        'ADP win in the table: its ECR snapshot is the last one the Wayback '
        'Machine archived that preseason, roughly two weeks staler than the '
        'early-September boards every other season uses. A staler board is a '
        'plausible reason to underperform, so that row is weaker evidence than '
        'the rest.</p>'
        f'<p class="cap" style="margin-top:22px"><b>Table 3.</b> {ax["label"]}. '
        'Same computation.</p>'
        + _anchor_table(ax) +
        f'<p class="note small" style="margin-top:8px">Here the experts pull '
        f'clearly ahead: ECR wins {ax["wins"]} of {ax["n"]} seasons by a mean of '
        f'{ax["mean"]:+.4f}, and the margin grows over time. Whether that is '
        'FantasyPros getting better, the ADP sample thinning out, or both, this '
        'data cannot settle.</p>'
        '<p style="margin-top:18px"><b>What that implies about my model, and it '
        f'is not flattering.</b> On the test seasons the model beats ADP by '
        f'{sp["total"]:+.4f}. But ECR on its own, with no model attached, already '
        f'beats ADP by {sp["from_anchor"]:+.4f} over those same seasons. The '
        f'regression contributes the remaining {sp["from_model"]:+.4f}. So '
        'roughly five sixths of my margin over the market is the expert list I '
        'started from, not the corrections I added.</p>'
        f'<p>It also resolves a puzzle. The model shows no edge at all on its own '
        f'tuning seasons ({win["shipped_tune_mean"]:.4f} against ADP&rsquo;s '
        f'{at["adp_mean"]:.4f}) yet {sp["total"]:+.4f} on test. Not a '
        'contradiction: in the tuning years the expert anchor was worth nothing '
        'over ADP, so a model standing on it had nothing inherited to win with. '
        f'In the test years the anchor was worth {sp["from_anchor"]:+.4f} before '
        'the model did anything at all. What changed between windows is not my '
        'model; it is what my model is standing on.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">Is a fantasy season just noise?</h2>'
        '<p>The obvious excuse for a mediocre score is that the outcome is '
        'random, so nobody could do better. I checked, and it is not true.</p>'
        '<p>Split each season into odd and even weeks and rank the same players '
        'twice. If seasons were mostly luck the two halves would disagree. They '
        f'agree at {rel["r_half_mean"]:.2f}, which corrects to about '
        f'{rel["r_full_mean"]:.2f} for a full season. In other words the thing '
        'being graded is measured well, and the distance between a 0.52 and a '
        'perfect board is real forecasting error, not scoring slop. There is '
        'genuine headroom up there. Nobody, expert or model, is reaching '
        f'it.</p><p class="note small">Reliability by position: '
        + ", ".join(f'{k} {v:.2f}' for k, v in rel["by_position"].items())
        + '. This is an upper bound, incidentally: a season-ending injury looks '
        '&ldquo;consistent&rdquo; across both halves even though it was '
        'unforecastable in August.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">Why the experts are so hard to beat</h2>'
        f'<p>Here is the number that reframed the project for me. Take my '
        f'features and use them to predict <i>the expert consensus itself</i> '
        f'instead of the season outcome. They explain '
        f'{red["features_to_anchor"]:.0%} of it.</p>'
        f'<p class="note small">That uses {red["features_to_anchor_n"]} of the '
        f'{win["n_features"]}. One feature is excluded on purpose: it is built '
        'from the expert rank itself (last season&rsquo;s points per game minus '
        'what the expert rank implied), so letting it predict the expert rank '
        'would be circular. Including it would read '
        f'{red["features_to_anchor_incl_anchor_derived"]:.0%} instead.</p>'
        '<p>Which makes sense once you say it out loud. ECR is roughly a hundred '
        'analysts reading the same box scores, the same depth charts, the same '
        'ages and the same beat reports I am feeding my regression. Age curves '
        'and vacated targets and rookie draft capital are not secrets. So '
        'two-thirds of my &ldquo;model&rdquo; is a reconstruction of work the '
        'experts already did, and only the leftover third is even eligible to be '
        'an edge.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">And there is nothing in the '
        'leftovers</h2>'
        '<p>That leftover third is exactly what the regression is trained on: '
        'not &ldquo;how many points will he score&rdquo; but &ldquo;where are '
        'the experts wrong&rdquo;. So I measured whether it gets that right on '
        'seasons it had never seen, fold by fold.</p>'
        f'<p>The correlation between the predicted miss and the actual miss is '
        f'<b>{oos["r_mean"]:.3f}</b>. Not 0.3, not 0.15. About 0.07, and '
        f'negative in {oos["n_folds_negative"]} of the {len(oos["folds"])} '
        'seasons tested. Measured as variance explained it is essentially zero '
        f'({oos["r2_oos_mean"]:+.4f}, i.e. very slightly worse than predicting '
        'nothing). In sample it looks like it is working, explaining '
        f'{red["features_to_residual_in_sample"]:.0%} of the miss; out of sample '
        'that evaporates. Which is the whole reason to insist on out-of-sample '
        'testing, and also why the tuning dials ended up pinned at &ldquo;trust '
        'this as little as possible&rdquo;.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">The experiment that convinced me: '
        'scrambled data</h2>'
        '<p>This is the one to remember if you take nothing else from the '
        'post.</p>'
        '<p>I took the training data and <b>shuffled the outcomes</b> within '
        'each season, so that a player&rsquo;s features were attached to a '
        'random other player&rsquo;s result. Every real relationship destroyed '
        'on purpose. Then I ran the identical pipeline: same ridge, same dials, '
        'same value conversion, same grading.</p>'
        f'<p>Three scrambled runs scored <b>{perm_means}</b>. My real model '
        f'scores <b>{perm["shipped_mean"]:.4f}</b>. Doing nothing at all, just '
        f'ranking by the expert anchor with no regression, scores '
        f'<b>{perm["null_mean"]:.4f}</b>.</p>'
        f'<p>One of the scrambled runs beat my real model by '
        f'{perm["best_over_shipped"]:+.4f}, which is larger than the entire edge '
        f'over ECR I would otherwise be claiming ({pe["mean"]:+.4f}). Read that '
        'as the measuring instrument rattling: pure noise fed through this '
        'pipeline lands in the same neighbourhood as real signal, so the '
        'neighbourhood is where the argument dies. Any &ldquo;improvement&rdquo; '
        'smaller than that rattle is unprovable, and nearly every improvement '
        'anyone reports in this hobby is smaller than that rattle.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">Even a real edge would be '
        'invisible</h2>'
        f'<p>Suppose the model really is {pe["mean"]:+.4f} better than the '
        'experts. Could this test see it? No. Season-to-season the gap bounces '
        f'around with a spread of {pe["sd"]:.4f}, so with '
        f'{pe["n"]} seasons the smallest edge detectable at conventional '
        f'standards is about <b>{pe["mde_80"]:+.4f}</b>, three times the effect. '
        f'To reliably confirm an edge this size I would need roughly '
        f'<b>{pe["seasons_needed_80"]} seasons</b> of NFL history graded the same '
        'way. I have eight, and I get one more per year.</p>'
        f'<p>The same arithmetic undercuts my nicer-looking result. Beating ADP '
        f'by {pa["mean"]:+.4f} in {pa["wins"]} of {pa["n"]} seasons gives a '
        f'one-sided p-value of {pa["p_one"]:g}, which looks publishable, except '
        f'that the smallest edge this design can detect is {pa["mde_80"]:+.4f}. '
        'When a test clears the bar on an effect it was not powerful enough to '
        'find, that is a lucky draw more than a demonstration.</p>'
        '<p><b>And it poisoned my feature selection.</b> My rule for keeping a '
        'new group of features was that it had to improve the validation score '
        'by +0.0020. But changing a single setting moves that score by '
        f'{fixed["paired_vs_shipped"]["se"]:.4f} to '
        f'{tightest["paired_vs_shipped"]["se"]:.4f} run to run. The bar was '
        'smaller than the wobble, which means every keep-or-drop call I made was '
        'roughly a coin flip. Two of them can be checked, because I later scored '
        'them on the test seasons: a player-trajectory block passed validation '
        'at +0.0059 and delivered +0.0003, and a wider hyperparameter search '
        'passed at +0.0068 and delivered &minus;0.0037. Two for two in the wrong '
        'direction. That is what coin flips look like.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">One thing I did find genuinely '
        'wrong</h2>'
        '<p>Not everything here is a null result. Auditing the loss function '
        'turned up a real design flaw.</p>'
        '<p>The model is <i>graded</i> on getting the order right, but it is '
        '<i>fit</i> by minimizing squared error on points. Those are not the '
        f'same target, and here is the damage: the {bu["share_under_50_pts"]:.1%} '
        'of players who got hurt or busted (under 50 points on the season) '
        f'account for <b>{bu["sq_error_share_of_under_50"]:.1%} of the total '
        'error the regression is trying to reduce</b>. So the model spends '
        'nearly three-quarters of its effort trying to predict which stars will '
        'tear an ACL, which nobody can do, and a quarter on ordering the healthy '
        'middle of the board, which is the actual job. There is a safety valve '
        f'meant to contain this, capping any single player&rsquo;s error at '
        f'{bu["clip"]:g} on the log scale, but it only ever triggers on '
        f'{bu["share_clip_binds"]:.1%} of players; it is set so loose it does '
        'nothing.</p>'
        f'<p>Tightening that cap helps, and monotonically: {sweep} as the cap '
        f'goes {clips}. The tightest setting is the first version of this model '
        'that beats the experts, the market, and doing-nothing simultaneously on '
        'the validation window. But I am not claiming it, because I just spent '
        'this whole post explaining why numbers that size are not claimable: '
        f'{tightest["paired_vs_shipped"]["wins"]} of '
        f'{tightest["paired_vs_shipped"]["n"]} seasons improve, one season '
        'supplies most of the gain, and holding the other dials fixed the cap '
        f'alone is worth {fixed["paired_vs_shipped"]["mean"]:+.4f}. It is a good '
        'hypothesis with a mechanism behind it. It is not a result.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">So what actually helps your draft?</h2>'
        '<p>Three things I would still defend, none of which are &ldquo;I have '
        'better player opinions than the experts&rdquo;.</p>'
        '<p><b>1. Convert to value over replacement, not points.</b> This is '
        'where nearly all of the real movement on my board comes from, and it is '
        'a modelling choice rather than a prediction. A tight end who scores 15 '
        'fewer points than a quarterback can be worth much more, because the '
        'quarterback you would otherwise stream is nearly as good and the tight '
        'end you would otherwise stream is not. Getting the exchange rate right '
        'between positions beats getting player 43 versus player 47 right.</p>'
        '<p><b>2. Draft timing is a separate question from value.</b> Who is '
        'worth the most and who you should take now are different, because '
        'positions dry up at different speeds. That is where running-back '
        'scarcity legitimately lives, and it is on the '
        '<a href="draftsim.html">draft page</a> rather than in the rankings.</p>'
        '<p><b>3. Treat any model board, including mine, as the expert '
        'consensus with a tilt.</b> That is what it is, measurably. '
        f'The <a href="rankings.html">{m["season"]} board</a> is public, every '
        'player row shows exactly which features moved him and by how much, and '
        'if a tilt looks stupid to you, it probably is; the evidence that it '
        'knows better than you is not there.</p>'
        f'<p>One more measurement worth stealing if you do this yourself. '
        'Comparing two boards by their season-average correlation throws away '
        'almost everything: a whole season of ~150 players collapses into one '
        'number, and then you have eight numbers. Compare them <i>per player</i> '
        'instead, asking how many ranks each board missed by. On my validation '
        f'seasons that is {plv["err_model"]:.1f} ranks of average miss for the '
        f'model against {plv["err_ecr"]:.1f} for the experts across '
        f'{plv["n"]:,} player-seasons, with the difference pinned inside about a '
        f'rank ([{plv["ci_lo"]:+.2f}, {plv["ci_hi"]:+.2f}]). Same conclusion, but '
        'it can see effects the season-average version never will.</p></section>'

        # ------------------------------------------------------------------
        '<section><h2 class="numbered">Method, and what to distrust</h2>'
        f'<p class="note">Value over replacement uses QB8 / RB30 / WR36 / TE8 '
        f'(QB and TE streaming-aware). Predictions for season <i>t</i> use only '
        'seasons before <i>t</i> plus season-<i>t</i> preseason facts: expert '
        'ranks, draft position, rosters, the April draft. No leakage, and the '
        'scrambled-data test above doubles as a check on that.</p>'
        '<ol class="method" id="method"></ol>'
        f'<p class="note small">The honest caveats: I have looked at the test '
        'seasons more than a pre-registered study would allow, so treat the '
        'p-values as descriptive. The value conversion is fixed for 12-team '
        'one-quarterback PPR and would move in other formats. Some rookies run '
        'on partly empty opportunity features. And the residual cap discussed '
        'above is still set where it was, not where the audit suggests.</p>'
        '<p class="note small">Full detail: <a href="methodology.html">methods</a>. '
        'Board: <a href="rankings.html">rankings</a>. Coefficients: '
        '<a href="features.html">features</a>. Every figure on this page is '
        'generated from the model artifacts by <code>bff.site</code> and '
        '<code>bff.methodology_audit</code>; none of it is typed in by '
        'hand.</p></section>'
        '</div>'
    )
    data = {"meta": p["meta"], "headline": p["headline"], "anchor": p["anchor"],
            "seasons": p["seasons"], "method": p["method"]}
    return _page("It&rsquo;s really hard to beat the ECR &mdash; betterffrank",
                 "index.html", body, data, JS_INDEX)


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


def render_draftsim(p: dict) -> str:
    """The site's Draft page ("Draft (beta)" in NAV): live draft simulator
    over the same board as rankings. Presentation only — never feeds anything
    scored."""
    body = (
        '<div class="sim-app">'

        # --- status bar (spans top) ---
        '<div class="status">'
        '<select id="sim-teams" aria-label="teams in league" title="teams in league">'
        '<option value="8">8 tm</option><option value="10">10 tm</option>'
        '<option value="12">12 tm</option></select>'
        '<select id="sim-flex" aria-label="flex spots" title="flex spots">'
        '<option value="1">1 FLEX</option><option value="2">2 FLEX</option></select>'
        '<select id="sim-seat" aria-label="your draft seat"></select>'
        '<button id="sim-undo">undo</button>'
        '<button id="sim-skip" title="advance the pick counter for a K/DST or a player not on this board">off-board</button>'
        '<button id="sim-reset">reset</button>'
        '<span class="sim-status" id="sim-status"></span>'
        '</div>'

        # --- roster rail (left) ---
        '<div class="panel p-roster">'
        '<div class="panel-hd">Your roster</div>'
        '<div class="panel-bd sim-roster" id="sim-roster"></div>'
        '</div>'

        # --- board (center) ---
        '<div class="panel p-board">'
        '<div class="panel-hd">Board <span class="hd-n" id="sim-count"></span></div>'
        '<div class="sim-controls">'
        '<input type="search" id="sim-search" placeholder="name or team" aria-label="filter">'
        '<span class="posfilter" id="sim-posfilter">'
        '<button data-pos="ALL" class="active">all</button>'
        '<button data-pos="QB">QB</button><button data-pos="RB">RB</button>'
        '<button data-pos="WR">WR</button><button data-pos="TE">TE</button></span>'
        '<label class="toggle"><input type="checkbox" id="sim-hide"> hide drafted</label>'
        '</div>'
        '<div class="panel-bd">'
        '<table class="sim-board"><thead><tr>'
        '<th></th>'
        '<th class="num sortable" data-sort="rank" title="our model rank">#</th>'
        '<th class="sortable" data-sort="name">Player</th><th>Pos</th><th>Team</th>'
        '<th class="num sortable" data-sort="adp" title="average draft position">ADP</th>'
        '<th class="num sortable" data-sort="ecr" title="expert consensus rank">ECR</th>'
        '<th class="num sortable" data-sort="vorp" title="value over replacement">VORP</th>'
        '<th class="num sortable" data-sort="vona" title="value lost by waiting to your next pick">VONA</th>'
        '</tr></thead>'
        '<tbody id="sim-board"></tbody></table></div>'
        '</div>'

        # --- right column: recs (top) + drafted (fills), placed by grid ---
        '<div class="panel p-recs">'
        '<div class="panel-hd">Next best <span class="sim-mode" id="sim-mode"></span></div>'
        '<div class="panel-bd" id="sim-recs"></div>'
        '</div>'
        '<div class="panel p-drafted">'
        '<div class="panel-hd">Drafted <span class="hd-n" id="sim-drafted-n">0</span></div>'
        '<div class="panel-bd sim-drafted" id="sim-drafted"></div>'
        '</div>'

        # --- collapsible how-it-works (kept out of the viewport) ---
        '<details class="sim-details"><summary>how recommendations work</summary>'
        '<p>Each available player is scored by the VORP he adds to <i>your starting lineup</i> '
        '(1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX) given who you already have. <b>Urgency</b> is what '
        'you lose by waiting &mdash; his lineup gain now minus the best same-position gain still '
        'available at your next turn (opponents drafting by ADP). Once every starter slot is '
        'filled it switches to best raw VORP, capped at 2 QB / 2 TE. Check a player off as he is '
        'drafted (in order); use <b>undo</b> to correct, <b>off-board</b> for a K/DST or a player '
        'not on this board. State is saved locally.</p></details>'

        '</div>'
    )
    data = {"meta": p["meta"], "draftsim": p["draftsim"]}
    return _page("betterffrank — draft simulator", "draftsim.html", body, data,
                 JS_DRAFTSIM, CSS_DRAFTSIM, show_footer=False)


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
        "draftsim.html": render_draftsim(payload),
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
