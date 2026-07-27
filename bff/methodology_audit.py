"""Methodology audit: how well can this experiment measure anything at all?

WHAT IT DOES. Nine diagnostics of the *procedure* rather than the model, all
computed on the tuning window (2013-2017), on training data, or as arithmetic
over the already-published `reports/scores_*.csv`. **No test look is spent** --
nothing here scores a new quantity on 2018-2025.

  reliability    split-half (odd vs even weeks) repeatability of the OUTCOME.
                 If the thing being graded were mostly noise, no forecast could
                 score well; it is not (0.89), so the observed error is real
                 forecast error, not measurement error.
  redundancy     R2 of the 53 features predicting the market ANCHOR. High means
                 the experts already price what the model knows.
  oos_residual   walk-forward correlation between the ridge's predicted
                 expert-miss and the actual expert-miss. This is the direct
                 measurement of "is there signal left over".
  geometry       ridge condition number and effective degrees of freedom -- how
                 much model the tuned alpha/shrink actually permits.
  bust           where the squared-error loss is spent. The fit minimizes
                 squared error on log points while the grade is an ordering,
                 and the bust tail dominates the former.
  power          minimum detectable effect of the S=8 sign-flip design, and how
                 many seasons the claimed effect would need.
  permutation    shuffle the training target within season and re-run. A
                 permuted model should land at the anchor-only null; how far it
                 SCATTERS is the noise floor of the whole evaluation.
  clip_sweep     tightening the residual clip (the `bust` finding's fix),
                 paired per fold against the shipped configuration.
  player_level   a per-player paired estimand instead of 8 season means, with a
                 season-clustered bootstrap -- the design that would have the
                 resolution the season-mean design lacks.

The clip sweep needs a fitter that varies the residual clip, which
`bff.model.fit_predict` hardcodes at +/-4, so `_fit` below mirrors that
function's matrix assembly (INCLUDING the contract that ppg_mismatch is the
last column). `_assert_reproduces_shipped` pins the copy to the real model:
clip=4 at the tuned hyperparameters must reproduce `bff.model.tune`'s mean
exactly, or this module refuses to write.

Deterministic: fixed seeds for the permutation and the bootstrap, so the JSON
is byte-stable across runs.

Run: uv run python -m bff.methodology_audit   -> reports/methodology_audit.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr, t as tdist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import bff.model as M
from bff.backtest import POSITIONS, build_pool, eval_season
from bff.compare import sign_flip_test

OUT = M.ROOT / "reports" / "methodology_audit.json"
PERMUTE_SEEDS = (0, 1, 2)
BOOT_N = 4000
BOOT_SEED = 0
CLIPS = (4.0, 2.0, 1.0, 0.5)
SHIPPED_CLIP = 4.0          # the value hardcoded in bff.model.fit_predict


# ---------------------------------------------------------------------------
# a local copy of fit_predict's matrix assembly, with the clip exposed
# ---------------------------------------------------------------------------

def _design(train: pl.DataFrame, test: pl.DataFrame):
    """(Xtr, Xte, tr_implied, te_implied) exactly as bff.model.fit_predict
    builds them. ppg_mismatch is derived here and appended LAST (contract)."""
    tr_imp = M.implied_expectation(train, train)
    te_imp = M.implied_expectation(train, test)
    sch_tr = np.where(train["season"].to_numpy() >= 2021, 17.0, 16.0)
    sch_te = np.where(test["season"].to_numpy() >= 2021, 17.0, 16.0)
    trm = (train["prev_ppg"].to_numpy() - np.expm1(tr_imp) / sch_tr) \
        * train["has_prior"].to_numpy()
    tem = (test["prev_ppg"].to_numpy() - np.expm1(te_imp) / sch_te) \
        * test["has_prior"].to_numpy()
    feats = [f for f in M.FEATURES if f != "ppg_mismatch"]
    Xtr = np.column_stack([train.select(feats).to_numpy(), trm])
    Xte = np.column_stack([test.select(feats).to_numpy(), tem])
    return Xtr, Xte, tr_imp, te_imp


def _split(df: pl.DataFrame, t: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter((pl.col("season") >= M.FIRST_TARGET) & (pl.col("season") < t))
    return train, df.filter(pl.col("season") == t)


def _fit(df: pl.DataFrame, t: int, alpha: float, shrink: float, clip: float,
         permute_seed: int | None = None) -> pl.DataFrame:
    """Walk-forward fit/score for season t with an adjustable residual clip."""
    train, test = _split(df, t)
    Xtr, Xte, tr_imp, te_imp = _design(train, test)
    resid = train["log_pts"].to_numpy() - tr_imp
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        seas = train["season"].to_numpy()
        for s in np.unique(seas):
            m = seas == s
            resid[m] = rng.permutation(resid[m])
    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=alpha).fit(scaler.transform(Xtr), np.clip(resid, -clip, clip))
    pred = model.predict(scaler.transform(Xte))
    return test.select("season", "gsis_id").with_columns(
        pl.Series("score", te_imp + shrink * pred))


def _folds(df: pl.DataFrame, alpha: float, shrink: float, clip: float,
           permute_seed: int | None = None) -> np.ndarray:
    """spearman_vorp per tune fold, through the canonical scoring pipeline."""
    adp, ecr, actuals, hist, pools = M.scoring_context()
    return np.array([
        eval_season(pools[s],
                    M.to_vorp(_fit(df, s, alpha, shrink, clip, permute_seed),
                              s, adp, ecr, hist),
                    actuals, s)["spearman_vorp"]
        for s in M.TUNE_SEASONS
    ])


def _grid_best(df: pl.DataFrame, clip: float) -> tuple[float, float, np.ndarray]:
    """Best (alpha, shrink) on the FROZEN grid for a given clip."""
    best = (-np.inf, None, None)
    for alpha in M.ALPHA_GRID:
        for shrink in M.SHRINK_GRID:
            f = _folds(df, alpha, shrink, clip)
            if f.mean() > best[0]:
                best = (f.mean(), (alpha, shrink), f)
    return best[1][0], best[1][1], best[2]


def _assert_reproduces_shipped(df: pl.DataFrame, alpha: float, shrink: float,
                               shipped_mean: float) -> np.ndarray:
    """_fit at the shipped clip must equal bff.model.fit_predict, or the local
    copy has drifted and every number below is suspect."""
    f = _folds(df, alpha, shrink, SHIPPED_CLIP)
    assert abs(f.mean() - shipped_mean) < 1e-12, (
        f"local fitter does not reproduce bff.model: {f.mean():.12f} "
        f"vs {shipped_mean:.12f}")
    return f


def _paired(delta: np.ndarray) -> dict:
    p_one, _ = sign_flip_test(delta.tolist())
    return {"mean": round(float(delta.mean()), 6),
            "sd": round(float(delta.std(ddof=1)), 6),
            "se": round(float(delta.std(ddof=1) / np.sqrt(len(delta))), 6),
            "wins": int((delta > 0).sum()), "n": int(len(delta)),
            "p_one": round(float(p_one), 4),
            "per_fold": [round(float(x), 6) for x in delta]}


# ---------------------------------------------------------------------------
# the nine diagnostics
# ---------------------------------------------------------------------------

def reliability() -> dict:
    """Split-half (odd vs even weeks) repeatability of the scored pool's season
    ranking. Spearman-Brown corrects a half-season r up to a full season.

    NOTE this is an UPPER bound on how repeatable the outcome is: a season-long
    injury is consistent across both halves, so it reads as 'reliable' variance
    even though it is unforecastable in August."""
    adp = pl.read_parquet(M.PROC / "adp.parquet")
    ecr = pl.read_parquet(M.PROC / "ecr.parquet")
    seasons = sorted(set(adp["season"].unique().to_list()))
    rows = []
    for s in [x for x in seasons if x >= min(M.TUNE_SEASONS)]:
        wk = M.RAW_STATS / f"player_stats_{s}.parquet"
        if not wk.exists():
            wk = M.RAW_STATS / f"stats_player_week_{s}.parquet"
        if not wk.exists():
            continue
        pool, _ = build_pool(s, adp, ecr)
        if pool.height == 0:
            continue
        w = (
            pl.read_parquet(wk)
            .filter(pl.col("season_type") == "REG")
            .select(pl.col("player_id").alias("gsis_id"), "week",
                    pl.col("fantasy_points_ppr").fill_null(0.0).alias("pts"))
            .join(pool.select("gsis_id", "position"), on="gsis_id", how="inner")
        )
        halves = [
            w.filter(pl.col("week") % 2 == k).group_by("gsis_id")
             .agg(pl.col("pts").sum().alias(nm))
            for k, nm in ((1, "odd"), (0, "even"))
        ]
        j = pool
        for h in halves:
            j = j.join(h, on="gsis_id", how="left")
        # a missed season IS the outcome: absent from the weekly table -> 0
        j = j.with_columns(pl.col("odd").fill_null(0.0), pl.col("even").fill_null(0.0))
        r_half = float(spearmanr(j["odd"], j["even"]).statistic)
        row = {"season": int(s), "n": j.height, "r_half": round(r_half, 4),
               "r_full": round(2 * r_half / (1 + r_half), 4)}
        for pos in POSITIONS:
            sub = j.filter(pl.col("position") == pos)
            rp = float(spearmanr(sub["odd"], sub["even"]).statistic)
            row[f"r_full_{pos}"] = round(2 * rp / (1 + rp), 4)
        rows.append(row)
    rf = np.array([r["r_full"] for r in rows])
    return {"rows": rows,
            "r_half_mean": round(float(np.mean([r["r_half"] for r in rows])), 4),
            "r_full_mean": round(float(rf.mean()), 4),
            "ceiling_mean": round(float(np.sqrt(np.clip(rf, 0, None)).mean()), 4),
            "by_position": {p: round(float(np.mean([r[f"r_full_{p}"] for r in rows])), 4)
                            for p in POSITIONS}}


def redundancy(df: pl.DataFrame) -> dict:
    """How much of the market anchor do the features already explain?"""
    tr = df.filter((pl.col("season") >= M.FIRST_TARGET)
                   & (pl.col("season") <= max(M.TUNE_SEASONS)))
    feats = [f for f in M.FEATURES if f in df.columns]
    X = StandardScaler().fit_transform(tr.select(feats).to_numpy())
    y = tr["log_pts"].to_numpy()
    anchor = tr["log_rank"].to_numpy().reshape(-1, 1)
    resid = y - M.implied_expectation(tr, tr)

    def r2(A, b):
        return round(float(LinearRegression().fit(A, b).score(A, b)), 4)

    return {"n_rows": tr.height, "n_features": len(feats),
            "features_to_anchor": r2(X, tr["log_rank"].to_numpy()),
            "features_to_outcome": r2(X, y),
            "anchor_linear_to_outcome": r2(anchor, y),
            "anchor_quadratic_to_outcome": round(1 - float(resid.var() / y.var()), 4),
            "anchor_plus_features_to_outcome": r2(np.column_stack([X, anchor]), y),
            "features_to_residual_in_sample": r2(X, resid)}


def oos_residual(df: pl.DataFrame, alpha: float, shrink: float) -> dict:
    """Does the ridge predict the expert-miss out of sample? (walk-forward)"""
    rows = []
    for t in M.TUNE_SEASONS:
        train, test = _split(df, t)
        Xtr, Xte, tr_imp, te_imp = _design(train, test)
        y_tr = train["log_pts"].to_numpy() - tr_imp
        y_te = test["log_pts"].to_numpy() - te_imp
        sc = StandardScaler().fit(Xtr)
        mo = Ridge(alpha=alpha).fit(sc.transform(Xtr),
                                    np.clip(y_tr, -SHIPPED_CLIP, SHIPPED_CLIP))
        p_te, p_tr = mo.predict(sc.transform(Xte)), mo.predict(sc.transform(Xtr))
        rows.append({
            "fold": int(t), "n_train": train.height, "n_test": test.height,
            "sd_residual": round(float(y_te.std()), 4),
            "r_pred_actual": round(float(np.corrcoef(p_te, y_te)[0, 1]), 4),
            "r2_oos": round(float(1 - ((y_te - shrink * p_te) ** 2).sum()
                                  / ((y_te - y_te.mean()) ** 2).sum()), 4),
            "r2_in": round(float(1 - ((y_tr - p_tr) ** 2).sum()
                                 / ((y_tr - y_tr.mean()) ** 2).sum()), 4)})
    return {"folds": rows,
            "r_mean": round(float(np.mean([r["r_pred_actual"] for r in rows])), 4),
            "r2_oos_mean": round(float(np.mean([r["r2_oos"] for r in rows])), 4),
            "n_folds_negative": int(sum(1 for r in rows if r["r2_oos"] < 0))}


def geometry(df: pl.DataFrame, alpha: float, shrink: float) -> dict:
    """How much model does the tuned regularization actually permit?"""
    tr = df.filter((pl.col("season") >= M.FIRST_TARGET)
                   & (pl.col("season") <= max(M.TUNE_SEASONS)))
    feats = [f for f in M.FEATURES if f in df.columns]
    X = StandardScaler().fit_transform(tr.select(feats).to_numpy())
    sv = np.linalg.svd(X, compute_uv=False)
    edof = float((sv ** 2 / (sv ** 2 + alpha)).sum())
    return {"n_rows": tr.height, "n_players": tr["gsis_id"].n_unique(),
            "n_seasons": tr["season"].n_unique(), "p": X.shape[1],
            "condition_number": round(float(sv.max() / sv.min()), 2),
            "alpha": alpha, "shrink": shrink,
            "effective_dof": round(edof, 1),
            "effective_dof_x_shrink": round(edof * shrink, 1)}


def bust(df: pl.DataFrame) -> dict:
    """Where the squared-error loss is actually spent."""
    tr = df.filter((pl.col("season") >= M.FIRST_TARGET)
                   & (pl.col("season") <= max(M.TUNE_SEASONS)))
    y = tr["log_pts"].to_numpy()
    resid = y - M.implied_expectation(tr, tr)
    pts = np.expm1(y)
    lo = pts < 50
    return {"n_rows": int(len(y)),
            "share_under_50_pts": round(float(lo.mean()), 4),
            "share_under_20_pts": round(float((pts < 20).mean()), 4),
            "sq_error_share_of_under_50": round(
                float((resid[lo] ** 2).sum() / (resid ** 2).sum()), 4),
            "share_clip_binds": round(float((np.abs(resid) > SHIPPED_CLIP).mean()), 4),
            "share_resid_over_2": round(float((np.abs(resid) > 2).mean()), 4),
            "clip": SHIPPED_CLIP}


def power() -> dict:
    """Resolution of the S=8 sign-flip design, from the PUBLISHED scores."""
    def series(name):
        d = pl.read_csv(M.ROOT / "reports" / f"scores_{name}.csv")
        d = d.filter((pl.col("metric") == "spearman_vorp")
                     & pl.col("season").cast(pl.Utf8).str.contains(r"^\d+$"))
        return d.with_columns(pl.col("season").cast(pl.Int64)).sort("season")

    sm, se_, sa = series("model"), series("ecr"), series("adp")
    seasons = sm["season"].to_list()
    m, e, a = (x["value"].to_numpy() for x in (sm, se_, sa))

    def one(delta, label):
        n = len(delta)
        sd = float(delta.std(ddof=1))
        se = sd / np.sqrt(n)
        p_one, p_two = sign_flip_test(delta.tolist())
        # paired, one-sided alpha=0.05, 80% power
        crit = tdist.ppf(0.95, n - 1) + tdist.ppf(0.80, n - 1)
        mde = crit * se
        need = None
        if delta.mean() > 0:
            for k in range(3, 100_001):
                c = tdist.ppf(0.95, k - 1) + tdist.ppf(0.80, k - 1)
                if c * sd / np.sqrt(k) <= delta.mean():
                    need = k
                    break
        return {"label": label, "mean": round(float(delta.mean()), 4),
                "sd": round(sd, 4), "se": round(se, 4),
                "wins": int((delta > 0).sum()), "n": n,
                "p_one": round(float(p_one), 4), "p_two": round(float(p_two), 4),
                "mde_80": round(float(mde), 4),
                "detected_below_own_mde": bool(delta.mean() < mde),
                "seasons_needed_80": need,
                "per_season": [round(float(x), 4) for x in delta]}

    return {"seasons": [int(x) for x in seasons],
            "sign_flip_floor": round(1 / 2 ** len(seasons), 4),
            "vs_ecr": one(m - e, "model - ECR"),
            "vs_adp": one(m - a, "model - ADP"),
            "ecr_vs_adp": one(e - a, "ECR - ADP")}


def permutation(df: pl.DataFrame, alpha: float, shrink: float,
                shipped: np.ndarray, null_mean: float) -> dict:
    """Shuffle the training target within season; the scatter is the noise floor."""
    runs = []
    for seed in PERMUTE_SEEDS:
        f = _folds(df, alpha, shrink, SHIPPED_CLIP, permute_seed=seed)
        runs.append({"seed": seed, "mean": round(float(f.mean()), 4),
                     "per_fold": [round(float(x), 4) for x in f]})
    means = np.array([r["mean"] for r in runs])
    return {"runs": runs, "shipped_mean": round(float(shipped.mean()), 4),
            "null_mean": round(null_mean, 4),
            "spread": round(float(means.max() - means.min()), 4),
            "n_beating_shipped": int((means > shipped.mean()).sum()),
            "n_beating_null": int((means > null_mean).sum()),
            "best_over_shipped": round(float(means.max() - shipped.mean()), 4)}


def clip_sweep(df: pl.DataFrame, shipped: np.ndarray,
               alpha: float, shrink: float) -> dict:
    """Tighter winsorization of the bust tail, paired per fold."""
    rows = []
    for clip in CLIPS:
        a, s, f = _grid_best(df, clip)
        rows.append({"clip": clip, "alpha": a, "shrink": s,
                     "mean": round(float(f.mean()), 4),
                     "per_fold": [round(float(x), 4) for x in f],
                     "paired_vs_shipped": _paired(f - shipped)})
    f_fixed = _folds(df, alpha, shrink, min(CLIPS))
    return {"rows": rows, "shipped_mean": round(float(shipped.mean()), 4),
            "tightest_at_shipped_hyperparams": {
                "clip": min(CLIPS), "alpha": alpha, "shrink": shrink,
                "mean": round(float(f_fixed.mean()), 4),
                "paired_vs_shipped": _paired(f_fixed - shipped)}}


def player_level(df: pl.DataFrame, alpha: float, shrink: float) -> dict:
    """A per-player paired estimand: |predicted VORP rank - realized VORP rank|,
    model vs ECR, with a season-clustered bootstrap. Tune window only."""
    adp, ecr, actuals, hist, pools = M.scoring_context()

    def ranked(preds, season):
        v = M.to_vorp(preds, season, adp, ecr, hist)
        return v.select("gsis_id", pl.col("score").rank(descending=True).alias("r"))

    frames = []
    for s in M.TUNE_SEASONS:
        pool = pools[s]
        truth = ranked(
            pool.join(actuals.filter(pl.col("season") == s).select("gsis_id", "pts_ppr"),
                      on="gsis_id", how="left")
                .with_columns(pl.col("pts_ppr").fill_null(0.0))
                .select("gsis_id", pl.col("pts_ppr").alias("score"),
                        pl.lit(s).cast(pl.Int64).alias("season")), s
        ).rename({"r": "true_rank"})
        mdl = ranked(M.fit_predict(df, s, alpha, shrink), s).rename({"r": "r_model"})
        ecr_preds = (
            ecr.filter((pl.col("season") == s) & pl.col("gsis_id").is_not_null()
                       & pl.col("position").is_in(POSITIONS))
            .sort("ecr_rank").unique(subset=["gsis_id"], keep="first")
            .select(pl.col("season").cast(pl.Int64), "gsis_id",
                    (-pl.col("ecr_rank").cast(pl.Float64)).alias("score"))
        )
        base = ranked(ecr_preds, s).rename({"r": "r_ecr"})
        j = (pool.join(truth, on="gsis_id", how="inner")
                 .join(mdl, on="gsis_id", how="left")
                 .join(base, on="gsis_id", how="left")
                 .drop_nulls(["r_model", "r_ecr"])
                 .with_columns(pl.lit(s).cast(pl.Int64).alias("season")))
        frames.append(j.select("season", "gsis_id", "tiebreak_rank",
                               "true_rank", "r_model", "r_ecr"))
    d = pl.concat(frames).with_columns(
        (pl.col("r_model") - pl.col("true_rank")).abs().alias("err_model"),
        (pl.col("r_ecr") - pl.col("true_rank")).abs().alias("err_ecr"),
    ).with_columns((pl.col("err_model") - pl.col("err_ecr")).alias("adv"))

    def block(sub, label):
        rng = np.random.default_rng(BOOT_SEED)
        seasons = sub["season"].unique().to_list()
        per = {s: sub.filter(pl.col("season") == s)["adv"].to_numpy() for s in seasons}
        boots = np.empty(BOOT_N)
        for i in range(BOOT_N):
            pick = rng.choice(seasons, len(seasons), replace=True)
            boots[i] = np.concatenate(
                [rng.choice(per[s], len(per[s]), replace=True) for s in pick]).mean()
        naive = float(sub["adv"].std(ddof=1) / np.sqrt(sub.height))
        return {"label": label, "n": sub.height,
                "err_model": round(float(sub["err_model"].mean()), 2),
                "err_ecr": round(float(sub["err_ecr"].mean()), 2),
                "advantage": round(float(sub["adv"].mean()), 3),
                "ci_lo": round(float(np.percentile(boots, 2.5)), 3),
                "ci_hi": round(float(np.percentile(boots, 97.5)), 3),
                "p_model_better": round(float((boots < 0).mean()), 3),
                "se_clustered": round(float(boots.std()), 3),
                "se_naive_iid": round(naive, 3),
                "cluster_inflation": round(float(boots.std() / naive), 2)}

    return {"note": "negative advantage = model closer to the truth than ECR",
            "all": block(d, "all pool players"),
            "top50": block(d.filter(pl.col("tiebreak_rank") <= 50), "top 50 by ADP")}


# ---------------------------------------------------------------------------

def build() -> dict:
    df = M.build_dataset()
    alpha, shrink, shipped_mean = M.tune(df)
    shipped = _assert_reproduces_shipped(df, alpha, shrink, shipped_mean)
    print(f"local fitter reproduces bff.model exactly ({shipped.mean():.4f})")

    # the NULL: anchor only, no ridge stage at all (features=[] in fit_predict)
    adp, ecr, actuals, hist, pools = M.scoring_context()
    null_folds = np.array([
        eval_season(pools[s],
                    M.to_vorp(M.fit_predict(df, s, alpha, shrink, features=[]),
                              s, adp, ecr, hist),
                    actuals, s)["spearman_vorp"]
        for s in M.TUNE_SEASONS
    ])

    out = {
        "window": {"tune": list(M.TUNE_SEASONS), "first_target": M.FIRST_TARGET,
                   "alpha": alpha, "shrink": shrink,
                   "shipped_tune_mean": round(float(shipped.mean()), 4),
                   "shipped_per_fold": [round(float(x), 4) for x in shipped],
                   "null_tune_mean": round(float(null_folds.mean()), 4),
                   "n_features": len(M.FEATURES)},
        "reliability": reliability(),
        "redundancy": redundancy(df),
        "oos_residual": oos_residual(df, alpha, shrink),
        "geometry": geometry(df, alpha, shrink),
        "bust": bust(df),
        "power": power(),
        "permutation": permutation(df, alpha, shrink, shipped,
                                  float(null_folds.mean())),
        "clip_sweep": clip_sweep(df, shipped, alpha, shrink),
        "player_level": player_level(df, alpha, shrink),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    audit = build()
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    r, q, o = audit["reliability"], audit["redundancy"], audit["oos_residual"]
    p, pm, c = audit["power"], audit["permutation"], audit["clip_sweep"]
    pl_ = audit["player_level"]["all"]
    print(f"  outcome reliability (split-half, Spearman-Brown) {r['r_full_mean']}")
    print(f"  features -> market anchor R2                     {q['features_to_anchor']}")
    print(f"  ridge residual out-of-sample r                   {o['r_mean']} "
          f"(R2 {o['r2_oos_mean']}, negative in {o['n_folds_negative']}/"
          f"{len(o['folds'])} folds)")
    print(f"  squared error carried by the bust tail           "
          f"{audit['bust']['sq_error_share_of_under_50']:.1%} of it, from "
          f"{audit['bust']['share_under_50_pts']:.1%} of players")
    print(f"  vs ECR: {p['vs_ecr']['mean']:+.4f}, MDE@80% {p['vs_ecr']['mde_80']:+.4f}, "
          f"seasons needed {p['vs_ecr']['seasons_needed_80']}")
    print(f"  permuted-target runs {[r['mean'] for r in pm['runs']]} vs shipped "
          f"{pm['shipped_mean']} / null {pm['null_mean']}")
    print(f"  clip sweep {[r['mean'] for r in c['rows']]} at clips {list(CLIPS)}")
    print(f"  player-level advantage {pl_['advantage']:+.3f} ranks "
          f"[{pl_['ci_lo']:+.3f}, {pl_['ci_hi']:+.3f}] on n={pl_['n']}")


if __name__ == "__main__":
    main()
