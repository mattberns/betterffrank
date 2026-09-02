"""Walk-forward evaluation against baselines that are actually hard to beat.

Fold structure: predict season s using only seasons < s. 2019 has no prior
season and is not scored. Six folds, ~910 picks.

The baselines matter more than the model. "Best available by ADP" is the
obvious one and it is too easy — it does not know that a team already has a
kicker, or that the draft is two rounds from over. Scored in-state during the
replay (`DraftState.baseline_ranks`):

  B0  best available by ADP                                14.9% / 31.6%
  B1  B0 masked by position caps and roster limits         14.9% / 31.6%
  B2  B1 plus "take an empty K/D-ST slot in the last two"  16.9% / 35.2%
  B4  the full model with every manager deviation zeroed   (fitted)

B2 costs nothing and is the honest bar. B4 versus the full model is the whole
premise of this project in one number — what per-manager modelling buys — and
it is reported whether or not it is zero.

K and D/ST are on the board here but were absent from the ADP board behind the
original 17.2%/36.0% figure, so every metric is ALSO reported for skill
positions only. Adding kickers to the board raises the headline without any
modelling; quoting only the combined number would be a con.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import predict as pred
from . import train as tr
from .league import POSITIONS

SKILL_IDX = {0, 1, 2, 3}
EPS = 1e-12
FOLD_SEASONS = range(2020, 2026)


def _rank_of(pids: np.ndarray, probs: np.ndarray, target: int) -> tuple[int, float]:
    order = np.argsort(-probs)
    hit = np.flatnonzero(pids[order] == target)
    if not len(hit):
        return len(pids), 0.0
    i = int(hit[0])
    return i, float(probs[order][i])


def score(
    f: tr.Fitted, examples: list[tr.Example], chain=None, label: str = "model"
) -> pl.DataFrame:
    rows = []
    for ex in examples:
        if ex.chosen_pid is None or ex.chosen_pos is None:
            continue
        # Scoring has no pick-by-pick history, so the manager's marginal
        # autodraft propensity is the best available belief. The live page does
        # better: it watches the picks and updates the belief as it goes.
        aw = chain.start(ex.manager) if chain is not None else 0.0
        pids, probs, outside = pred.joint_probs(f, ex, aw)
        rank, p = _rank_of(pids, probs, ex.chosen_pid)
        pp = pred.position_probs(f, ex, False)
        best_pos = max(pp, key=pp.get) if pp else -1
        wpr = -1
        if ex.chosen_pos in ex.cand:
            q = pred.player_probs(f, ex, ex.chosen_pos)[:-1]
            cand = ex.cand[ex.chosen_pos]
            hit = np.flatnonzero(cand[np.argsort(-q)] == ex.chosen_pid)
            wpr = int(hit[0]) if len(hit) else len(cand)
        rows.append(
            {
                "label": label, "season": ex.season, "overall_pick": ex.overall_pick,
                "round": ex.round, "manager": ex.manager, "is_auto": ex.is_auto,
                "position": POSITIONS[ex.chosen_pos], "skill": ex.chosen_pos in SKILL_IDX,
                "rank": rank, "prob": p, "pos_correct": best_pos == ex.chosen_pos,
                "within_pos_rank": wpr, "outside_mass": outside,
                "b0": ex.b0_rank, "b1": ex.b1_rank, "b2": ex.b2_rank,
            }
        )
    return pl.DataFrame(rows)


def summarise(df: pl.DataFrame, rank_col: str = "rank", label: str | None = None) -> dict:
    skill = df.filter(pl.col("skill"))
    r = df[rank_col].to_numpy()
    rs = skill[rank_col].to_numpy()
    out = {
        "model": label or df["label"][0],
        "n": df.height,
        "top1": round(float(np.mean(r == 0)), 4),
        "top3": round(float(np.mean(r < 3)), 4),
        "top5": round(float(np.mean(r < 5)), 4),
        "skill_top1": round(float(np.mean(rs == 0)), 4) if len(rs) else None,
        "skill_top3": round(float(np.mean(rs < 3)), 4) if len(rs) else None,
    }
    if rank_col == "rank":
        out["nll"] = round(float(-np.log(np.maximum(df["prob"].to_numpy(), EPS)).mean()), 4)
        out["pos_acc"] = round(float(df["pos_correct"].mean()), 4)
    return out


def by_bucket(df: pl.DataFrame, rank_col: str = "rank") -> pl.DataFrame:
    return (
        df.with_columns(
            bucket=pl.when(pl.col("round") <= 3).then(pl.lit("R1-3"))
            .when(pl.col("round") <= 8).then(pl.lit("R4-8"))
            .otherwise(pl.lit("R9+"))
        )
        .group_by("bucket")
        .agg(
            n=pl.len(),
            top1=(pl.col(rank_col) == 0).mean().round(4),
            top3=(pl.col(rank_col) < 3).mean().round(4),
            b2_top1=(pl.col("b2") == 0).mean().round(4),
            b2_top3=(pl.col("b2") < 3).mean().round(4),
        )
        .sort("bucket")
    )


def walk_forward(
    examples: list[tr.Example],
    picks: pl.DataFrame,
    tau: dict[str, float] | None = None,
    seasons=FOLD_SEASONS,
    with_manager: bool = True,
    fit_chain=None,
    verbose: bool = True,
    blocks: frozenset[str] | None = None,
) -> pl.DataFrame:
    """Fit on seasons < s, score season s. Nothing in fold s is ever seen by
    the fit that predicts it — including the standardisation."""
    out = []
    for s in seasons:
        train_ex = [e for e in examples if e.season < s]
        test_ex = [e for e in examples if e.season == s]
        if not train_ex or not test_ex:
            continue
        f = tr.train(train_ex, tau=tau, with_manager=with_manager, blocks=blocks)
        chain = None
        if fit_chain is not None:
            chain = fit_chain(picks.filter(pl.col("season") < s))
        got = score(f, test_ex, chain, label="model")
        out.append(got)
        if verbose:
            r = got["rank"].to_numpy()
            print(f"  {s}: n={got.height:3d}  top1={np.mean(r==0):.3f}  top3={np.mean(r<3):.3f}"
                  f"  lambda={f.lam:.3f}  train={len(train_ex)}")
    return pl.concat(out) if out else pl.DataFrame()


def ablate(
    examples: list[tr.Example], picks: pl.DataFrame, fit_chain=None,
    tau: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Leave-one-block-out over the per-manager channels.

    Every row is a full walk-forward refit, so this is the only honest answer to
    "what did those parameters buy". The `all off` row is what `--b4` always
    claimed to be and, until the `blocks` refactor, was not: the old boolean
    reached the position stage only, so `reach[m]` was fitted underneath it.
    """
    rows = []
    full = walk_forward(examples, picks, tau=tau, fit_chain=fit_chain,
                        blocks=tr.DEFAULT_BLOCKS, verbose=False)
    base = summarise(full, label="everything on")
    rows.append(base)
    for block in sorted(tr.DEFAULT_BLOCKS):
        got = walk_forward(examples, picks, tau=tau, fit_chain=fit_chain,
                           blocks=tr.DEFAULT_BLOCKS - {block}, verbose=False)
        rows.append(summarise(got, label=f"- manager_{block}"))
    off = walk_forward(examples, picks, tau=tau, fit_chain=fit_chain,
                       blocks=tr.NO_BLOCKS, verbose=False)
    rows.append(summarise(off, label="all off (true B4)"))
    out = pl.DataFrame(rows)
    b = rows[0]
    return out.with_columns(
        d_top1=(pl.col("top1") - b["top1"]).round(4),
        d_top3=(pl.col("top3") - b["top3"]).round(4),
        d_nll=(pl.col("nll") - b["nll"]).round(4),
    ).select("model", "n", "top1", "top3", "nll", "d_top1", "d_top3", "d_nll")


def baseline_table(df: pl.DataFrame) -> pl.DataFrame:
    """Every model-free rule on exactly the folds the model was scored on."""
    rows = [
        summarise(df, "b0", "B0 best-available ADP"),
        summarise(df, "b1", "B1 + cap/roster mask"),
        summarise(df, "b2", "B2 + endgame K/DST"),
        summarise(df, "rank", "model"),
    ]
    return pl.DataFrame(rows)
