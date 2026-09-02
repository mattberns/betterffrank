"""Scoring: from a fitted model and a draft state to a distribution over players.

The same function serves evaluation, simulation and the live page, because all
three hand it the same `Example` object — and an `Example` is built from a
`DraftState` by exactly the code path that built the training set.
"""

from __future__ import annotations

import numpy as np

from . import features as feat
from .choice_model import segment_logsumexp
from .league import POSITIONS
from .replay import DraftState
from .train import (
    ASC_POSITIONS, CHANNEL_COL, MANAGER_EARLY, MANAGER_POS, Example, Fitted,
)


def player_utilities(f: Fitted, ex: Example, p: int) -> np.ndarray:
    """Utilities for the candidates at position p, with the outside option
    appended as the final element."""
    X0 = f.plr_std.apply(ex.plr_X[p])
    L = f.plr_layout
    u_adp = X0[:, feat.PLR_FEATURES.index("u_adp")]
    u = X0 @ f.gamma[: L.n_feat] + u_adp * f.gamma[L.pos_adp + p]
    mi = f.managers.get(ex.manager)
    if mi is not None:
        for name, off in L.channels:
            u = u + X0[:, CHANNEL_COL[name]] * f.gamma[off + mi]
    return np.append(u, f.gamma[L.outside + p])


def player_probs(f: Fitted, ex: Example, p: int) -> np.ndarray:
    u = player_utilities(f, ex, p)
    u = u - u.max()
    e = np.exp(u)
    return e / e.sum()


def inclusive_value(f: Fitted, ex: Example, p: int) -> float:
    u = player_utilities(f, ex, p)
    return float(segment_logsumexp(u, np.array([0, len(u)]))[0])


def position_probs(f: Fitted, ex: Example, is_auto: bool = False) -> dict[int, float]:
    theta = f.theta_auto if is_auto else f.theta_human
    L = f.auto_layout if is_auto else f.pos_layout
    X0 = f.pos_std.apply(ex.pos_X)
    mi = f.managers.get(ex.manager)
    u = X0 @ theta[: L.n_feat]
    for i, p in enumerate(ex.positions):
        name = POSITIONS[p]
        if name in ASC_POSITIONS:
            u[i] += theta[L.asc + ASC_POSITIONS.index(name)]
        u[i] += theta[L.lam] * inclusive_value(f, ex, p)
        if mi is not None and L.manager_pos >= 0 and name in MANAGER_POS:
            u[i] += theta[L.manager_pos + mi * len(MANAGER_POS) + MANAGER_POS.index(name)]
        if (
            mi is not None
            and L.manager_early >= 0
            and name in MANAGER_EARLY
            and ex.round <= feat.EARLY_ROUNDS
        ):
            u[i] += theta[
                L.manager_early + mi * len(MANAGER_EARLY) + MANAGER_EARLY.index(name)
            ]
    u = u - u.max()
    e = np.exp(u)
    e /= e.sum()
    return {p: float(e[i]) for i, p in enumerate(ex.positions)}


def joint_probs(
    f: Fitted, ex: Example, auto_weight: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """(pids, probabilities, outside mass).

    `auto_weight` mixes the autodraft and human position models; the player
    stage is shared, so only the position stage is mixed. Measured: within a
    position, auto and human behave the same (38.9% vs 40.2% top-1).
    """
    pp_h = position_probs(f, ex, False)
    pp = pp_h
    if auto_weight > 0:
        pp_a = position_probs(f, ex, True)
        pp = {p: (1 - auto_weight) * pp_h[p] + auto_weight * pp_a[p] for p in pp_h}

    pids, probs = [], []
    outside = 0.0
    for p in ex.positions:
        q = player_probs(f, ex, p)
        cand = ex.cand[p]
        pids.extend(cand.tolist())
        probs.extend((pp[p] * q[:-1]).tolist())
        outside += pp[p] * float(q[-1])
    return np.array(pids, dtype=np.int32), np.array(probs), outside


def example_from_state(
    st: DraftState, seat: int | None = None, manager: str | None = None, kappa: float = 1.0
) -> Example | None:
    """Build a scoring example from a live state — the same construction the
    training set used, which is what keeps train and serve honest."""
    seat = st.seat_on_clock() if seat is None else seat
    if seat < 0:
        return None
    cs = st.choice_set(seat, kappa)
    if not cs:
        return None
    positions, pos_X = feat.position_features(st, seat, cs)
    plr_X = {p: feat.player_features(st, seat, cs[p]) for p in positions}
    return Example(
        season=st.board.season,
        overall_pick=st.pick_no,
        round=(st.pick_no - 1) // st.cfg.teams + 1,
        seat=seat,
        manager=manager if manager is not None else st.managers[seat],
        is_auto=False,
        chosen_pid=None,
        chosen_pos=None,
        in_window=False,
        positions=positions,
        pos_X=pos_X,
        cand={p: cs[p] for p in positions},
        plr_X=plr_X,
        chosen_idx=-1,
    )


def top_k(f: Fitted, ex: Example, k: int = 3, auto_weight: float = 0.0):
    pids, probs, _ = joint_probs(f, ex, auto_weight)
    order = np.argsort(-probs)[:k]
    return [(int(pids[i]), float(probs[i])) for i in order]
