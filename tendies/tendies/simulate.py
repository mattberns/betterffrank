"""Forward simulation: who is still on the board when your turn comes round.

This is the number the page exists to show, so two details matter more than
they look:

**The autodraft state is drawn ONCE per simulated path and then carried
forward through the Markov chain.** Re-drawing it at every pick would collapse
to the marginal rate and understate the variance of exactly these survival
probabilities. Autodraft is persistent — P(auto | previous auto) = 0.906 — so a
path where someone has walked away should keep them away.

**Everything is seeded from the state.** The same board position always
produces the same numbers, on any machine, in either language. A survival
probability that jitters when nothing changed destroys trust in a live draft.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import predict as pred
from .autopick import AutoChain
from .replay import DraftState
from .rng import Mulberry32, hash32
from .train import Fitted

DEFAULT_SIMS = 2000

# Survival recalibration, p_cal = sigmoid(a + b*logit(p)).
#
# The raw simulation OVER-PREDICTS survival: measured -1.7pp overall and 6-11pp
# through the middle of the range. The mechanism is that picks are drawn
# independently given the state, so simulated drafts contain fewer positional
# runs than real ones, and a player who is "next up" everywhere survives more
# often in simulation than in life.
#
# Fitted on 2022-2023 and tested on 2024-2025, this map cuts calibration error
# from ECE 0.0174 to 0.0067. Coefficients fitted on two seasons and on four
# agree to three decimals, which is what makes it a bias worth correcting
# rather than noise worth ignoring.
RECAL = (-0.2950, 1.1220)


def recalibrate(p: np.ndarray, coef=RECAL) -> np.ndarray:
    a, b = coef
    q = np.clip(p, 1e-4, 1 - 1e-4)
    z = a + b * np.log(q / (1 - q))
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Survival:
    pids: np.ndarray            # candidate pids
    p_available: np.ndarray     # P(still on the board at my next turn)
    horizon: int                # picks between now and that turn
    n_sims: int
    taken_by: np.ndarray        # P(taken) attributed per pick offset, for display


def state_seed(st: DraftState, base: int = 20482930) -> int:
    key = f"{st.board.season}|{st.cfg.teams}|{','.join(str(p) for p in st.picks)}"
    return (base ^ hash32(key)) & 0xFFFFFFFF


def simulate(
    st: DraftState,
    f: Fitted,
    chain: AutoChain | None = None,
    seat: int | None = None,
    n_sims: int = DEFAULT_SIMS,
    auto_override: dict[int, float] | None = None,
    seed: int | None = None,
    watch_top: int = 60,
    calibrated: bool = True,
) -> Survival:
    """P(each currently-available player survives to `seat`'s next turn)."""
    seat = st.seat_on_clock() if seat is None else seat
    horizon = st.horizon(seat)
    if horizon <= 0:
        horizon = 0

    # Watch the players a drafter could plausibly be waiting on: the top of the
    # available board. Watching everyone makes the calibration metric
    # meaningless — 90% of the board is never at risk, so a "99% available"
    # bin swamps every decision-relevant one.
    watch = np.array([], dtype=np.int32)
    if horizon:
        pool = np.concatenate([st.available(p, watch_top) for p in range(len(st.board.by_pos))])
        watch = pool[np.argsort(st.board.adp_rank[pool])][:watch_top]
    if horizon == 0 or len(watch) == 0:
        return Survival(watch, np.ones(len(watch)), 0, n_sims, np.zeros(len(watch)))

    survived = np.zeros(len(watch), dtype=np.int32)
    index = {int(pid): i for i, pid in enumerate(watch)}
    base_seed = seed if seed is not None else state_seed(st)

    # Autodraft belief per seat, before any simulated pick.
    belief = {}
    for s in range(st.cfg.teams):
        mgr = st.managers[s] if s < len(st.managers) else ""
        if auto_override is not None and s in auto_override:
            belief[s] = auto_override[s]
        elif chain is not None:
            belief[s] = chain.start(mgr)
        else:
            belief[s] = 0.0

    for sim in range(n_sims):
        rng = Mulberry32((base_seed + sim * 2654435761) & 0xFFFFFFFF)
        sim_st = st.copy()
        # draw each manager's autodraft state once, then let it persist
        auto_state = {s: rng.random() < belief[s] for s in range(st.cfg.teams)}
        for _ in range(horizon):
            s = sim_st.seat_on_clock()
            if s < 0:
                break
            mgr = sim_st.managers[s] if s < len(sim_st.managers) else ""
            if chain is not None and (auto_override is None or s not in auto_override):
                auto_state[s] = rng.random() < chain.step(mgr, auto_state.get(s, False))
            ex = pred.example_from_state(sim_st, s, mgr)
            if ex is None:
                break
            pids, probs, outside = pred.joint_probs(f, ex, 1.0 if auto_state.get(s) else 0.0)
            total = probs.sum() + outside
            if total <= 0:
                break
            j = rng.pick(np.append(probs, outside) / total)
            if j >= len(pids):           # outside option: someone off-window went
                sim_st.advance(None)
            else:
                sim_st.advance(int(pids[j]))
        gone = sim_st.taken
        for pid, i in index.items():
            if not gone[pid]:
                survived[i] += 1

    raw = survived / n_sims
    return Survival(
        pids=watch,
        p_available=recalibrate(raw) if calibrated else raw,
        horizon=horizon,
        n_sims=n_sims,
        taken_by=1.0 - raw,
    )


def survival_frame(sv: Survival, st: DraftState):
    import polars as pl

    from .league import POSITIONS

    return pl.DataFrame(
        {
            "pid": sv.pids,
            "name": [st.board.name[p] for p in sv.pids],
            "position": [POSITIONS[st.board.pos[p]] for p in sv.pids],
            "adp_rank": [int(st.board.adp_rank[p]) for p in sv.pids],
            "vorp": [float(st.board.vorp[p]) for p in sv.pids],
            "p_available": sv.p_available,
        }
    ).sort("adp_rank")
