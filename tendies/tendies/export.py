"""Export the fitted model and the live board as JSON the browser can score.

This is the train/serve boundary, so it is written defensively. The feature
NAME LISTS ship with the coefficients, and `web/engine.js` asserts they match
its own constants at load time. Without that, adding a feature in Python and
forgetting it in JS would silently shift every coefficient by one slot and the
page would show confident nonsense with no error anywhere.

Floats are rounded at write time so the artifact diffs cleanly between runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from . import features as feat
from . import simulate
from . import vorp as vorp_mod
from .autopick import AutoChain
from .config import ECR_SOURCE_BY_SEASON
from .league import POSITIONS, LeagueConfig
from .replay import WINDOW, Board
from .train import (
    ASC_POSITIONS, MANAGER_EARLY, MANAGER_POS, PLAYER_CHANNELS, Fitted,
)

SCHEMA = 4
ROUND_TO = 9

# How many standard errors two candidates' scores must differ by before the page
# is willing to ORDER them rather than call them tied. 1.96 is the two-sided 5%
# normal quantile; it is here rather than in engine.js so the artifact records
# the threshold it was rendered under.
RESOLUTION_Z = 1.96


def _r(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), ROUND_TO)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def board_payload(board_df: pl.DataFrame) -> list[dict]:
    """The live board. `pid` is the array index AND `adp_rank - 1`; the JS side
    relies on that identity for its availability bitmap."""
    out = []
    for r in board_df.sort("pid").iter_rows(named=True):
        out.append(
            {
                "pid": int(r["pid"]),
                "name": r["name"],
                "pos": POSITIONS.index(r["position"]),
                "team": r["pro_team"] or "",
                "bye": int(r["bye"] or 0),
                "adp": _r(r["adp"]),
                "adpRank": int(r["adp_rank"]),
                "posRank": int(r["pos_rank"]),
                "ecr": int(r["ecr_rank"]) if r["ecr_rank"] is not None else None,
                "vorp": _r(r["vorp"]),
                "vorpAdp": _r(r.get("vorp_adp", r["vorp"])),
                # How much of each number the isotonic curve can actually
                # resolve. The page groups candidates into ties on these rather
                # than ordering differences it cannot measure -- see vorp.py's
                # docstring for the measurement behind it.
                "vorpSe": _r(r["vorp_se"]),
                "edgeSe": _r(r["edge_se"]),
                "rookie": int(r.get("rookie") or 0),
                "age": _r(r.get("age") or 0.0),
                "durability": _r(r.get("durability") or 0.0),
                "prevOwner": r.get("prev_owner") or "",
            }
        )
    return out


def _block(v: np.ndarray, start: int, width: int) -> list:
    """One manager block, sliced to its EXACT width.

    Both of these were open-ended slices (`gamma[off:]`, `theta[off:]`) until
    the layout refactor. Appending any coefficient after the last block would
    have been swallowed into it, and `assertContract` could not have caught it:
    that guard compares FEATURE NAME LISTS, which would still have matched. The
    page would have applied one manager coefficient as another, silently.
    """
    return [] if start < 0 else _r(v[start : start + width])


def model_payload(
    f: Fitted, chain: AutoChain | None, cfg: LeagueConfig, flow=None
) -> dict:
    pl_, po = f.plr_layout, f.pos_layout
    n_mgr = len(f.managers)
    asc = list(ASC_POSITIONS)
    return {
        "schema": SCHEMA,
        "positions": list(POSITIONS),
        "window": {p: WINDOW[p] for p in POSITIONS},
        "league": {
            "teams": cfg.teams,
            "rounds": cfg.rounds,
            "starters": cfg.starters,
            "flex": cfg.flex,
            "bench": cfg.bench,
            "limits": cfg.limits,
            "rosterSize": cfg.roster_size,
        },
        "player": {
            "features": list(feat.PLR_FEATURES),
            "mean": _r(f.plr_std.mean),
            "sd": _r(f.plr_std.sd),
            "beta": _r(f.gamma[: pl_.n_feat]),
            "posAdp": _r(f.gamma[pl_.pos_adp : pl_.pos_adp + len(POSITIONS)]),
            "outside": _r(f.gamma[pl_.outside : pl_.outside + len(POSITIONS)]),
            # Per-manager channels, as an ordered LIST so JS applies them in
            # the same order Python did. Each is a coefficient on the named
            # STANDARDISED player feature. `reach` is in here like the rest.
            "channels": [
                {"name": name, "feature": dict(PLAYER_CHANNELS)[name],
                 "beta": _block(f.gamma, off, n_mgr)}
                for name, off in pl_.channels
            ],
        },
        "position": {
            "features": list(feat.POS_FEATURES),
            "mean": _r(f.pos_std.mean),
            "sd": _r(f.pos_std.sd),
            "ascPositions": asc,
            "human": {
                "beta": _r(f.theta_human[: po.n_feat]),
                "asc": _r(f.theta_human[po.asc : po.asc + po.n_asc]),
                "lambda": _r(f.theta_human[po.lam]),
                "managerPos": list(MANAGER_POS),
                "managerAsc": _block(
                    f.theta_human, po.manager_pos, n_mgr * len(MANAGER_POS)
                ),
                "managerEarlyPos": list(MANAGER_EARLY),
                "managerEarly": _block(
                    f.theta_human, po.manager_early, n_mgr * len(MANAGER_EARLY)
                ),
            },
            "auto": {
                "beta": _r(f.theta_auto[: po.n_feat]),
                "asc": _r(f.theta_auto[po.asc : po.asc + po.n_asc]),
                "lambda": _r(f.theta_auto[po.lam]),
            },
        },
        "managers": {m: i for m, i in f.managers.items()},
        "autoChain": chain.to_dict() if chain else None,
        "endgameRounds": feat.ENDGAME_ROUNDS,
        "earlyRounds": feat.EARLY_ROUNDS,
        "recal": list(simulate.RECAL),
        # Standard errors below which the recommendation panel reports a tie
        # instead of a rank. See RESOLUTION_Z.
        "resolutionZ": RESOLUTION_Z,
        # Positions with no VORP curve. The lineup objective is blind to them
        # (every kicker is worth the same 0), so the plan has to treat them as
        # a roster-legality constraint instead of a price -- see engine.js.
        "noCurve": [p for p in POSITIONS if p not in vorp_mod.CURVE_POSITIONS],
        # Positions whose replacement level IS a streaming total (see vorp.py).
        # The empty-slot floor is clamped at 0 for these, or the streaming
        # already inside their VORP gets charged a second time.
        "streamable": [p for p in POSITIONS if p in vorp_mod.STREAMABLE],
        # Half-PPR points a season that ROSTERING a body at a streamable
        # position is worth over punting it (vorp.HOLD_GAIN, derived by
        # streaming.hold_gain). engine.js adds it to every occupied dedicated
        # slot at the position in lineupValueWith; positions absent here get 0.
        "holdGain": {p: float(vorp_mod.HOLD_GAIN[p]) for p in POSITIONS
                     if p in vorp_mod.HOLD_GAIN},
        # Which expert series priced each season. Stamped into the artifact on
        # purpose: a terminal line nobody reads is what let the 2026-09-01
        # source switch through, and this shows up as a diff instead.
        "ecrSource": dict(sorted(ECR_SOURCE_BY_SEASON.items())),
        # Measured positional flow: mean cumulative picks by position at each
        # point of a draft. Two jobs -- far-horizon availability, and the
        # empty-slot floor in `lineupValue`. See depletion.py for why an
        # ADP-order assumption is not good enough.
        "flow": None if flow is None else [[round(float(x), 4) for x in row] for row in flow],
    }


def write(
    path: Path,
    board_df: pl.DataFrame,
    f: Fitted,
    chain: AutoChain | None,
    cfg: LeagueConfig,
    meta: dict | None = None,
    flow=None,
) -> Path:
    payload = {
        "meta": meta or {},
        "model": model_payload(f, chain, cfg, flow),
        "board": board_payload(board_df),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path


def fixtures(
    board: Board, cfg: LeagueConfig, f: Fitted, chain, states: list[list[int | None]],
    managers: list[str], path: Path,
) -> Path:
    """Golden fixtures for the Python/JS parity test: a set of draft states with
    the probabilities Python computes for each. `tests/parity.mjs` replays the
    same states through the JS engine and the two must agree."""
    from . import predict as pred
    from .replay import DraftState

    cases = []
    for i, picks in enumerate(states):
        st = DraftState(board, cfg, managers)
        for pid in picks:
            st.advance(pid)
        seat = st.seat_on_clock()
        if seat < 0:
            continue
        ex = pred.example_from_state(st, seat)
        if ex is None:
            continue
        pos_p = pred.position_probs(f, ex, False)
        pids, probs, outside = pred.joint_probs(f, ex, 0.0)
        order = np.argsort(-probs)[:20]
        cases.append(
            {
                "id": f"case{i:02d}",
                "picks": [None if p is None else int(p) for p in picks],
                "seat": int(seat),
                "manager": ex.manager,
                "expect": {
                    "positions": [int(p) for p in ex.positions],
                    "posProbs": _r([pos_p[p] for p in ex.positions]),
                    "posFeatures": _r(ex.pos_X),
                    "outside": _r(outside),
                    "top": [[int(pids[j]), _r(float(probs[j]))] for j in order],
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "cases": cases}, indent=1))
    return path
