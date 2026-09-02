"""The feature contract. `web/features.js` must implement this identically.

Two stages, because the choice factorises that way:

  POSITION stage — one row per feasible position at this state.
  PLAYER stage   — one row per candidate within a position.

Everything here is a function of state only: the board, who is already gone,
and the drafting seat's roster. Three things are therefore banned, and each has
bitten someone:

* **`adp_delta` from the picks table.** It is `overall_pick - adp_rank` for the
  player who was actually taken, so it exists only for the realised choice.
  Joining it would hand the model the answer. The per-candidate analogue
  (`adp_rank_j - current_pick`) is computed here instead.
* **Season outcomes.** Points reach features only through `vorp`, which comes
  from a curve fitted on seasons strictly before this one.
* **Nested simulation.** Look-ahead uses the deterministic "players leave in
  ADP order" approximation, the same one `bff/vona.py` uses. A simulation
  inside a feature would make the outer simulation quadratic and make training
  features stochastic, breaking byte-stability and JS parity together.

Numeric conventions that MUST match in JS or the two sides drift silently:
natural log everywhere; counts are floats before standardisation; a missing bye
scores 0 rather than NaN; softmax is always max-subtracted.
"""

from __future__ import annotations

import numpy as np

from .league import FLEX_POSITIONS, POS_INDEX, POSITIONS
from .replay import DraftState

POS_FEATURES = (
    "first_at_pos",       # 1 if the seat has none of this position yet
    "starter_need",       # dedicated starter slots still unfilled here
    "flex_need",          # unfilled FLEX slots, if this position can fill them
    "log_count",          # log1p(count already rostered here)
    "best_vorp",          # VORP of the best available at this position (/100)
    "vona",               # that VORP minus the best still here at my next turn
    "n_gone",             # how many at this position go before my next turn
    "adp_lead",           # (best available's ADP rank - this pick) / 10
    "run5",               # picks at this position in the last 5
    "run10",              # picks at this position in the last 10
    "endgame_k",          # 1 if K and the draft is nearly over
    "endgame_dst",        # 1 if D/ST and the draft is nearly over
    "early_qb",           # 1 if QB and the draft is still young
    "early_te",           # 1 if TE and the draft is still young
)

# Deliberately NOT features: log_h (picks until my next turn), rounds_left, and
# starters_filled. All three are properties of the OCCASION, identical for every
# position in the choice set, so they cancel in the conditional-logit softmax and
# are structurally unidentified — their fitted coefficients came out at exactly
# 0.000 and a leave-one-out pass moved the likelihood by exactly 0.0000. They
# were not merely weak; they could never have done anything. Occasion-level
# information can only enter through an interaction with position, which is what
# `endgame_k` / `endgame_dst` and `early_qb` / `early_te` are: a point in the
# draft crossed with a position. "This manager takes quarterbacks early" is an
# occasion-level statement about a position, so this is the ONLY shape it can
# take — a bare round term would cancel exactly like `rounds_left` did.
#
# Note the deliberate asymmetry between the two pairs. `endgame_*` counts
# BACKWARD from the end (`rounds_left`), because "the last two rounds" has to
# mean the same thing in the 16-round 2019-20 drafts and the 15-round ones
# since. `early_*` counts FORWARD from round 1, because "round 6" already does.

PLR_FEATURES = (
    "u_adp",              # -log((adp+10)/(best_adp+10)); 0 for the best available
    "posrank_gap",        # within-position slots below the best available / 10
    "ecr_lean",           # log((ecr+10)/(adp+10)); + = market takes him earlier
    "ecr_missing",        # 1 when no expert rank exists (always for K/D-ST)
    "vorp_gap",           # VORP below the best available at this position / 100
    "bye_clash",          # my starters already on this bye week, capped at 3
    "same_team",          # 1 if I already roster someone on his NFL team
    "is_rookie",          # 1 if this is his NFL entry year
    "age",                # years older than typical at his position this season
    "was_mine",           # 1 if THIS seat drafted him last season
    "durability",         # share of last season's games played, vs typical
)

ENDGAME_ROUNDS = 2        # "nearly over" = this many rounds left

# "Still young" = this round or earlier. Set from the league's own timing: the
# mean round of a team-season's first QB is 6.6 and of its first TE 6.7, so a
# threshold at 6 splits the mass roughly in half, which is where an indicator
# carries the most information. It is a choice, and it may only be moved on
# training folds.
EARLY_ROUNDS = 6


def position_features(
    st: DraftState, seat: int, cs: dict[int, np.ndarray]
) -> tuple[list[int], np.ndarray]:
    """(positions, matrix[n_positions, len(POS_FEATURES)])."""
    board = st.board
    pick = st.pick_no
    h = st.horizon(seat)
    counts = st.counts[seat]
    run5, run10 = st.runs(5), st.runs(10)
    rounds_left = st.cfg.rounds - ((pick - 1) // st.cfg.teams)
    rnd = (pick - 1) // st.cfg.teams + 1
    # Under ADP-order depletion, everyone with adp_rank < pick + h is gone by
    # my next turn. This is the same approximation bff/vona.py makes.
    horizon_rank = pick + h

    positions = sorted(cs)
    out = np.zeros((len(positions), len(POS_FEATURES)), dtype=np.float64)
    for i, p in enumerate(positions):
        cand = cs[p]
        name = POSITIONS[p]
        best = cand[0]
        best_vorp = float(board.vorp[best])
        survivors = cand[board.adp_rank[cand] >= horizon_rank]
        vorp_later = float(board.vorp[survivors[0]]) if len(survivors) else best_vorp
        n_gone = int(len(cand) - len(survivors))
        dedicated = st.cfg.starters.get(name, 0)
        flex_need = 0.0
        if name in FLEX_POSITIONS and int(counts[p]) >= dedicated:
            used = sum(
                max(0, int(counts[POS_INDEX[q]]) - st.cfg.starters.get(q, 0))
                for q in FLEX_POSITIONS
            )
            flex_need = float(max(0, st.cfg.flex - used))
        out[i] = (
            1.0 if counts[p] == 0 else 0.0,
            float(max(0, dedicated - int(counts[p]))),
            flex_need,
            float(np.log1p(int(counts[p]))),
            best_vorp / 100.0,
            (best_vorp - vorp_later) / 100.0,
            float(n_gone),
            float(int(board.adp_rank[best]) - pick) / 10.0,
            float(run5[p]),
            float(run10[p]),
            1.0 if (name == "K" and rounds_left <= ENDGAME_ROUNDS) else 0.0,
            1.0 if (name == "DST" and rounds_left <= ENDGAME_ROUNDS) else 0.0,
            1.0 if (name == "QB" and rnd <= EARLY_ROUNDS) else 0.0,
            1.0 if (name == "TE" and rnd <= EARLY_ROUNDS) else 0.0,
        )
    return positions, out


def player_features(st: DraftState, seat: int, cand: np.ndarray) -> np.ndarray:
    """matrix[n_candidates, len(PLR_FEATURES)] for one position's candidates."""
    board = st.board
    roster = st.rosters[seat]
    my_byes = [int(board.bye[q]) for q in roster if board.bye[q] > 0]
    my_teams = {board.pro_team[q] for q in roster}

    adp = board.adp_rank[cand].astype(np.float64)
    best_adp = adp[0]
    pos_rank = board.pos_rank[cand].astype(np.float64)
    vorp = board.vorp[cand]
    ecr = board.ecr_rank[cand]
    has_ecr = ~np.isnan(ecr)

    out = np.zeros((len(cand), len(PLR_FEATURES)), dtype=np.float64)
    out[:, 0] = -np.log((adp + 10.0) / (best_adp + 10.0))
    out[:, 1] = (pos_rank - pos_rank[0]) / 10.0
    lean = np.zeros(len(cand))
    lean[has_ecr] = np.log((ecr[has_ecr] + 10.0) / (adp[has_ecr] + 10.0))
    out[:, 2] = lean
    out[:, 3] = (~has_ecr).astype(np.float64)
    out[:, 4] = (vorp - vorp[0]) / 100.0
    if my_byes:
        byes = board.bye[cand]
        out[:, 5] = [min(3.0, float(my_byes.count(int(b)))) if b > 0 else 0.0 for b in byes]
    out[:, 6] = [1.0 if board.pro_team[q] in my_teams and board.pro_team[q] else 0.0
                 for q in cand]
    out[:, 7] = board.rookie[cand].astype(np.float64)
    out[:, 8] = board.age[cand]
    me = st.managers[seat] if seat < len(st.managers) else ""
    if me:
        out[:, 9] = [1.0 if board.prev_owner[q] == me else 0.0 for q in cand]
    out[:, 10] = board.durability[cand]
    return out


class Standardizer:
    """Fold-specific means and SDs. Global standardisation would leak the test
    fold's distribution into training; small, but real."""

    def __init__(self, mean: np.ndarray, sd: np.ndarray) -> None:
        self.mean, self.sd = mean, sd

    @staticmethod
    def fit(x: np.ndarray, protect: tuple[int, ...] = ()) -> "Standardizer":
        mean = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0
        for i in protect:                 # keep indicators as 0/1
            mean[i], sd[i] = 0.0, 1.0
        return Standardizer(mean, sd)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.sd

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "sd": self.sd.tolist()}


# Indicator columns keep their raw 0/1 scale so their coefficients stay readable.
# `is_rookie` in particular is multiplied by the per-manager `rookie[m]` column,
# so leaving it on 0/1 makes that coefficient read directly as a log-odds shift.
POS_INDICATORS = tuple(
    POS_FEATURES.index(f)
    for f in ("first_at_pos", "endgame_k", "endgame_dst", "early_qb", "early_te")
)
PLR_INDICATORS = tuple(
    PLR_FEATURES.index(f)
    for f in ("ecr_missing", "same_team", "is_rookie", "was_mine")
)
