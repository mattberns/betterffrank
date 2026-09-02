"""Build design matrices from replayed states and fit the two stages.

Fitting order matters and is the standard sequential (limited-information)
estimator for a nested logit:

  1. player stage on every pick   -> gamma
  2. inclusive value I_p from gamma, for every feasible position
  3. position stage on human picks, with I_p as a regressor -> theta, lambda
  4. position stage on autodraft picks, pooled across managers

Examples are built ONCE for all seasons and filtered per fold, because a
feature depends only on the state, never on the fold. Standardisation is
re-fit inside each fold — using global means would leak the test fold's
distribution into training. Small leak, but the kind that survives review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from . import features as feat
from .choice_model import GROUP_CODE, ChoiceData, ModelSpec, fit, segment_logsumexp
from .league import POSITIONS
from .replay import Board, replay_season

# Position deviations on the ADP-following coefficient, and manager deviations
# on the position constants. WR is the reference position.
REF_POSITION = "WR"
MANAGER_POS = ("QB", "RB", "TE")

# Positions that get a per-manager EARLY deviation on top of the flat one. The
# flat `asc[QB|m]` says how often he takes a quarterback; `asc[QB|m|early]` says
# whether he takes it in the first six rounds. They are identified apart because
# every manager picks in both regimes — see REPORT.
MANAGER_EARLY = ("QB", "TE")

# PLAYER-STAGE per-manager channels: (block name, the PLR_FEATURE it scales).
#
# Each one is "this manager weights this player attribute differently from the
# league". They are all the same algebra — a design column equal to the
# STANDARDISED feature, switched on for that manager's occasions — so they are
# declared once here and the layout, the spec, the design matrix, the payload
# and the JS all loop over this tuple instead of naming offsets. `reach` is not
# a special case: `acc += z * (posAdp[p] + reach)` is exactly a coefficient on
# standardised `u_adp`.
#
# APPEND, do not reorder: the order fixes the coefficient layout.
PLAYER_CHANNELS = (
    ("reach", "u_adp"),          # follows ADP more or less steeply than the league
    ("rookie", "is_rookie"),     # appetite for first-year players
    ("expert", "ecr_lean"),      # drafts off the expert list vs off the market
    ("stack", "same_team"),      # stacks his own NFL teams vs diversifies
    ("age", "age"),              # prefers young or old for the position
    ("mine", "was_mine"),        # re-drafts the players he had last season
    ("durability", "durability"),  # avoids or ignores last season's absentees
)
CHANNEL_COL = {name: feat.PLR_FEATURES.index(f) for name, f in PLAYER_CHANNELS}
assert all(f in feat.PLR_FEATURES for _, f in PLAYER_CHANNELS)

# Every per-manager channel, each independently droppable. `--b4` used to be a
# boolean that only reached the position stage, so `reach[m]` was fitted even
# when the run claimed every manager deviation was off; a set of block names
# makes "which deviations are in this fit" a single fact that the layouts, the
# specs, the payload and the ablation all read from.
MANAGER_BLOCKS = ("pos", "timing") + tuple(n for n, _ in PLAYER_CHANNELS)
ALL_BLOCKS = frozenset(MANAGER_BLOCKS)
NO_BLOCKS: frozenset[str] = frozenset()

# The channels that are actually FITTED. `stack`, `mine` and `durability` are
# implemented, exported and testable, but left OFF: the per-channel leave-one-out
# in REPORT prices them at -0.0006, -0.0004 and -0.0060 nats, i.e. two inert and
# one actively harmful. Thirteen parameters each, on a signal that is not there.
#
# Their POOLED features stay in `PLR_FEATURES` — `same_team` -0.393 and
# `was_mine` +0.371 are two of the larger coefficients in the model. What is
# dropped is only the claim that managers differ from each other on them, and
# for `was_mine` the trait side agrees: its split-half reliability is -0.76.
#
# This is selection on the six evaluated folds, which REPORT's optimism section
# names. Flip it back with `blocks=ALL_BLOCKS`.
OFF_BLOCKS = frozenset({"stack", "mine", "durability"})
DEFAULT_BLOCKS = ALL_BLOCKS - OFF_BLOCKS


@dataclass(frozen=True)
class PlayerLayout:
    """Column offsets for the player stage. Every consumer indexes through this.

    The offsets used to be written out as `off + 2 * len(POSITIONS) + mi` in
    four places, and `export.model_payload` sliced the last block open-endedly
    (`gamma[off + 2 * len(POSITIONS):]`). Appending any column after that block
    would have been swallowed into it silently — the feature-name contract that
    `engine.js` asserts covers the FEATURE lists, not the manager blocks, so the
    page would have applied one manager coefficient as another with no error
    anywhere. Named offsets and explicit widths remove the whole failure mode.
    """

    n_feat: int
    pos_adp: int
    outside: int
    channels: tuple[tuple[str, int], ...]   # (block, start offset), in fit order
    n_managers: int
    total: int

    def start(self, block: str) -> int:
        """Offset of a channel's block, or -1 if it is not in this fit."""
        for name, off in self.channels:
            if name == block:
                return off
        return -1


@dataclass(frozen=True)
class PositionLayout:
    n_feat: int
    asc: int
    n_asc: int
    lam: int
    manager_pos: int    # -1 when the block is not in this fit
    manager_early: int  # -1 when the block is not in this fit
    n_managers: int
    total: int


def player_layout(n_managers: int, blocks: frozenset[str] = DEFAULT_BLOCKS) -> PlayerLayout:
    n_feat = len(feat.PLR_FEATURES)
    pos_adp = n_feat
    outside = pos_adp + len(POSITIONS)
    cur = outside + len(POSITIONS)
    channels = []
    for name, _ in PLAYER_CHANNELS:
        if name in blocks:
            channels.append((name, cur))
            cur += n_managers
    return PlayerLayout(n_feat, pos_adp, outside, tuple(channels), n_managers, cur)


def position_layout(
    n_managers: int, blocks: frozenset[str] = DEFAULT_BLOCKS
) -> PositionLayout:
    n_feat = len(feat.POS_FEATURES)
    n_asc = len(POSITIONS) - 1
    asc = n_feat
    lam = asc + n_asc
    cur = lam + 1
    manager_pos = manager_early = -1
    if "pos" in blocks:
        manager_pos, cur = cur, cur + n_managers * len(MANAGER_POS)
    if "timing" in blocks:
        manager_early, cur = cur, cur + n_managers * len(MANAGER_EARLY)
    return PositionLayout(
        n_feat, asc, n_asc, lam, manager_pos, manager_early, n_managers, cur
    )


@dataclass
class Example:
    season: int
    overall_pick: int
    round: int
    seat: int
    manager: str
    is_auto: bool
    chosen_pid: int | None
    chosen_pos: int | None
    in_window: bool
    positions: list[int]
    pos_X: np.ndarray                    # [n_positions, len(POS_FEATURES)]
    cand: dict[int, np.ndarray]
    plr_X: dict[int, np.ndarray]
    chosen_idx: int                      # index within cand[chosen_pos], -1 = outside
    b0_rank: int = -1                    # model-free baselines, scored in-state
    b1_rank: int = -1
    b2_rank: int = -1


def build_examples(
    boards_by_season: dict[int, Board],
    linked: pl.DataFrame,
    cfgs: dict,
    kappa: float = 1.0,
) -> list[Example]:
    out: list[Example] = []
    for season in sorted(cfgs):
        picks = linked.filter(pl.col("season") == season)
        if picks.is_empty():
            continue
        board = boards_by_season[season]
        for st, row, pid, chosen_pos, in_window in replay_season(
            board, cfgs[season], picks, kappa
        ):
            seat = st.seat_on_clock()
            cs = st.choice_set(seat, kappa)
            if not cs:
                continue
            positions, pos_X = feat.position_features(st, seat, cs)
            plr_X = {p: feat.player_features(st, seat, cs[p]) for p in positions}
            b0 = b1 = b2 = -1
            if pid is not None:
                b0, b1, b2 = st.baseline_ranks(seat, pid)
            idx = -1
            if pid is not None and chosen_pos in cs:
                hit = np.flatnonzero(cs[chosen_pos] == pid)
                idx = int(hit[0]) if len(hit) else -1
            out.append(
                Example(
                    season=season,
                    overall_pick=int(row["overall_pick"]),
                    round=int(row["round"]),
                    seat=seat,
                    manager=row["franchise_id"],
                    is_auto=bool(row["auto_pick"]),
                    chosen_pid=pid,
                    chosen_pos=chosen_pos,
                    in_window=bool(in_window),
                    positions=positions,
                    pos_X=pos_X,
                    cand={p: cs[p] for p in positions},
                    plr_X=plr_X,
                    chosen_idx=idx,
                    b0_rank=b0, b1_rank=b1, b2_rank=b2,
                )
            )
    return out


# --------------------------------------------------------------------------
# player stage
# --------------------------------------------------------------------------


def player_spec(managers: list[str], blocks: frozenset[str] = DEFAULT_BLOCKS) -> ModelSpec:
    spec = ModelSpec()
    for f in feat.PLR_FEATURES:
        spec.add(f"g_{f}")
    for p in POSITIONS:                       # how steeply each position follows ADP
        spec.add(f"g_u_adp[{p}]", GROUP_CODE["ridge"])
    for p in POSITIONS:                       # outside option, one constant each
        spec.add(f"outside[{p}]")
    for name, _ in PLAYER_CHANNELS:
        if name in blocks:
            code = GROUP_CODE["manager_reach" if name == "reach" else "manager_taste"]
            for m in managers:
                spec.add(f"{name}[{m}]", code)
    assert len(spec.names) == player_layout(len(managers), blocks).total
    return spec


def player_rows(
    ex: Example, layout: PlayerLayout, managers: dict[str, int],
    std: feat.Standardizer | None
) -> tuple[np.ndarray, int] | None:
    """Design rows for one example's chosen position, plus the outside row."""
    if ex.chosen_pos is None:
        return None
    p = ex.chosen_pos
    if p not in ex.plr_X:
        return None
    X0 = ex.plr_X[p]
    if std is not None:
        X0 = std.apply(X0)
    n = X0.shape[0]
    X = np.zeros((n + 1, layout.total))
    X[:n, : layout.n_feat] = X0
    u_adp_col = feat.PLR_FEATURES.index("u_adp")
    X[:n, layout.pos_adp + p] = X0[:, u_adp_col]
    X[n, layout.outside + p] = 1.0                             # outside constant
    mi = managers.get(ex.manager)
    if mi is not None:
        for name, off in layout.channels:
            X[:n, off + mi] = X0[:, CHANNEL_COL[name]]
    chosen = ex.chosen_idx if ex.chosen_idx >= 0 else n
    return X, chosen


def player_data(
    examples: list[Example], spec: ModelSpec, layout: PlayerLayout,
    managers: dict[str, int], std: feat.Standardizer | None,
) -> ChoiceData:
    Xs, ptr, chosen = [], [0], []
    for ex in examples:
        got = player_rows(ex, layout, managers, std)
        if got is None:
            continue
        X, c = got
        chosen.append(ptr[-1] + c)
        Xs.append(X)
        ptr.append(ptr[-1] + X.shape[0])
    if not Xs:
        return ChoiceData(np.zeros((0, layout.total)), np.array([0]), np.array([], int))
    return ChoiceData(np.vstack(Xs), np.array(ptr), np.array(chosen), names=tuple(spec.names))


def inclusive_values(
    ex: Example, layout: PlayerLayout, managers: dict[str, int],
    std: feat.Standardizer | None, gamma: np.ndarray
) -> dict[int, float]:
    """I_p = log sum exp(utility) over each position's candidates + outside."""
    out = {}
    u_adp_col = feat.PLR_FEATURES.index("u_adp")
    mi = managers.get(ex.manager)
    for p in ex.positions:
        X0 = ex.plr_X[p]
        if std is not None:
            X0 = std.apply(X0)
        u = X0 @ gamma[: layout.n_feat]
        u = u + X0[:, u_adp_col] * gamma[layout.pos_adp + p]
        if mi is not None:
            for name, off in layout.channels:
                u = u + X0[:, CHANNEL_COL[name]] * gamma[off + mi]
        u = np.append(u, gamma[layout.outside + p])
        out[p] = float(segment_logsumexp(u, np.array([0, len(u)]))[0])
    return out


# --------------------------------------------------------------------------
# position stage
# --------------------------------------------------------------------------


ASC_POSITIONS = [p for p in POSITIONS if p != REF_POSITION]


def position_spec(managers: list[str], blocks: frozenset[str] = DEFAULT_BLOCKS) -> ModelSpec:
    spec = ModelSpec()
    for f in feat.POS_FEATURES:
        spec.add(f"t_{f}")
    for p in ASC_POSITIONS:
        spec.add(f"asc[{p}]")
    spec.add("lambda_iv")
    if "pos" in blocks:
        for m in managers:
            for p in MANAGER_POS:
                spec.add(f"asc[{p}|{m}]", GROUP_CODE["manager_pos"])
    if "timing" in blocks:
        for m in managers:
            for p in MANAGER_EARLY:
                spec.add(f"asc[{p}|{m}|early]", GROUP_CODE["manager_timing"])
    assert len(spec.names) == position_layout(len(managers), blocks).total
    return spec


def position_rows(
    ex: Example, layout: PositionLayout, managers: dict[str, int],
    std: feat.Standardizer | None, iv: dict[int, float]
) -> tuple[np.ndarray, int] | None:
    if ex.chosen_pos is None or ex.chosen_pos not in ex.positions:
        return None
    X0 = ex.pos_X
    if std is not None:
        X0 = std.apply(X0)
    n = len(ex.positions)
    X = np.zeros((n, layout.total))
    X[:, : layout.n_feat] = X0
    for i, p in enumerate(ex.positions):
        name = POSITIONS[p]
        if name in ASC_POSITIONS:
            X[i, layout.asc + ASC_POSITIONS.index(name)] = 1.0
        X[i, layout.lam] = iv[p]
    mi = managers.get(ex.manager)
    if mi is not None and layout.manager_pos >= 0:
        base = layout.manager_pos + mi * len(MANAGER_POS)
        for i, p in enumerate(ex.positions):
            name = POSITIONS[p]
            if name in MANAGER_POS:
                X[i, base + MANAGER_POS.index(name)] = 1.0
    if mi is not None and layout.manager_early >= 0 and ex.round <= feat.EARLY_ROUNDS:
        base = layout.manager_early + mi * len(MANAGER_EARLY)
        for i, p in enumerate(ex.positions):
            name = POSITIONS[p]
            if name in MANAGER_EARLY:
                X[i, base + MANAGER_EARLY.index(name)] = 1.0
    return X, ex.positions.index(ex.chosen_pos)


def position_data(
    examples: list[Example], spec: ModelSpec, layout: PositionLayout,
    managers: dict[str, int], std: feat.Standardizer | None,
    ivs: list[dict[int, float]],
) -> ChoiceData:
    Xs, ptr, chosen = [], [0], []
    for ex, iv in zip(examples, ivs):
        got = position_rows(ex, layout, managers, std, iv)
        if got is None:
            continue
        X, c = got
        chosen.append(ptr[-1] + c)
        Xs.append(X)
        ptr.append(ptr[-1] + X.shape[0])
    if not Xs:
        return ChoiceData(np.zeros((0, layout.total)), np.array([0]), np.array([], int))
    return ChoiceData(np.vstack(Xs), np.array(ptr), np.array(chosen), names=tuple(spec.names))


# --------------------------------------------------------------------------
# the fitted object
# --------------------------------------------------------------------------


# `manager_timing` is shrunk harder than the others on purpose. Each
# `asc[QB|m|early]` is identified off 1-3 events for most managers (see REPORT),
# so its prior has to do most of the work; at 0.25 the fit chases noise.
DEFAULT_TAU = {
    "manager_pos": 0.25,
    "manager_reach": 0.25,
    "manager_taste": 0.25,     # every player-attribute channel but reach
    "manager_timing": 0.15,
    "ridge": 1.0,
}


@dataclass
class Fitted:
    managers: dict[str, int]
    plr_spec: ModelSpec
    plr_std: feat.Standardizer
    gamma: np.ndarray
    pos_spec: ModelSpec
    pos_std: feat.Standardizer
    theta_human: np.ndarray
    theta_auto: np.ndarray
    blocks: frozenset[str]
    tau: dict[str, float]

    @property
    def lam(self) -> float:
        return float(self.theta_human[self.pos_spec.index("lambda_iv")])

    @property
    def with_manager(self) -> bool:
        """Any per-manager channel at all. `predict` gates on this."""
        return bool(self.blocks)

    @property
    def plr_layout(self) -> PlayerLayout:
        return player_layout(len(self.managers), self.blocks)

    @property
    def pos_layout(self) -> PositionLayout:
        """The HUMAN position layout. The auto model is fitted pooled, so it
        always uses the no-manager layout."""
        return position_layout(len(self.managers), self.blocks)

    @property
    def auto_layout(self) -> PositionLayout:
        return position_layout(len(self.managers), NO_BLOCKS)


def train(
    examples: list[Example],
    tau: dict[str, float] | None = None,
    with_manager: bool | None = None,
    blocks: frozenset[str] | None = None,
) -> Fitted:
    """`blocks` names the per-manager channels in this fit; `with_manager` is
    kept as the old boolean shim (True -> all channels, False -> none)."""
    if blocks is None:
        blocks = DEFAULT_BLOCKS if with_manager in (None, True) else NO_BLOCKS
    tau = tau or DEFAULT_TAU
    managers = {m: i for i, m in enumerate(sorted({e.manager for e in examples if e.manager}))}

    usable = [e for e in examples if e.chosen_pos is not None]
    plr_std = feat.Standardizer.fit(
        np.vstack([e.plr_X[p] for e in usable for p in e.positions]), feat.PLR_INDICATORS
    )
    plr_layout = player_layout(len(managers), blocks)
    plr_spec = player_spec(list(managers), blocks)
    pdata = player_data(usable, plr_spec, plr_layout, managers, plr_std)
    gamma = fit(pdata, plr_spec.penalty_vector(tau))

    ivs = [inclusive_values(e, plr_layout, managers, plr_std, gamma) for e in usable]
    pos_std = feat.Standardizer.fit(
        np.vstack([e.pos_X for e in usable]), feat.POS_INDICATORS
    )

    def fit_position(subset_flags, blks):
        sub = [e for e, f in zip(usable, subset_flags) if f]
        sub_iv = [v for v, f in zip(ivs, subset_flags) if f]
        spec = position_spec(list(managers), blks)
        layout = position_layout(len(managers), blks)
        data = position_data(sub, spec, layout, managers, pos_std, sub_iv)
        bounds = [(None, None)] * len(spec.names)
        bounds[layout.lam] = (0.0, 1.2)                # RUM-consistent range
        x0 = np.zeros(len(spec.names))
        x0[layout.lam] = 1.0
        return spec, fit(data, spec.penalty_vector(tau), bounds=bounds, x0=x0)

    human = [not e.is_auto for e in usable]
    auto = [e.is_auto for e in usable]
    pos_spec, theta_human = fit_position(human, blocks)
    _, theta_auto = fit_position(auto, NO_BLOCKS) if any(auto) else (None, None)
    if theta_auto is None:
        theta_auto = np.zeros(position_layout(len(managers), NO_BLOCKS).total)

    return Fitted(
        managers=managers, plr_spec=plr_spec, plr_std=plr_std, gamma=gamma,
        pos_spec=pos_spec, pos_std=pos_std, theta_human=theta_human,
        theta_auto=theta_auto, blocks=frozenset(blocks), tau=tau,
    )
