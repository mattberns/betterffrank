"""The draft-state engine. Everything else is built on this.

One code path serves three callers, and that symmetry is the whole point:

    training    advance() replays the historical pick
    simulation  advance() applies a pick sampled from the model
    live page   advance() applies the pick the user clicked

The state at pick n is therefore built from picks 1..n-1 in every path, so
there is no train/serve asymmetry available to get wrong, and no feature can
be "accidentally offline-only" — if it were, simulation would crash.

Two rules protect that and are easy to violate later:

1. **No feature may consult the future.** Look-ahead (how many players at this
   position will be gone by my next turn) uses the deterministic ADP-order
   depletion assumption, the same one `bff/vona.py` makes. Never a nested
   simulation inside a feature: it makes the outer simulation quadratic and
   makes training features stochastic, which breaks byte-stability and
   Python/JS parity in one move.
2. **No feature may read a season outcome.** VORP enters only through the
   walk-forward curve, which filters to seasons < t internally.

Position caps are a MASK on the choice set, not a penalty in the utility: ESPN
enforces them, so a capped position is not a bad choice, it is not a choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .league import POS_INDEX, POSITIONS, LeagueConfig, snake_seats

# Per-position candidate windows. Wider for the dense positions; a global
# top-N starves the thin ones, which leaves a feasible position with an
# undefined inclusive value. Scaled by `kappa` at call time.
WINDOW = {"QB": 10, "RB": 20, "WR": 20, "TE": 10, "K": 10, "DST": 10}


@dataclass(frozen=True)
class Board:
    """One season's player universe as flat arrays indexed by pid."""

    season: int
    name: list[str]
    pro_team: list[str]
    pos: np.ndarray          # int8, index into POSITIONS
    adp_rank: np.ndarray     # int32, == pid + 1
    pos_rank: np.ndarray     # int32, within-position ordinal
    ecr_rank: np.ndarray     # float64, nan where absent (always for K/DST)
    vorp: np.ndarray         # float64, 0.0 for K/DST by design
    bye: np.ndarray          # int16, 0 = unknown
    rookie: np.ndarray       # int8, 1 = NFL entry year == this season
    age: np.ndarray          # float64, years vs typical at his position
    durability: np.ndarray   # float64, share of last season played, vs typical
    prev_owner: list[str]    # franchise that drafted him last season, "" if none
    by_pos: list[np.ndarray]  # pids at each position, ADP order
    slot_of: np.ndarray      # int32, index of each pid within its by_pos array

    @property
    def n(self) -> int:
        return len(self.name)

    @staticmethod
    def from_frame(df: pl.DataFrame) -> "Board":
        df = df.sort("pid")
        pos = np.array([POS_INDEX[p] for p in df["position"]], dtype=np.int8)
        by_pos, slot_of = [], np.zeros(df.height, dtype=np.int32)
        for i in range(len(POSITIONS)):
            pids = np.flatnonzero(pos == i).astype(np.int32)
            by_pos.append(pids)
            slot_of[pids] = np.arange(len(pids), dtype=np.int32)
        ecr = df["ecr_rank"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        return Board(
            season=int(df["season"][0]),
            name=df["name"].to_list(),
            pro_team=[t or "" for t in df["pro_team"].to_list()],
            pos=pos,
            adp_rank=df["adp_rank"].to_numpy().astype(np.int32),
            pos_rank=df["pos_rank"].to_numpy().astype(np.int32),
            ecr_rank=ecr,
            vorp=df["vorp"].to_numpy().astype(np.float64) if "vorp" in df.columns
            else np.zeros(df.height),
            bye=df["bye"].fill_null(0).to_numpy().astype(np.int16),
            rookie=df["rookie"].fill_null(0).to_numpy().astype(np.int8)
            if "rookie" in df.columns
            else np.zeros(df.height, dtype=np.int8),
            age=df["age"].fill_null(0.0).to_numpy().astype(np.float64)
            if "age" in df.columns
            else np.zeros(df.height),
            durability=df["durability"].fill_null(0.0).to_numpy().astype(np.float64)
            if "durability" in df.columns
            else np.zeros(df.height),
            prev_owner=[o or "" for o in df["prev_owner"].to_list()]
            if "prev_owner" in df.columns
            else [""] * df.height,
            by_pos=by_pos,
            slot_of=slot_of,
        )


class DraftState:
    """Mutable draft position. `advance` and `undo` are exact inverses."""

    def __init__(
        self,
        board: Board,
        cfg: LeagueConfig,
        managers: list[str],
        seats: list[int] | None = None,
    ) -> None:
        self.board = board
        self.cfg = cfg
        self.managers = managers                       # seat index 0..teams-1
        self.seats = seats or snake_seats(cfg.teams, cfg.rounds)
        self.picks: list[int | None] = []
        self.taken = np.zeros(board.n, dtype=bool)
        self.counts = np.zeros((cfg.teams, len(POSITIONS)), dtype=np.int16)
        self.rosters: list[list[int]] = [[] for _ in range(cfg.teams)]
        self.cursor = np.zeros(len(POSITIONS), dtype=np.int32)
        self.recent: list[int] = []                    # position index per pick, oldest first

    # ---- geometry --------------------------------------------------------

    @property
    def pick_no(self) -> int:
        """1-based overall pick currently on the clock."""
        return len(self.picks) + 1

    def seat_on_clock(self) -> int:
        """0-based seat index, or -1 past the end of the draft."""
        i = len(self.picks)
        return self.seats[i] - 1 if i < len(self.seats) else -1

    def next_pick_for(self, seat: int, after: int | None = None) -> int | None:
        """Overall pick number (1-based) of that seat's next turn."""
        start = (after if after is not None else len(self.picks))
        for i in range(start, len(self.seats)):
            if self.seats[i] - 1 == seat:
                return i + 1
        return None

    def horizon(self, seat: int) -> int:
        """How many picks happen before this seat picks again.

        If the seat is ON THE CLOCK the answer is about its FOLLOWING turn —
        "who survives until I pick again" — so the current pick is skipped.
        Otherwise it is the wait until this seat's upcoming turn. In a 10-team
        snake this runs 1 at the turn to 18 at the wrap.
        """
        i = len(self.picks)
        on_clock = i < len(self.seats) and self.seats[i] - 1 == seat
        nxt = self.next_pick_for(seat, after=i + 1 if on_clock else i)
        if nxt is None:
            return 0
        return nxt - self.pick_no - (1 if on_clock else 0)

    # ---- choice set ------------------------------------------------------

    def _advance_cursor(self, p: int) -> None:
        arr = self.board.by_pos[p]
        c = self.cursor[p]
        while c < len(arr) and self.taken[arr[c]]:
            c += 1
        self.cursor[p] = c

    def available(self, p: int, k: int) -> np.ndarray:
        """Top-k available pids at position p, in ADP order."""
        self._advance_cursor(p)
        arr = self.board.by_pos[p]
        out = np.empty(k, dtype=np.int32)
        n = 0
        i = self.cursor[p]
        while i < len(arr) and n < k:
            pid = arr[i]
            if not self.taken[pid]:
                out[n] = pid
                n += 1
            i += 1
        return out[:n]

    def open_positions(self, seat: int) -> list[int]:
        """Positions the seat may still legally draft."""
        counts = self.counts[seat]
        out = []
        for p, name in enumerate(POSITIONS):
            if counts[p] >= self.cfg.limits.get(name, 99):
                continue
            self._advance_cursor(p)
            if self.cursor[p] < len(self.board.by_pos[p]):
                out.append(p)
        return out

    def choice_set(self, seat: int, kappa: float = 1.0) -> dict[int, np.ndarray]:
        """{position index: candidate pids}. Capped and exhausted positions are
        absent entirely — they are not choices."""
        return {
            p: self.available(p, max(1, int(round(WINDOW[POSITIONS[p]] * kappa))))
            for p in self.open_positions(seat)
        }

    # ---- mutation --------------------------------------------------------

    def advance(self, pid: int | None) -> None:
        """Apply one pick to the seat on the clock. `None` records a pick of a
        player absent from the board without attributing him a position."""
        seat = self.seat_on_clock()
        self.picks.append(pid)
        if pid is None:
            self.recent.append(-1)
            return
        assert not self.taken[pid], f"pid {pid} already drafted"
        self.taken[pid] = True
        p = int(self.board.pos[pid])
        if seat >= 0:
            self.counts[seat, p] += 1
            self.rosters[seat].append(pid)
        self.recent.append(p)

    def undo(self) -> None:
        if not self.picks:
            return
        pid = self.picks.pop()
        self.recent.pop()
        if pid is None:
            return
        seat = self.seat_on_clock()
        self.taken[pid] = False
        p = int(self.board.pos[pid])
        if seat >= 0:
            self.counts[seat, p] -= 1
            self.rosters[seat].remove(pid)
        # the cursor may only move backwards on undo
        self.cursor[p] = min(self.cursor[p], self.board.slot_of[pid])

    def copy(self) -> "DraftState":
        """Fork for simulation. Board and seats are shared by reference."""
        st = DraftState.__new__(DraftState)
        st.board, st.cfg, st.managers, st.seats = self.board, self.cfg, self.managers, self.seats
        st.picks = list(self.picks)
        st.taken = self.taken.copy()
        st.counts = self.counts.copy()
        st.rosters = [list(r) for r in self.rosters]
        st.cursor = self.cursor.copy()
        st.recent = list(self.recent)
        return st

    def board_rank_of(self, pid: int, positions: set[int] | None = None) -> int:
        """0-based rank of `pid` among available players in ADP order.

        Cheap because the board IS the ADP order (pid == adp_rank - 1), so
        "how many available players are ahead of him" is a count of untaken
        lower pids. `positions` restricts the count, which is how the
        cap-masked baseline is scored.
        """
        ahead = ~self.taken[:pid]
        if positions is not None:
            ahead = ahead & np.isin(self.board.pos[:pid], list(positions))
        return int(ahead.sum())

    def baseline_ranks(self, seat: int, pid: int) -> tuple[int, int, int]:
        """Rank of the actual pick under the three model-free rules.

        B0 best available by ADP; B1 also respects position caps and roster
        limits; B2 additionally takes an empty K/D-ST slot in the last two
        rounds. B1 and B2 cost nothing and are the honest bar — a model that
        cannot beat "don't draft a third kicker" has not earned its keep.
        """
        b0 = self.board_rank_of(pid)
        allowed = set(self.open_positions(seat))
        b1 = self.board_rank_of(pid, allowed)
        rounds_left = self.cfg.rounds - ((self.pick_no - 1) // self.cfg.teams)
        forced = set()
        if rounds_left <= 2:
            for name in ("K", "DST"):
                p = POS_INDEX[name]
                if p in allowed and self.counts[seat, p] < self.cfg.starters.get(name, 1):
                    forced.add(p)
        if forced:
            p = int(self.board.pos[pid])
            b2 = self.board_rank_of(pid, forced) if p in forced else (
                sum(int((~self.taken[self.board.by_pos[q]]).sum()) for q in forced) + b1
            )
        else:
            b2 = b1
        return b0, b1, b2

    def runs(self, window: int) -> np.ndarray:
        """Picks at each position within the last `window` overall picks."""
        out = np.zeros(len(POSITIONS), dtype=np.float64)
        for p in self.recent[-window:]:
            if p >= 0:
                out[p] += 1
        return out


def replay_season(
    board: Board,
    cfg: LeagueConfig,
    picks: pl.DataFrame,
    kappa: float = 1.0,
):
    """Walk one season's picks, yielding the state as it was BEFORE each.

    Yields `(state, row, pid, chosen_pos, in_window)`. The caller builds
    features from `state` and then the generator advances it, so a feature
    physically cannot see the pick it is predicting.
    """
    seats_by_pick = picks.sort("overall_pick")["team_id"].to_list()
    seat_ids = sorted(set(seats_by_pick))
    seat_of_team = {t: i for i, t in enumerate(seat_ids)}
    managers = [""] * len(seat_ids)
    for row in picks.unique(subset=["team_id"]).iter_rows(named=True):
        managers[seat_of_team[row["team_id"]]] = row["franchise_id"]

    order = [seat_of_team[t] + 1 for t in seats_by_pick]
    st = DraftState(board, cfg, managers, seats=order)

    for row in picks.sort("overall_pick").iter_rows(named=True):
        pid = row["pid"]
        pid = None if pid is None else int(pid)
        seat = st.seat_on_clock()
        in_window = False
        chosen_pos = None
        if pid is not None:
            chosen_pos = int(board.pos[pid])
            cs = st.choice_set(seat, kappa)
            in_window = chosen_pos in cs and pid in cs[chosen_pos]
        yield st, row, pid, chosen_pos, in_window
        st.advance(pid)
