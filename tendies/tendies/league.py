"""Per-season league rules, read from ESPN rather than assumed.

This league's rules MOVED, and hardcoding today's settings would silently
corrupt every older training row without changing any metric:

    2019      1 FLEX, 7 bench, 16 rounds
    2020      1 FLEX, 7 bench (+IR), 16 rounds
    2021-22   1 FLEX, 6 bench, 15 rounds
    2023-25   2 FLEX, 5 bench, 15 rounds

Position limits are hard caps that BIND in the data (observed maxima QB 4,
RB 8, WR 7, TE 3 against limits QB 4 / RB 8 / WR 8 / TE 3 / K 3 / D-ST 3), so
they belong in the choice set as a mask, not in the utility as a penalty.

Careful: `lineupSlotCounts` is keyed by LINEUP SLOT id and `positionLimits` by
POSITION id, and the two numbering schemes overlap without agreeing (slot 4 is
WR, position 4 is TE). Reading either with the other's map produces a plausible
dict that is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# lineupSlotCounts keys
SLOT_ID = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DST", 17: "K", 20: "BE", 21: "IR", 23: "FLEX"}
# positionLimits keys -- a DIFFERENT numbering
POSITION_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
FLEX_POSITIONS = ("RB", "WR", "TE")

DEFAULT_LIMITS = {"QB": 4, "RB": 8, "WR": 8, "TE": 3, "K": 3, "DST": 3}


@dataclass(frozen=True)
class LeagueConfig:
    season: int
    teams: int
    rounds: int
    starters: dict[str, int]           # dedicated lineup slots by position
    flex: int                          # FLEX slots, filled from FLEX_POSITIONS
    bench: int
    limits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LIMITS))

    @property
    def roster_size(self) -> int:
        return sum(self.starters.values()) + self.flex + self.bench

    def starter_need(self, counts: dict[str, int] | list[int]) -> dict[str, int]:
        """Dedicated starter slots still unfilled at each position."""
        get = (lambda p: counts[POS_INDEX[p]]) if isinstance(counts, list) else counts.get
        return {p: max(0, n - (get(p) or 0)) for p, n in self.starters.items()}


def _unwrap(payload):
    return payload[0] if isinstance(payload, list) and payload else payload


def from_payload(payload: dict, season: int, rounds: int | None = None) -> LeagueConfig | None:
    p = _unwrap(payload)
    settings = (p or {}).get("settings") or {}
    rs = settings.get("rosterSettings") or {}
    slots_raw = rs.get("lineupSlotCounts") or {}
    if not slots_raw:
        return None
    slots = {SLOT_ID.get(int(k), str(k)): int(v) for k, v in slots_raw.items() if int(v) > 0}
    limits = {
        POSITION_ID[int(k)]: int(v)
        for k, v in (rs.get("positionLimits") or {}).items()
        if int(k) in POSITION_ID and int(v) > 0
    }
    starters = {p_: slots.get(p_, 0) for p_ in POSITIONS}
    teams = int(settings.get("size") or 10)
    bench = slots.get("BE", 0)
    total = sum(starters.values()) + slots.get("FLEX", 0) + bench
    return LeagueConfig(
        season=season,
        teams=teams,
        rounds=rounds if rounds is not None else total,
        starters=starters,
        flex=slots.get("FLEX", 0),
        bench=bench,
        limits=limits or dict(DEFAULT_LIMITS),
    )


def load(
    cache_dir: Path, season: int, rounds: int | None = None, fallback: int | None = None
) -> LeagueConfig:
    """League config for a season, falling back to the newest season that has
    settings (2026's payload carries none until the league is set up)."""
    path = cache_dir / f"league_{season}.json"
    if path.exists():
        cfg = from_payload(json.loads(path.read_text()), season, rounds)
        if cfg is not None:
            return cfg
    if fallback is not None:
        borrowed = load(cache_dir, fallback, rounds)
        print(f"  {season}: no roster settings in payload — using {fallback}'s")
        return LeagueConfig(
            season=season, teams=borrowed.teams, rounds=rounds or borrowed.rounds,
            starters=borrowed.starters, flex=borrowed.flex, bench=borrowed.bench,
            limits=borrowed.limits,
        )
    raise FileNotFoundError(f"no league settings for {season} in {cache_dir}")


def draft_order(cache_dir: Path, season: int) -> list[str] | None:
    """Seat order for a season, taken from ESPN once the commissioner sets it.

    Before the draft, ESPN already publishes the full pick list with `teamId`
    filled in and `playerId` at -1 — so the order is knowable in advance, and
    the page should not make you retype it. Returns owner ids by seat, or None
    while the order is still unset.
    """
    path = cache_dir / f"league_{season}.json"
    if not path.exists():
        return None
    payload = _unwrap(json.loads(path.read_text()))
    picks = ((payload or {}).get("draftDetail") or {}).get("picks") or []
    if not picks:
        return None
    first_round = {}
    for p in picks:
        if int(p.get("roundId") or 0) != 1:
            continue
        first_round[int(p.get("roundPickNumber") or 0)] = int(p.get("teamId") or 0)
    if not first_round:
        return None
    team_by_seat = [first_round[k] for k in sorted(first_round)]
    owners = {}
    for t in payload.get("teams") or []:
        o = t.get("primaryOwner") or ((t.get("owners") or [None])[0])
        owners[int(t["id"])] = o
    return [owners.get(t, "") for t in team_by_seat]


def snake_seats(teams: int, rounds: int) -> list[int]:
    """Seat (1-based) owning each overall pick. Verified strict snake in all
    seven seasons — no third-round reversal, no traded picks."""
    seats = []
    for r in range(rounds):
        order = range(1, teams + 1) if r % 2 == 0 else range(teams, 0, -1)
        seats.extend(order)
    return seats
