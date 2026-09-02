"""How the board empties: the measured positional flow through a draft.

The recommendation has to answer "will there still be a kicker when I get back
here?" fifteen rounds out, and it cannot afford to simulate that far. The cheap
alternative everyone reaches for -- assume players leave in ADP order -- is
wrong in exactly the place it is needed:

    kicker ADPs on the 2026 board run 120-257 and defenses 125-372, so under
    ADP depletion every kicker is gone by pick 150 and Tyler Shough (ADP 313)
    gets drafted.

Real drafts do not work that way. Nobody takes a kicker before round 11 and
then everybody takes one at once. So instead of assuming an order, measure it:
for each position, the average cumulative number drafted by each point of the
draft, over this league's own history.

    avg cumulative drafted by pick   QB    RB    WR    TE     K   DST
      100                           9.8  37.0  42.8   9.8   0.0   0.6
      140                          13.2  46.6  54.4  12.8   4.0   9.0
      150                          13.4  47.8  55.0  13.8  10.0  10.0

The same table does two jobs, which is why it is worth measuring properly:

1. **Far-horizon availability.** Applied as an INCREMENT from the current pick
   (`gone_q(now) + N(n) - N(now)`), so it self-corrects for a draft that has
   already run RB-heavy rather than replaying an average draft.
2. **The empty-slot price.** Evaluated past the last pick it says who is still
   undrafted at each position, and the best of those is what you actually start
   if you leave a slot open -- floored at 0 where the position is streamable, so
   the streaming already inside `vorp.REPL_RANKS` is not charged twice. That
   number is the floor in `lineupValue`; see engine.js. Deriving both from one
   table is what keeps them consistent.

Counting uses the pick's own `position` column, not the linked board row, so
the handful of picks that never matched a board player still count against
their position.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import vorp as vorp_mod
from .league import POS_INDEX, POSITIONS


def season_shapes(linked: pl.DataFrame) -> pl.DataFrame:
    """(season, teams, picks) for each drafted season."""
    return (
        linked.group_by("season")
        .agg(teams=pl.col("team_id").n_unique(), picks=pl.len())
        .sort("season")
    )


def _cumulative(picks: pl.DataFrame, n_picks: int) -> np.ndarray:
    """(n_pos, n_picks + 1) cumulative count by overall pick; column 0 is zero."""
    out = np.zeros((len(POSITIONS), n_picks + 1), dtype=np.float64)
    rows = picks.select("overall_pick", "position").sort("overall_pick")
    for overall, pos in rows.iter_rows():
        p = POS_INDEX.get(pos)
        if p is None or not 1 <= overall <= n_picks:
            continue
        out[p, overall] += 1.0
    return np.cumsum(out, axis=1)


def flow_table(
    linked: pl.DataFrame, teams: int, rounds: int, min_seasons: int = 3
) -> tuple[np.ndarray, list[int]]:
    """Mean cumulative picks by position at each point of a draft.

    Returns `(table, seasons_used)` where `table` is (n_pos, steps + 1) and
    column `i` is the average count drafted through overall pick `i` of a
    `steps`-pick draft. Consumers index it by FRACTION of the draft, so a
    league that changes its round count still gets a sensible curve.

    Only seasons with the same shape are averaged when there are enough of
    them; a 16-round draft distributes its kickers differently from a 15-round
    one, and that difference is the whole point of the table. Falling back to
    every season resamples them onto the target length by fraction.
    """
    steps = teams * rounds
    shapes = season_shapes(linked)
    same = shapes.filter((pl.col("teams") == teams) & (pl.col("picks") == steps))
    use = same if same.height >= min_seasons else shapes
    seasons = use["season"].to_list()

    acc = np.zeros((len(POSITIONS), steps + 1), dtype=np.float64)
    for season, n_picks in zip(seasons, use["picks"].to_list()):
        cum = _cumulative(linked.filter(pl.col("season") == season), int(n_picks))
        if n_picks == steps:
            acc += cum
        else:
            # resample by fraction of the draft, then rescale so the endpoint
            # is the number of picks this draft actually makes
            src = np.linspace(0.0, 1.0, int(n_picks) + 1)
            dst = np.linspace(0.0, 1.0, steps + 1)
            for p in range(len(POSITIONS)):
                acc[p] += np.interp(dst, src, cum[p]) * (steps / float(n_picks))
    return acc / max(1, len(seasons)), [int(s) for s in seasons]


def gone_by(table: np.ndarray, pick: int, total: int) -> np.ndarray:
    """Interpolated cumulative count at overall `pick` of a `total`-pick draft."""
    steps = table.shape[1] - 1
    x = np.clip(pick / max(1, total), 0.0, 1.0) * steps
    lo = int(np.floor(x))
    hi = min(lo + 1, steps)
    w = x - lo
    return table[:, lo] * (1.0 - w) + table[:, hi] * w


def empty_values(board, table: np.ndarray, total: int) -> dict[str, float]:
    """What it costs to leave a starting slot open, in VORP.

    Not 0 (which means "a replacement-level player" and is what the engine used
    to assume) but what you would actually field, and that is the best player at
    the position who goes UNDRAFTED -- read off the flow table past the last
    pick.

    At a STREAMABLE position the floor is `max(0, that)`, and the zero is
    load-bearing rather than defensive. `vorp.REPL_RANKS` already prices QB and
    TE against a streaming total, so an empty QB slot is worth exactly 0 by
    construction; charging it the best-undrafted quarterback instead (-5.8 on
    the 2026 board, TE -19.5) discounts the same streaming twice and pays the
    drafter for it. The `max` still lets the LIVE board raise the floor: if the
    room punts quarterbacks and a QB4 goes undrafted, leaving the slot open
    really is worth his VORP.

    Reference implementation for the JS derivation in engine.js -- the parity
    fixtures check the two agree.
    """
    gone = gone_by(table, total, total)
    out = {}
    for p, name in enumerate(POSITIONS):
        pids = board.by_pos[p]
        start = int(round(gone[p]))
        rest = board.vorp[pids[start:]] if start < len(pids) else np.array([])
        best = float(rest.max()) if len(rest) else 0.0
        out[name] = max(0.0, best) if name in vorp_mod.STREAMABLE else best
    return out


def report(linked: pl.DataFrame, teams: int, rounds: int) -> str:
    table, seasons = flow_table(linked, teams, rounds)
    total = teams * rounds
    lines = [
        f"positional flow from {len(seasons)} seasons "
        f"({min(seasons)}-{max(seasons)}), {teams} teams x {rounds} rounds",
        "  pick  " + "".join(f"{p:>7s}" for p in POSITIONS),
    ]
    for n in (25, 50, 75, 100, 120, 140, total):
        g = gone_by(table, n, total)
        lines.append(f"  {n:>4d}  " + "".join(f"{v:7.1f}" for v in g))
    return "\n".join(lines)
