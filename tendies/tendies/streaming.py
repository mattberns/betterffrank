"""Where each position's replacement level comes from — the derivation behind
`vorp.REPL_RANKS`.

Replacement is *the best option you would field at that position for free*. That
is one definition, but it has two arms, because the free option is not the same
kind of thing at every position:

- **QB and TE are streamable.** One starter each in a 10-team league, so ~13 of
  each are drafted and the waiver pool stays deep. A manager who punts the
  position does not roster one man all year; he starts whoever looks best that
  week. So the free option is a STREAMING TOTAL, and it is worth more than any
  single undrafted player's season.
- **RB and WR are not.** Fifty-odd of each come off the board (2 starters plus a
  flex that resolves to RB/WR), the pool behind them is startable-in-name-only,
  and there is no matchup rotation that fixes it. The free option is simply THE
  BEST UNDRAFTED PLAYER.

Both arms are measured here, on this league's own drafts, and both are expressed
as a slot on the VORP curve — because `vorp.vorp_of` subtracts `curve_at(curve,
REPL_RANKS[pos])`, so a rank measured against anything else (a realised
order statistic, say) is not the quantity that gets subtracted.

## The streaming policy, and the haircut

Free pool = players whose within-position ECR slot is worse than `n_owned`. Each
REG week, start the free player with the best PRIOR-weeks form (week 1: best ECR)
and bank his actual half-PPR score. No hindsight: form only ever looks backwards.

That policy is still optimistic in one specific way, and the parent repo flags
it too (`bff/streaming.py`): it hoards the year's breakout for free. In 2018 it
picks up Mahomes in week 2 and keeps him; in a real ten-team league one of nine
rivals claims him. `drop_top` removes the season's biggest free-pool scorers
before streaming, which is the cheapest honest correction, and the sweep in
`main` is there so the sensitivity is visible rather than assumed. At the
n_owned this league actually drafts (QB 14, TE 14), over 2018-2025:

    QB  drop 0 -> 305.9 pts = QB3    drop 1 -> 257.2 = QB7    drop 2 -> 239.9 = QB16
    TE  drop 0 -> 137.2 pts = TE3    drop 1 -> 118.8 = TE6    drop 2 -> 100.2 = TE15

`drop_top=1` is what ships. Zero assumes you win every breakout waiver; two
assumes you win none and that the pool's second-best is also gone. That spread
is the honest uncertainty in these two constants and it is not small — a reader
who believes he wins the breakout should be running QB3/TE3, which would price
quarterbacks and tight ends near zero all the way down the board.

## What this changed, and what it did not

QB7 did NOT move and TE went only 7 -> 6: they were already at the streaming
value, which is why "quarterbacks look overvalued on the board" was a
misdiagnosis. RB25 -> RB43 and WR30 -> WR51 are the fix. Those two were priced
against roster-demand depth (the last starter), which is 38 and 30 points better
than what is actually free, so every back and receiver read that much too low and
QB/TE floated to the top of a column that meant a different thing in each row.

Prints, writes nothing, and is not imported by the fitted pipeline — the same
status `bff/streaming.py` has in the parent repo.

    uv run tendies streaming
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import vorp as vorp_mod
from .config import ACTUALS_PARQUET, CURVE_ADP_PARQUET, STATS_RAW_DIR

REG_WEEKS = 17
# How far either side of the MEASURED roster count to sweep n_owned. The centre
# is not a constant here on purpose: it is however many of the position this
# league actually drafts, read off its own boards, and `main` marks it.
N_OWNED_OFFSETS = (-2, 0, +2)
DROP_TOP = (0, 1, 2)
SHIPPED_DROP_TOP = 1
# How far a streaming total may sit above its reported curve slot before the
# slot stops being a fair summary of it. See `curve_slot`: the curve steps, so a
# total can land between two slots and the rank then hides points.
RESIDUAL_WARN = 5.0


def _weekly_path(season: int):
    """2025 is a differently named file upstream, the same special case the
    parent repo carries in `bff/streaming.py` and `bff/actuals.py`."""
    return (STATS_RAW_DIR / "stats_player_week_2025.parquet" if season == 2025
            else STATS_RAW_DIR / f"player_stats_{season}.parquet")


def _has_weekly(season: int) -> bool:
    return _weekly_path(season).exists()


def weekly(season: int, pos: str) -> pl.DataFrame:
    """(gsis_id, week, pts) — half-PPR weekly scores at one position.

    Half-PPR is the mean of the two nflverse columns rather than a re-scoring
    from components: `fantasy_points` is standard and `fantasy_points_ppr` adds
    one per reception, so their average adds half of one. Anything computed from
    receptions by hand would disagree with `actuals.pts_half`, which is what the
    curve is built from.
    """
    d = pl.read_parquet(_weekly_path(season))
    return (
        d.filter((pl.col("position") == pos) & (pl.col("season_type") == "REG")
                 & (pl.col("week") <= REG_WEEKS))
        .select(pl.col("player_id").alias("gsis_id"), "week",
                ((pl.col("fantasy_points") + pl.col("fantasy_points_ppr")) / 2).alias("pts"))
    )


def slot_history(ecr: pl.DataFrame | None = None) -> pl.DataFrame:
    """`vorp.market_slot_points` plus the gsis_id it drops.

    The extra column is the whole point: streaming needs to know which PLAYERS
    are inside `n_owned`, not just how many. Built through `vorp.slot_key` so
    the slots are the same ECR-indexed ones the curve uses.
    """
    from . import dataset as ds

    if ecr is None:
        ecr, _ = ds.load_ecr()
    boards = pl.read_parquet(CURVE_ADP_PARQUET).join(
        ecr.drop_nulls("gsis_id").unique(subset=["season", "gsis_id"], keep="first"),
        on=["season", "gsis_id"], how="left",
    )
    act = pl.read_parquet(ACTUALS_PARQUET).select(
        "season", "gsis_id", pl.col("pts_half").alias("pts"))
    df = (
        boards.filter(pl.col("position").is_in(vorp_mod.CURVE_POSITIONS)
                      & pl.col("gsis_id").is_not_null())
        .join(act, on=["season", "gsis_id"], how="left")
        .with_columns(pl.col("pts").fill_null(0.0))
    )
    return vorp_mod.slot_key(df).select("season", "position", "pos_rank", "pts", "gsis_id")


def stream_total(hist: pl.DataFrame, season: int, pos: str, n_owned: int,
                 drop_top: int = SHIPPED_DROP_TOP) -> float:
    """Half-PPR points a position-punting manager banks by form-streaming the
    free pool, after `drop_top` of its biggest scorers are claimed by rivals."""
    h = hist.filter((pl.col("season") == season) & (pl.col("position") == pos))
    rank = dict(zip(h["gsis_id"].to_list(), h["pos_rank"].to_list()))
    owned = [g for g, r in rank.items() if r <= n_owned]
    free = weekly(season, pos).filter(~pl.col("gsis_id").is_in(owned))
    if drop_top:
        claimed = (free.group_by("gsis_id").agg(pl.col("pts").sum())
                   .sort("pts", descending=True)["gsis_id"].to_list()[:drop_top])
        free = free.filter(~pl.col("gsis_id").is_in(claimed))
    total = 0.0
    for w in sorted(free["week"].unique().to_list()):
        cur = free.filter(pl.col("week") == w)
        if cur.height == 0:
            continue
        # form is PRIOR weeks only -- week 1 has none, and falls back to ECR
        prior = (free.filter(pl.col("week") < w).group_by("gsis_id")
                 .agg(pl.col("pts").mean().alias("form")))
        cand = (cur.join(prior, on="gsis_id", how="left")
                .with_columns(pl.col("gsis_id").replace_strict(rank, default=10**6).alias("pr"))
                .sort(["form", "pr"], descending=[True, False], nulls_last=True))
        total += float(cand["pts"][0])
    return total


def curve_slot(curve: np.ndarray, value: float) -> int:
    """The first curve slot worth no more than `value` — i.e. the replacement
    rank whose subtracted curve value equals this free option.

    A SLOT IS A LOSSY ENCODING OF A TOTAL, and this curve makes that bite. It is
    a step function with long flat runs (isotonic regression pools adjacent
    violators) and both streamable positions plateau right where their streaming
    total lands: the 2026 QB curve holds 250 from QB8 all the way to QB15, then
    drops to 221. A total of 250.1 reads QB8 and 249.9 reads QB16, and the value
    actually subtracted differs by 29 points across that hair.

    So check the RESIDUAL (`total - curve_at(slot)`) before trusting a slot.
    `main` prints it and flags anything over `RESIDUAL_WARN`. The shipped
    constants sit +2.1 (QB) and +1.2 (TE) points from their slots, which is why
    they are safe to quote as ranks; the drop_top=2 rows land mid-step and are
    not.
    """
    below = curve <= value
    return int(np.argmax(below)) + 1 if below.any() else len(curve)


def slot_spread(curve: np.ndarray, totals: list[float]) -> tuple[int, int, float]:
    """(lowest slot, highest slot, points spread) over a set of streaming totals.

    What the slot readout's instability is actually worth. A wide slot range with
    a narrow points spread means the choice of replacement rank inside the range
    barely moves the board, which is the case at both streamable positions.
    """
    slots = [curve_slot(curve, t) for t in totals]
    lo, hi = min(slots), max(slots)
    return lo, hi, float(curve[lo - 1] - curve[hi - 1])


def undrafted_depth(boards: pl.DataFrame, linked: pl.DataFrame) -> pl.DataFrame:
    """(season, position, drafted, best_undrafted_slot) from the REAL drafts.

    The non-streamable arm. Note it is not "the n-th slot after n were drafted":
    managers do not draft in expert order, so the best player left is reliably
    better than the count implies (48 backs off the board leaves ~RB43).
    """
    rows = []
    for s in sorted(linked["season"].unique().to_list()):
        b = boards.filter(pl.col("season") == s)
        p = linked.filter(pl.col("season") == s)
        taken = p["pid"].drop_nulls().to_list()
        for pos in vorp_mod.CURVE_POSITIONS:
            left = b.filter((pl.col("position") == pos) & (~pl.col("pid").is_in(taken)))
            rows.append({
                "season": s, "position": pos,
                "drafted": p.filter(pl.col("position") == pos).height,
                "best_undrafted_slot": int(left["pos_slot"].min()) if left.height else None,
            })
    return pl.DataFrame(rows)


def main(league_id: int | None = None) -> None:
    from . import dataset as ds
    from .config import DEFAULT_LEAGUE_ID

    boards, linked, _ = ds.build_all(league_id or DEFAULT_LEAGUE_ID)
    hist = slot_history()
    live = max(int(s) for s in boards["season"].unique().to_list())
    curve = vorp_mod.build_curve(hist.select("season", "position", "pos_rank", "pts"), live)
    # The streaming arm needs boards and weekly scores, not this league's picks,
    # so it runs on every scored board season -- 2018 included, which the draft
    # history (first draft 2019) does not reach.
    seasons = [s for s in sorted(int(x) for x in boards["season"].unique().to_list())
               if s < live and _has_weekly(s)]

    depth = undrafted_depth(boards, linked)
    drafts = sorted(int(s) for s in linked["season"].unique().to_list())
    print(f"\nBEST UNDRAFTED SLOT — this league's real drafts, {drafts[0]}-{drafts[-1]}")
    print(f"{'pos':>4s} | {'mean drafted':>12s} | {'mean best undrafted':>19s} | per season")
    free_slot, n_owned = {}, {}
    for pos in vorp_mod.CURVE_POSITIONS:
        d = depth.filter(pl.col("position") == pos).drop_nulls("best_undrafted_slot")
        free_slot[pos] = float(d["best_undrafted_slot"].mean())
        n_owned[pos] = int(round(float(d["drafted"].mean())))
        per = " ".join(f"{v:>3d}" for v in d["best_undrafted_slot"].to_list())
        print(f"{pos:>4s} | {float(d['drafted'].mean()):12.1f} | "
              f"{pos}{free_slot[pos]:<18.1f} | {per}")

    for pos in sorted(vorp_mod.STREAMABLE):
        centre = n_owned[pos]
        grid = [centre + o for o in N_OWNED_OFFSETS]
        print(f"\n{pos} STREAMING — mean half-PPR points banked by punting the position "
              f"({seasons[0]}-{seasons[-1]}), and the {pos} curve slot worth the same")
        print(f"{'drop_top':>8s} | " + "  ".join(
            f"n_owned={n}{'*' if n == centre else ' '}" for n in grid))
        shipped = {}
        for drop in DROP_TOP:
            line = f"{drop:>8d} | "
            for n in grid:
                tot = float(np.mean([stream_total(hist, s, pos, n, drop) for s in seasons]))
                shipped.setdefault(drop, {})[n] = tot
                line += f"  {tot:6.1f} -> {pos}{curve_slot(curve[pos], tot):<4d}"
            print(line)
        print(f"  * n_owned={centre} is what this league actually drafts; the shipped "
              f"replacement is that column at drop_top={SHIPPED_DROP_TOP}")
        at = shipped[SHIPPED_DROP_TOP][centre]
        slot = curve_slot(curve[pos], at)
        lo, hi, spread = slot_spread(curve[pos], list(shipped[SHIPPED_DROP_TOP].values()))
        resid = at - vorp_mod.curve_at(curve[pos], slot)
        print(f"    -> {at:.1f} pts = {pos}{slot} (residual {resid:+.1f} pts — "
              + ("the rank is a fair statement of the total"
                 if resid < RESIDUAL_WARN else
                 f"OVER {RESIDUAL_WARN:.0f}: the total lands mid-step and no slot equals it, "
                 "so quote the points, not the rank")
              + "; see curve_slot)")
        print(f"       across the n_owned sweep {pos}{lo}-{pos}{hi}, worth {spread:.0f} points "
              f"of curve, and across drop_top {pos}{curve_slot(curve[pos], shipped[0][centre])}-"
              f"{pos}{curve_slot(curve[pos], shipped[2][centre])}. Both are real "
              "uncertainty, not a rounding choice.")
        print(f"  best single undrafted {pos} is {pos}{free_slot[pos]:.1f} "
              f"({vorp_mod.curve_at(curve[pos], round(free_slot[pos])):.0f} pts) — streaming "
              f"beats it, which is why {pos} takes the streaming arm")

    print("\nSHIPPED (vorp.REPL_RANKS): " + ", ".join(
        f"{p}={vorp_mod.REPL_RANKS[p]}" for p in vorp_mod.CURVE_POSITIONS))
    print(f"  QB/TE from the streaming arm at drop_top={SHIPPED_DROP_TOP}; "
          "RB/WR from the best-undrafted arm above.")
    print("  curve value at each replacement rank (%d curve): " % live + ", ".join(
        f"{p}{vorp_mod.REPL_RANKS[p]}={vorp_mod.curve_at(curve[p], vorp_mod.REPL_RANKS[p]):.0f}"
        for p in vorp_mod.CURVE_POSITIONS))
