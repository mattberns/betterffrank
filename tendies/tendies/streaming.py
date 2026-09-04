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

## The policy's OTHER defect, and why it does not move the constants

The bare prior-weeks mean is a bad predictor in September and it benches a stud
after two quiet games. Measured 2026-09-03 over 2013-2025, it drops the
consensus TE1 in 10 of 13 seasons at a cost of 12.5 points a season, and it
fails `check_dominance` — hold-and-stream lands BELOW hold-alone in 75 of 182
tight-end and 92 of 182 quarterback player-seasons, worst -86 points. Always
starting your own man is free, so a rule that cannot match it is discarding
points, and any measurement built on it understates.

`SHRINK_WEEKS` mixes preseason prior into the form estimate and roughly halves
the violation rate. **It does not move REPL_RANKS**: the streaming totals rise
+6.2 (QB) and +5.5 (TE) and both stay on the same curve slot, QB7 and TE6. So
the shipped constants are stable under the repair, which is why `stream_total`
still defaults to shrink=0 and the derivation above is unchanged. Where it
matters is `option_value` — see the block comment in `main`.

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

from functools import lru_cache

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

# Pseudo-weeks of PRESEASON prior mixed into each player's in-season form before
# the weekly start decision. Zero reproduces the bare prior-weeks mean, which is
# what the shipped REPL_RANKS were derived under and what `stream_total`
# therefore still defaults to.
#
# The bare mean is a bad predictor in September and it benches a stud after two
# quiet games. Measured 2026-09-03 over 2013-2025: it dropped the consensus TE1
# in 10 of 13 seasons at a cost of 12.5 points a season, and violated
# `check_dominance`'s identity in 75 of 182 tight-end and 92 of 182 quarterback
# player-seasons. A policy that loses to "just start my guy" is not a fair
# measure of what holding a player is worth, so every option-value number taken
# from the unshrunk rule is contaminated DOWNWARD by an unknown amount.
#
# SHRINK_WEEKS is swept, not asserted, for the same reason `drop_top` is: it is a
# claim about waiver behaviour that the data can bound but not pin. `main` prints
# the whole grid. Nothing in the shipped pipeline reads either constant.
#
# THE VIOLATION RATE NEVER FLATTENS, so it does not select a value. Swept
# 2026-09-03: QB 51/36/34/26/18% and TE 41/34/29/25/20% at shrink 0/1/2/4/8,
# falling monotonically throughout. An earlier version of this comment claimed a
# plateau; there is none in the swept range, and 4.0 is a midpoint, not an
# optimum.
#
# What makes the measurement survive that anyway is that the KNOB'S ENDPOINTS ARE
# BOTH DEFENSIBLE and the answer is the same at both. At shrink 0 the rule is the
# bare prior-weeks mean. As shrink grows, form collapses to the preseason prior,
# and since the rostered man is inside `n_owned` while the pool is outside, the
# policy converges on "start your guy every week, stream his bye" -- which is
# knob-free and is what a real manager does. So this interpolates between two
# sensible policies rather than running off, and the option value is POSITIVE AND
# SIGNIFICANT at every value except shrink 0, the one policy independently known
# to be broken.
#
# TE is the striking case: +20.2 / +19.6 / +19.0 / +17.6 at shrink 1/2/4/8
# (t 3.0-3.5) against +10.0 (t 1.52) at shrink 0. Flat across a factor of eight.
# So the TE result is NOT an artefact of the knob -- the knob only has to leave
# the broken setting. Do not repeat the concern that shrinkage manufactured it;
# it was a reasonable worry and the sweep answers it.
SHRINK_WEEKS = 4.0
SHRINK_SWEEP = (0.0, 1.0, 2.0, 4.0, 8.0)


def _weekly_path(season: int):
    """2025 is a differently named file upstream, the same special case the
    parent repo carries in `bff/streaming.py` and `bff/actuals.py`."""
    return (STATS_RAW_DIR / "stats_player_week_2025.parquet" if season == 2025
            else STATS_RAW_DIR / f"player_stats_{season}.parquet")


def _has_weekly(season: int) -> bool:
    return _weekly_path(season).exists()


# The option-value sweep asks for the same (season, position) frame hundreds of
# times -- 14 slots x 13 seasons x 2 shrink settings x 2 positions -- and each
# call was re-reading a whole weekly parquet off disk. That is what killed the
# 2026-09-03 run. Polars frames are treated as immutable everywhere here, so
# handing out the same object is safe.
@lru_cache(maxsize=None)
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


def season_curve(hist: pl.DataFrame, season: int, pos: str):
    """`vorp.build_curve` for one season, cached across the option-value sweep.

    Keyed on the history's shape and season, not on the frame itself (polars
    frames are unhashable). Every caller in this module passes the one `hist`
    that `main` built, so the key is sufficient.
    """
    key = (id(hist), int(season))
    hit = _CURVE_CACHE.get(key)
    if hit is None:
        hit = vorp_mod.build_curve(
            hist.select("season", "position", "pos_rank", "pts"), season)
        _CURVE_CACHE[key] = hit
    return hit.get(pos)


_CURVE_CACHE: dict = {}


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


def _form(pool: pl.DataFrame, week: int, rank: dict, curve=None,
          shrink: float = 0.0) -> pl.DataFrame:
    """(gsis_id, form) — each player's start-worthiness going INTO `week`.

    PRIOR weeks only, so there is no hindsight at any `shrink`. With
    `shrink == 0` this is the bare prior-weeks mean the shipped constants were
    derived under, and week 1 has no prior at all (null, which sorts last and
    hands the decision to the ECR tiebreak).

    With `shrink > 0` the mean is pulled toward the points-per-game the player's
    own PRESEASON slot implies, `shrink` pseudo-weeks strong. `curve` is built
    from seasons strictly before this one, so the prior is preseason information
    and the rule stays walk-forward. An unranked free-pool player clamps to the
    end of the curve, which is the correct low prior for a waiver body.
    """
    prior = (pool.filter(pl.col("week") < week).group_by("gsis_id")
             .agg(pl.col("pts").mean().alias("m"), pl.len().alias("n")))
    if curve is None or shrink <= 0:
        return prior.select("gsis_id", pl.col("m").alias("form"))
    ppg = {g: vorp_mod.curve_at(curve, rank.get(g, 10**6)) / REG_WEEKS
           for g in pool["gsis_id"].unique().to_list()}
    # every pool player gets a form, not only those with a prior week: at
    # shrink > 0 week 1 is the preseason prior rather than a null that the ECR
    # tiebreak has to resolve.
    return (
        pool.select("gsis_id").unique()
        .join(prior, on="gsis_id", how="left")
        .with_columns(
            pl.col("m").fill_null(0.0), pl.col("n").fill_null(0),
            pl.col("gsis_id").replace_strict(ppg, default=0.0).alias("p0"),
        )
        .select(
            "gsis_id",
            ((pl.col("m") * pl.col("n") + pl.col("p0") * shrink)
             / (pl.col("n") + shrink)).alias("form"),
        )
    )


def _free_pool(hist: pl.DataFrame, season: int, pos: str, n_owned: int,
               drop_top: int, exclude: str | None = None):
    """(free-pool weekly frame, within-position rank map) for one season."""
    h = hist.filter((pl.col("season") == season) & (pl.col("position") == pos))
    rank = dict(zip(h["gsis_id"].to_list(), h["pos_rank"].to_list()))
    owned = [g for g, r in rank.items() if r <= n_owned]
    free = weekly(season, pos).filter(~pl.col("gsis_id").is_in(owned))
    if exclude is not None:
        free = free.filter(pl.col("gsis_id") != exclude)
    if drop_top:
        claimed = (free.group_by("gsis_id").agg(pl.col("pts").sum())
                   .sort("pts", descending=True)["gsis_id"].to_list()[:drop_top])
        free = free.filter(~pl.col("gsis_id").is_in(claimed))
    return free, rank


def stream_total(hist: pl.DataFrame, season: int, pos: str, n_owned: int,
                 drop_top: int = SHIPPED_DROP_TOP, curve=None,
                 shrink: float = 0.0) -> float:
    """Half-PPR points a position-punting manager banks by form-streaming the
    free pool, after `drop_top` of its biggest scorers are claimed by rivals.

    `curve`/`shrink` default OFF, so the shipped `REPL_RANKS` derivation is
    unchanged. Pass them to see what the shrunk rule (see `SHRINK_WEEKS`) would
    make the floor; that is a live question and `main` prints both.
    """
    free, rank = _free_pool(hist, season, pos, n_owned, drop_top)
    total = 0.0
    for w in sorted(free["week"].unique().to_list()):
        cur = free.filter(pl.col("week") == w)
        if cur.height == 0:
            continue
        cand = (cur.join(_form(free, w, rank, curve, shrink), on="gsis_id", how="left")
                .with_columns(pl.col("gsis_id").replace_strict(rank, default=10**6).alias("pr"))
                .sort(["form", "pr"], descending=[True, False], nulls_last=True))
        total += float(cand["pts"][0])
    return total


def option_value(hist: pl.DataFrame, season: int, pos: str, n_owned: int,
                 floor: float, drop_top: int = SHIPPED_DROP_TOP, curve=None,
                 shrink: float = 0.0) -> list[tuple]:
    """Per drafted slot: `(slot, his own total, roster-him-AND-stream,
    what the ENGINE prices, what the OLD test priced)`.

    `lineupValueWith` prices an occupied slot at `max(his value, the streaming
    floor)`, and that is not the same quantity as the season total of WEEKLY
    maxima. Roster a tight end and you do not stop streaming: each week you start
    whichever of him and the best free tight end looks better. So the model
    understates an occupied streamable slot by the option value of holding both,
    and at a slot below the floor it understates it by the whole thing — a tight
    end worth less than streaming prices identically to an EMPTY slot, which is
    why no such tight end can ever earn a pick.

    **The baseline is `max(curve[slot], floor)`, and getting that wrong is what
    sank the first version of this measurement.** The engine prices a slot from
    the CURVE, which is an expectation; the old test compared against
    `max(his REALIZED total, floor)`. `E[max(X, c)] > max(E[X], c)` for any X
    with spread, so the old baseline was inflated by that Jensen gap — measured
    2026-09-03 at +13.0 points over TE8-14 and +29.6 over QB8-14, against effects
    of +9 and +28. That is the whole reason the test read zero. The old
    quantity is still returned as the fifth element so the two are comparable.

    The policy is the no-hindsight rule `stream_total` uses, applied to the
    rostered player and the free pool ALIKE: `shrink` pseudo-weeks of preseason
    prior mixed into a prior-weeks mean, then start whoever grades higher and
    bank his actual points. Applying it symmetrically matters — the unshrunk
    rule benched the consensus TE1 in 10 of 13 seasons because his two quiet
    Septembers outweighed nothing at all. His bye and inactive weeks are
    streamed. Rostering him also removes him from everyone else's free pool,
    which is why `free` excludes him.
    """
    h = hist.filter((pl.col("season") == season) & (pl.col("position") == pos))
    rank = dict(zip(h["gsis_id"].to_list(), h["pos_rank"].to_list()))
    wk = weekly(season, pos)
    shrunk = curve is not None and shrink > 0
    out = []
    for slot in range(1, n_owned + 1):
        mine = [g for g, r in rank.items() if r == slot]
        if not mine:
            continue
        me = mine[0]
        free, _ = _free_pool(hist, season, pos, n_owned, drop_top, exclude=me)
        mypts = {int(r["week"]): float(r["pts"])
                 for r in wk.filter(pl.col("gsis_id") == me).iter_rows(named=True)}
        my_p0 = vorp_mod.curve_at(curve, slot) / REG_WEEKS if shrunk else None
        both = 0.0
        for w in sorted(free["week"].unique().to_list()):
            cur = free.filter(pl.col("week") == w)
            if cur.height == 0:
                continue
            cand = (cur.join(_form(free, w, rank, curve, shrink), on="gsis_id", how="left")
                    .with_columns(pl.col("gsis_id")
                                  .replace_strict(rank, default=10**6).alias("pr"))
                    .sort(["form", "pr"], descending=[True, False], nulls_last=True))
            spts, sform = float(cand["pts"][0]), cand["form"][0]
            m = mypts.get(w)
            if m is None:
                both += spts                       # his bye or a scratch: stream it
                continue
            pri = [mypts[x] for x in mypts if x < w]
            if shrunk:
                mf = (float(np.sum(pri)) + my_p0 * shrink) / (len(pri) + shrink)
            else:
                mf = float(np.mean(pri)) if pri else None
            both += m if (mf is not None and (sform is None or mf > float(sform))) else spts
        alone = sum(mypts.values())
        priced = max(vorp_mod.curve_at(curve, slot), floor) if curve is not None else max(alone, floor)
        out.append((slot, alone, both, priced, max(alone, floor)))
    return out


def check_dominance(rows: list[tuple]) -> tuple[int, int, float]:
    """`(violations, n, worst)` of the identity HOLD+STREAM >= HOLD ALONE.

    Always starting your own player is a policy available to the streamer at no
    information cost, so a streaming rule that cannot match it is losing points
    it did not have to lose. Ex-post regret makes the odd violation inevitable;
    a rate near half the sample does not. This is the gate the option-value
    number has to pass before it can mean anything, because a policy that
    destroys value understates what holding a player is worth.
    """
    d = [both - alone for _, alone, both, _, _ in rows]
    return sum(x < -1e-6 for x in d), len(d), (min(d) if d else 0.0)


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
    # ...but the OPTION-VALUE measurement below is not limited to the ADP board's
    # range, and reading `boards` for it threw away five seasons for nothing.
    # It needs a curve slot and a weekly score, which is `hist` (2012-) and
    # data/raw/stats (1999-); `boards` is the half-PPR window, which starts in
    # 2018 for a publisher reason with no bearing here. The first curve season
    # cannot be scored (nothing precedes it), hence the +1.
    opt_seasons = [s for s in sorted(int(x) for x in hist["season"].unique().to_list())
                   if min(int(x) for x in hist["season"].unique().to_list()) < s < live
                   and _has_weekly(s)]

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

    # ---------------------------------------------------------------------
    # OPTION VALUE OF ROSTERING A BODY AT A STREAMABLE SLOT. Still not shipped,
    # but the reason changed on 2026-09-03 and the old reason was wrong.
    #
    # The gap is real and structural. `lineupValueWith` prices an occupied slot
    # at `max(his value, the floor)`, which is not the season total of WEEKLY
    # maxima — you roster a tight end AND keep streaming. So a tight end below
    # the streaming floor prices identically to an EMPTY tight end slot, and no
    # such tight end can ever earn a pick however good he looks. That is why the
    # draft page will not take Travis Kelce in round 12 with an empty TE slot.
    #
    # The first measurement said the gap was worth nothing (TE +1.6 +/- 3.8,
    # QB +5.7 +/- 6.1) and it was measuring the wrong difference TWICE:
    #
    #   1. WRONG BASELINE. It compared against `max(his REALIZED total, floor)`
    #      while the engine prices `max(curve[slot], floor)`. `E[max(X,c)] >
    #      max(E[X],c)`, so the old baseline carried a Jensen gap of +13.0 points
    #      at TE8-14 and +29.6 at QB8-14 — larger than the effect. Correcting it
    #      alone flips TE9-12 from -6.2 to +7.7.
    #   2. TOO FEW SEASONS. It read `boards`, the half-PPR ADP window (2018-),
    #      for a quantity that needs only a curve slot and a weekly score. Those
    #      run from 2012 and 1999. Eight seasons where thirteen were sitting
    #      there. See `opt_seasons`.
    #
    # Corrected, on 13 seasons and swept over the policy knob, the flat block
    # reads QB +28.2 -> +40.7 and TE +10.0 -> +17.6 across shrink 0 -> 8. Both
    # are positive at every setting; both are significant at every setting except
    # TE at shrink 0, which is the policy `check_dominance` rejects anyway. So
    # the direction is settled and the magnitude is bracketed, roughly +20 to +40
    # points on an occupied slot the engine currently prices at zero gradient.
    #
    # The unshrunk rule fails `check_dominance` hard: hold+stream comes in BELOW
    # hold-alone in 75 of 182 tight-end and 92 of 182 quarterback player-seasons,
    # worst -86 points. Always starting your own man costs no information, so a
    # rule that cannot match it is throwing away points. The visible symptom is
    # the top of the board: the bare prior-weeks mean benches the consensus TE1
    # in 10 of 13 seasons at -12.5 points a season, which would DEMOTE Bowers if
    # it were shipped as a curve. That is why shrink 0 is not the answer even
    # though it is the most conservative-looking column.
    #
    # Two things to settle before any of this reaches the objective:
    #   - The shrunk rule also moves `stream_total`, hence REPL_RANKS, hence the
    #     board. It is deliberately NOT wired into the shipped derivation; the
    #     sweep prints both so the cost of switching is visible first.
    #   - RB and WR have waiver access too. A correction applied only to QB/TE
    #     tilts the board back toward them, which is exactly the failure the
    #     2026-09-01 replacement repair removed. Fix all four or none.
    print(f"\nOPTION VALUE OF ROSTERING A BODY (not shipped — see the comment in "
          f"streaming.main), {opt_seasons[0]}-{opt_seasons[-1]}, n={len(opt_seasons)} seasons")
    print("  the engine prices an occupied slot at max(curve[slot], floor); this is "
          "what rostering him AND streaming actually banks")

    def stat(a):
        a = np.asarray(a, dtype=float)
        return (a.mean(), a.std(ddof=1) / np.sqrt(len(a)), len(a)) if len(a) > 1 else (0.0, 0.0, len(a))

    verdict: dict = {}
    for pos in sorted(vorp_mod.STREAMABLE):
        print(f"\n  {pos}  (floor {pos}{vorp_mod.REPL_RANKS[pos]})")
        print(f"    {'shrink':>6s} | {'dominance viol':>14s} | {'ALL slots':>18s} | "
              f"{'the flat block':>18s} | {'old baseline':>14s}")
        for sk in SHRINK_SWEEP:
            rows, per = [], []
            for s in opt_seasons:
                cp = season_curve(hist, s, pos)
                if cp is None:
                    continue
                fl = vorp_mod.curve_at(cp, vorp_mod.REPL_RANKS[pos])
                r = option_value(hist, s, pos, n_owned[pos], fl, SHIPPED_DROP_TOP, cp, sk)
                rows += r
                blk = [x[2] - x[3] for x in r if x[0] > vorp_mod.REPL_RANKS[pos]]
                if blk:
                    per.append(float(np.mean(blk)))
            viol, n, worst = check_dominance(rows)
            ma, sa, _ = stat([x[2] - x[3] for x in rows])
            mb, sb, _ = stat([x[2] - x[3] for x in rows if x[0] > vorp_mod.REPL_RANKS[pos]])
            mo, so, _ = stat([x[2] - x[4] for x in rows])
            mp, sp, _ = stat(per)
            t = 0.0 if sp == 0 else mp / sp
            verdict.setdefault(pos, {})[sk] = (viol / max(1, n), mb, t)
            print(f"    {sk:6.1f} | {viol:5d}/{n:<4d} {worst:6.0f} | {ma:+7.1f} +/- {sa:4.1f}"
                  f"       | {mb:+7.1f} +/- {sb:4.1f}       | {mo:+6.1f} +/- {so:4.1f}")
            print(f"           |  per season over the block: {mp:+6.1f} +/- {sp:4.1f}  "
                  f"t {t:4.2f}  ({sum(x > 0 for x in per)}/{len(per)} seasons)")
    print("\n  -> read DOWN each column, not across: the option value and the policy's "
          "own\n     quality move together, so a shrink picked on the effect size would be "
          "selection.")
    for pos in sorted(vorp_mod.STREAMABLE):
        v = verdict[pos]
        lo, hi = v[SHRINK_SWEEP[0]], v[SHRINK_WEEKS]
        print(f"     {pos}: violations {lo[0]:.0%} -> {hi[0]:.0%} over shrink "
              f"{SHRINK_SWEEP[0]:.0f}-{SHRINK_WEEKS:.0f}; the block reads "
              f"{lo[1]:+.1f} (t {lo[2]:.2f}) -> {hi[1]:+.1f} (t {hi[2]:.2f})")
    print("     Nothing is added to the objective: the effect is real but its SIZE is a "
          "function\n     of a policy constant this data cannot pin. See the block comment "
          "above.")

    print(f"\nWHAT THE SHRUNK RULE WOULD DO TO THE FLOOR (not applied; REPL_RANKS is "
          f"derived at shrink=0)")
    for pos in sorted(vorp_mod.STREAMABLE):
        base = float(np.mean([stream_total(hist, s, pos, n_owned[pos], SHIPPED_DROP_TOP)
                              for s in seasons]))
        alt = float(np.mean([
            stream_total(hist, s, pos, n_owned[pos], SHIPPED_DROP_TOP,
                         season_curve(hist, s, pos), SHRINK_WEEKS)
            for s in seasons]))
        print(f"  {pos}  shrink 0 -> {base:6.1f} pts = {pos}{curve_slot(curve[pos], base)}"
              f"   |   shrink {SHRINK_WEEKS:.0f} -> {alt:6.1f} pts = "
              f"{pos}{curve_slot(curve[pos], alt)}   ({alt - base:+.1f})")

    print("\nSHIPPED (vorp.REPL_RANKS): " + ", ".join(
        f"{p}={vorp_mod.REPL_RANKS[p]}" for p in vorp_mod.CURVE_POSITIONS))
    print(f"  QB/TE from the streaming arm at drop_top={SHIPPED_DROP_TOP}; "
          "RB/WR from the best-undrafted arm above.")
    print("  curve value at each replacement rank (%d curve): " % live + ", ".join(
        f"{p}{vorp_mod.REPL_RANKS[p]}={vorp_mod.curve_at(curve[p], vorp_mod.REPL_RANKS[p]):.0f}"
        for p in vorp_mod.CURVE_POSITIONS))
