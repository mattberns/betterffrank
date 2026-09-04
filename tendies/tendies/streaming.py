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

## Holding a body below the floor, and what the engine misses

`lineupValueWith` prices an occupied streamable slot at `max(his VORP, 0)`, so a
tight end worth less than streaming prices identically to an EMPTY slot and no
such tight end can earn a pick on value. `hold_gain` measures the PAIRED quantity

    gain(k) = (hold slot k AND stream around him) - (stream only)

same season, same free pool, same weekly form rule, 2013-2025. Below the floor
it reads TE +6 to +13 and QB +13 to +19 a season (t 2-3), flat across the
below-floor slots and stable across drop_top 1/2 and shrink 0/4. The engine
credited 0 until 2026-09-04; it now adds `vorp.HOLD_GAIN` (QB 15, TE 6 -- the
drop_top=1 / shrink=0 corner, rounded) to every occupied streamable slot.
RB44-49 / WR52-56 read +1 to +6 / +1 to +3, but against a one-slot
stream of their free pool that banks RB14-25 / WR27-40 -- which is the proof
that a single-slot streaming sim is NOT a replacement estimator at a
multi-starter position, so the RB/WR line is a check and not a comparable.

An earlier version of this measurement (2026-09-03) read +20 to +40 and called
it option value. Decomposed, PURE option value -- hold+stream over the better
of hold-alone or stream-alone -- is NEGATIVE (TE -2.6, QB -9.8); the rest was
the realized stream sitting above `curve@REPL` on the widened window (+9.2 TE,
+19.3 QB) plus the hindsight max of two realized paths. Neither is a value of
holding anyone. See the block comment in `main`.

`SHRINK_WEEKS` mixes preseason prior into the form estimate. It repairs a real
policy defect -- the bare prior-weeks mean benches the consensus TE1 in 10 of
13 seasons and fails `check_dominance` in 75/182 TE and 92/182 QB
player-seasons -- and **it does not move REPL_RANKS**: the streaming totals rise
+6.2 (QB) and +5.5 (TE) and stay on QB7 and TE6. So `stream_total` still
defaults to shrink=0 and the derivation above is unchanged.

## What this changed, and what it did not

QB7 did NOT move and TE went only 7 -> 6: they were already at the streaming
value, which is why "quarterbacks look overvalued on the board" was a
misdiagnosis. RB25 -> RB43 and WR30 -> WR51 are the fix. Those two were priced
against roster-demand depth (the last starter), which is 38 and 30 points better
than what is actually free, so every back and receiver read that much too low and
QB/TE floated to the top of a column that meant a different thing in each row.

Prints, writes nothing, and is not imported by the fitted pipeline — the same
status `bff/streaming.py` has in the parent repo.

    uv run python -m tendies streaming
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
# player-seasons. The violation rate falls monotonically with shrink (QB
# 51/36/34/26/18%, TE 41/34/29/25/20% at 0/1/2/4/8) and never flattens, so it
# does not select a value; 4.0 is a midpoint, not an optimum.
#
# It is swept, not asserted, for the same reason `drop_top` is: it is a claim
# about waiver behaviour that the data can bound but not pin. Both endpoints are
# defensible -- shrink 0 is the bare form rule, and as shrink grows the policy
# converges on "start your guy every week, stream his bye" -- and `hold_gain`'s
# below-floor answer is the same at both (see `main`). Nothing in the shipped
# pipeline reads this constant.
SHRINK_WEEKS = 4.0
# Slots per row in the by-slot gain table `main` prints.
GAIN_BAND = 4


def _weekly_path(season: int):
    """2025 is a differently named file upstream, the same special case the
    parent repo carries in `bff/streaming.py` and `bff/actuals.py`."""
    return (STATS_RAW_DIR / "stats_player_week_2025.parquet" if season == 2025
            else STATS_RAW_DIR / f"player_stats_{season}.parquet")


def _has_weekly(season: int) -> bool:
    return _weekly_path(season).exists()


# The hold-gain sweep asks for the same (season, position) frame repeatedly --
# 13 seasons x several policy settings x 4 positions -- and each call was
# re-reading a whole weekly parquet off disk. That is what killed the 2026-09-03
# run. Polars frames are treated as immutable everywhere here, so handing out
# the same object is safe.
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
    """`vorp.build_curve` for one season, cached across the hold-gain sweep.

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
               drop_top: int):
    """(free-pool weekly frame, within-position rank map) for one season."""
    h = hist.filter((pl.col("season") == season) & (pl.col("position") == pos))
    rank = dict(zip(h["gsis_id"].to_list(), h["pos_rank"].to_list()))
    owned = [g for g, r in rank.items() if r <= n_owned]
    free = weekly(season, pos).filter(~pl.col("gsis_id").is_in(owned))
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


def hold_gain(hist: pl.DataFrame, season: int, pos: str, n_owned: int,
              drop_top: int = SHIPPED_DROP_TOP, curve=None,
              shrink: float = 0.0) -> list[tuple]:
    """Per drafted slot: `(slot, his own total, hold-him-AND-stream, stream only)`.

    The difference of the last two is what rostering the player at `slot` is
    worth over punting the position: same season, same free pool, same weekly
    rule, the only change being whether you own him. No curve enters it, so
    neither the walk-forward curve's bias on a given window nor the floor's
    calibration can inflate it -- both of which sank the earlier version of this
    measurement (see the block comment in `main`).

    Policy, applied to the rostered player and the free pool alike: `shrink`
    pseudo-weeks of preseason prior mixed into a prior-weeks mean, start whoever
    grades higher, bank his actual points. His bye and inactive weeks are
    streamed. At `shrink == 0` week 1 has no form and the ECR tiebreak decides,
    which is the rule the shipped REPL_RANKS were derived under.

    The streaming leg is computed ONCE per season and shared by every slot:
    `_free_pool` already excludes everyone inside `n_owned`, so the free pool
    does not depend on which owned player is being valued. (The earlier code
    passed `exclude=me` and rebuilt it per slot; that argument was a no-op and
    the rebuild was most of the runtime.)
    """
    h = hist.filter((pl.col("season") == season) & (pl.col("position") == pos))
    rank = dict(zip(h["gsis_id"].to_list(), h["pos_rank"].to_list()))
    free, _ = _free_pool(hist, season, pos, n_owned, drop_top)
    if free.height == 0:
        return []
    shrunk = curve is not None and shrink > 0
    spts, sform = {}, {}
    for w in sorted(free["week"].unique().to_list()):
        cur = free.filter(pl.col("week") == w)
        if cur.height == 0:
            continue
        cand = (cur.join(_form(free, w, rank, curve, shrink), on="gsis_id", how="left")
                .with_columns(pl.col("gsis_id").replace_strict(rank, default=10**6).alias("pr"))
                .sort(["form", "pr"], descending=[True, False], nulls_last=True))
        spts[w], sform[w] = float(cand["pts"][0]), cand["form"][0]
    stream_only = float(sum(spts.values()))

    wk = weekly(season, pos)
    mine: dict[int, str] = {}
    for g, r in rank.items():
        if r <= n_owned:
            mine.setdefault(int(r), g)
    out = []
    for slot in sorted(mine):
        mypts = {int(r["week"]): float(r["pts"])
                 for r in wk.filter(pl.col("gsis_id") == mine[slot]).iter_rows(named=True)}
        my_p0 = vorp_mod.curve_at(curve, slot) / REG_WEEKS if shrunk else None
        both = 0.0
        for w, s in spts.items():
            m = mypts.get(w)
            if m is None:
                both += s                           # his bye or a scratch: stream it
                continue
            pri = [mypts[x] for x in mypts if x < w]
            if shrunk:
                mf = (float(np.sum(pri)) + my_p0 * shrink) / (len(pri) + shrink)
            else:
                mf = float(np.mean(pri)) if pri else None
            f = sform[w]
            both += m if (mf is not None and (f is None or mf > float(f))) else s
        out.append((slot, float(sum(mypts.values())), both, stream_only))
    return out


def check_dominance(rows: list[tuple]) -> tuple[int, int, float]:
    """`(violations, n, worst)` of the identity HOLD+STREAM >= HOLD ALONE.

    Always starting your own player is a policy available to the streamer at no
    information cost, so a streaming rule that cannot match it is losing points
    it did not have to lose. Ex-post regret makes the odd violation inevitable;
    a rate near half the sample does not. A policy that loses to "just start my
    guy" understates `hold_gain`, which is why the rate is printed next to it.
    Takes `hold_gain` rows: `(slot, alone, both, stream_only)`.
    """
    d = [both - alone for _, alone, both, _ in rows]
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
    # WHAT HOLDING A BODY BELOW THE FLOOR IS WORTH -- the derivation behind
    # vorp.HOLD_GAIN. The measurement lives here so the number is reproducible
    # and so nobody re-derives the two wrong versions of it.
    #
    # `lineupValueWith` prices an occupied slot at `max(his VORP, empty)`, and
    # at a streamable position `empty` is 0, so a tight end below the floor
    # prices identically to NO tight end and can never earn a pick on value.
    # That is why the draft page will not take Kelce in round 12 with an empty
    # TE slot. The question is what he is actually worth over punting.
    #
    # Two earlier answers were wrong in opposite directions:
    #   - 2026-09-02: `both - max(his REALIZED total, floor)` on players who
    #     finished at or below the floor -- selects on the outcome, read ~0.
    #   - 2026-09-03: `both - max(curve[slot], floor)` on 2013-2025 -- read +20
    #     to +40 and called it option value. Decomposed, PURE option value
    #     (hold+stream over the better of hold-alone or stream-alone) is
    #     NEGATIVE (TE -2.6, QB -9.8 below the floor at shrink 4); the +20-40
    #     was the realized stream sitting above curve@REPL on the widened
    #     window (+9.2 TE, +19.3 QB) plus E[max] of two realized paths, which
    #     is hindsight nobody banks. Neither is a value of holding anyone.
    #
    # The quantity that survives is PAIRED: same season, same free pool, same
    # weekly form rule, hold him or not. `hold_gain` returns it; below the
    # shipped floor it reads (drop_top 1/2 x shrink 0/4, 13 seasons):
    #     TE7-14   +6.1 / +9.8 / +7.9 / +12.9   (t 2.2-3.3)
    #     QB8-14  +14.6 / +13.0 / +18.8 / +16.4  (t 1.8-2.9)
    # flat across the below-floor slots -- a TE13-14 held and streamed around
    # reads +7 to +16, TE9-12 +4 to +12 -- and the engine credited 0. Above the
    # floor the comparison is knob-dependent (TE1-4: 7 under the credit at
    # shrink 0, 13 over it at shrink 4), so only the below-floor block is a
    # finding.
    #
    # RB and WR are printed as a check, not a comparable. Their below-floor
    # bodies read +1 to +6 (RB44-49) and +1 to +3 (WR52-56), but against a
    # ONE-SLOT stream of the free pool that banks RB14-25 / WR27-40 -- far
    # above the shipped RB43/WR51 floors. That is the demonstration that a
    # single-slot streaming sim is not a replacement estimator at a
    # multi-starter position (2-3 backs start at once, and the weekly waiver
    # hand is contested by nine rivals, which drop_top -- season-total leaders
    # only -- does not model). RB/WR bench bodies earn through `insurance` in
    # engine.js instead. So the "fix all four or none" constraint from the
    # 09-03 pass dissolves: the positions do not share a mechanism.
    #
    # SHIPPED 2026-09-04 as vorp.HOLD_GAIN = {QB: 15, TE: 6}: the drop_top=1 /
    # shrink=0 corner (the setting REPL_RANKS is derived under), rounded.
    # engine.js adds it to every OCCUPIED dedicated slot at the position
    # (uniform, so within-position order and the elite premium are untouched;
    # only occupied-vs-empty moves). Ten-seat mock before/after on the 2026
    # board and the 2025 fixture: no pick moved on any seat, every lineup +21.0,
    # margin over best-by-ADP unchanged (+22.3 / +80.6); the R13 TE/QB rows read
    # now 6.0 / 15.0 instead of 0.0. As predicted: on a TE block this flat and
    # deep (TE13-14 survive 15 picks at 86-98%) it changes what the slot is
    # WORTH, not when the DP fills it -- until a board's round-10-12
    # alternatives are worth under six points, where it will now take the TE.
    print(f"\nHOLDING A BODY BELOW THE FLOOR (the derivation behind vorp.HOLD_GAIN — see "
          f"the comment in streaming.main), {opt_seasons[0]}-{opt_seasons[-1]}, "
          f"n={len(opt_seasons)} seasons")
    print("  gain = (hold slot k AND stream around him) - (stream only), paired within "
          "season; credit = max(0, curve[k] - floor), what the engine priced before "
          "HOLD_GAIN")

    def stat(a):
        a = np.asarray(a, dtype=float)
        return ((a.mean(), a.std(ddof=1) / np.sqrt(len(a)), len(a)) if len(a) > 1
                else ((a.mean() if len(a) else 0.0), 0.0, len(a)))

    def per_season(rows):
        per: dict = {}
        for r in rows:
            per.setdefault(r[0], []).append(r[3] - r[4])
        return [float(np.mean(v)) for v in per.values()]

    settings = [(d, sk) for d in (SHIPPED_DROP_TOP, SHIPPED_DROP_TOP + 1)
                for sk in (0.0, SHRINK_WEEKS)]
    for pos in vorp_mod.CURVE_POSITIONS:
        repl, cl = vorp_mod.REPL_RANKS[pos], curve[pos]
        floor_live = vorp_mod.curve_at(cl, repl)
        streamable = pos in vorp_mod.STREAMABLE
        print(f"\n  {pos}  floor {pos}{repl} = {floor_live:.0f} pts on the {live} curve"
              + ("" if streamable else "  (NOT streamable: a check, not a comparable)"))
        for drop, sk in settings:
            if not streamable and drop != SHIPPED_DROP_TOP:
                continue
            rows = []
            for s in opt_seasons:
                cp = season_curve(hist, s, pos)
                if cp is None:
                    continue
                rows += [(s,) + r for r in hold_gain(hist, s, pos, n_owned[pos], drop, cp, sk)]
            so = [r[4] for r in rows if r[1] == 1]
            ms = float(np.mean(so)) if so else 0.0
            viol, n, worst = check_dominance([r[1:] for r in rows])
            below = [r for r in rows if r[1] > repl]
            mb, eb, _ = stat([r[3] - r[4] for r in below])
            ps = per_season(below)
            mp, ep, _ = stat(ps)
            print(f"    drop_top={drop} shrink={sk:.0f}: stream only {ms:6.1f} pts = "
                  f"{pos}{curve_slot(cl, ms):<3d} | hold+stream < hold alone {viol}/{n}, "
                  f"worst {worst:.0f} | below floor {pos}{repl + 1}-{pos}{n_owned[pos]}: "
                  f"gain {mb:+.1f} +/-{eb:.1f}, per-season t "
                  f"{0.0 if ep == 0 else mp / ep:.2f} ({sum(x > 0 for x in ps)}/{len(ps)})")
            if streamable and drop == SHIPPED_DROP_TOP:
                print(f"      {'slots':>10s} | {'n':>3s} | {'gain':>15s} | {'credit':>6s} | "
                      f"{'gain-credit':>11s} | {'alone-stream':>15s}")
                lo = 1
                while lo <= n_owned[pos]:
                    hi = min(lo + GAIN_BAND - 1, n_owned[pos])
                    rs = [r for r in rows if lo <= r[1] <= hi]
                    if rs:
                        mg, eg, nn = stat([r[3] - r[4] for r in rs])
                        ma, ea, _ = stat([r[2] - r[4] for r in rs])
                        cred = float(np.mean([
                            max(0.0, vorp_mod.curve_at(cl, r[1]) - floor_live) for r in rs]))
                        print(f"      {pos}{lo:>2d}-{pos}{hi:<4d} | {nn:3d} | {mg:+7.1f} +/-{eg:4.1f}"
                              f" | {cred:6.1f} | {mg - cred:+8.1f}    | {ma:+7.1f} +/-{ea:4.1f}")
                    lo = hi + 1
    print("\n  -> a below-floor TE body is worth roughly +6-13 over punting and a QB body "
          "+13-19; shipped as\n     vorp.HOLD_GAIN QB 15 / TE 6 (the drop_top=1 / shrink=0 "
          "corner). RB/WR are paired against a\n     ONE-SLOT stream that banks RB14-25 / "
          "WR27-40, which is why they are not streamable and why\n     their line is a "
          "check, not a comparable.")

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
