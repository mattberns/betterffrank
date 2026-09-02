"""Value over replacement for a 10-team half-PPR league.

The method is the parent repo's drafted-slot curve (`bff/vorp.py`), restated
here rather than imported: that module hard-codes `pts_ppr` and a POSITIONS
tuple with no K/D-ST, and tendies deliberately carries no dependency on `bff`.
The mechanism is the same and the reasoning behind it is worth repeating,
because the alternative is a trap:

  For each prior season, rank the players within each position by PRESEASON
  EXPERT CONSENSUS (not by what they went on to score), average the actual
  points at each within-position slot, and fit the result monotone
  non-increasing. `curve[pos][k-1]` is then "what the player the experts rated
  k-th at this position is worth in expectation".

The ranking is ECR, not ADP, and that is deliberate — see `slot_key`. ADP is
where the market ended up; ECR is the better estimate of what a player is
worth, by the parent repo's own measurement. The market rank still drives the
BEHAVIOUR features (`u_adp`, `adp_lead`), because what a manager will do is a
different question from what a player is worth.

Three deliberate departures from the parent's implementation:

**Monotonicity is imposed by isotonic regression (PAVA), not by
`np.minimum.accumulate` over a rolling mean.** The raw curve is genuinely
non-monotone at the top — over 2012-2024 the market's RB1 averages 173 half-PPR
points and its RB3 averages 229, because the consensus first back busts often
enough to matter. A running minimum propagates that dip forward and clamps
every later slot to it: on this league's 7-season history the entire RB curve
collapsed to a single value, making every RB's VORP identical. Isotonic
regression finds the closest non-increasing curve in least squares instead, so
a low slot-1 average pulls its neighbours toward it rather than flattening
everything behind it.

**The curve is built from the parent's deep market history (2012-), not from
this league's 7 boards.** Each slot then averages ~13 seasons rather than ~6,
which is the difference between a usable curve and noise. It stays
leakage-safe: the `season < t` filter is unchanged.

**Slots are indexed by ECR, with ADP only as a fallback for players the expert
boards do not reach.** See `slot_key`.

The trap is ranking by realised points instead — that prices the hindsight
order statistic, over-values the noisy mid-QB/TE tails, and (in the parent
repo, measured) shoved mid-round TEs and QBs 20-40 board slots too high.

VORP = curve[pos][player's within-position slot] - curve[pos][replacement].

**Replacement is the best option you would field at that position for FREE**, and
that is the only definition under which the resulting numbers are comparable
across positions. It has two arms, because the free option is a different kind of
thing at a streamable position than at a scarce one — QB/TE replacement is a
STREAMING TOTAL, RB/WR replacement is the BEST UNDRAFTED PLAYER. Both are
measured on this league's own drafts by `streaming.py`, which is the record
behind the four constants.

The trap to avoid here is a per-position mix of definitions, and this module
shipped one until 2026-09-01: QB/TE were streaming-aware while RB/WR sat at
roster-demand depth (RB25/WR30, the last starter). Roster-demand depth is 38
points better than a free back and 30 better than a free receiver, so every RB
and WR read that much too low, the board sorted quarterbacks and tight ends to
the top, and the VORP column meant a different thing in each row. RB and WR are
what the repair moved: QB7 was already exactly the streaming value and did not
budge, TE went 7 -> 6 (six points).

K and D/ST get NO curve: `actuals.parquet` carries no points for them, so they
are valued flat at zero and their draft timing is modelled purely by the position
stage.

**Every VORP ships with the standard error of its own estimate, and consumers
are expected to use it.** This is not decoration. A slot mean averages ~14
seasons of one draft position, and at the back of the board the season-to-season
spread swamps any slot-to-slot trend: WR30 has ranged from 36.4 to 251.3
half-PPR points, sd 64.6 on a mean of 127.6, so a single slot's mean carries an
SE of 15-19 points. Regressing points on slot over WR30-39 gives +1.50 pts/slot
(SE 1.66, t +0.90) — the point estimate is UPWARD, and the 95% interval on the
WR30 -> WR39 change runs -16 to +43. There is no measurable decline across that
range, which is exactly why PAVA pools those ten slots into one block, and
pooling is the estimator working rather than failing: 140 observations drop the
block's SE to 4.8 points against 15-19 for any slot alone.

The consequence is that the curve resolves the TOP of the board and not the
back of it. Measured on the 2012-2025 history behind the 2026 board:

    WR30-39 (VORP 29.2) vs WR50-54 (0.0)   +32.2 +/- 8.2    t 3.92   resolved
    RB34-36 (34.6)      vs RB42-46 (0.0)   +19.1 +/- 10.6   t 1.81   NOT
    RB34-36 (34.6)      vs RB37-41 (6.9)   +12.2 +/- 11.2   t 1.09   NOT
    WR30-39 (29.2)      vs WR40-42 (21.4)   +7.8 +/-  9.4   t 0.83   NOT

Row two is the entire spread of available backs at a round-8 pick, best to
worst, and it does not clear. So a ranked list built on differences of that size
is a ranked list of noise, and `web/engine.js` groups candidates into ties
rather than pretending to order them. `vorp_se` and `edge_se` are what it uses;
they are the SE of a DIFFERENCE of two block means, which is 0 when PAVA pooled
both slots into the same block (the two estimates are then literally the same
number).

Do NOT "fix" a flat block by adding an ECR tiebreak inside it. That invents
signal the history rejects at t = 0.90, and it is the first thing anyone reaches
for on seeing ten identical numbers.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import isotonic_regression

# Positions that get a VORP curve. K and DST are deliberately absent.
CURVE_POSITIONS = ("QB", "RB", "WR", "TE")

# Positions where punting the slot means STREAMING it rather than rostering one
# man. One starter each in a 10-team league, so ~13 come off the board and the
# waiver pool stays deep enough to rotate on matchup. Exported to the browser in
# the payload (see export.model_payload) so engine.js does not hard-code it.
STREAMABLE = frozenset({"QB", "TE"})

# 10 teams. Every one of these is READ OFF `tendies streaming`, at the column
# for the number of the position this league actually drafts -- none is rounded
# to taste. QB/TE are the streaming total's curve-slot equivalent under the
# drop_top=1 haircut (QB 257.2 pts = QB7, TE 118.8 = TE6); RB/WR are the mean
# best-undrafted slot in this league's 2019-2025 drafts (RB42.6, WR50.7).
#
# Do NOT set RB/WR back to roster-demand depth to "match" the parent repo: that
# repo's RB30/WR36 exists to score a cross-position Spearman on a 12-team board,
# not to price a draft, and mixing the two definitions is the bug this replaced.
REPL_RANKS = {"QB": 7, "RB": 43, "WR": 51, "TE": 6, "K": 1, "DST": 1}


def slot_key(df: pl.DataFrame) -> pl.DataFrame:
    """Within-position ordering used to index the curve: EXPERT rank first,
    market rank only as a fallback.

    ADP is where the market ended up, not what the player is worth, and the
    parent repo measured the difference: expert consensus beats ADP by +0.0145
    mean spearman_vorp on its 2018-2025 test window, winning 7 of 8 seasons and
    widening to +0.055 by 2025. So a value curve should be indexed by ECR.

    ECR boards are shallower than the deep ADP history, so players without an
    expert rank sort after every ranked player, in ADP order among themselves.
    K and D/ST have no ECR at all and get no curve either way.
    """
    by = ["season", "position"] if "season" in df.columns else ["position"]
    return (
        df.with_columns(_ecr=pl.col("ecr_rank").fill_null(10**6))
        .sort([*by, "_ecr", "adp_rank"])
        .with_columns(pos_rank=(pl.int_range(pl.len()).over(by) + 1).cast(pl.Int64))
        .drop("_ecr")
    )


# A curve history that has lost its ECR join is not a worse curve, it is a
# different quantity: `slot_key` would index by MARKET rank while `attach_vorp`
# looks players up by EXPERT rank. That happened once already (REPORT.md, "the
# curve's own slots were indexed by MARKET rank") and cost 0.3 of a standard
# error, which is small enough to go unnoticed and wrong enough to matter.
MIN_CURVE_ECR_COVERAGE = 0.60      # measured: ppr join 0.756 overall / 0.629 low


def market_slot_points(
    boards: pl.DataFrame, actuals: pl.DataFrame, pts_col: str = "pts_half",
    require_ecr: bool = True,
) -> pl.DataFrame:
    """(season, position, pos_rank, pts) — the market's slot-k pick at each
    position, tagged with what he actually scored.

    `boards` is any stacked market board (needs season, position, adp_rank,
    gsis_id, and — unless `require_ecr=False` — ecr_rank); pass the parent's
    deep history, not just this league's boards. `actuals` needs season,
    gsis_id and `pts_col`. A board player with no actuals row scored zero,
    which is the honest reading: he was drafted and returned nothing.
    """
    act = actuals.select("season", "gsis_id", pl.col(pts_col).alias("pts"))
    df = boards.filter(
        pl.col("position").is_in(CURVE_POSITIONS) & pl.col("gsis_id").is_not_null()
    ).join(act, on=["season", "gsis_id"], how="left").with_columns(
        pl.col("pts").fill_null(0.0)
    )
    if "ecr_rank" not in df.columns:
        if require_ecr:
            raise ValueError(
                "curve history has no ecr_rank column; slot_key would index the "
                "curve by MARKET rank while attach_vorp looks players up by "
                "EXPERT rank. Join ECR on first, or pass require_ecr=False if "
                "you genuinely want a market-indexed curve."
            )
        df = df.with_columns(ecr_rank=pl.lit(None, dtype=pl.Int32))
    elif require_ecr:
        cov = float(df["ecr_rank"].is_not_null().mean())
        if cov < MIN_CURVE_ECR_COVERAGE:
            raise ValueError(
                f"only {cov:.1%} of curve-history rows carry an ECR "
                f"(floor {MIN_CURVE_ECR_COVERAGE:.0%}); the join matched almost "
                "nothing, so the curve would be market-indexed in all but name."
            )
    return slot_key(df).select("season", "position", "pos_rank", "pts")


# Depth floors for the curve history. These exist so that swapping
# CURVE_ADP_PARQUET to the half-PPR board becomes a one-line config change that
# either passes or is refused, rather than a silent halving of the sample.
# Measured 2026-09-01: fp_adp_ppr gives 15 seasons and 8 prior at the earliest
# scored fold; fp_adp_half today gives 9 and 2, and that second number moves the
# fold-2020 RB slot 1 from 226.8 points to 298.9. Both floors clear once the
# half board reaches 2014.
CURVE_MIN_SEASONS = 12
CURVE_MIN_PRIOR_SEASONS = 6
# Not binding on either current board (half gives QB 60 / RB 129 / WR 162 /
# TE 75 per season); this is what stops a publisher-pruned board from flattening
# the curve so far that `curve_at`'s clamp does the work the data should.
CURVE_MIN_SLOTS = {p: 2 * REPL_RANKS[p] for p in CURVE_POSITIONS}


def check_depth(hist: pl.DataFrame, first_board_season: int) -> None:
    """Refuse a curve history too thin to average. Raises, never warns: a thin
    curve is invisible in every downstream number it touches."""
    seasons = sorted(int(s) for s in hist["season"].unique().to_list())
    if len(seasons) < CURVE_MIN_SEASONS:
        raise ValueError(
            f"curve history spans {len(seasons)} seasons ({seasons[0]}-{seasons[-1]}), "
            f"floor is {CURVE_MIN_SEASONS}"
        )
    prior = [s for s in seasons if s < first_board_season]
    if len(prior) < CURVE_MIN_PRIOR_SEASONS:
        raise ValueError(
            f"only {len(prior)} curve seasons precede the earliest board season "
            f"{first_board_season} (floor {CURVE_MIN_PRIOR_SEASONS}); the earliest "
            "fold would price its board off almost nothing"
        )
    depth = (
        hist.filter(pl.col("season") < first_board_season)
        .group_by("position").agg(slots=pl.col("pos_rank").max())
    )
    for row in depth.iter_rows(named=True):
        floor = CURVE_MIN_SLOTS.get(row["position"])
        if floor is not None and (row["slots"] or 0) < floor:
            raise ValueError(
                f"{row['position']} curve reaches slot {row['slots']} at the "
                f"earliest fold, floor is {floor} (2x replacement rank)"
            )


# Two fitted slots belong to the same PAVA block iff their fitted values are
# equal. Blocks are strictly decreasing by construction, so equality is exact up
# to float noise and this tolerance never has to separate two real blocks.
_BLOCK_TOL = 1e-9


def _blocks(fit: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive 0-based index ranges of the isotonic fit's flat blocks."""
    edges = [0]
    for i in range(1, len(fit)):
        if abs(float(fit[i]) - float(fit[i - 1])) > _BLOCK_TOL:
            edges.append(i)
    edges.append(len(fit))
    return [(edges[k], edges[k + 1] - 1) for k in range(len(edges) - 1)]


def _block_se(sub: pl.DataFrame, fit: np.ndarray) -> np.ndarray:
    """Standard error of each fitted slot, by slot index.

    A slot's fitted value is the mean of every observation PAVA pooled into its
    block, so its SE is that pooled mean's SE — which is the whole reason the
    pooling is worth doing. A block with fewer than two observations falls back
    to the position's own spread (one season is not an average).
    """
    by: dict[int, np.ndarray] = {
        int(k): np.asarray(v, dtype=float)
        for k, v in sub.group_by("pos_rank").agg(pl.col("pts")).iter_rows()
    }
    fallback = float(sub["pts"].std(ddof=1) or 0.0)
    se = np.zeros(len(fit), dtype=float)
    for lo, hi in _blocks(fit):
        parts = [by[k] for k in range(lo + 1, hi + 2) if k in by]
        vals = np.concatenate(parts) if parts else np.empty(0)
        se[lo : hi + 1] = (
            float(vals.std(ddof=1)) / np.sqrt(vals.size) if vals.size > 1 else fallback
        )
    return se


def build_curve_full(
    hist: pl.DataFrame, season: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(curves, standard errors) by position, from seasons STRICTLY BEFORE
    `season`. That filter is the leakage guard; do not relax it to "<=" to get
    a smoother curve.

    The SEs are not optional extra colour — see the module docstring. The curve
    resolves the top of the board and not the back of it, and a consumer that
    ranks on unresolved differences is ranking noise.
    """
    prior = hist.filter(pl.col("season") < season)
    curves: dict[str, np.ndarray] = {}
    ses: dict[str, np.ndarray] = {}
    for pos in CURVE_POSITIONS:
        raw = prior.filter(pl.col("position") == pos)
        sub = raw.group_by("pos_rank").agg(pl.col("pts").mean()).sort("pos_rank")
        if sub.height == 0:
            continue
        grid = np.arange(1, int(sub["pos_rank"].max()) + 1)
        vals = np.interp(grid, sub["pos_rank"].to_numpy(), sub["pts"].to_numpy())
        fit = isotonic_regression(vals, increasing=False).x
        curves[pos] = fit
        ses[pos] = _block_se(raw, fit)
    return curves, ses


def build_curve(hist: pl.DataFrame, season: int) -> dict[str, np.ndarray]:
    """pos -> expected points by within-position draft slot. The curve alone;
    `build_curve_full` also returns each slot's standard error."""
    return build_curve_full(hist, season)[0]


def curve_at(curve: np.ndarray, rank: int) -> float:
    """Expected points at a within-position slot, clamped at both ends."""
    return float(curve[min(max(rank, 1), len(curve)) - 1])


def pair_se(curve: np.ndarray, se: np.ndarray, a: int, b: int) -> float:
    """SE of the DIFFERENCE between two slots' fitted values.

    Exactly 0 when PAVA pooled both slots into one block: the two estimates are
    then the same number, so their difference is 0 with no uncertainty at all —
    not "an unknown amount". Otherwise the blocks are disjoint sets of
    observations, so the variances add.
    """
    if abs(curve_at(curve, a) - curve_at(curve, b)) < _BLOCK_TOL:
        return 0.0
    return float(np.hypot(curve_at(se, a), curve_at(se, b)))


def vorp_se_of(
    curves: dict[str, np.ndarray], ses: dict[str, np.ndarray],
    position: str, pos_rank: int,
) -> float:
    """SE of `vorp_of(...)`, which is a difference of two block means."""
    curve, se = curves.get(position), ses.get(position)
    if curve is None or se is None:
        return 0.0
    return pair_se(curve, se, pos_rank, REPL_RANKS[position])


def edge_se_of(
    curves: dict[str, np.ndarray], ses: dict[str, np.ndarray],
    position: str, ecr_slot: int, adp_slot: int,
) -> float:
    """SE of the expert-vs-market edge. The replacement term cancels, so this is
    the SE between the two slots the same player occupies on the two boards."""
    curve, se = curves.get(position), ses.get(position)
    if curve is None or se is None:
        return 0.0
    return pair_se(curve, se, ecr_slot, adp_slot)


def vorp_of(curves: dict[str, np.ndarray], position: str, pos_rank: int) -> float:
    """VORP of the player taken at within-position slot `pos_rank`.

    Returns 0.0 for K/DST (no curve by design) and for any position the curve
    could not be built for (the first season, which has no prior data).
    """
    curve = curves.get(position)
    if curve is None:
        return 0.0
    return curve_at(curve, pos_rank) - curve_at(curve, REPL_RANKS[position])


def attach_vorp(
    board: pl.DataFrame, curves: dict[str, np.ndarray], ses: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Add `pos_slot` (within-position EXPERT rank), `vorp`, `vorp_adp`, and the
    standard error of each.

    The slot is the player's rank among his position by ECR, not by ADP — a
    player the experts rate a round higher than the market is worth what the
    experts say, which is the whole point of pricing off ECR.

    `vorp_adp` prices the same player at his MARKET slot instead. The gap
    `vorp - vorp_adp` is the expert-vs-market edge in points rather than in
    rank places, which is what the draft page needs to rank bench picks: a
    player the market underrates is the bet worth making with a spare slot,
    and rank units are not comparable across positions or across the board.

    `vorp_se` and `edge_se` say how much of either number the history can
    actually resolve. `ses` is REQUIRED rather than defaulted, because a
    defaulted zero reads downstream as "this estimate is exact" — the loudest
    possible wrong answer, and the one the resolution floor exists to prevent.
    """
    b = slot_key(board).rename({"pos_rank": "pos_slot"}).with_columns(
        pos_rank=pl.col("adp_rank").rank("ordinal").over("position").cast(pl.Int32)
    )
    pos = b["position"].to_list()
    ecr_slots, adp_slots = b["pos_slot"].to_list(), b["pos_rank"].to_list()
    vorp = [vorp_of(curves, p, s) for p, s in zip(pos, ecr_slots)]
    vorp_adp = [vorp_of(curves, p, s) for p, s in zip(pos, adp_slots)]
    vorp_se = [vorp_se_of(curves, ses, p, s) for p, s in zip(pos, ecr_slots)]
    edge_se = [
        edge_se_of(curves, ses, p, e, a) for p, e, a in zip(pos, ecr_slots, adp_slots)
    ]
    return b.with_columns(
        vorp=pl.Series(vorp, dtype=pl.Float64),
        vorp_adp=pl.Series(vorp_adp, dtype=pl.Float64),
        vorp_se=pl.Series(vorp_se, dtype=pl.Float64),
        edge_se=pl.Series(edge_se, dtype=pl.Float64),
    )
