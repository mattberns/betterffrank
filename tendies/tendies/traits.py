"""Per-manager draft traits, and whether each one is real.

A trait is a number you can compute about a manager from his picks. That is the
easy half. The hard half is that a cross-sectional spread proves nothing: with
13 franchises and ~100 picks each, traits that are pure noise still spread out.

So every trait ships with a **split-half reliability**: compute it on the odd
seasons of a manager's own career and again on the even ones, correlate the two
vectors across managers. A trait that is a property of the person correlates
with itself; a trait that is a property of that year's board does not.
Spearman-Brown steps the half-length r up to a full-length measurement. With
seven managers the SE on r is about 0.45, so a bare r is not a finding — a
seeded permutation p-value ships beside it and the ORDER of the traits is what
should be read, not the third decimal.

Three rules the first version of this analysis got wrong, recorded so they stay
fixed:

* **Human picks only, for every trait that is about preference.** For three
  managers, half of the measured "tendency" was ESPN's autodraft robot.
  `cmd_report` had no filter at all until this module landed.
* **The timing traits are the exception, and they need their own rule.** "Round
  of his first QB" measured on human picks alone is wrong in a specific way: a
  manager whose first QB came from autodraft in round 7 would score as though
  he waited until his next human QB in round 12. So first-round traits are
  measured over ALL of that team-season's picks, and a team-season whose first
  pick at the position was an autodraft pick is DROPPED — the timing was not
  his decision. A season with no such pick at all is right-censored at one past
  the last round, because "never" is the extreme of "late", not missing data.
* **Split within a manager's own career, by season.** Splitting picks (odd/even
  or randomly) puts both halves inside the same draft, so a trait scores as
  reliable for agreeing with itself about one season's board — measured, that
  collapses RB share to +0.13 and inflates reach rate. Odd/even seasons rather
  than first-half/second-half, because chronological halves confound a stable
  trait with career drift.

## On the numbers this replaces

REPORT.md previously published RB share +0.664, reach rate +0.032 and mean ADP
delta -0.222, computed by hand with no code behind them. The TRAIT definitions
reproduce exactly here (Hawxhurst .523, S. Patterson .327, league average .398 —
pinned by `test_rb_share_matches_published_values`), but no split reproduces
those correlations: four candidate methods span -0.43 to +0.90 on the same
data. The two ADP-derived ones cannot reproduce even in principle — they were
measured against the FFC board that `config.DEFAULT_FALLBACK_BOARDS` has since
superseded with FantasyPros. The numbers here are the pinned-method ones and
they are what REPORT now quotes. In particular the published "reach rate does
not repeat" conclusion does NOT survive: under this method it repeats strongly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

# "Early" for a share trait. Rounds 1-8 is where a 15-round draft decides
# anything; past it every manager is taking the same handful of names.
SHARE_ROUNDS = 8

# A reach is a pick this far ahead of the market.
REACH = -10

# A manager enters the reliability table only if BOTH halves of his career
# carry this many human picks. Gating on human picks rather than on seasons is
# the point: a manager with seven seasons that were mostly autodraft has no
# measurable taste. On this league it selects seven franchises — the same seven
# the old ad-hoc analysis used, so the headline sample is unchanged.
MIN_PICKS = 25

PERMUTATIONS = 10_000
SEED = 0


@dataclass(frozen=True)
class Trait:
    fn: Callable[[pl.DataFrame], float | None]
    label: str


def _human(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(~pl.col("auto_pick"))


def _share(df: pl.DataFrame, position: str) -> float | None:
    early = _human(df).filter(pl.col("round") <= SHARE_ROUNDS)
    return float((early["position"] == position).mean()) if early.height else None


def _first_round(df: pl.DataFrame, position: str) -> float | None:
    """Mean round of the FIRST pick at `position`, over team-seasons.

    Grouped by (franchise, season), not season, so the same function is correct
    for one manager and for the league — pooling by season alone would measure
    "when the first QB left the board", a different quantity entirely.

    Measured over ALL picks (see the module docstring): a team-season whose
    first pick at the position was an autodraft is dropped, one with no such
    pick is censored at last round + 1.
    """
    vals = []
    for _, team_season in df.group_by(["franchise_id", "season"]):
        hit = team_season.filter(pl.col("position") == position).sort("overall_pick")
        if not hit.height:
            vals.append(float(team_season["round"].max()) + 1.0)
        elif bool(hit["auto_pick"][0]):
            continue                      # ESPN's decision, not his
        else:
            vals.append(float(hit["round"][0]))
    return float(np.mean(vals)) if vals else None


def _reach_rate(df: pl.DataFrame) -> float | None:
    x = _human(df).drop_nulls("adp_delta_filled")
    return float((x["adp_delta_filled"] <= REACH).mean()) if x.height else None


def _mean_adp_delta(df: pl.DataFrame) -> float | None:
    x = _human(df).drop_nulls("adp_delta_filled")
    return float(x["adp_delta_filled"].mean()) if x.height else None


def _rookie_rate(df: pl.DataFrame) -> float | None:
    h = _human(df)
    return float(h["rookie"].mean()) if h.height else None


def _expert_lean(df: pl.DataFrame) -> float | None:
    """Mean (ADP rank - ECR rank) over his picks: positive means the players he
    takes are ranked better by the experts than by the market, i.e. he drafts
    off the expert list. Only picks with both ranks count."""
    x = _human(df).drop_nulls(["adp_rank_filled", "ecr_rank"])
    return float((x["adp_rank_filled"] - x["ecr_rank"]).mean()) if x.height else None


def _stack_rate(df: pl.DataFrame) -> float | None:
    """Share of his picks that share an NFL team with someone he already took
    THAT season. Computed in pick order, so it is the same quantity the model's
    `same_team` feature sees."""
    hits = tot = 0
    for _, team_season in df.group_by(["franchise_id", "season"]):
        seen: set[str] = set()
        for r in team_season.sort("overall_pick").iter_rows(named=True):
            if r["auto_pick"]:
                continue                    # ESPN does not stack on purpose
            team = r["pro_team"]
            if team:
                tot += 1
                if team in seen:
                    hits += 1
                seen.add(team)
    return hits / tot if tot else None


def _mean_age(df: pl.DataFrame) -> float | None:
    """Mean age of his picks, relative to the typical player at that position
    that season (the board's centred `age`). Negative = drafts young."""
    x = _human(df).filter(pl.col("has_age") == 1)
    return float(x["age"].mean()) if x.height else None


def _reunion_rate(df: pl.DataFrame) -> float | None:
    """Share of his picks he also drafted the previous season. Restricted to
    seasons where he HAD a previous draft, so joining the league late does not
    read as disloyalty."""
    x = _human(df).filter(pl.col("had_prev_draft") == 1)
    return float(x["was_mine"].mean()) if x.height else None


def _mean_durability(df: pl.DataFrame) -> float | None:
    """Mean share of last season played, relative to typical at that position.
    Negative = he takes players who missed time."""
    x = _human(df).filter(pl.col("rookie") == 0)
    return float(x["durability"].mean()) if x.height else None


TRAITS: dict[str, Trait] = {
    "rb_share_r1_8": Trait(lambda d: _share(d, "RB"), "RB share, rounds 1-8"),
    "qb_share_r1_8": Trait(lambda d: _share(d, "QB"), "QB share, rounds 1-8"),
    "te_share_r1_8": Trait(lambda d: _share(d, "TE"), "TE share, rounds 1-8"),
    "first_qb_round": Trait(lambda d: _first_round(d, "QB"), "mean round of first QB"),
    "first_te_round": Trait(lambda d: _first_round(d, "TE"), "mean round of first TE"),
    "rookie_rate": Trait(_rookie_rate, "share of picks spent on rookies"),
    "expert_lean": Trait(_expert_lean, "mean ADP rank minus ECR rank of his picks"),
    "stack_rate": Trait(_stack_rate, "share of picks sharing an NFL team with his own"),
    "mean_age": Trait(_mean_age, "mean age vs typical at the position"),
    "reunion_rate": Trait(_reunion_rate, "share of picks he also drafted last season"),
    "mean_durability": Trait(_mean_durability, "mean share of last season played"),
    "reach_rate": Trait(_reach_rate, f"share of picks {abs(REACH)}+ ahead of ADP"),
    "mean_adp_delta": Trait(_mean_adp_delta, "mean picks later than ADP"),
}


BOARD_COLS = ("rookie", "age", "has_age", "durability", "prev_owner", "ecr_rank")


def attach_board(picks: pl.DataFrame, boards: pl.DataFrame) -> pl.DataFrame:
    """Player attributes come from the BOARD, via the `pid` link, so these
    traits and the model's features can never disagree about a player. A pick
    with no board row (four in this league's history) scores at the neutral
    value.

    `was_mine` is derived here rather than read: `prev_owner` is a property of
    the PLAYER, and whether it counts depends on WHO is picking.
    """
    if "was_mine" in picks.columns:
        return picks
    b = boards.select("season", "pid", *BOARD_COLS).unique(subset=["season", "pid"])
    prev = (
        picks.select("franchise_id", drafted=pl.col("season") + 1)
        .unique()
        .with_columns(had_prev_draft=pl.lit(1, dtype=pl.Int8))
        .rename({"drafted": "season"})
    )
    return (
        picks.join(b, on=["season", "pid"], how="left")
        .join(prev, on=["franchise_id", "season"], how="left")
        .with_columns(
            pl.col("rookie").fill_null(0),
            pl.col("age").fill_null(0.0),
            pl.col("has_age").fill_null(0),
            pl.col("durability").fill_null(0.0),
            pl.col("had_prev_draft").fill_null(0),
            was_mine=(pl.col("prev_owner") == pl.col("franchise_id"))
            .fill_null(False)
            .cast(pl.Int8),
        )
    )


def per_manager(picks: pl.DataFrame) -> pl.DataFrame:
    """One row per franchise, one column per trait."""
    rows = []
    for (fid,), grp in picks.group_by(["franchise_id"], maintain_order=True):
        human = _human(grp)
        row = {
            "franchise_id": fid,
            "franchise_label": grp["franchise_label"][0],
            "seasons": grp["season"].n_unique(),
            "human": human.height,
        }
        for name, trait in TRAITS.items():
            v = trait.fn(grp)
            row[name] = None if v is None else round(v, 3)
        rows.append(row)
    return pl.DataFrame(rows).sort("human", descending=True)


def league_average(picks: pl.DataFrame) -> dict[str, float | None]:
    return {name: t.fn(picks) for name, t in TRAITS.items()}


def _permutation_p(a: list[float], b: list[float], r: float) -> float:
    """Two-sided: how often does shuffling one half's manager labels beat |r|?"""
    rng = np.random.default_rng(SEED)
    a_, b_ = np.asarray(a), np.asarray(b)
    hits = 0
    for _ in range(PERMUTATIONS):
        perm = rng.permutation(b_)
        if np.std(perm) == 0:
            continue
        if abs(float(np.corrcoef(a_, perm)[0, 1])) >= abs(r):
            hits += 1
    return (hits + 1) / (PERMUTATIONS + 1)


def halves(picks: pl.DataFrame, min_picks: int = MIN_PICKS) -> dict[str, tuple]:
    """Each qualifying manager's career split into odd and even seasons."""
    out = {}
    for (fid,), grp in picks.group_by(["franchise_id"]):
        seasons = sorted(grp["season"].unique().to_list())
        odd = grp.filter(pl.col("season").is_in(seasons[0::2]))
        even = grp.filter(pl.col("season").is_in(seasons[1::2]))
        if min(_human(odd).height, _human(even).height) >= min_picks:
            out[fid] = (odd, even)
    return out


def split_half(picks: pl.DataFrame, min_picks: int = MIN_PICKS) -> pl.DataFrame:
    """Odd seasons vs even seasons of each manager's own career.

    `r` is Pearson across managers; `full_r` is Spearman-Brown, what the same
    trait would score measured over a whole career instead of half of one; `p`
    is a seeded two-sided permutation test on the manager labels.
    """
    keep = halves(picks, min_picks)
    rows = []
    for name, trait in TRAITS.items():
        a, b = [], []
        for odd, even in keep.values():
            x, y = trait.fn(odd), trait.fn(even)
            if x is not None and y is not None:
                a.append(x)
                b.append(y)
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            r, sb, p = float("nan"), float("nan"), float("nan")
        else:
            r = float(np.corrcoef(a, b)[0, 1])
            # Spearman-Brown steps a half-length reliability up to full length,
            # and that only means anything for a POSITIVE one. At r <= 0 the
            # formula still returns a number (-0.76 becomes -6.4) and the number
            # is meaningless, so it is not reported.
            sb = 2 * r / (1 + r) if r > 0 else float("nan")
            p = _permutation_p(a, b, r)
        rows.append(
            {
                "trait": name,
                "managers": len(a),
                "r": None if r != r else round(r, 3),
                "full_r": None if sb != sb else round(sb, 3),
                "p": None if p != p else round(p, 3),
                "label": trait.label,
            }
        )
    return pl.DataFrame(rows).sort("r", descending=True, nulls_last=True)
