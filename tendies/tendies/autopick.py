"""Autodraft as a two-state Markov chain, not a per-pick coin flip.

Measured on this league: P(auto | previous pick was auto) = 0.906 over 160
transitions; P(auto | previous was human) = 0.024 over 840. Of 70 team-seasons,
47 contain no autopicks at all, and 13 of the rest are a pure suffix — someone
left the room and never came back. Two are 100% auto from pick one.

That shape matters in two places:

* **Prediction.** A manager's autodraft state is persistent and partly
  observable: once you have seen them autopick, they will almost certainly
  autopick again. A per-pick rate throws that away.
* **Simulation.** The state must be drawn ONCE per simulated path and carried
  forward. Re-drawing it at every pick collapses to the marginal rate and
  understates the variance of exactly the survival probabilities the page
  displays.

The chain is supervised — `auto_pick` is observed in training — so there is no
latent-variable estimation here, just counting with a shrinkage prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# Beta prior strength for per-manager rates. 20 was measured best out-of-sample
# for the marginal rate (log-lik/pick -0.368 vs -0.448 pooled).
PRIOR_STRENGTH = 20.0


@dataclass
class AutoChain:
    p_start: dict[str, float]      # P(auto on the manager's first pick)
    p_h_to_a: dict[str, float]     # P(auto | previous human)
    p_a_to_a: float                # P(auto | previous auto), pooled
    league_start: float
    league_h_to_a: float

    def start(self, manager: str) -> float:
        return self.p_start.get(manager, self.league_start)

    def step(self, manager: str, was_auto: bool) -> float:
        if was_auto:
            return self.p_a_to_a
        return self.p_h_to_a.get(manager, self.league_h_to_a)

    def to_dict(self) -> dict:
        return {
            "p_start": self.p_start,
            "p_h_to_a": self.p_h_to_a,
            "p_a_to_a": self.p_a_to_a,
            "league_start": self.league_start,
            "league_h_to_a": self.league_h_to_a,
        }


def fit_chain(picks: pl.DataFrame, prior: float = PRIOR_STRENGTH) -> AutoChain:
    """Count transitions within each (season, franchise) pick sequence."""
    df = picks.sort("season", "franchise_id", "overall_pick")
    starts: dict[str, list[int]] = {}
    h2a: dict[str, list[int]] = {}
    a2a = [0, 0]
    for (season, mgr), g in df.group_by(["season", "franchise_id"], maintain_order=True):
        seq = g.sort("overall_pick")["auto_pick"].to_list()
        if not seq:
            continue
        starts.setdefault(mgr, []).append(int(bool(seq[0])))
        for prev, cur in zip(seq, seq[1:]):
            if prev:
                a2a[0] += int(bool(cur))
                a2a[1] += 1
            else:
                h2a.setdefault(mgr, []).append(int(bool(cur)))

    lg_start = float(np.mean([v for vals in starts.values() for v in vals])) if starts else 0.0
    lg_h2a = float(np.mean([v for vals in h2a.values() for v in vals])) if h2a else 0.0

    def shrunk(counts: dict[str, list[int]], league: float, strength: float) -> dict[str, float]:
        return {
            m: (float(np.sum(v)) + strength * league) / (len(v) + strength)
            for m, v in counts.items()
        }

    return AutoChain(
        # first-pick evidence is one observation per season, so shrink it harder
        p_start=shrunk(starts, lg_start, 3.0),
        p_h_to_a=shrunk(h2a, lg_h2a, prior),
        p_a_to_a=(a2a[0] + 0.9 * 5) / (a2a[1] + 5) if a2a[1] else 0.9,
        league_start=lg_start,
        league_h_to_a=lg_h2a,
    )


def belief_update(
    prior_belief: float, p_auto_pick: float, p_human_pick: float
) -> float:
    """Bayes on the observed pick: an autodraft-looking pick raises the belief
    that this manager is autodrafting. Used by the live page when the user has
    not asserted the state."""
    num = prior_belief * p_auto_pick
    den = num + (1 - prior_belief) * p_human_pick
    return float(num / den) if den > 0 else prior_belief


def summary(chain: AutoChain, picks: pl.DataFrame) -> pl.DataFrame:
    obs = (
        picks.group_by("franchise_id", "franchise_label")
        .agg(picks=pl.len(), observed_rate=pl.col("auto_pick").mean().round(3))
        .sort("observed_rate", descending=True)
    )
    return obs.with_columns(
        p_start=pl.col("franchise_id").map_elements(chain.start, return_dtype=pl.Float64).round(3),
        p_h_to_a=pl.col("franchise_id")
        .map_elements(lambda m: chain.step(m, False), return_dtype=pl.Float64).round(3),
    )
