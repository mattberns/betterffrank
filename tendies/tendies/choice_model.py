"""Nested-logit draft-choice model, fitted by penalised maximum likelihood.

Structure, and why this one:

    P(player j) = P(position of j | state) x P(j | position, state)

Written with the inclusive value I_p = log sum_k exp(w_k'g) carried into the
position stage with coefficient lambda, this is McFadden's nested logit. It
strictly nests both obvious alternatives: lambda = 1 IS a joint conditional
logit over every available player, and lambda = 0 is two independent stages.
So lambda is not a modelling flourish, it is the one number that says which of
those two was right, and it costs one parameter to find out.

The substantive reason to nest by position is IIA. A flat conditional logit
over ~250 players asserts that removing the best available RB shifts demand to
other positions in proportion to their current shares. That is plainly false —
it shifts to the next RB. Nesting relaxes exactly that, along exactly the
partition where it fails.

**Out-of-window picks get an outside option, they are not dropped.** Each
position's candidate list carries an extra alternative with its own constant,
and a pick outside the window scores against it. Dropping those rows would
condition the likelihood on "the pick was a top-20 name", which is selection on
the label and makes the model overconfident about the head of the board.

**Auto and human picks are fitted separately, and only where they differ.**
Measured: within a position, autodraft picks the top available 38.9% of the
time against humans' 40.2% — indistinguishable. The difference is entirely in
WHICH POSITION gets taken. So the player stage is shared and only the position
stage is split, which saves ~7 parameters on 177 auto picks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


# Weak ridge on every coefficient; separation control only (see penalty_vector).
BASE_RIDGE = 0.1


@dataclass
class ChoiceData:
    """Alternatives for a set of choice occasions, flattened and grouped.

    Rows are sorted by `group`; `ptr` holds group boundaries so the segment
    softmax can use reduceat. `chosen` is the absolute row index picked in each
    group.
    """

    X: np.ndarray                  # [n_rows, n_features]
    ptr: np.ndarray                # [n_groups + 1]
    chosen: np.ndarray             # [n_groups]
    weight: np.ndarray | None = None
    names: tuple[str, ...] = ()

    @property
    def n_groups(self) -> int:
        return len(self.ptr) - 1


def segment_logsumexp(u: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Max-subtracted per group. The subtraction is not an optimisation: JS and
    Python disagree in the tails without it."""
    mx = np.maximum.reduceat(u, ptr[:-1])
    mx[np.diff(ptr) == 0] = 0.0
    shifted = np.exp(u - np.repeat(mx, np.diff(ptr)))
    return mx + np.log(np.add.reduceat(shifted, ptr[:-1]))


def segment_softmax(u: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    lse = segment_logsumexp(u, ptr)
    return np.exp(u - np.repeat(lse, np.diff(ptr)))


def _nll_and_grad(beta, data: ChoiceData, penalty: np.ndarray):
    u = data.X @ beta
    lse = segment_logsumexp(u, data.ptr)
    w = data.weight if data.weight is not None else np.ones(data.n_groups)
    nll = -float(np.sum(w * (u[data.chosen] - lse)))
    p = segment_softmax(u, data.ptr)
    grad = (data.X * (p * np.repeat(w, np.diff(data.ptr)))[:, None]).sum(axis=0)
    grad -= (data.X[data.chosen] * w[:, None]).sum(axis=0)
    nll += 0.5 * float(np.sum(penalty * beta**2))
    grad += penalty * beta
    return nll, grad


def fit(
    data: ChoiceData,
    penalty: np.ndarray,
    bounds: list[tuple[float | None, float | None]] | None = None,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Penalised conditional logit. L-BFGS-B from zeros; deterministic."""
    k = data.X.shape[1]
    if data.n_groups == 0:
        return np.zeros(k)
    res = minimize(
        _nll_and_grad,
        np.zeros(k) if x0 is None else x0,
        args=(data, penalty),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    beta = res.x
    if np.max(np.abs(beta)) > 20:
        raise RuntimeError(
            f"coefficient magnitude {np.max(np.abs(beta)):.1f} — separation; "
            "increase the penalty or drop the offending column"
        )
    return beta


def log_likelihood(data: ChoiceData, beta: np.ndarray) -> np.ndarray:
    """Per-group log-likelihood; the model's own scoring metric."""
    u = data.X @ beta
    return u[data.chosen] - segment_logsumexp(u, data.ptr)


@dataclass
class ModelSpec:
    """Column layout for one stage: which columns are penalised, and how."""

    names: list[str] = field(default_factory=list)
    penalty: list[float] = field(default_factory=list)

    def add(self, name: str, penalty: float = 0.0) -> int:
        self.names.append(name)
        self.penalty.append(penalty)
        return len(self.names) - 1

    def index(self, name: str) -> int:
        return self.names.index(name)

    def penalty_vector(self, tau: dict[str, float], base: float = BASE_RIDGE) -> np.ndarray:
        """`penalty` holds a group id (as a float code) or 0 for unpenalised.
        Converted to 1/tau^2 at fit time so tau can be tuned per fold.

        Every column also carries `base`, a weak ridge that exists only to stop
        separation: a position nobody ever chose in a small subset (the auto
        model has 177 picks) otherwise sends its constant to -infinity. At this
        strength it is invisible for any coefficient the data identifies.
        """
        out = np.full(len(self.names), base)
        for i, code in enumerate(self.penalty):
            if code:
                out[i] += 1.0 / max(tau.get(_GROUP_BY_CODE[int(code)], 1e-3), 1e-6) ** 2
        return out


# Penalty groups. Manager deviations shrink toward the pooled model; the
# structural coefficients are unpenalised (beyond a numerical ridge).
#
# APPEND, NEVER REORDER OR INSERT. `GROUP_CODE` is positional (`i + 1`) and the
# code is stored as a float in `ModelSpec.penalty`, so moving an entry silently
# re-assigns every column's shrinkage to a different group.
GROUPS = ("manager_pos", "manager_reach", "manager_rookie", "manager_timing",
          "manager_taste", "ridge")
_GROUP_BY_CODE = {i + 1: g for i, g in enumerate(GROUPS)}
GROUP_CODE = {g: i + 1 for i, g in enumerate(GROUPS)}
