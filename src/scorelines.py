"""Scoreline probabilities from two Poisson rates.

This module is the bridge between "expected goals" and "win/draw/loss". Given
λ_home and λ_away it builds the joint score matrix P(home = i, away = j) under
independent Poisson counts, and derives outcomes from it.

Two details that are easy to get wrong:

1. **Tail mass.** The 0..5 display grid does not sum to 1; the tail carries the
   rest. We build on a wider grid (MAX_GOALS = 15) and renormalise, which keeps
   the residual error around 1e-7 for realistic rates. Outcome probabilities are
   computed from that wider grid, never from the 5x5 display grid.
2. **Independence is a modelling assumption.** Home and away goals are not
   perfectly independent (that is exactly what Dixon-Coles' rho corrects for).
   Step 4 adds the correction; this module stays the clean baseline.
"""

from __future__ import annotations

from math import lgamma

import numpy as np

# Wide enough that the truncation error is negligible for rates up to ~6 goals.
MAX_GOALS = 15

# --------------------------------------------------------------------------
# Rate ceiling
# --------------------------------------------------------------------------
# The log link has no upper bound, so a huge rating gap extrapolates into
# nonsense: Brazil vs San Marino came out at 11-16 expected goals, i.e. a
# "most likely scoreline" no one has ever seen in a competitive international.
#
# `cap_rate` fixes that without touching ordinary predictions. It is the
# identity up to `knee`, then bends smoothly towards `ceiling`:
#
#     lam <= knee : lam
#     lam >  knee : knee + (ceiling - knee) * (1 - exp(-(lam - knee)/(ceiling - knee)))
#
# continuous and with slope 1 at the knee, asymptotic to `ceiling`, and never
# above it. Both constants are chosen on the validation block, never the test
# block.
RATE_KNEE = 3.5
RATE_CEILING = 6.0


def cap_rate(
    lam: np.ndarray | float,
    knee: float | None = None,
    ceiling: float | None = None,
) -> np.ndarray:
    """Squash extreme expected-goal rates towards `ceiling`, leaving the rest.

    Rates at or below `knee` are returned unchanged. Above it the curve rises
    with slope 1 at the knee and flattens towards `ceiling`, so a 1.2-goal
    favourite is untouched while an 11-goal extrapolation becomes believable.
    """
    knee = RATE_KNEE if knee is None else float(knee)
    ceiling = RATE_CEILING if ceiling is None else float(ceiling)
    if ceiling <= knee:
        raise ValueError("ceiling must be above knee")

    values = np.asarray(lam, dtype=float)
    span = ceiling - knee
    excess = np.maximum(values - knee, 0.0)
    capped = knee + span * (1.0 - np.exp(-excess / span))
    return np.where(values > knee, capped, values)


def _poisson_pmf(lam: float, k_max: int) -> np.ndarray:
    """Poisson probabilities for k = 0..k_max, computed in log space.

    Log space avoids `lam ** k` overflowing for large k and underflowing for
    small lam. lgamma(n + 1) = log(n!) without building huge factorials.
    """
    k = np.arange(k_max + 1, dtype=float)
    if lam <= 0:
        pmf = np.zeros(k_max + 1)
        pmf[0] = 1.0
        return pmf
    log_pmf = k * np.log(lam) - lam - np.array([lgamma(i + 1.0) for i in k])
    return np.exp(log_pmf)


def score_matrix(lam_home: float, lam_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home = i, away = j) for i, j in 0..max_goals, renormalised to sum to 1.

    Rows are home goals, columns away goals, so `matrix[i, j]` is the home side
    scoring i and the away side j.
    """
    mat = np.outer(_poisson_pmf(float(lam_home), max_goals),
                   _poisson_pmf(float(lam_away), max_goals))
    total = mat.sum()
    if total <= 0:  # pragma: no cover - only reachable with invalid inputs
        raise ValueError("degenerate score matrix")
    return mat / total


def outcome_probabilities_from_matrix(mat: np.ndarray) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) from a score matrix."""
    n = mat.shape[0]
    p_home = mat[np.tril_indices(n, -1)].sum()   # i > j
    p_draw = np.trace(mat)                        # i == j
    p_away = mat[np.triu_indices(n, 1)].sum()     # i < j
    return float(p_home), float(p_draw), float(p_away)


def outcome_probabilities(
    lam_home: float, lam_away: float, max_goals: int = MAX_GOALS
) -> tuple[float, float, float]:
    """Win/draw/win probabilities directly from the two rates."""
    return outcome_probabilities_from_matrix(score_matrix(lam_home, lam_away, max_goals))


def apply_dixon_coles_correction(
    mat: np.ndarray, lam_home: float, lam_away: float, rho: float
) -> np.ndarray:
    """Reweight the four low-score cells with the Dixon-Coles tau factors.

        tau(0,0) = 1 - lam_h * lam_a * rho
        tau(0,1) = 1 + lam_h * rho
        tau(1,0) = 1 + lam_a * rho
        tau(1,1) = 1 - rho
        tau(x,y) = 1 otherwise

    With the usual small negative rho (about -0.05) this lifts 0-0 and 1-1 and
    trims 1-0 / 0-1, which is precisely where independent Poisson is known to
    mis-fit football. The adjusted matrix is renormalised because the tau factors
    are not a normalised distribution on their own.
    """
    out = mat.copy()
    out[0, 0] *= 1.0 - lam_home * lam_away * rho
    out[0, 1] *= 1.0 + lam_home * rho
    out[1, 0] *= 1.0 + lam_away * rho
    out[1, 1] *= 1.0 - rho
    out = np.clip(out, 0.0, None)
    total = out.sum()
    if total <= 0:  # pragma: no cover - only reachable with a pathological rho
        return mat
    return out / total


def dixon_coles_matrix(
    lam_home: float, lam_away: float, rho: float, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Independent-Poisson score matrix with the Dixon-Coles correction applied."""
    return apply_dixon_coles_correction(
        score_matrix(lam_home, lam_away, max_goals), lam_home, lam_away, rho
    )


def dixon_coles_outcome_probabilities(
    lam_home: float, lam_away: float, rho: float, max_goals: int = MAX_GOALS
) -> tuple[float, float, float]:
    """Win/draw/win from the Dixon-Coles corrected matrix."""
    return outcome_probabilities_from_matrix(
        dixon_coles_matrix(lam_home, lam_away, rho, max_goals)
    )


def most_likely_scorelines(
    lam_home: float, lam_away: float, max_goals: int = MAX_GOALS, top_n: int = 5
) -> list[tuple[int, int, float]]:
    """The `top_n` most probable (home, away, probability) scorelines."""
    mat = score_matrix(lam_home, lam_away, max_goals)
    idx = np.dstack(np.unravel_index(np.argsort(mat, axis=None)[::-1], mat.shape))[0]
    return [(int(i), int(j), float(mat[i, j])) for i, j in idx[:top_n]]


def display_matrix(lam_home: float, lam_away: float, max_goals: int = 5) -> np.ndarray:
    """The 0..max_goals grid used for a heatmap (rows home, columns away).

    Renormalised over the display grid so it reads as a probability table, which
    is why its cells are slightly larger than the corresponding cells of the
    full matrix. Use `score_matrix` for anything numerical.
    """
    return score_matrix(lam_home, lam_away, max_goals=max_goals)
