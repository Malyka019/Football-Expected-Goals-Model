"""Poisson regression on the engineered features (step 4, model 2).

Why Poisson?
------------
A match's goals are a count of rare, roughly independent events in a fixed time
window -- the textbook setting for a Poisson process. Three consequences that
matter here:

1. The variance equals the mean, so the model understands that a 1.4-goal
   favourite will sometimes score 0 or 3 without that being an "error".
2. Rates cannot go negative, which a linear regression on goals cannot promise.
3. The log link makes the coefficients multiplicative: exp(beta) is a ratio of
   goal rates. "Being 100 Elo points stronger multiplies your scoring rate by
   exp(beta)." That is the language football analysts actually use.

This model is *symmetric by construction*: it is fitted on team-rows, where
`elo_team` is the row's own team and `elo_opponent` the opponent. The same
coefficients describe both sides of a fixture, so the model cannot disagree with
itself about how good a team is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import GoalModel

ELO_CENTRE = 1500.0
ELO_SCALE = 100.0

# Tournament dummies; "friendly" is the reference category, so each coefficient
# reads as "goals relative to a friendly".
REFERENCE_TOURNAMENT = "friendly"
TOURNAMENT_DUMMIES = [
    "world_cup", "continental_final", "qualifier", "nations_league", "other",
]


def build_design(rows: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the GLM.

    Notes on the choices:
    - Elo is centred and divided by 100, so a coefficient is "per 100 rating
      points" instead of "per point" (the latter is ~0.0003 and unreadable).
    - attack/defence enter in logs, because they are multiplicative ratios to
      the league average; their coefficients should land near 1.0 if the scaling
      is right, which is a useful sanity check rather than a fitted free-for-all.
    - elo_diff is deliberately omitted: it is exactly elo_team - elo_opponent,
      so including it would make the design matrix singular.
    """
    X = pd.DataFrame(index=rows.index)

    X["elo_team"] = (rows["elo_team"].astype(float) - ELO_CENTRE) / ELO_SCALE
    X["elo_opponent"] = (rows["elo_opponent"].astype(float) - ELO_CENTRE) / ELO_SCALE
    X["home_advantage"] = rows["home_advantage"].astype(float)

    X["form_gf_10"] = rows["form_gf_10"].astype(float)
    X["form_ga_10"] = rows["form_ga_10"].astype(float)
    X["form_points_10"] = rows["form_points_10"].astype(float)

    X["log_attack"] = np.log(rows["attack"].astype(float).clip(lower=0.05))
    X["log_defence"] = np.log(rows["defence"].astype(float).clip(lower=0.05))

    X["h2h_played"] = np.log1p(rows["h2h_played"].astype(float))
    X["h2h_gd"] = rows["h2h_gd"].astype(float)
    X["h2h_win"] = rows["h2h_win"].astype(float)
    X["log_history"] = np.log1p(rows["history_matches"].astype(float))

    categories = rows["tournament_category"].astype(str)
    for level in TOURNAMENT_DUMMIES:
        X[f"t_{level}"] = (categories == level).astype(float)

    return X


class PoissonGLM(GoalModel):
    """`goals ~ Poisson` with a log link, fitted by maximum likelihood."""

    name = "poisson_glm"

    def fit(self, rows: pd.DataFrame) -> "PoissonGLM":
        design = build_design(rows)

        # A constant column (e.g. a tournament category with no matches in the
        # training window) is collinear with the intercept and makes the design
        # rank-deficient. Drop those, and remember the surviving columns so that
        # prediction builds exactly the same matrix.
        self.columns_ = [
            col for col in design.columns if design[col].std(ddof=0) > 0
        ]
        design = design[self.columns_]

        endog = rows["goals_for"].astype(float).to_numpy()
        exog = sm.add_constant(design, has_constant="add")

        self.model_ = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()
        self.params_ = pd.Series(
            np.asarray(self.model_.params, dtype=float),
            index=["const"] + self.columns_,
        )
        self.is_fitted = True
        return self

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        design = build_design(rows)[self.columns_]
        mu = self.model_.predict(sm.add_constant(design, has_constant="add"))
        # A lower floor keeps the score matrix sane; the ceiling is shared with
        # every other model so extreme mismatches stay believable.
        return self._finalise_rates(np.clip(np.asarray(mu, dtype=float), 0.01, 20.0))

    def summary(self):
        """statsmodels' full regression summary (coefficients, z, p-values)."""
        self._check_fitted()
        return self.model_.summary()
