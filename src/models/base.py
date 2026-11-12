"""Shared interface for every goals model (steps 3-5).

A tiny, boring contract. Having one is what makes the comparison table in step 6
possible without special-casing: every model takes the same team-level table and
returns one predicted rate per row. Nothing else.

    model = SomeModel().fit(train_rows)
    lambdas = model.predict_lambda(rows)      # len(lambdas) == len(rows)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .. import scorelines


class GoalModel(ABC):
    """Fit on team-match rows, predict one Poisson rate per row."""

    name: str = "base"

    def __init__(self) -> None:
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, rows: pd.DataFrame) -> "GoalModel":
        """Estimate parameters from training rows (never from validation/test)."""

    @abstractmethod
    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        """Expected goals for each row's team in that match."""

    def outcome_probabilities(
        self, lam_home: float, lam_away: float, max_goals: int = scorelines.MAX_GOALS
    ) -> tuple[float, float, float]:
        """Win/draw/win from the model's own scoreline distribution.

        Independent Poisson by default. Dixon-Coles overrides this so its
        rho-corrected probabilities are what get scored, which is the whole
        point of the correction.
        """
        return scorelines.outcome_probabilities(lam_home, lam_away, max_goals=max_goals)

    def _finalise_rates(self, rates: np.ndarray) -> np.ndarray:
        """Apply the shared rate ceiling to a model's expected goals.

        Every model returns through here so that a pathological extrapolation
        (a 1,000-point rating gap) cannot escape as a 15-goal prediction. The
        ceiling is the identity for ordinary rates, so normal predictions are
        unchanged; see `scorelines.cap_rate`.
        """
        return scorelines.cap_rate(rates)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before predicting")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, {state})"
