"""Baseline models: the bar every real model has to clear (step 3).

All three are deliberately dumb. A baseline is not there to win; it is there to
answer "how much of this is just the league being the league?". If a Poisson
model cannot beat the league average on log loss, it has learned nothing and the
features are noise.

Fitted values come from the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import GoalModel


class LeagueAverageBaseline(GoalModel):
    """Every team, every venue: the training-set goals-per-team-per-match.

    This is the "historical average goals" baseline. It knows nothing about the
    teams, the venue or the tournament, so any model that loses to it is broken.
    """

    name = "baseline_league_average"

    def fit(self, rows: pd.DataFrame) -> "LeagueAverageBaseline":
        self.mean_goals_ = float(rows["goals_for"].mean())
        self.is_fitted = True
        return self

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.full(len(rows), self.mean_goals_, dtype=float)


class HomeAwayBaseline(LeagueAverageBaseline):
    """Split the average by venue instead of ignoring it.

    Two rates: what a team scores when it has home advantage, and what it scores
    when it does not (away sides, and both teams at a neutral venue). Still no
    knowledge of who is playing.
    """

    name = "baseline_home_away"

    def fit(self, rows: pd.DataFrame) -> "HomeAwayBaseline":
        has_home = rows["home_advantage"].to_numpy().astype(bool)
        self.mean_with_home_ = float(rows.loc[has_home, "goals_for"].mean())
        self.mean_without_home_ = float(rows.loc[~has_home, "goals_for"].mean())
        self.is_fitted = True
        return self

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        has_home = rows["home_advantage"].to_numpy().astype(bool)
        return np.where(
            has_home, self.mean_with_home_, self.mean_without_home_
        ).astype(float)


class FormBaseline(GoalModel):
    """A hand-made rate with no fitting: own recent scoring, opponent's recent conceding.

        lambda = (team form: goals scored over last 10 + opponent form: goals conceded
                  over last 10) / 2

    This is roughly what a pundit does. It is the interesting baseline, because
    it already uses the engineered features -- so a fitted Poisson model has to
    beat *this*, not just the league average, to justify its existence.
    """

    name = "baseline_form"

    def fit(self, rows: pd.DataFrame) -> "FormBaseline":
        self.fallback_ = float(rows["goals_for"].mean())
        self.is_fitted = True
        return self

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()

        opponent = (
            rows[["match_id", "team", "form_ga_10"]]
            .rename(columns={"team": "opponent", "form_ga_10": "opp_form_ga_10"})
        )
        merged = rows.merge(opponent, on=["match_id", "opponent"], how="left", sort=False)

        own = merged["form_gf_10"].to_numpy(float)
        opp = merged["opp_form_ga_10"].to_numpy(float)
        opp = np.where(np.isnan(opp), self.fallback_, opp)

        return np.clip(0.5 * (own + opp), 0.05, None)
