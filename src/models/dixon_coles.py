"""Dixon-Coles (1997), fitted from scratch (step 4, model 3).

The paper contributed two things on top of a plain Poisson model, and both are
implemented here.

**(a) Time decay.** A match from 2009 should not count as much as one from last
month. Each match gets weight

        w = 0.5 ** (age_in_days / halflife_days)

with a two-year halflife by default, so a match two years old counts half as
much as today's. This is what makes the model follow form instead of averaging a
team's entire history.

**(b) The tau correction for low scores.** Independent Poisson systematically
mis-prices 0-0, 1-0, 0-1 and 1-1. Dixon-Coles multiply those four cells of the
score matrix by

        tau(0,0) = 1 - lam_h*lam_a*rho      tau(1,1) = 1 - rho
        tau(0,1) = 1 + lam_h*rho            tau(1,0) = 1 + lam_a*rho

and leave every other cell alone. With the usual small negative rho, draws
become slightly more likely than independent Poisson thinks -- which matches
what actually happens in football.

The parameterisation
--------------------
    lam_home = exp(delta + attack[home] - defence[away] + gamma * home_flag)
    lam_away = exp(delta + attack[away] - defence[home])

Adding a constant to every attack and taking it off the intercept leaves every
prediction unchanged, so without something to break that symmetry the likelihood
has a flat direction. The textbook fix is a sum-to-zero constraint, but on the
real dataset (317 teams) that reparameterisation makes *every* gradient component
depend on a single shared parameter, which destroys the conditioning: L-BFGS
crawled for 1500 iterations without converging.

Instead we fit with a small **ridge penalty** (0.5 * ridge * ||attack||^2 + same
for defence), which identifies the model and conditions the problem, then
**recentre** the fitted parameters afterwards:

    attack' = attack - mean(attack)      defence' = defence - mean(defence)
    delta'  = delta + mean(attack) - mean(defence)

That transformation is exactly neutral on the likelihood -- the predicted rates
are bit-identical -- it just puts the parameters back in the interpretable
"sum to zero" form. `test_dc_recentring_does_not_change_predictions` pins that
down.

Fitting
-------
Weighted maximum likelihood over 2*n_teams + 3 parameters, using L-BFGS-B with an
**analytic gradient** (see `_nll`). The gradient includes the tau term's
dependence on the rates, not just its dependence on rho -- that coupling is easy
to miss and produces a subtly wrong rho if you ignore it.
`tests/test_models.py` checks the analytic gradient against finite differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import scorelines
from .base import GoalModel

HALFLIFE_DAYS = 730.0        # two years
RHO_BOUNDS = (-0.35, 0.35)   # keeps tau positive for realistic rates
DEFAULT_RIDGE = 1.0          # penalty on attack/defence, for identifiability


def time_weights(
    dates: pd.Series | np.ndarray,
    reference,
    halflife_days: float = HALFLIFE_DAYS,
) -> np.ndarray:
    """Exponential decay weight per match: 1.0 today, 0.5 at one halflife."""
    age_days = (pd.Timestamp(reference) - pd.to_datetime(pd.Series(dates))).dt.days.to_numpy(float)
    age_days = np.clip(age_days, 0.0, None)
    return np.power(0.5, age_days / float(halflife_days))


class DixonColes(GoalModel):
    """Weighted Poisson MLE with team attack/defence, home advantage and rho."""

    name = "dixon_coles"

    def __init__(
        self,
        halflife_days: float = HALFLIFE_DAYS,
        max_iter: int = 1500,
        ridge: float = DEFAULT_RIDGE,
    ):
        super().__init__()
        self.halflife_days = float(halflife_days)
        self.max_iter = int(max_iter)
        self.ridge = float(ridge)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, rows: pd.DataFrame) -> "DixonColes":
        """Fit on team-match rows (two per match, as produced by to_long_format)."""
        home = rows.loc[rows["is_home"] == 1,
                        ["match_id", "team", "opponent", "date",
                         "home_advantage", "goals_for"]]
        away = rows.loc[rows["is_home"] == 0, ["match_id", "goals_for"]]

        matches = home.merge(
            away, on="match_id", how="inner", suffixes=("_h", "_a"),
            validate="one_to_one",
        ).sort_values("match_id")

        teams = sorted(set(matches["team"]) | set(matches["opponent"]))
        self.teams_ = teams
        self.team_index_ = {team: i for i, team in enumerate(teams)}
        n = len(teams)

        home_idx = matches["team"].map(self.team_index_).to_numpy(int)
        away_idx = matches["opponent"].map(self.team_index_).to_numpy(int)
        home_goals = matches["goals_for_h"].to_numpy(float)
        away_goals = matches["goals_for_a"].to_numpy(float)
        home_flag = matches["home_advantage"].to_numpy(float)
        weights = time_weights(matches["date"], matches["date"].max(), self.halflife_days)

        # Only the four low-score cells get the tau treatment.
        is_00 = (home_goals == 0) & (away_goals == 0)
        is_01 = (home_goals == 0) & (away_goals == 1)
        is_10 = (home_goals == 1) & (away_goals == 0)
        is_11 = (home_goals == 1) & (away_goals == 1)

        ridge = self.ridge

        def _nll(params: np.ndarray) -> tuple[float, np.ndarray]:
            """Negative weighted log-likelihood and its exact gradient."""
            attack = params[:n]
            defence = params[n:2 * n]
            delta, gamma, rho = params[-3], params[-2], params[-1]

            eta_home = delta + attack[home_idx] - defence[away_idx] + gamma * home_flag
            eta_away = delta + attack[away_idx] - defence[home_idx]
            lam_home = np.clip(np.exp(eta_home), 1e-8, 30.0)
            lam_away = np.clip(np.exp(eta_away), 1e-8, 30.0)

            # --- Poisson part (constants dropped) -------------------------
            nll = float(np.sum(weights * (
                lam_home - home_goals * np.log(lam_home)
                + lam_away - away_goals * np.log(lam_away)
            )))
            grad_eta_home = weights * (lam_home - home_goals)
            grad_eta_away = weights * (lam_away - away_goals)

            # --- tau part -------------------------------------------------
            tau_00 = np.clip(1.0 - lam_home * lam_away * rho, 1e-10, None)
            tau_01 = np.clip(1.0 + lam_home * rho, 1e-10, None)
            tau_10 = np.clip(1.0 + lam_away * rho, 1e-10, None)
            tau_11 = max(1.0 - rho, 1e-10)

            log_tau = (is_00 * np.log(tau_00) + is_01 * np.log(tau_01)
                       + is_10 * np.log(tau_10) + is_11 * np.log(tau_11))
            nll -= float(np.sum(weights * log_tau))

            # d(-log tau)/d eta for the cells that multiply lam_home / lam_away
            coupling = weights * is_00 * (lam_home * lam_away * rho / tau_00)
            grad_eta_home += coupling - weights * is_01 * (lam_home * rho / tau_01)
            grad_eta_away += coupling - weights * is_10 * (lam_away * rho / tau_10)

            grad_rho = float(np.sum(weights * (
                is_00 * (lam_home * lam_away / tau_00)
                - is_01 * (lam_home / tau_01)
                - is_10 * (lam_away / tau_10)
                + is_11 / tau_11
            )))

            # --- ridge penalty (identifiability + conditioning) ------------
            nll += 0.5 * ridge * (float(attack @ attack) + float(defence @ defence))

            grad_attack = (
                np.bincount(home_idx, weights=grad_eta_home, minlength=n)
                + np.bincount(away_idx, weights=grad_eta_away, minlength=n)
                + ridge * attack
            )
            grad_defence = -(
                np.bincount(away_idx, weights=grad_eta_home, minlength=n)
                + np.bincount(home_idx, weights=grad_eta_away, minlength=n)
            ) + ridge * defence

            grad = np.concatenate([
                grad_attack,
                grad_defence,
                [grad_eta_home.sum() + grad_eta_away.sum(),
                 float(np.dot(grad_eta_home, home_flag)),
                 grad_rho],
            ])
            return nll, grad

        # --- starting values --------------------------------------------
        mean_goals = float(np.average(
            np.concatenate([home_goals, away_goals]),
            weights=np.concatenate([weights, weights]),
        ))
        x0 = np.zeros(2 * n + 3)
        x0[-3] = np.log(max(mean_goals, 0.1))   # delta
        x0[-2] = 0.20                            # home advantage
        x0[-1] = -0.05                           # rho

        bounds = (
            [(-4.0, 4.0)] * (2 * n)
            + [(-2.0, 3.0), (-1.0, 2.0), RHO_BOUNDS]
        )

        result = minimize(
            _nll, x0, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-10},
        )

        # Keep the closure so the gradient test can re-evaluate the objective.
        self._objective_ = _nll
        self.result_ = result
        self.halflife_days_ = self.halflife_days
        self.ridge_ = ridge
        self.n_teams_ = n
        self.n_matches_ = len(matches)
        self.converged_ = bool(result.success)

        attack = result.x[:n]
        defence = result.x[n:2 * n]
        delta = float(result.x[-3])

        # Recentre: neutral on the likelihood, and puts the parameters back into
        # the interpretable sum-to-zero form.
        attack_mean = float(attack.mean())
        defence_mean = float(defence.mean())
        self.attack_ = attack - attack_mean
        self.defence_ = defence - defence_mean
        self.delta_ = delta + attack_mean - defence_mean
        self.home_gamma_ = float(result.x[-2])
        self.rho_ = float(result.x[-1])

        # Padded copies so an unseen team falls back to "average" (0.0).
        self._attack_padded_ = np.append(self.attack_, 0.0)
        self._defence_padded_ = np.append(self.defence_, 0.0)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _index(self, names: pd.Series) -> np.ndarray:
        n = len(self.teams_)
        return np.array([self.team_index_.get(str(t), n) for t in names], dtype=int)

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        home_bonus = rows["home_advantage"].astype(float).to_numpy()
        eta = (
            self.delta_
            + self._attack_padded_[self._index(rows["team"])]
            - self._defence_padded_[self._index(rows["opponent"])]
            + self.home_gamma_ * home_bonus
        )
        return self._finalise_rates(np.clip(np.exp(eta), 0.01, 20.0))

    def outcome_probabilities(
        self, lam_home: float, lam_away: float, max_goals: int = scorelines.MAX_GOALS
    ) -> tuple[float, float, float]:
        """Win/draw/win from the rho-corrected score distribution."""
        return scorelines.dixon_coles_outcome_probabilities(
            lam_home, lam_away, self.rho_, max_goals
        )

    def predict_score_matrix(
        self, lam_home: float, lam_away: float, max_goals: int = 5
    ) -> np.ndarray:
        """Display-ready scoreline matrix including the rho correction."""
        return scorelines.dixon_coles_matrix(
            lam_home, lam_away, self.rho_, max_goals=max_goals
        )

    def __getstate__(self) -> dict:
        """Drop the likelihood closure before pickling.

        `_objective_` is a local closure (used only by the gradient test), and
        closures cannot be pickled. The fitted parameters are all we need to
        make predictions, so removing it makes the model savable.
        """
        state = self.__dict__.copy()
        state.pop("_objective_", None)
        return state

    def team_table(self) -> pd.DataFrame:
        """Per-team attack/defence parameters, for inspection and plots."""
        self._check_fitted()
        return pd.DataFrame(
            {"attack": self.attack_, "defence": self.defence_}, index=self.teams_
        )
