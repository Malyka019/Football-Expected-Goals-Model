"""Tests for step 4: Poisson GLM and the from-scratch Dixon-Coles fit.

The important one is `test_dc_gradient_matches_finite_differences`: a hand-derived
analytic gradient is the easiest place in this project to be confidently wrong,
so it is checked against numerical differentiation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, features, scorelines  # noqa: E402
from src.models import DixonColes, PoissonGLM  # noqa: E402
from src.models.dixon_coles import time_weights  # noqa: E402
from src.models.poisson_glm import build_design  # noqa: E402


@pytest.fixture(scope="module")
def feature_table(tmp_path_factory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("data") / "results.csv"
    data.generate_synthetic_results(path, n_matches=700, seed=5)
    matches = data.load_results(path)
    return features.compute_features(matches)


# --------------------------------------------------------------------------
# tau correction
# --------------------------------------------------------------------------


def test_tau_correction_lifts_draws_and_trims_1_0():
    lam_home, lam_away, rho = 1.4, 1.1, -0.1
    base = scorelines.score_matrix(lam_home, lam_away)
    adjusted = scorelines.dixon_coles_matrix(lam_home, lam_away, rho)

    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[0, 0] > base[0, 0]     # 0-0 lifted
    assert adjusted[1, 1] > base[1, 1]     # 1-1 lifted
    assert adjusted[0, 1] < base[0, 1]     # 0-1 trimmed
    assert adjusted[1, 0] < base[1, 0]     # 1-0 trimmed

    # Cells outside the 2x2 block keep their relative shape.
    ratio = adjusted[2, 2] / base[2, 2]
    assert adjusted[4, 3] / base[4, 3] == pytest.approx(ratio, rel=1e-9)


def test_zero_rho_reproduces_independent_poisson():
    base = scorelines.score_matrix(1.3, 0.9)
    adjusted = scorelines.dixon_coles_matrix(1.3, 0.9, rho=0.0)
    np.testing.assert_allclose(adjusted, base, atol=1e-12)


def test_dixon_coles_outcome_probabilities_sum_to_one():
    for rho in (-0.2, -0.05, 0.0, 0.1):
        p = scorelines.dixon_coles_outcome_probabilities(1.5, 1.2, rho)
        assert sum(p) == pytest.approx(1.0)
        assert all(0.0 <= x <= 1.0 for x in p)


# --------------------------------------------------------------------------
# time decay
# --------------------------------------------------------------------------


def test_time_weights_halve_at_one_halflife():
    reference = pd.Timestamp("2022-12-18")
    dates = pd.Series(["2022-12-18", "2021-12-18", "2020-12-19", "2025-01-01"])
    weights = time_weights(dates, reference, halflife_days=365)

    assert weights[0] == pytest.approx(1.0)          # today
    assert weights[1] == pytest.approx(0.5)          # one halflife ago
    assert weights[2] == pytest.approx(0.25, rel=0.01)
    assert weights[3] == pytest.approx(1.0)          # future dates clipped


# --------------------------------------------------------------------------
# Dixon-Coles: the gradient, then the fit
# --------------------------------------------------------------------------


def test_dc_gradient_matches_finite_differences(feature_table):
    model = DixonColes().fit(feature_table)
    objective = model._objective_

    rng = np.random.default_rng(0)
    params = model.result_.x + rng.normal(0.0, 0.01, model.result_.x.shape)

    _, analytic = objective(params)

    eps = 1e-6
    numeric = np.empty_like(params)
    for i in range(len(params)):
        upper = params.copy()
        lower = params.copy()
        upper[i] += eps
        lower[i] -= eps
        numeric[i] = (objective(upper)[0] - objective(lower)[0]) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-4)


def test_dc_fit_is_identified_and_reasonable(feature_table):
    model = DixonColes().fit(feature_table)

    # Sum-to-zero constraints identify the attack/defence scale.
    assert model.attack_.sum() == pytest.approx(0.0, abs=1e-8)
    assert model.defence_.sum() == pytest.approx(0.0, abs=1e-8)

    lower, upper = -0.35, 0.35
    assert lower <= model.rho_ <= upper
    assert np.isfinite(model.result_.fun)
    assert len(model.teams_) == feature_table["team"].nunique()

    lam = model.predict_lambda(feature_table)
    assert len(lam) == len(feature_table)
    assert np.all(lam > 0) and np.all(np.isfinite(lam))
    # The fit is *weighted* by recency, so predicted rates track recent scoring
    # rather than the all-time mean. Expect them close, not identical.
    assert lam.mean() == pytest.approx(feature_table["goals_for"].mean(), rel=0.2)


def test_dc_recentring_does_not_change_predictions(feature_table):
    """Shifting attacks by c and the intercept by -c must leave rates untouched.

    This is the transformation the fit applies after optimising, so if it were
    wrong every prediction would be off.
    """
    model = DixonColes().fit(feature_table)
    rows = feature_table.iloc[:25]
    before = model.predict_lambda(rows)

    shift = 0.7
    model.attack_ = model.attack_ + shift
    model.delta_ = model.delta_ - shift
    model._attack_padded_ = np.append(model.attack_, 0.0)

    np.testing.assert_allclose(before, model.predict_lambda(rows), rtol=1e-12)


def test_dc_handles_unseen_teams(feature_table):
    model = DixonColes().fit(feature_table)
    rows = feature_table.iloc[[0, 1]].copy()
    rows["team"] = ["Atlantis", "Atlantis"]
    rows["opponent"] = ["Atlantis", "Atlantis"]

    lam = model.predict_lambda(rows)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0)


def test_dc_predicting_before_fitting_raises(feature_table):
    with pytest.raises(RuntimeError):
        DixonColes().predict_lambda(feature_table)


def test_dc_outcome_probabilities_use_its_own_rho(feature_table):
    model = DixonColes().fit(feature_table)
    p_model = model.outcome_probabilities(1.4, 1.1)
    p_independent = scorelines.outcome_probabilities(1.4, 1.1)
    assert sum(p_model) == pytest.approx(1.0)
    # The correction is small but not a no-op.
    assert not np.allclose(p_model, p_independent, atol=1e-6)


def test_dc_score_matrix_is_display_ready(feature_table):
    model = DixonColes().fit(feature_table)
    mat = model.predict_score_matrix(1.5, 1.0, max_goals=5)
    assert mat.shape == (6, 6)
    assert mat.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Poisson GLM
# --------------------------------------------------------------------------


def test_glm_design_omits_the_collinear_difference(feature_table):
    design = build_design(feature_table)
    assert "elo_diff" not in design.columns
    assert "elo_team" in design.columns and "elo_opponent" in design.columns
    assert design.notna().all().all()


def test_glm_fits_and_predicts_sensible_rates(feature_table):
    model = PoissonGLM().fit(feature_table)

    # Stronger teams must score more: the own-Elo coefficient is positive, and
    # the opponent's is negative.
    assert model.params_["elo_team"] > 0
    assert model.params_["elo_opponent"] < 0

    lam = model.predict_lambda(feature_table)
    assert len(lam) == len(feature_table)
    assert np.all(lam > 0) and np.all(np.isfinite(lam))
    assert lam.mean() == pytest.approx(feature_table["goals_for"].mean(), abs=0.05)


def test_glm_predicting_before_fitting_raises(feature_table):
    with pytest.raises(RuntimeError):
        PoissonGLM().predict_lambda(feature_table)
