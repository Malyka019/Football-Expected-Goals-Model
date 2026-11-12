"""Baseline tests: they are simple, so the assertions are exact numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import FormBaseline, HomeAwayBaseline, LeagueAverageBaseline  # noqa: E402


@pytest.fixture
def rows() -> pd.DataFrame:
    """Two matches, four team-rows. Goals per team: 2, 0, 1, 1 (mean 1.0)."""
    return pd.DataFrame({
        "match_id": [1, 1, 2, 2],
        "team": ["A", "B", "A", "B"],
        "opponent": ["B", "A", "B", "A"],
        "is_home": [1, 0, 1, 0],
        "home_advantage": [1, 0, 1, 0],
        "goals_for": [2, 0, 1, 1],
        "form_gf_10": [2.0, 0.0, 1.5, 1.0],
        "form_ga_10": [0.0, 2.0, 1.0, 1.5],
    })


def test_league_average_returns_the_training_mean(rows):
    model = LeagueAverageBaseline().fit(rows)
    lam = model.predict_lambda(rows)
    assert np.allclose(lam, 1.0)
    assert len(lam) == len(rows)


def test_home_away_baseline_splits_the_mean_by_venue(rows):
    model = HomeAwayBaseline().fit(rows)
    lam = model.predict_lambda(rows)
    # Rows with home advantage: 2 and 1 -> 1.5. Without: 0 and 1 -> 0.5.
    assert np.allclose(lam, [1.5, 0.5, 1.5, 0.5])


def test_form_baseline_averages_own_scoring_and_opponent_conceding(rows):
    model = FormBaseline().fit(rows)
    lam = model.predict_lambda(rows)
    # row 1: A scores 2.0/10, B concedes 2.0/10 -> 2.0
    # row 2: B scores 0.0/10, A concedes 0.0/10 -> 0.0, floored at 0.05
    # row 3: A scores 1.5, B concedes 1.5 -> 1.5
    # row 4: B scores 1.0, A concedes 1.0 -> 1.0
    assert np.allclose(lam, [2.0, 0.05, 1.5, 1.0])


def test_form_baseline_preserves_row_order(rows):
    """A left join on (match_id, opponent) must not reorder predictions."""
    shuffled = pd.concat([rows.iloc[[2, 3]], rows.iloc[[0, 1]]], ignore_index=True)
    model = FormBaseline().fit(rows)
    lam = model.predict_lambda(shuffled)
    assert np.allclose(lam, [1.5, 1.0, 2.0, 0.05])


def test_predicting_before_fitting_raises(rows):
    for model in (LeagueAverageBaseline(), HomeAwayBaseline(), FormBaseline()):
        with pytest.raises(RuntimeError):
            model.predict_lambda(rows)


def test_baselines_never_predict_negative_rates(rows):
    for model in (LeagueAverageBaseline(), HomeAwayBaseline(), FormBaseline()):
        lam = model.fit(rows).predict_lambda(rows)
        assert np.all(lam > 0)
        assert np.all(np.isfinite(lam))
