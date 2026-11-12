"""Metric tests: hand-computed values, so a silent formula change fails loudly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, evaluate, scorelines  # noqa: E402


# --------------------------------------------------------------------------
# Goal-level metrics
# --------------------------------------------------------------------------


def test_poisson_deviance_hand_computed():
    # y=0, mu=1  ->  2 * (0 - (0 - 1)) = 2
    assert evaluate.poisson_deviance(np.array([0.0]), np.array([1.0])) == pytest.approx(2.0)
    # y == mu    ->  0
    assert evaluate.poisson_deviance(np.array([2.0]), np.array([2.0])) == pytest.approx(0.0)
    # y=3, mu=1  ->  2 * (3 log 3 - 2)
    assert evaluate.poisson_deviance(np.array([3.0]), np.array([1.0])) == pytest.approx(
        2 * (3 * np.log(3) - 2)
    )
    # Averaging is a plain mean over team-match rows.
    both = evaluate.poisson_deviance(np.array([0.0, 2.0]), np.array([1.0, 2.0]))
    assert both == pytest.approx(1.0)


def test_mae_and_rmse_hand_computed():
    y = np.array([0.0, 3.0])
    mu = np.array([1.0, 1.0])
    assert evaluate.mean_absolute_error(y, mu) == pytest.approx(1.5)
    assert evaluate.root_mean_squared_error(y, mu) == pytest.approx(np.sqrt((1 + 4) / 2))


# --------------------------------------------------------------------------
# Score matrix
# --------------------------------------------------------------------------


def test_score_matrix_sums_to_one_and_has_poisson_marginals():
    lam_home, lam_away = 1.4, 1.1
    mat = scorelines.score_matrix(lam_home, lam_away, max_goals=15)

    assert mat.sum() == pytest.approx(1.0)
    # Marginals must match the Poisson pmf up to the renormalisation.
    np.testing.assert_allclose(mat.sum(axis=1), scorelines._poisson_pmf(lam_home, 15), atol=1e-6)
    np.testing.assert_allclose(mat.sum(axis=0), scorelines._poisson_pmf(lam_away, 15), atol=1e-6)


def test_outcome_probabilities_sum_to_one():
    for lam_h, lam_a in [(1.4, 1.1), (0.3, 3.5), (2.0, 2.0)]:
        p = scorelines.outcome_probabilities(lam_h, lam_a)
        assert sum(p) == pytest.approx(1.0)


def test_equal_rates_are_symmetric():
    p_home, p_draw, p_away = scorelines.outcome_probabilities(1.6, 1.6)
    assert p_home == pytest.approx(p_away)
    # A draw is the single most likely *scoreline*, never the most likely outcome
    # when both sides are evenly matched -- worth knowing before quoting "draw".
    assert p_draw < p_home + p_away


def test_more_expected_goals_means_higher_win_probability():
    low = scorelines.outcome_probabilities(1.0, 1.0)
    high = scorelines.outcome_probabilities(2.2, 1.0)
    assert high[0] > low[0]
    assert high[2] < low[2]


def test_display_matrix_uses_the_wide_grid_for_probabilities():
    """The 0-5 heatmap must not be used for outcome probabilities."""
    narrow = scorelines.display_matrix(3.5, 3.0, max_goals=5)
    wide = scorelines.score_matrix(3.5, 3.0, max_goals=15)
    assert narrow.shape == (6, 6)
    assert narrow.sum() == pytest.approx(1.0)
    # Truncating at 5 goals loses real mass when rates are high.
    assert narrow[5, 5] != pytest.approx(wide[5, 5])


def test_most_likely_scorelines_sorted():
    lines = scorelines.most_likely_scorelines(1.5, 0.9, top_n=4)
    probs = [p for _, _, p in lines]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0


# --------------------------------------------------------------------------
# Outcome-level metrics
# --------------------------------------------------------------------------


def test_log_loss_hand_computed():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    outcomes = np.array([0, 2])
    expected = -0.5 * (np.log(0.5) + np.log(0.7))
    assert evaluate.log_loss(probs, outcomes) == pytest.approx(expected)

    perfect = evaluate.log_loss(np.array([[0.0, 0.0, 1.0]]), np.array([2]))
    assert perfect == pytest.approx(0.0, abs=1e-9)


def test_ranked_probability_score_hand_computed():
    probs = np.array([[0.5, 0.3, 0.2]])
    # home win: 0.5 * [(0.5 - 1)^2 + (0.8 - 1)^2] = 0.145
    assert evaluate.ranked_probability_score(probs, np.array([0])) == pytest.approx(0.145)
    # Perfect forecast.
    assert evaluate.ranked_probability_score(
        np.array([[1.0, 0.0, 0.0]]), np.array([0])
    ) == pytest.approx(0.0)
    # Uniform forecast scores 2/9 whatever happens, by symmetry.
    uniform = np.full((3, 3), 1 / 3)
    assert evaluate.ranked_probability_score(uniform, np.array([0, 1, 2])) == pytest.approx(2 / 9)


def test_rps_punishes_the_wrong_side_more_than_the_middle():
    """RPS respects ordering: predicting home when away wins is worse than predicting a draw."""
    home_lean = np.array([[0.8, 0.15, 0.05]])
    draw_lean = np.array([[0.15, 0.7, 0.15]])
    assert evaluate.ranked_probability_score(home_lean, np.array([2])) > \
        evaluate.ranked_probability_score(draw_lean, np.array([2]))


def test_accuracy_uses_the_argmax_outcome():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert evaluate.accuracy(probs, np.array([0, 2])) == 1.0
    assert evaluate.accuracy(probs, np.array([1, 1])) == 0.0


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _preds_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [0, 1],
        "date": pd.to_datetime(["2022-11-22", "2022-11-23"]),
        "lambda_home": [1.8, 0.6],
        "lambda_away": [0.7, 2.1],
        "home_score": [2, 0],
        "away_score": [1, 3],
    })


def test_evaluate_predictions_end_to_end():
    metrics = evaluate.evaluate_predictions(_preds_frame())

    assert metrics["n_matches"] == 2
    assert metrics["n_team_rows"] == 4
    assert metrics["accuracy"] == 1.0        # the favoured side won both
    assert 0.0 < metrics["log_loss"] < 2.0
    assert 0.0 < metrics["rps"] < 0.5
    assert metrics["mean_predicted_goals"] == pytest.approx((1.8 + 0.7 + 0.6 + 2.1) / 4)
    for key in evaluate.METRIC_KEYS:
        assert key in metrics


def test_evaluate_predictions_rejects_incomplete_tables():
    with pytest.raises(ValueError):
        evaluate.evaluate_predictions(pd.DataFrame({"lambda_home": [1.0]}))


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    preds = _preds_frame()
    a = evaluate.bootstrap_metrics(preds, n_boot=200, seed=1)
    b = evaluate.bootstrap_metrics(preds, n_boot=200, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["metric"]) == set(evaluate.METRIC_KEYS)
    assert (a["ci_low"] <= a["value"] + 1e-12).all()
    assert (a["value"] <= a["ci_high"] + 1e-12).all()

    # A different seed moves the interval but not the point estimate.
    c = evaluate.bootstrap_metrics(preds, n_boot=200, seed=2)
    np.testing.assert_allclose(a["value"], c["value"])


def test_pivot_predictions_roundtrips_to_one_row_per_match():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2022-11-22", "2022-11-23"]),
        "home_team": ["Argentina", "France"],
        "away_team": ["Saudi Arabia", "Australia"],
        "home_score": [1, 4],
        "away_score": [2, 1],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "neutral": [True, True],
        "year": [2022, 2022],
        "match_id": [0, 1],
        "tournament_category": ["world_cup", "world_cup"],
    })
    long = data.to_long_format(matches)
    preds = evaluate.pivot_predictions(long, np.array([1.5, 1.0, 2.0, 0.9]))

    assert len(preds) == 2
    assert preds["home_score"].tolist() == [1, 4]
    assert preds["away_score"].tolist() == [2, 1]
    # Row order of the long table must not scramble the home/away assignment.
    merged = preds.set_index("match_id")
    assert merged.loc[0, "lambda_home"] == 1.5
    assert merged.loc[0, "lambda_away"] == 1.0


def test_markdown_table_renders_numbers_and_strings():
    table = pd.DataFrame({"model": ["a"], "accuracy": [0.5]})
    text = evaluate.to_markdown(table)
    assert text.startswith("| model | accuracy |")
    assert "0.5000" in text


def test_rps_per_match_averages_to_the_overall_rps():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    outcomes = np.array([0, 2])

    per_match = evaluate.rps_per_match(probs, outcomes)
    assert per_match.shape == (2,)
    assert per_match[0] == pytest.approx(0.145)
    assert per_match.mean() == pytest.approx(
        evaluate.ranked_probability_score(probs, outcomes)
    )


def test_calibration_bins_pools_all_three_outcomes():
    probs = np.full((10, 3), 1 / 3)
    outcomes = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    table = evaluate.calibration_bins(probs, outcomes, n_bins=10)
    assert table["n"].sum() == 30          # 10 matches x 3 outcomes
    assert table["mean_predicted"].iloc[0] == pytest.approx(1 / 3, abs=1e-6)
    assert ((table["observed"] >= 0) & (table["observed"] <= 1)).all()
