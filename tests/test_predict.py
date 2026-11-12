"""Tests for step 7: predict_match, fixture construction and artifacts."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, predict  # noqa: E402


@pytest.fixture(scope="module")
def matches(tmp_path_factory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("predict") / "results.csv"
    data.generate_synthetic_results(path, n_matches=900, seed=21)
    return data.load_results(path)


@pytest.fixture(scope="module")
def predictor(matches) -> predict.Predictor:
    return predict.Predictor.from_data("poisson_glm", matches=matches)


# --------------------------------------------------------------------------
# State and fixture construction
# --------------------------------------------------------------------------


def test_team_state_has_one_complete_row_per_team(predictor):
    assert len(predictor.state) == len(predictor.teams)
    assert predictor.state[predict.STATE_COLUMNS].notna().all().all()
    assert predictor.state["elo"].between(1000, 2500).all()


def test_fixture_rows_encode_venue_correctly(predictor):
    rows = predict.fixture_rows(
        predictor.state, predictor.matches, "Brazil", "Argentina",
        neutral=True, tournament_category="world_cup",
    )
    assert len(rows) == 2
    assert rows["home_advantage"].tolist() == [0, 0]
    assert rows["neutral"].all()

    home_rows = predict.fixture_rows(
        predictor.state, predictor.matches, "Brazil", "Argentina",
        neutral=False, tournament_category="friendly",
    )
    assert home_rows["home_advantage"].tolist() == [1, 0]


def test_head_to_head_is_from_the_first_teams_perspective(predictor):
    stats = predict.head_to_head(predictor.matches, "Brazil", "Argentina")
    reversed_stats = predict.head_to_head(predictor.matches, "Argentina", "Brazil")
    assert stats["played"] == reversed_stats["played"]
    assert stats["gd"] == pytest.approx(-reversed_stats["gd"])


# --------------------------------------------------------------------------
# predict_match
# --------------------------------------------------------------------------


def test_prediction_is_a_complete_distribution(predictor):
    result = predictor.predict_match("Brazil", "Argentina", neutral=True)

    assert result.expected_goals_a > 0 and result.expected_goals_b > 0
    assert result.p_win_a + result.p_draw + result.p_win_b == pytest.approx(1.0, abs=1e-9)
    assert result.score_matrix.shape == (6, 6)
    assert result.score_matrix.to_numpy().sum() == pytest.approx(100.0, abs=1e-6)
    assert len(result.top_scorelines) == 5
    assert result.most_likely_scoreline == (
        f"{result.top_scorelines[0][0]}-{result.top_scorelines[0][1]}"
    )
    assert "expected goals" in result.summary()
    assert result.as_dict()["team_a"] == "Brazil"


def test_home_advantage_raises_the_home_sides_rate(predictor):
    neutral = predictor.predict_match("Brazil", "Argentina", neutral=True)
    at_home = predictor.predict_match("Brazil", "Argentina", neutral=False)

    assert at_home.expected_goals_a > neutral.expected_goals_a
    assert at_home.p_win_a > neutral.p_win_a


def test_the_stronger_side_is_favoured_in_both_orderings(predictor):
    strong, weak = predictor.state["elo"].sort_values(ascending=False).head(2).index

    forward = predictor.predict_match(strong, weak, neutral=True)
    backward = predictor.predict_match(weak, strong, neutral=True)

    assert forward.expected_goals_a > forward.expected_goals_b
    assert backward.expected_goals_b > backward.expected_goals_a


def test_a_team_cannot_play_itself(predictor):
    with pytest.raises(ValueError):
        predictor.predict_match("Brazil", "Brazil")


def test_tournament_changes_the_prediction_only_where_the_model_uses_it(predictor):
    friendly = predictor.predict_match("Brazil", "Argentina", tournament="Friendly")
    world_cup = predictor.predict_match("Brazil", "Argentina", tournament="FIFA World Cup")
    assert friendly.tournament != world_cup.tournament
    assert friendly.p_win_a + friendly.p_draw + friendly.p_win_b == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Team-name resolution
# --------------------------------------------------------------------------


def test_exact_and_near_miss_team_names_resolve(predictor):
    teams = predictor.teams
    assert predict.resolve_team("Brazil", teams) == "Brazil"
    assert predict.resolve_team("Argentia", teams) == "Argentina"


def test_unknown_team_fails_with_suggestions(predictor):
    with pytest.raises(KeyError, match="Did you mean|Unknown team"):
        predict.resolve_team("Xylophonia", predictor.teams)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


def test_pickled_model_predicts_identically(predictor):
    rows = predict.fixture_rows(
        predictor.state, predictor.matches, "Brazil", "Argentina",
        neutral=True, tournament_category="world_cup",
    )
    before = predictor.model.predict_lambda(rows)
    restored = pickle.loads(pickle.dumps(predictor.model))
    np.testing.assert_allclose(before, restored.predict_lambda(rows))


def test_dixon_coles_is_picklable(matches):
    dc = predict.Predictor.from_data("dixon_coles", matches=matches)
    rows = predict.fixture_rows(
        dc.state, dc.matches, "Brazil", "Argentina",
        neutral=True, tournament_category="world_cup",
    )
    before = dc.model.predict_lambda(rows)
    restored = pickle.loads(pickle.dumps(dc.model))

    # The likelihood closure is dropped on pickling, but predictions survive.
    assert not hasattr(restored, "_objective_")
    np.testing.assert_allclose(before, restored.predict_lambda(rows))


def test_save_and_load_artifact_roundtrip(predictor, tmp_path, monkeypatch):
    monkeypatch.setattr(
        predict, "artifact_path", lambda name: tmp_path / f"{name}.pkl"
    )
    path = predict.save_artifact(predictor.model, {"model": "poisson_glm", "trained_on": ["train"]})
    assert path.exists()

    model, metadata = predict.load_artifact("poisson_glm")
    assert metadata["trained_on"] == ["train"]

    rows = predict.fixture_rows(
        predictor.state, predictor.matches, "Brazil", "Argentina",
        neutral=False, tournament_category="friendly",
    )
    np.testing.assert_allclose(
        predictor.model.predict_lambda(rows), model.predict_lambda(rows)
    )
