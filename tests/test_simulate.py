"""Tests for step 8: the Monte Carlo World Cup.

The statistical tests use a hand-made rate matrix where the answer is obvious
(one team is far stronger), so they do not depend on the synthetic generator's
particular latent strengths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, predict, simulate  # noqa: E402


@pytest.fixture(scope="module")
def predictor(tmp_path_factory) -> predict.Predictor:
    path = tmp_path_factory.mktemp("sim") / "results.csv"
    data.generate_synthetic_results(path, n_matches=700, seed=31)
    return predict.Predictor.from_data("poisson_glm", matches=data.load_results(path))


@pytest.fixture(scope="module")
def field(predictor) -> list[str]:
    return predictor.state.sort_values("elo", ascending=False).index[:32].tolist()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def test_rate_matrix_is_one_call_and_well_formed(predictor, field):
    lam = simulate.build_rate_matrix(predictor, field)

    assert lam.shape == (32, 32)
    assert np.all(np.diag(lam) == 0.0), "a team does not play itself"
    off_diagonal = lam[~np.eye(32, dtype=bool)]
    assert np.all(off_diagonal > 0)
    assert np.all(off_diagonal < 20)


def test_draw_groups_is_a_partition_and_respects_pots(predictor, field):
    strength = predictor.state.loc[field, "elo"].to_numpy()
    groups = simulate.draw_groups(field, seed=3, strength=strength)

    assert groups.shape == (8, 4)
    assert sorted(groups.ravel().tolist()) == list(range(32)), "each team exactly once"

    # The four strongest sides must be in four different groups.
    strongest = set(np.argsort(-strength)[:4].tolist())
    assert all(len(strongest & set(group.tolist())) <= 1 for group in groups)


def test_draw_groups_rejects_the_wrong_number_of_teams():
    with pytest.raises(ValueError):
        simulate.draw_groups(["A", "B", "C"], seed=1)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def test_simulation_is_reproducible(predictor, field):
    lam = simulate.build_rate_matrix(predictor, field)
    groups = simulate.draw_groups(field, seed=7)

    first = simulate.simulate(lam, groups, n_sims=150, seed=11)
    second = simulate.simulate(lam, groups, n_sims=150, seed=11)
    np.testing.assert_array_equal(first["stage"], second["stage"])


def test_probabilities_are_normalised_and_monotone(predictor, field):
    lam = simulate.build_rate_matrix(predictor, field)
    groups = simulate.draw_groups(field, seed=7)
    outcome = simulate.simulate(lam, groups, n_sims=150, seed=11)

    table = simulate.summarise(outcome["stage"], field, 150)

    assert table["p_champion"].sum() == pytest.approx(1.0)
    assert (table["p_champion"] <= table["p_final"] + 1e-12).all()
    assert (table["p_final"] <= table["p_sf"] + 1e-12).all()
    assert (table["p_sf"] <= table["p_qf"] + 1e-12).all()
    assert (table["p_qf"] <= table["p_r16"] + 1e-12).all()
    # Exactly 16 teams survive the group stage in every simulation.
    assert table["p_r16"].sum() == pytest.approx(16.0)


def test_a_dominant_team_almost_always_wins():
    lam = np.full((32, 32), 1.3)
    lam[0, :] = 4.0        # team 0 scores freely
    lam[:, 0] = 0.15       # and concedes almost nothing
    np.fill_diagonal(lam, 0.0)
    groups = np.arange(32).reshape(8, 4)

    outcome = simulate.simulate(lam, groups, n_sims=300, seed=5)
    table = simulate.summarise(outcome["stage"], [f"T{i}" for i in range(32)], 300)

    assert table.set_index("team").loc["T0", "p_champion"] > 0.95


def test_knockout_ties_are_always_resolved():
    rng = np.random.default_rng(2)
    lam = np.full((4, 4), 1.2)
    np.fill_diagonal(lam, 0.0)

    home = np.array([[0], [1], [2], [3]])
    away = np.array([[1], [0], [3], [2]])

    winners = simulate._knockout(lam, home, away, rng)

    assert winners.shape == (4, 1)
    pairs = set(zip(home[:, 0].tolist(), away[:, 0].tolist()))
    for winner, pair in zip(winners[:, 0].tolist(), pairs):
        assert winner in pair, "a draw must still produce one of the two teams"


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_run_simulation_end_to_end(predictor):
    result = simulate.run_simulation(predictor, n_sims=120, seed=1)

    assert len(result.teams) == 32
    assert result.groups.shape == (8, 4)
    assert result.n_sims == 120
    assert result.table["p_champion"].sum() == pytest.approx(1.0)
    assert list(result.table.columns) == [
        "team", "p_champion", "se_champion", "p_final", "p_sf", "p_qf", "p_r16",
    ]
