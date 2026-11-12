"""Tests for step 5: LightGBM with a Poisson objective."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, features  # noqa: E402
from src.models.gbm_poisson import GBMPoisson, build_design, tune_gbm  # noqa: E402


@pytest.fixture(scope="module")
def feature_table(tmp_path_factory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("gbm") / "results.csv"
    data.generate_synthetic_results(path, n_matches=600, seed=13)
    return features.compute_features(data.load_results(path))


def test_design_uses_an_explicit_categorical(feature_table):
    design = build_design(feature_table)
    # Explicit categories mean train and predict can never disagree on encoding.
    assert str(design["tournament_category"].dtype) == "category"
    assert set(design["tournament_category"].cat.categories) == set(
        __import__("src.config", fromlist=["config"]).TOURNAMENT_CATEGORIES
    )
    assert design.drop(columns=["tournament_category"]).notna().all().all()


def test_fit_predict_and_determinism(feature_table):
    model = GBMPoisson(params={"n_estimators": 60}).fit(feature_table)
    lam = model.predict_lambda(feature_table)

    assert len(lam) == len(feature_table)
    assert np.all(lam > 0) and np.all(np.isfinite(lam))
    assert lam.mean() == pytest.approx(feature_table["goals_for"].mean(), rel=0.25)

    again = GBMPoisson(params={"n_estimators": 60}).fit(feature_table)
    np.testing.assert_allclose(lam, again.predict_lambda(feature_table))


def test_feature_importance_is_a_percentage(feature_table):
    model = GBMPoisson(params={"n_estimators": 60}).fit(feature_table)
    importance = model.feature_importance()

    assert importance.sum() == pytest.approx(100.0)
    assert (importance >= 0).all()
    assert importance.index.tolist() == model.feature_names_ or set(
        importance.index
    ) == set(model.feature_names_)


def test_predict_before_fit_raises(feature_table):
    with pytest.raises(RuntimeError):
        GBMPoisson().predict_lambda(feature_table)


def test_tuning_returns_a_sorted_table_and_valid_params(feature_table):
    split = int(len(feature_table) * 0.75)
    train, validation = feature_table.iloc[:split], feature_table.iloc[split:]

    table, best = tune_gbm(
        train, validation, grid=[{"num_leaves": 7}, {"num_leaves": 15}]
    )
    assert len(table) == 2
    assert table["rps"].is_monotonic_increasing
    assert best["num_leaves"] in (7, 15)
    # The tuner must not leak the test block: it only ever sees these two frames.
    assert set(table.columns) >= {"num_leaves", "rps", "log_loss", "poisson_deviance"}
