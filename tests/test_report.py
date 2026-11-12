"""Tests for step 6: the reporting layer's plumbing.

These do not re-test the metrics (test_metrics.py does that); they check that the
report assembles the right shape, keeps intervals around point estimates, and
never presents fallback data as real results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, report  # noqa: E402


def fake_predictions() -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame({
        "split": ["test", "test", "test"],
        "date": pd.to_datetime(["2022-11-22", "2022-11-23", "2022-11-24"]),
        "home_team": ["Argentina", "France", "Brazil"],
        "away_team": ["Saudi Arabia", "Australia", "Serbia"],
        "lambda_home": [1.8, 1.6, 1.4],
        "lambda_away": [0.6, 0.9, 1.0],
        "home_score": [1, 4, 2],
        "away_score": [2, 1, 0],
        "p_home": [0.55, 0.58, 0.48],
        "p_draw": [0.25, 0.24, 0.27],
        "p_away": [0.20, 0.18, 0.25],
        "outcome": [2, 0, 0],
        "rps": [0.12, 0.05, 0.09],
    })
    return {"poisson_glm": frame}


def test_summary_table_has_intervals_around_every_point_estimate():
    summary = report.summary_table(fake_predictions(), "test", n_boot=100)

    assert summary["model"].tolist() == ["poisson_glm"]
    assert int(summary.loc[0, "n_matches"]) == 3
    for metric in ("rps", "log_loss", "accuracy"):
        assert summary.loc[0, f"{metric}_low"] <= summary.loc[0, metric]
        assert summary.loc[0, metric] <= summary.loc[0, f"{metric}_high"]


def test_display_summary_formats_numbers_for_markdown():
    summary = report.summary_table(fake_predictions(), "test", n_boot=50)
    display = report._display_summary(summary)

    assert display.loc[0, "Model"] == "Poisson GLM"
    # "0.123 [0.100, 0.150]" style, so the interval reads as one cell.
    assert "[" in display.loc[0, "RPS"] and "]" in display.loc[0, "RPS"]


def test_showcase_picks_the_last_match_of_the_block():
    predictions = fake_predictions()
    name, row = report._pick_showcase(predictions, block="test")
    assert name == "poisson_glm"
    assert row["date"] == pd.Timestamp("2022-11-24")


def test_showcase_returns_none_for_a_missing_block():
    assert report._pick_showcase(fake_predictions(), block="world_cup_2018") is None


def test_synthetic_marker_distinguishes_fallback_data(tmp_path):
    """Without this marker a report could present fake numbers as real results."""
    csv = tmp_path / "results.csv"
    assert not data.is_synthetic_data(csv)

    data.generate_synthetic_results(csv, n_matches=50, seed=1)
    assert data.is_synthetic_data(csv)
    assert data.load_results(csv).attrs["synthetic"] is True
