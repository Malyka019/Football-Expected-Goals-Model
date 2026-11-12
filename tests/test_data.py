"""Tests for the data layer: cleaning, tournament mapping, long-format invariants.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data  # noqa: E402


@pytest.fixture(scope="module")
def raw_matches() -> pd.DataFrame:
    """A small synthetic match table with a few deliberate defects."""
    df = pd.DataFrame({
        "date": ["2010-06-11", "2010-06-12", "2010-06-12", "2014-07-13",
                 "2018-06-16", "2022-11-22", None],
        "home_team": ["Argentina", "Brazil ", "Brazil", "Germany",
                      "France", "Argentina", "Spain"],
        "away_team": ["Nigeria", "North Korea", "North Korea", "Argentina",
                      "Australia", "Saudi Arabia", "Italy"],
        "home_score": [1, 2, 2, 1, 2, 1, 0],
        "away_score": [0, 1, 1, 0, 1, 2, 0],
        # duplicate fixture above (2010-06-12) plus a row with no score
        "tournament": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup",
                       "FIFA World Cup", "FIFA World Cup",
                       "FIFA World Cup qualification", "Friendly"],
        "city": ["Johannesburg"] * 7,
        "country": ["South Africa"] * 7,
        "neutral": ["TRUE", "TRUE", "TRUE", "TRUE", "FALSE", "FALSE", "FALSE"],
    })
    df.loc[6, "home_score"] = np.nan  # unusable target -> must be dropped
    return df


def test_cleaning_drops_bad_and_duplicate_rows(raw_matches):
    clean, report = data.clean_results(raw_matches)

    # 7 rows - 1 (unparseable date AND missing score, same row) - 1 duplicate = 5
    assert len(clean) == 5
    assert report["dropped_missing_values"] == 1
    assert report["dropped_duplicates"] == 1
    assert clean["home_score"].notna().all()
    assert clean["date"].is_monotonic_increasing
    assert clean["match_id"].is_unique


def test_neutral_flag_normalised_to_bool(raw_matches):
    clean, _ = data.clean_results(raw_matches)
    assert clean["neutral"].dtype == bool
    # 3 of the 5 surviving rows were flagged TRUE (one neutral row was the duplicate)
    assert clean["neutral"].sum() == 3


def test_team_names_stripped(raw_matches):
    clean, _ = data.clean_results(raw_matches)
    assert "Brazil " not in set(clean["home_team"])
    assert "Brazil" in set(clean["home_team"])


def test_tournament_classification():
    cases = {
        "FIFA World Cup": "world_cup",
        "FIFA World Cup qualification": "qualifier",
        "UEFA Euro": "continental_final",
        "UEFA Euro qualification": "qualifier",
        "Copa America": "continental_final",
        "UEFA Nations League": "nations_league",
        "Friendly": "friendly",
        "British Home Championship": "other",
    }
    for name, expected in cases.items():
        assert config.classify_tournament(name) == expected, name

    assert config.tournament_importance("world_cup") > config.tournament_importance("friendly")


def test_long_format_is_two_rows_per_match_with_consistent_goals(raw_matches):
    matches, _ = data.clean_results(raw_matches)
    long = data.to_long_format(matches)

    assert len(long) == 2 * len(matches)
    assert long.groupby("match_id").size().eq(2).all()
    assert long["goals_for"].notna().all()
    assert long["goal_diff"].equals(long["goals_for"] - long["goals_against"])

    # Each team's goals against must be the opponent's goals for, same match.
    merged = long.merge(long, on=["match_id"], suffixes=("", "_opp"))
    pairs = merged[merged["team"] != merged["team_opp"]]
    assert (pairs["goals_for"] == pairs["goals_against_opp"]).all()
    assert (pairs["opponent"] == pairs["team_opp"]).all()


def test_results_and_home_advantage_flags(raw_matches):
    matches, _ = data.clean_results(raw_matches)
    long = data.to_long_format(matches)

    # Result must agree with the sign of the goal difference.
    assert (long.loc[long["result"] == "W", "goal_diff"] > 0).all()
    assert (long.loc[long["result"] == "D", "goal_diff"] == 0).all()
    assert (long.loc[long["result"] == "L", "goal_diff"] < 0).all()

    # Home advantage exists only for the nominal home side at a non-neutral venue.
    neutral_ids = set(matches.loc[matches["neutral"], "match_id"])
    assert (long.loc[long["match_id"].isin(neutral_ids), "home_advantage"] == 0).all()
    assert long["home_advantage"].isin([0, 1]).all()
    assert long["home_advantage"].sum() == matches["neutral"].eq(False).sum()


def test_synthetic_generator_roundtrips(tmp_path):
    csv = tmp_path / "results.csv"
    data.generate_synthetic_results(csv, n_matches=300, seed=7)
    assert csv.exists()

    matches = data.load_results(csv)
    long = data.to_long_format(matches)

    assert 0 < len(matches) <= 300
    assert len(long) == 2 * len(matches)
    assert matches["neutral"].any() and (~matches["neutral"]).any()
    for col in ("world_cup", "friendly", "qualifier"):
        assert (matches["tournament_category"] == col).any()
    # The generator is seeded, so the file must be reproducible.
    first = csv.read_bytes()
    data.generate_synthetic_results(csv, n_matches=300, seed=7)
    assert csv.read_bytes() == first


def test_load_results_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_results(tmp_path / "nope.csv")
