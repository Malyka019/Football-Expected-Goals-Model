"""Leakage tests for the feature layer.

Two kinds of test live here:

1. **Physical invariants** -- facts that must hold by construction (the current
   match is not inside its own form window, Elo is updated after it is read,
   head-to-head history excludes the match itself).
2. **Truncation invariance** -- the strong one. Recompute the features on a
   dataset truncated at date T and assert every row before T is unchanged. If a
   feature ever reaches forward in time, this fails. A second variant randomises
   all matches *after* T and checks the same thing, which also catches features
   built from a whole-dataset aggregate (a global mean, a global z-score).

Run with:  pytest -q tests/test_no_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, features  # noqa: E402

# Every column whose value must be identical under truncation.
COMPARE_COLUMNS = [
    "elo_team", "elo_opponent", "elo_diff", "elo_home_pre", "elo_away_pre",
    "history_matches",
    "form_gf_5", "form_ga_5", "form_points_5", "form_win_5",
    "form_gf_10", "form_ga_10", "form_points_10", "form_win_10",
    "attack", "defence", "h2h_played", "h2h_gd", "h2h_win",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_matches(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal valid match table from (date, home, away, hs, as) rows."""
    df = pd.DataFrame(
        rows, columns=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "Friendly"
    df["neutral"] = False
    df["year"] = df["date"].dt.year
    df["match_id"] = np.arange(len(df))
    df["tournament_category"] = df["tournament"].map(config.classify_tournament)
    return df


@pytest.fixture(scope="module")
def synthetic_matches(tmp_path_factory) -> pd.DataFrame:
    path = tmp_path_factory.mktemp("data") / "results.csv"
    data.generate_synthetic_results(path, n_matches=900, seed=11)
    matches = data.load_results(path)
    assert len(matches) > 500
    return matches


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------


def test_goal_difference_multiplier():
    assert features.goal_difference_multiplier(0) == 1.0
    assert features.goal_difference_multiplier(1) == 1.0
    assert features.goal_difference_multiplier(2) == 1.5
    assert features.goal_difference_multiplier(3) == pytest.approx(14 / 8)
    assert features.goal_difference_multiplier(6) == pytest.approx(17 / 8)


def test_elo_records_pre_rating_before_updating():
    matches = make_matches([
        ("2015-01-01", "A", "B", 3, 0),
        ("2015-02-01", "A", "B", 1, 0),
    ])
    elo = features.add_elo_ratings(matches)

    # First match: both start at the baseline, so the "previous" rating is 1500.
    assert elo.loc[0, "elo_home_pre"] == pytest.approx(features.START_RATING)
    assert elo.loc[0, "elo_away_pre"] == pytest.approx(features.START_RATING)
    # The winner's next pre-match rating equals its post-match rating from before.
    assert elo.loc[1, "elo_home_pre"] == pytest.approx(elo.loc[0, "elo_home_post"])
    assert elo.loc[1, "elo_home_pre"] > features.START_RATING


def test_elo_updates_are_zero_sum_per_match():
    matches = make_matches([
        ("2015-01-01", "A", "B", 2, 1),
        ("2015-02-01", "B", "A", 0, 0),
        ("2015-03-01", "A", "C", 0, 4),
    ])
    elo = features.add_elo_ratings(matches)
    home_delta = elo["elo_home_post"] - elo["elo_home_pre"]
    away_delta = elo["elo_away_post"] - elo["elo_away_pre"]
    assert np.allclose(home_delta + away_delta, 0.0)
    # A draw gives the home side half the expected... so with equal ratings the
    # home team gains points only if home advantage was priced in and it failed.
    assert elo.loc[1, "elo_home_post"] < elo.loc[1, "elo_home_pre"]


def test_home_advantage_absent_at_neutral_venue():
    base = [("2015-01-01", "A", "B", 1, 0)]
    home = features.add_elo_ratings(make_matches(base))
    neutral_matches = make_matches(base)
    neutral_matches["neutral"] = True
    neutral = features.add_elo_ratings(neutral_matches)

    home_gain = home.loc[0, "elo_home_post"] - home.loc[0, "elo_home_pre"]
    neutral_gain = neutral.loc[0, "elo_home_post"] - neutral.loc[0, "elo_home_pre"]
    # Beating an equal side at a neutral venue is more informative, so it moves
    # the rating further than the same win with home advantage priced in.
    assert neutral_gain > home_gain > 0


def test_tournament_importance_scales_elo_movement():
    friendly = features.add_elo_ratings(make_matches([("2015-01-01", "A", "B", 1, 0)]))
    wc = make_matches([("2015-01-01", "A", "B", 1, 0)])
    wc["tournament"] = "FIFA World Cup"
    wc["tournament_category"] = "world_cup"
    world_cup = features.add_elo_ratings(wc)

    friendly_gain = friendly.loc[0, "elo_home_post"] - friendly.loc[0, "elo_home_pre"]
    wc_gain = world_cup.loc[0, "elo_home_post"] - world_cup.loc[0, "elo_home_pre"]
    assert wc_gain > friendly_gain


# --------------------------------------------------------------------------
# Rolling form
# --------------------------------------------------------------------------


def test_rolling_form_excludes_the_current_match():
    """A team scoring 2, then 4, then 1 must show 3.0 in the 2-match window."""
    matches = make_matches([
        ("2015-01-01", "A", "B", 2, 0),
        ("2015-02-01", "A", "B", 4, 1),
        ("2015-03-01", "A", "B", 1, 0),
    ])
    form = features.add_rolling_form(data.to_long_format(matches), windows=(2,))
    a_rows = form[form["team"] == "A"].sort_values("date").reset_index(drop=True)

    assert a_rows.loc[0, "history_matches"] == 0
    assert a_rows.loc[0, "form_gf_2"] != a_rows.loc[0, "form_gf_2"]  # NaN, no history
    assert a_rows.loc[1, "form_gf_2"] == pytest.approx(2.0)          # only the first match
    assert a_rows.loc[2, "form_gf_2"] == pytest.approx(3.0)          # (2+4)/2, not (2+4+1)/3
    assert a_rows.loc[2, "form_ga_2"] == pytest.approx(0.5)
    assert a_rows.loc[2, "form_points_2"] == pytest.approx(3.0)


def test_rolling_form_window_never_includes_future_matches():
    matches = make_matches([
        ("2015-01-01", "A", "B", 1, 0),
        ("2015-02-01", "A", "C", 1, 0),
        ("2015-03-01", "A", "D", 9, 9),
    ])
    form = features.add_rolling_form(data.to_long_format(matches),
                                     windows=(5,)).sort_values(["team", "date"])
    a_mid = form[(form["team"] == "A") & (form["date"] == "2015-02-01")].iloc[0]
    assert a_mid["form_gf_5"] == pytest.approx(1.0), "the 9-goal match is in the future"


# --------------------------------------------------------------------------
# Head to head
# --------------------------------------------------------------------------


def test_head_to_head_excludes_the_current_meeting():
    matches = make_matches([
        ("2015-01-01", "A", "B", 3, 0),
        ("2015-02-01", "B", "A", 1, 1),   # reversed fixture, still counts as h2h
    ])
    h2h = features.add_head_to_head(matches)

    assert h2h.loc[0, "h2h_played"] == 0
    assert h2h.loc[0, "h2h_gd_home"] == 0.0

    # Second meeting, listed B at home: history is one A win by 3.
    assert h2h.loc[1, "h2h_played"] == 1
    assert h2h.loc[1, "h2h_gd_home"] == pytest.approx(-3.0)   # from B's perspective
    assert h2h.loc[1, "h2h_win_home"] == pytest.approx(0.0)
    assert h2h.loc[1, "h2h_draw"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Strength
# --------------------------------------------------------------------------


def test_strength_features_are_bounded_and_positive(synthetic_matches):
    feats = features.compute_features(synthetic_matches)
    assert feats["attack"].between(0.2, 5.0).all()
    assert feats["defence"].between(0.2, 5.0).all()
    assert feats["attack"].notna().all()
    assert feats["defence"].notna().all()


def test_stronger_side_has_higher_attack_ratio():
    """A team that keeps scoring should show attack > 1 and their opponents < 1."""
    matches = make_matches([
        ("2015-01-01", "A", "B", 3, 0),
        ("2015-02-01", "A", "B", 3, 0),
        ("2015-03-01", "A", "B", 3, 0),
        ("2015-04-01", "A", "B", 3, 0),
        ("2015-05-01", "A", "B", 3, 0),
        ("2015-06-01", "A", "B", 3, 0),
    ])
    feats = features.compute_features(matches)
    last = feats[feats["date"] == "2015-06-01"].set_index("team")

    # The opponent adjustment is damped, not full strength, so the side that
    # keeps winning 3-0 still shows attack > 1 rather than being normalised
    # back to exactly 1.0 by its own opponent's conceding rate.
    assert last.loc["A", "attack"] > 1.0
    assert last.loc["A", "attack"] > last.loc["B", "attack"]
    assert last.loc["B", "defence"] > last.loc["A", "defence"]
    assert last.loc["A", "elo_diff"] > 0


# --------------------------------------------------------------------------
# The strong tests: truncation invariance
# --------------------------------------------------------------------------


def test_features_are_unchanged_when_the_future_is_removed(synthetic_matches):
    full = features.compute_features(synthetic_matches)

    for cutoff in ("1996-01-01", "2005-06-01", "2014-01-01"):
        cutoff_ts = pd.Timestamp(cutoff)
        past = synthetic_matches[synthetic_matches["date"] <= cutoff_ts]
        if len(past) < 50:
            continue

        truncated = features.compute_features(past)
        expected = (full[full["date"] <= cutoff_ts]
                    .sort_values(["match_id", "is_home"]).reset_index(drop=True))
        actual = (truncated.sort_values(["match_id", "is_home"]).reset_index(drop=True))

        assert len(expected) == len(actual)
        pd.testing.assert_frame_equal(
            expected[COMPARE_COLUMNS], actual[COMPARE_COLUMNS],
            check_dtype=False, atol=1e-9, obj=f"features before {cutoff}",
        )


def test_features_are_unchanged_when_future_results_are_randomised(synthetic_matches):
    """Stronger than truncation: it catches whole-dataset aggregates."""
    cutoff = pd.Timestamp("2013-01-01")
    full = features.compute_features(synthetic_matches)

    rng = np.random.default_rng(0)
    mutated = synthetic_matches.copy()
    future = mutated["date"] > cutoff
    mutated.loc[future, "home_score"] = rng.integers(0, 7, int(future.sum()))
    mutated.loc[future, "away_score"] = rng.integers(0, 7, int(future.sum()))

    mutated_feats = features.compute_features(mutated)

    expected = (full[full["date"] <= cutoff]
                .sort_values(["match_id", "is_home"]).reset_index(drop=True))
    actual = (mutated_feats[mutated_feats["date"] <= cutoff]
              .sort_values(["match_id", "is_home"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(
        expected[COMPARE_COLUMNS], actual[COMPARE_COLUMNS],
        check_dtype=False, atol=1e-9,
    )

    # And the mutation really did happen, so the test is not vacuous.
    assert (mutated.loc[future, "home_score"].values
            != synthetic_matches.loc[future, "home_score"].values).any()


# --------------------------------------------------------------------------
# Assembly sanity
# --------------------------------------------------------------------------


def test_feature_table_shape_and_no_nulls(synthetic_matches):
    feats = features.compute_features(synthetic_matches)

    assert len(feats) == 2 * len(synthetic_matches)
    assert feats.groupby("match_id").size().eq(2).all()
    assert feats[features.FEATURE_COLUMNS].notna().all().all()
    assert np.allclose(feats["elo_diff"], feats["elo_team"] - feats["elo_opponent"])
    # elo_diff must point the right way: the stronger side in a match has the
    # positive difference in exactly one of its two rows.
    per_match = feats.groupby("match_id")["elo_diff"].sum()
    assert np.allclose(per_match, 0.0)
