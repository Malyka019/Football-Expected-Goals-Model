"""Tests for the expected-goals ceiling (`scorelines.cap_rate`).

The ceiling exists because a log-linear rate has no upper bound: Brazil vs San
Marino extrapolated to 16 expected goals. These tests pin the properties that
make it safe to apply to every prediction:

* ordinary rates pass through untouched (no calibration damage),
* the curve never falls and never exceeds the ceiling,
* it joins the identity smoothly at the knee,
* and an absurd rate lands somewhere a football fan would believe.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import scorelines


def test_rates_below_the_knee_are_untouched():
    rates = np.array([0.05, 0.4, 1.0, 2.5, scorelines.RATE_KNEE])
    np.testing.assert_allclose(scorelines.cap_rate(rates), rates, rtol=0, atol=0)


def test_cap_is_monotone_and_bounded():
    rates = np.linspace(0.0, 50.0, 500)
    capped = scorelines.cap_rate(rates)

    assert np.all(np.diff(capped) >= -1e-12), "capping must not reorder predictions"
    assert capped.max() < scorelines.RATE_CEILING
    assert capped.max() > scorelines.RATE_CEILING - 0.5, "the ceiling should be reached"


def test_cap_never_increases_a_rate():
    rates = np.linspace(0.0, 30.0, 300)
    assert np.all(scorelines.cap_rate(rates) <= rates + 1e-12)


def test_cap_is_continuous_at_the_knee():
    knee = scorelines.RATE_KNEE
    below = float(scorelines.cap_rate(knee - 1e-6))
    above = float(scorelines.cap_rate(knee + 1e-6))
    assert abs(above - below) < 1e-5


def test_extreme_rate_becomes_believable():
    # The Brazil vs San Marino extrapolation before the cap.
    assert float(scorelines.cap_rate(15.93)) < 6.5


def test_ceiling_must_exceed_knee():
    with pytest.raises(ValueError):
        scorelines.cap_rate(1.0, knee=5.0, ceiling=5.0)
