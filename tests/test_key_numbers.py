"""
Unit tests for Key Numbers Discrete Margin Engine and Totals-Conditioned Variance.
"""

import math
import pytest
from src.key_numbers import (
    MARGIN_FREQUENCIES,
    calculate_spread_probabilities,
    calculate_totals_conditioned_sigma,
    discrete_margin_probabilities,
    is_integer_spread,
    payout_multiplier,
)


def test_is_integer_spread():
    assert is_integer_spread(-3.0) is True
    assert is_integer_spread(7.0) is True
    assert is_integer_spread(0.0) is True
    assert is_integer_spread(-3.5) is False
    assert is_integer_spread(2.5) is False


def test_totals_conditioned_sigma():
    base_sigma = 13.43
    # Standard total ~ 44.5 should return base sigma
    sigma_standard = calculate_totals_conditioned_sigma(base_sigma, total_line=44.5)
    assert pytest.approx(sigma_standard, 1e-4) == base_sigma

    # Low total (e.g. 36.0 defensive battle / windy game) -> compressed sigma
    sigma_low = calculate_totals_conditioned_sigma(base_sigma, total_line=36.0)
    assert sigma_low < base_sigma
    assert pytest.approx(sigma_low, 1e-3) == base_sigma * math.sqrt(36.0 / 44.5)

    # High total (e.g. 52.0 shootout) -> expanded sigma
    sigma_high = calculate_totals_conditioned_sigma(base_sigma, total_line=52.0)
    assert sigma_high > base_sigma
    assert pytest.approx(sigma_high, 1e-3) == base_sigma * math.sqrt(52.0 / 44.5)

    # None total returns base sigma
    assert calculate_totals_conditioned_sigma(base_sigma, None) == base_sigma


def test_discrete_margin_probabilities_sum_to_one():
    mu = 2.5
    sigma = 13.4
    probs = discrete_margin_probabilities(mu, sigma)

    # Total probability over discrete margins should sum to 1.0
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, 1e-3) == 1.0

    # Margins 3 and 7 should have significant weights reflecting NFL scoring structure
    assert probs[3] > probs[4]
    assert probs[7] > probs[8]
    assert probs[6] > probs[5]


def test_discrete_margin_probabilities_mean_preservation():
    # Assert discrete expectation equals mu to within 1e-4 points without mean-shift distortion
    for mu in [3.5, -6.5, 0.0, 7.2, -10.5]:
        dist = discrete_margin_probabilities(mu, sigma=13.4)
        expected_margin = sum(m * p for m, p in dist.items())
        assert abs(expected_margin - mu) < 1e-4


def test_calculate_spread_probabilities():
    mu = 3.5
    sigma = 13.4

    # Half point spread (-3.5) -> push probability must be 0.0
    probs_half = calculate_spread_probabilities(mu, sigma, home_spread=-3.5)
    assert probs_half["p_push"] == 0.0
    assert pytest.approx(probs_half["p_home_win"] + probs_half["p_away_win"], 1e-4) == 1.0

    # Integer spread (-3.0) -> push probability must be > 0.0 (typically 5-10% around key number 3)
    probs_int = calculate_spread_probabilities(mu, sigma, home_spread=-3.0)
    assert probs_int["p_push"] > 0.04
    assert probs_int["p_push"] < 0.15
    assert pytest.approx(probs_int["p_home_win"] + probs_int["p_away_win"] + probs_int["p_push"], 1e-4) == 1.0
    assert pytest.approx(probs_int["cond_home_win"] + probs_int["cond_away_win"], 1e-4) == 1.0


def test_payout_multiplier():
    assert pytest.approx(payout_multiplier(-110.0), 1e-4) == 100.0 / 110.0
    assert pytest.approx(payout_multiplier(100.0), 1e-4) == 1.0
    assert pytest.approx(payout_multiplier(120.0), 1e-4) == 1.20
    assert payout_multiplier(0.0) == 1.0
