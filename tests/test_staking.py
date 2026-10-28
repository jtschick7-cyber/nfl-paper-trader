"""
Unit tests for Push-Aware Kelly Staking and Slate Bankroll Management.
"""

import pytest
from src.staking import (
    DEFAULT_MAX_SLATE_UNITS,
    american_to_decimal_odds,
    cap_slate_stakes,
    compute_push_aware_stake,
    full_kelly_fraction,
)


def test_american_to_decimal_odds():
    # Even money
    assert pytest.approx(american_to_decimal_odds(100.0), 1e-4) == 2.0
    assert pytest.approx(american_to_decimal_odds(-100.0), 1e-4) == 2.0

    # Underdog +150 -> $2.50 total return
    assert pytest.approx(american_to_decimal_odds(150.0), 1e-4) == 2.50

    # Favorite -110 -> 1.9091
    assert pytest.approx(american_to_decimal_odds(-110.0), 1e-4) == 1.0 + 100.0 / 110.0

    # Deep favorite -200 -> 1.50
    assert pytest.approx(american_to_decimal_odds(-200.0), 1e-4) == 1.50


def test_full_kelly_fraction_closed_form():
    # Standard coin flip at even money (w=0.5, p=0.0, l=0.5, b=1.0) -> 0.0 edge
    assert full_kelly_fraction(0.5, 0.0, 0.5, decimal_odds=2.0) == 0.0

    # 60% win at even money (w=0.6, p=0.0, l=0.4, b=1.0) -> f = (0.6*1 - 0.4)/(1*1) = 0.20
    assert pytest.approx(full_kelly_fraction(0.6, 0.0, 0.4, decimal_odds=2.0), 1e-6) == 0.20

    # Negative edge -> 0.0 stake
    assert full_kelly_fraction(0.4, 0.0, 0.6, decimal_odds=2.0) == 0.0

    # Win with 10% push probability: w=0.54, p=0.10, l=0.36, b=1.0
    # Formula: (0.54*1 - 0.36) / (1 * (0.54 + 0.36)) = 0.18 / 0.90 = 0.20
    assert pytest.approx(full_kelly_fraction(0.54, 0.10, 0.36, decimal_odds=2.0), 1e-6) == 0.20


def test_push_aware_stake_sizing():
    # 55% conditional win, 5% push, -110 odds
    p_push = 0.05
    w = 0.55 * (1.0 - p_push)
    stake_info = compute_push_aware_stake(w, p_push, -110.0, kelly_multiplier=0.25)

    assert stake_info["recommended_units"] > 0.0
    assert stake_info["recommended_units"] <= 3.0
    assert stake_info["win_prob"] == pytest.approx(w)

    # Negative EV bet should recommend 0.0 units
    neg_ev = compute_push_aware_stake(0.45, 0.05, -110.0)
    assert neg_ev["recommended_units"] == 0.0


def test_cap_slate_stakes():
    bets = [
        {"team": "KC", "recommended_units": 4.0},
        {"team": "BUF", "recommended_units": 5.0},
        {"team": "DET", "recommended_units": 6.0},
    ]
    # Total raw = 15.0u, cap = 10.0u
    capped = cap_slate_stakes(bets, max_slate_exposure=10.0)
    total_capped = sum(b["recommended_units"] for b in capped)
    assert pytest.approx(total_capped, 1e-4) == 10.0

    # Proportions preserved (within 2 decimal place unit rounding)
    assert pytest.approx(capped[0]["recommended_units"] / capped[2]["recommended_units"], abs=0.01) == 4.0 / 6.0

    # Uncapped when below limit
    small_bets = [{"team": "KC", "recommended_units": 2.0}]
    uncapped = cap_slate_stakes(small_bets, max_slate_exposure=10.0)
    assert uncapped[0]["recommended_units"] == 2.0


def test_cap_slate_stakes_per_week_grouping():
    # Bets across multiple weeks
    raw_bets = [
        {"week": 7, "team": "KC", "recommended_units": 6.0},
        {"week": 7, "team": "BUF", "recommended_units": 6.0},
        {"week": 8, "team": "DET", "recommended_units": 7.0},
        {"week": 8, "team": "GB", "recommended_units": 5.0},
    ]
    # If capped per-week independently:
    from collections import defaultdict
    by_week = defaultdict(list)
    for b in raw_bets:
        by_week[b["week"]].append(dict(b))

    capped_all = []
    for w, w_bets in by_week.items():
        capped_all.extend(cap_slate_stakes(w_bets, max_slate_exposure=10.0))

    # Week 7 total should be 10.0u, Week 8 total should be 10.0u
    w7_total = sum(b["recommended_units"] for b in capped_all if b["week"] == 7)
    w8_total = sum(b["recommended_units"] for b in capped_all if b["week"] == 8)
    assert pytest.approx(w7_total, 1e-4) == 10.0
    assert pytest.approx(w8_total, 1e-4) == 10.0
    # Combined across both weeks should be 20.0u (NOT compressed to 10u)
    assert pytest.approx(sum(b["recommended_units"] for b in capped_all), 1e-4) == 20.0
