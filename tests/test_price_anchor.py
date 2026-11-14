"""R5: the price-consistent market anchor (m*, its fit, the diagnostic calibrator)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.key_numbers import calculate_spread_probabilities, market_implied_mean
from src.price_anchor import (anchor_mean, calibrator_slope_se, devig_home, fit_price_anchor)

SIGMA = 12.81


def test_devig_home_two_way_and_missing():
    q, assumed = devig_home(-110, -110)
    assert q == pytest.approx(0.5) and not assumed
    q, assumed = devig_home(110, -130)
    implied_h, implied_a = 100 / 210, 130 / 230
    assert q == pytest.approx(implied_h / (implied_h + implied_a)) and not assumed
    for h, a in [(None, -110), (-110, float("nan")), (0, -110), ("", -110), (-110, None)]:
        assert devig_home(h, a) == (0.5, True)


@pytest.mark.parametrize("home_line,q,total", [
    (-3.0, 0.47, None), (-2.5, 0.5, None), (7.0, 0.53, None), (-10.0, 0.45, 41.5), (0.0, 0.5, 52.5), (3.5, 0.54, 37.0),
])
def test_anchor_mean_matches_r2_market_implied_mean(home_line, q, total):
    m = anchor_mean(home_line, q, SIGMA, total)
    assert m == pytest.approx(market_implied_mean(home_line, q, SIGMA, total), abs=1e-6)
    p = calculate_spread_probabilities(m, SIGMA, home_line, total_line=total)["cond_home_win"]
    assert p == pytest.approx(q, abs=1e-6)


def test_anchor_mean_sits_off_the_line_where_the_engine_says():
    # -2.5 at even prices: the engine's mass on 3 pulls m* above 2.5 (spec: 1.465 off is
    # possible). The anchor is the price-consistent mean, not -home_line.
    m = anchor_mean(-2.5, 0.5, SIGMA)
    assert abs(m - 2.5) > 0.5


def _synthetic_oos(variant, beta, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    home_line = rng.choice([-7.0, -6.5, -3.5, -3.0, -2.5, -1.0, 1.5, 3.0, 4.0, 7.0], n).astype(float)
    home_odds = rng.choice([-120.0, -110.0, -105.0, 100.0], n)
    away_odds = np.where(home_odds == -120.0, 100.0, np.where(home_odds == -105.0, -115.0,
                         np.where(home_odds == 100.0, -120.0, -110.0)))
    df = pd.DataFrame({"home_line": home_line, "home_spread_odds": home_odds, "away_spread_odds": away_odds})
    q = np.array([devig_home(h, a)[0] for h, a in zip(home_odds, away_odds)])
    m_star = np.array([anchor_mean(hl, qq, 13.0) for hl, qq in zip(home_line, q)])
    d = rng.normal(0.0, 3.0, n)
    ref = m_star if variant == "A" else -home_line
    df["oos_mu"] = ref + d
    df["actual_home_margin"] = m_star + beta * d + rng.normal(0.0, 13.0, n)
    return df


@pytest.mark.parametrize("variant", ["A", "B"])
def test_fit_price_anchor_recovers_beta(variant):
    df = _synthetic_oos(variant, beta=0.3)
    fit = fit_price_anchor(df, variant=variant, use_total=False)
    assert fit["n_beta"] == len(df)
    assert fit["beta"] == pytest.approx(0.3, abs=3 * fit["beta_se"])
    assert fit["sigma"] == pytest.approx(13.0, abs=0.5)
    assert fit["n_price_assumed"] == 0
    assert len(fit["m_star"]) == len(df) and len(fit["mu"]) == len(df)
    # diagnostic calibrator is recorded, never applied
    assert fit["calibrator"] == ("identity", None)
    assert math.isfinite(fit["calibrator_diagnostic_slope"]) and fit["calibrator_diagnostic_se"] > 0


def test_fit_price_anchor_flags_missing_prices():
    df = _synthetic_oos("A", beta=0.0, n=600)
    df.loc[:9, "home_spread_odds"] = np.nan
    fit = fit_price_anchor(df, variant="A", use_total=False)
    assert fit["n_price_assumed"] == 10


def test_calibrator_slope_se_matches_logistic_regression_scale():
    # A well-specified logistic slope of 1 on 20k symmetric rows: SE near 1/sqrt(info).
    rng = np.random.default_rng(3)
    n = 20000
    p = rng.uniform(0.3, 0.7, n)
    win = rng.uniform(size=n) < p
    ev = pd.DataFrame({"p_raw": np.r_[p, 1 - p], "actual_win": np.r_[win, ~win],
                       "push": False, "weight": 0.5})
    se = calibrator_slope_se(ev, 0.0, 1.0)
    x = np.log(p / (1 - p))
    expected = 1.0 / math.sqrt(float(np.sum(p * (1 - p) * x * x)))
    assert se == pytest.approx(expected, rel=0.05)
