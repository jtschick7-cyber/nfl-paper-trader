"""R4: the shared decision rule reproduces the loggers' former inline arithmetic exactly."""

import itertools

import pytest

from src.decision import PASS, PICK, shadow_decision
from src.key_numbers import calculate_spread_probabilities
from src.market_blend import blend_margin
from src.model_v2_core import american_to_implied_prob_scalar, calibrate_probability
from src.odds_utils import american_to_implied_prob

V3_SIGMA = 12.814507752638638
V3_BETA = 0.03693757162735288
V3_SLOPE = 0.36136814753333507
V4_SIGMA = 12.81431565188366
V4_BETA = 0.042324548578395654


# --- the inline code paths as they stood before R4 (copied from the loggers) --------

def _inline_v2(mu, home_line, home_odds, away_odds, sigma, calibrator):
    spread_res = calculate_spread_probabilities(mu, sigma, home_line)
    p_home_discrete = float(spread_res["cond_home_win"])
    p_home_cal = float(calibrate_probability(calibrator, p_home_discrete))
    p_away_cal = 1.0 - p_home_cal
    edge_home = p_home_cal - american_to_implied_prob_scalar(home_odds)
    edge_away = p_away_cal - american_to_implied_prob_scalar(away_odds)
    if edge_home >= edge_away:
        return mu, p_home_discrete, p_home_cal, edge_home, edge_away, "HOME", home_line, home_odds, edge_home
    return mu, p_home_discrete, p_home_cal, edge_home, edge_away, "AWAY", -home_line, away_odds, edge_away


def _inline_v3(mu, home_line, home_odds, away_odds, sigma, market_beta, calibrator):
    if market_beta is not None:
        mu = blend_margin(mu, home_line, market_beta)
    spread_res = calculate_spread_probabilities(mu, sigma, home_line)
    p_home_discrete = float(spread_res["cond_home_win"])
    if isinstance(calibrator, tuple):
        p_home_cal = float(calibrate_probability(calibrator, p_home_discrete))
    elif hasattr(calibrator, "predict_proba"):
        try:
            p_home_cal = float(calibrator.predict_proba([[p_home_discrete]])[0][1])
        except Exception:
            p_home_cal = p_home_discrete
    else:
        p_home_cal = p_home_discrete
    p_away_cal = 1.0 - p_home_cal
    edge_home = p_home_cal - american_to_implied_prob(home_odds)
    edge_away = p_away_cal - american_to_implied_prob(away_odds)
    if edge_home >= edge_away:
        return mu, p_home_discrete, p_home_cal, edge_home, edge_away, "HOME", home_line, home_odds, edge_home
    return mu, p_home_discrete, p_home_cal, edge_home, edge_away, "AWAY", -home_line, away_odds, edge_away


def _inline_v4(mu, home_line, home_odds, away_odds, sigma_base, market_beta, calibrator, total_line):
    if market_beta is not None:
        mu = blend_margin(mu, home_line, market_beta)
    spread_res = calculate_spread_probabilities(mu, sigma_base, home_line, total_line=total_line)
    p_home_discrete = float(spread_res["cond_home_win"])
    p_home_cal = float(calibrate_probability(calibrator, p_home_discrete))
    p_away_cal = 1.0 - p_home_cal
    edge_h = p_home_cal - american_to_implied_prob(home_odds)
    edge_a = p_away_cal - american_to_implied_prob(away_odds)
    if edge_h >= edge_a:
        return mu, p_home_discrete, p_home_cal, edge_h, edge_a, "HOME", home_line, home_odds, edge_h
    return mu, p_home_discrete, p_home_cal, edge_h, edge_a, "AWAY", -home_line, away_odds, edge_a


def _as_tuple(d):
    return (d["mu_blend"], d["p_home_discrete"], d["p_home_cal"], d["edge_home"], d["edge_away"],
            d["best_side"], d["best_line"], d["best_odds"], d["best_edge"])


def _assert_same(a, b):
    for x, y in zip(a, b):
        if isinstance(x, str):
            assert x == y
        else:
            assert x == pytest.approx(y, abs=1e-9)


RAW_MUS = [-14.2, -3.0, 0.0, 2.7, 9.9]
LINES = [-7.0, -3.0, -2.5, 0.0, 3.5, 10.0]  # integer and half-point
PRICES = [(-110, -110), (110, -130), (-105, -115), (125, -145)]


class _Proba:
    def predict_proba(self, X):
        return [[0.4, 0.6]]


@pytest.mark.parametrize("raw_mu,home_line,prices", itertools.product(RAW_MUS, LINES, PRICES))
def test_v2_matches_inline(raw_mu, home_line, prices):
    cal = ("logistic", (0.01, 0.8))
    got = shadow_decision(raw_mu, home_line, *prices, sigma=13.4977, market_beta=None, calibrator=cal)
    _assert_same(_as_tuple(got), _inline_v2(raw_mu, home_line, *prices, 13.4977, cal))


@pytest.mark.parametrize("raw_mu,home_line,prices", itertools.product(RAW_MUS, LINES, PRICES))
@pytest.mark.parametrize("calibrator", [("logistic", (0.0, V3_SLOPE)), ("identity", None), ("no_edge", (0.0, -0.1)), _Proba(), None])
def test_v3_matches_inline(raw_mu, home_line, prices, calibrator):
    got = shadow_decision(raw_mu, home_line, *prices, sigma=V3_SIGMA, market_beta=V3_BETA, calibrator=calibrator)
    _assert_same(_as_tuple(got), _inline_v3(raw_mu, home_line, *prices, V3_SIGMA, V3_BETA, calibrator))


def test_v3_without_beta_matches_inline():
    cal = ("logistic", (0.0, V3_SLOPE))
    got = shadow_decision(4.0, -3.0, -110, -110, sigma=V3_SIGMA, market_beta=None, calibrator=cal)
    _assert_same(_as_tuple(got), _inline_v3(4.0, -3.0, -110, -110, V3_SIGMA, None, cal))


@pytest.mark.parametrize("raw_mu,home_line,prices", itertools.product(RAW_MUS, LINES, PRICES))
@pytest.mark.parametrize("total_line", [None, 37.5, 44.0, 52.5])
def test_v4_matches_inline(raw_mu, home_line, prices, total_line):
    cal = ("logistic", (0.0, 0.42))
    got = shadow_decision(raw_mu, home_line, *prices, sigma=V4_SIGMA, market_beta=V4_BETA,
                          calibrator=cal, total_line=total_line)
    _assert_same(_as_tuple(got), _inline_v4(raw_mu, home_line, *prices, V4_SIGMA, V4_BETA, cal, total_line))


def test_decision_and_raw_lean():
    d = shadow_decision(5.0, -3.0, 110, -130, sigma=V3_SIGMA, market_beta=V3_BETA,
                        calibrator=("logistic", (0.0, V3_SLOPE)))
    assert d["raw_lean_pts"] == pytest.approx(2.0)
    assert d["decision"] == (PICK if d["best_edge"] >= 0.02 else PASS)
    assert shadow_decision(5.0, -3.0, 110, -130, sigma=V3_SIGMA, market_beta=V3_BETA,
                           calibrator=("logistic", (0.0, V3_SLOPE)), min_edge=1.0)["decision"] == PASS


# --- locks on today's rule ---------------------------------------------------------

V3_TODAY = dict(sigma=12.8145, market_beta=0.0369, calibrator=("logistic", (0.0, 0.3614)))


@pytest.mark.parametrize("disagreement", [-60, -30, -10, -2, 0, 2, 10, 30, 60])
def test_v3_today_cannot_bet_at_standard_price(disagreement):
    # R5 is expected to change this: under today's beta and calibrator slope no model
    # input up to 60 points of disagreement clears 2% at -110/-110 on a dead number.
    home_line = -5.5
    d = shadow_decision(-home_line + disagreement, home_line, -110, -110, **V3_TODAY)
    assert d["decision"] == PASS


@pytest.mark.xfail(strict=True, reason="R5: zero disagreement must not produce a pick from price alone")
def test_r5_zero_disagreement_long_price_is_a_pass():
    home_line = -3.0
    d = shadow_decision(-home_line, home_line, 110, -130, **V3_TODAY)
    assert d["decision"] == PASS
