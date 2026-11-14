"""R4: the shared decision rule reproduces the loggers' former inline arithmetic exactly."""

import itertools
import math

import joblib
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


# --- locks on the rule -------------------------------------------------------------

# The pre-R5 v3 artifact (line anchor + logistic calibrator). Still the rule for any row
# whose artifact has no `anchor` field, and for v2.
V3_LEGACY = dict(sigma=12.8145, market_beta=0.0369, calibrator=("logistic", (0.0, 0.3614)))

ARTIFACT_PATHS = {"v3": "models/nfl_spread_model_v3.pkl", "v4": "models/nfl_spread_model_v4.pkl"}


def _artifact_kw(model):
    a = joblib.load(ARTIFACT_PATHS[model])
    return dict(sigma=float(a["sigma"]), market_beta=a["market_beta"], calibrator=a["calibrator"],
                anchor=a.get("anchor"), anchor_variant=a.get("anchor_variant"))


@pytest.mark.parametrize("disagreement", [-60, -30, -10, -2, 0, 2, 10, 30, 60])
def test_legacy_v3_rule_cannot_bet_at_standard_price(disagreement):
    # The R4 lock, kept for the pre-R5 rule: under the old beta and calibrator slope no
    # model input up to 60 points of disagreement clears 2% at -110/-110 on a dead number.
    home_line = -5.5
    d = shadow_decision(-home_line + disagreement, home_line, -110, -110, **V3_LEGACY)
    assert d["decision"] == PASS


@pytest.mark.parametrize("disagreement", [-60, -30, -10, -2, 0, 2, 10, 30, 60])
def test_r5_v3_needs_about_67_points_to_bet_at_standard_price(disagreement):
    # R5 replacement for the R4 lock: under the retrained v3 (price anchor, beta 0.0212,
    # identity calibrator) a dead number at -110/-110 still passes up to 60 points of
    # disagreement against m*, but the rule can bet: 67.5 points clears 2% on either side.
    home_line = -5.5
    kw = _artifact_kw("v3")
    m_star = shadow_decision(0.0, home_line, -110, -110, **kw)["anchor_mean"]
    assert shadow_decision(m_star + disagreement, home_line, -110, -110, **kw)["decision"] == PASS
    for d, side in ((67.5, "HOME"), (-67.5, "AWAY")):
        r = shadow_decision(m_star + d, home_line, -110, -110, **kw)
        assert r["decision"] == PICK and r["best_side"] == side


def test_r5_zero_disagreement_long_price_is_a_pass():
    # R4 locked this as xfail: the old rule picked +110 at -3 with no model input.
    home_line = -3.0
    d = shadow_decision(-home_line, home_line, 110, -130, **_artifact_kw("v3"))
    assert d["decision"] == PASS


# --- R5: price-consistent anchor -----------------------------------------------------

GRID_LINES = [x / 2 for x in range(-20, 21)]  # -10 to +10 by 0.5
GRID_PRICES = [(-130, 110), (-120, 100), (-110, -110), (-105, -115), (100, -120), (105, -125),
               (110, -130), (115, -140)]
PRICE_RULE = dict(sigma=12.81, market_beta=0.05, calibrator=("identity", None), anchor="price")


def _q_home(h, a):
    ih, ia = american_to_implied_prob(h), american_to_implied_prob(a)
    return ih / (ih + ia)


@pytest.mark.parametrize("variant", ["A", "B"])
@pytest.mark.parametrize("home_line", GRID_LINES)
def test_zero_disagreement_gives_market_probability_and_passes(home_line, variant):
    # Zero disagreement is measured against m* in variant A and against the line in B.
    for h, a in GRID_PRICES:
        q = _q_home(h, a)
        line_probe = shadow_decision(-home_line, home_line, h, a, anchor_variant=variant, **PRICE_RULE)
        assert line_probe["decision"] == PASS, (home_line, h, a)
        assert line_probe["q_home"] == pytest.approx(q, abs=1e-12)
        assert not line_probe["price_assumed"]
        zero_mu = line_probe["anchor_mean"] if variant == "A" else -home_line
        d = shadow_decision(zero_mu, home_line, h, a, anchor_variant=variant, **PRICE_RULE)
        assert d["mu_blend"] == pytest.approx(d["anchor_mean"], abs=1e-9)
        assert d["p_home_cal"] == pytest.approx(q, abs=1e-6), (home_line, h, a)
        assert d["decision"] == PASS, (home_line, h, a)


@pytest.mark.parametrize("total_line", [37.5, 44.0, 52.5])
def test_zero_disagreement_with_total_conditioned_sigma(total_line):
    for home_line in (-7.0, -3.0, -2.5, 1.5, 6.5):
        for h, a in GRID_PRICES:
            m_star = shadow_decision(0.0, home_line, h, a, total_line=total_line, anchor_variant="A",
                                     **PRICE_RULE)["anchor_mean"]
            d = shadow_decision(m_star, home_line, h, a, total_line=total_line, anchor_variant="A", **PRICE_RULE)
            assert d["p_home_cal"] == pytest.approx(_q_home(h, a), abs=1e-6)
            assert d["decision"] == PASS


def test_variant_a_beta_zero_is_exactly_the_market_probability_for_any_model_input():
    rule = dict(PRICE_RULE, market_beta=0.0)
    for raw_mu in (-20.0, 0.0, 14.0):
        d = shadow_decision(raw_mu, -3.0, 115, -140, anchor_variant="A", **rule)
        assert d["p_home_cal"] == pytest.approx(_q_home(115, -140), abs=1e-6)
        assert d["decision"] == PASS


@pytest.mark.parametrize("variant", ["A", "B"])
@pytest.mark.parametrize("home_line,disagreement,side", [(-6.5, 6.0, "HOME"), (-6.5, -6.0, "AWAY"),
                                                         (3.0, 6.0, "HOME"), (3.0, -6.0, "AWAY")])
def test_six_points_of_disagreement_at_half_beta_bets_the_models_side(home_line, disagreement, side, variant):
    # The anchored rule is not simply incapable of betting.
    rule = dict(PRICE_RULE, market_beta=0.5)
    d = shadow_decision(-home_line + disagreement, home_line, -110, -110, anchor_variant=variant, **rule)
    assert d["decision"] == PICK and d["best_side"] == side


def test_missing_price_assumes_even_market_and_flags_it():
    d = shadow_decision(-3.0, -3.0, None, -110, anchor_variant="A", **PRICE_RULE)
    assert d["price_assumed"] and d["q_home"] == 0.5
    assert d["decision"] == PASS


def test_price_anchor_refuses_a_fitted_calibrator():
    with pytest.raises(ValueError):
        shadow_decision(3.0, -3.0, -110, -110, sigma=12.81, market_beta=0.05,
                        calibrator=("logistic", (0.0, 0.36)), anchor="price")


@pytest.mark.parametrize("model", ["v3", "v4"])
def test_retrained_artifact_loads_and_is_price_anchored(model):
    a = joblib.load(ARTIFACT_PATHS[model])
    assert a["anchor"] == "price" and a["anchor_variant"] in ("A", "B")
    assert a["calibrator"] == ("identity", None)
    for key in ("market_beta", "market_beta_se", "calibrator_diagnostic_slope", "calibrator_diagnostic_se"):
        assert math.isfinite(float(a[key])), key
    assert 0.0 <= a["market_beta"] <= 1.0
    kw = _artifact_kw(model)
    total = 44.5 if model == "v4" else None
    for home_line in (-7.0, -3.0, -2.5, 0.0, 3.5):
        for h, aw in [(-110, -110), (110, -130), (115, -140), (-130, 110)]:
            line_probe = shadow_decision(-home_line, home_line, h, aw, total_line=total, **kw)
            assert line_probe["decision"] == PASS
            zero_mu = line_probe["anchor_mean"] if kw["anchor_variant"] == "A" else -home_line
            d = shadow_decision(zero_mu, home_line, h, aw, total_line=total, **kw)
            assert d["p_home_cal"] == pytest.approx(_q_home(h, aw), abs=1e-6)
            assert d["decision"] == PASS
