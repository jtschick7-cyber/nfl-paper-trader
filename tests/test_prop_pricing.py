"""P4: player_receptions pricing (src/props/prop_pricing.py)."""

import math

import pytest
from scipy.stats import poisson

from src.props import prop_pricing as pp
from src.total_pricing import devig_two_way


# ---- 1. normalize_name ------------------------------------------------------------------------

def test_normalize_name_apostrophes_and_suffixes():
    assert pp.normalize_name("Ja'Marr Chase") == pp.normalize_name("JaMarr Chase") == "jamarr chase"
    assert pp.normalize_name("Ja’Marr Chase") == "jamarr chase"
    assert pp.normalize_name("Michael Pittman Jr.") == pp.normalize_name("Michael Pittman") == "michael pittman"
    assert pp.normalize_name("  A.J.   Brown ") == "aj brown"
    for suffix in ("Sr", "II", "III", "IV", "V", "jr"):
        assert pp.normalize_name(f"Odell Beckham {suffix}") == "odell beckham"
    assert pp.normalize_name("Jr Smith") == "jr smith"  # only a trailing suffix is dropped


# ---- 2. pricing -------------------------------------------------------------------------------

@pytest.mark.parametrize("line", [2.5, 4.5, 6.5])
@pytest.mark.parametrize("lam", [1.3, 3.9, 6.2])
def test_poisson_round_trip(line, lam):
    p_over, p_under, _ = pp.prop_probabilities(lam, line)
    q_over = p_over / (p_over + p_under)
    assert abs(pp.implied_lambda(line, q_over) - lam) < 1e-6


def test_negative_binomial_round_trip():
    p_over, p_under, _ = pp.prop_probabilities(4.2, 4.5, r=pp.NB_SENSITIVITY_R)
    assert abs(pp.implied_lambda(4.5, p_over / (p_over + p_under), r=pp.NB_SENSITIVITY_R) - 4.2) < 1e-6
    # NB(r=8) is over-dispersed: fatter tails than the Poisson at the same mean
    assert pp.prop_probabilities(4.0, 8.5, r=8.0)[0] > pp.prop_probabilities(4.0, 8.5)[0]


def test_zero_disagreement_at_minus_110_is_minus_4_5_pct():
    q, _ = devig_two_way(-110, -110)
    lam = pp.implied_lambda(4.5, q)
    for side in ("OVER", "UNDER"):
        _, push, ev = pp.price_prop_offer(lam, side, 4.5, -110)
        assert push == 0.0
        assert ev == pytest.approx(-0.04545, abs=2e-4)


def test_integer_line_push_is_poisson_pmf():
    p_over, p_under, p_push = pp.prop_probabilities(4.3, 4)
    assert p_push == pytest.approx(poisson.pmf(4, 4.3), abs=1e-12)
    assert p_over == pytest.approx(poisson.sf(4, 4.3), abs=1e-12)
    assert p_under == pytest.approx(poisson.cdf(3, 4.3), abs=1e-12)
    assert p_over + p_under + p_push == pytest.approx(1.0, abs=1e-12)
    assert pp.prop_probabilities(4.3, 4.5)[2] == 0.0


def test_implied_lambda_is_bisection_on_the_frozen_bracket():
    assert pp.implied_lambda(4.5, 0.999999999) == pytest.approx(pp.LAMBDA_HI, abs=1e-5)
    assert pp.implied_lambda(4.5, 1e-12) == pytest.approx(pp.LAMBDA_LO, abs=1e-5)


# ---- 3. cross-line ----------------------------------------------------------------------------

def test_cross_line_prices_follow_from_one_lambda():
    q, _ = devig_two_way(-130, 100)
    lam = pp.implied_lambda(4.5, q)
    o45, u45, _ = pp.prop_probabilities(lam, 4.5)
    assert o45 / (o45 + u45) == pytest.approx(q, abs=1e-6)

    p_win, p_push, ev = pp.price_prop_offer(lam, "OVER", 3.5, -200)
    assert p_push == 0.0
    assert p_win == pytest.approx(o45 + poisson.pmf(4, lam), abs=1e-9)  # P(R>3) = P(R>4) + P(R=4)
    assert ev == pytest.approx(p_win * 0.5 - (1 - p_win), abs=1e-9)

    p_win, _, ev = pp.price_prop_offer(lam, "UNDER", 5.5, -150)
    assert p_win == pytest.approx(poisson.cdf(5, lam), abs=1e-9)
    assert ev == pytest.approx(p_win * (100 / 150) - (1 - p_win), abs=1e-9)


def test_integer_line_offer_ev_returns_stake_on_push():
    lam = 4.0
    p_win, p_push, ev = pp.price_prop_offer(lam, "OVER", 4, 100)
    p_loss = 1 - p_win - p_push
    assert ev == pytest.approx(p_win * 1.0 - p_loss, abs=1e-12)


# ---- prop_quotes ------------------------------------------------------------------------------

def _resp(books):
    return {"id": "e1", "commence_time": "2026-11-26T01:00:00Z", "home_team": "Los Angeles Rams",
            "away_team": "Green Bay Packers", "bookmakers": books}


def test_prop_quotes_rows_per_book_player_line():
    resp = _resp([
        {"key": "pinnacle", "markets": [{"key": "player_receptions", "last_update": "2026-11-24T21:13:06Z",
                                         "outcomes": [
            {"name": "Over", "description": "Ja'Marr Chase", "price": -135, "point": 5.5},
            {"name": "Under", "description": "Ja'Marr Chase", "price": 106, "point": 5.5},
            {"name": "Over", "description": "Ja'Marr Chase", "price": 150, "point": 6.5},
            {"name": "Under", "description": "Ja'Marr Chase", "price": -190, "point": 6.5}]}]},
        {"key": "betrivers", "markets": [{"key": "player_receptions", "last_update": "2026-11-24T20:00:00Z",
                                          "outcomes": [
            {"name": "Over", "description": "Tee Higgins", "price": -120, "point": 4.5}]}]},
        {"key": "fanduel", "markets": [{"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "Tee Higgins", "price": -110, "point": 55.5}]}]},
    ])
    rows = pp.prop_quotes(resp)
    assert len(rows) == 3
    pin = sorted((r for r in rows if r["book"] == "pinnacle"), key=lambda r: r["line"])
    assert [(r["player"], r["line"], r["over_odds"], r["under_odds"]) for r in pin] == [
        ("jamarr chase", 5.5, -135, 106), ("jamarr chase", 6.5, 150, -190)]
    assert pin[0]["player_name"] == "Ja'Marr Chase"
    assert pin[0]["last_update"] == "2026-11-24T21:13:06Z"
    br = [r for r in rows if r["book"] == "betrivers"][0]
    assert br["over_odds"] == -120 and br["under_odds"] is None
