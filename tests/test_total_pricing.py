"""P2: totals pricing (src/total_pricing.py)."""

import pytest

from src.total_pricing import (
    TOTAL_SIGMA, devig_two_way, price_total_offer, total_implied_mean, total_probabilities, total_quotes,
)


@pytest.mark.parametrize("line", [41.0, 43.5, 47.0])
@pytest.mark.parametrize("mu", [38.2, 44.0, 51.7])
def test_implied_mean_round_trip(line, mu):
    p_over, p_under, _ = total_probabilities(mu, line)
    q_over = p_over / (p_over + p_under)
    assert total_implied_mean(line, q_over, TOTAL_SIGMA) == pytest.approx(mu, abs=1e-6)


@pytest.mark.parametrize("line", [43.5, 44.0])
def test_zero_disagreement_at_minus_110_costs_the_vig(line):
    q_over, assumed = devig_two_way(-110, -110)
    assert q_over == pytest.approx(0.5) and not assumed
    mu = total_implied_mean(line, q_over, TOTAL_SIGMA)
    for side in ("OVER", "UNDER"):
        _, _, ev = price_total_offer(mu, side, line, -110, TOTAL_SIGMA)
        assert ev == pytest.approx(-0.045, abs=0.002)


def test_push_mass_at_the_mode():
    p_over, p_under, p_push = total_probabilities(44.0, 44.0)
    assert 0.029 < p_push < 0.031
    assert p_over == pytest.approx(p_under)
    assert p_over + p_under + p_push == pytest.approx(1.0)
    assert total_probabilities(44.0, 43.5)[2] == 0.0


def test_devig_assumes_minus_110_when_a_price_is_missing():
    assert devig_two_way(None, -110) == (0.5, True)
    assert devig_two_way(-110, 0) == (0.5, True)
    q, assumed = devig_two_way(-130, 110)
    assert 0.5 < q < 0.6 and not assumed


def _totals(key, over_pt, under_pt, over_px=-110, under_px=-110, last_update="2026-11-26T12:00:00Z"):
    return {"key": key, "last_update": "2026-11-26T11:00:00Z", "markets": [
        {"key": "spreads", "outcomes": []},
        {"key": "totals", "last_update": last_update, "outcomes": [
            {"name": "Over", "price": over_px, "point": over_pt},
            {"name": "Under", "price": under_px, "point": under_pt}]}]}


def test_total_quotes_skips_mismatched_points_and_reads_last_update():
    ev = {"id": "e", "bookmakers": [_totals("a", 44.5, 44.5, -105, -115), _totals("b", 44.5, 45.5),
                                    {"key": "c", "markets": [{"key": "spreads", "outcomes": []}]}]}
    q = total_quotes(ev)
    assert q == [{"book": "a", "line": 44.5, "over_odds": -105.0, "under_odds": -115.0,
                  "last_update": "2026-11-26T12:00:00Z"}]
