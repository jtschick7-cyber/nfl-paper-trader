import pandas as pd
import pytest

from src.compare_models import clv_fair_probability, clv_win_probability, closing_reference_line, compute_metrics, grade_single_bet


def test_clv_probability_values_key_numbers():
    # Taking +3.5 when it closes +3 beats a move between dead numbers (+5.5 -> +5)
    off_three = clv_win_probability(3.5, 3.0)
    dead = clv_win_probability(5.5, 5.0)
    assert off_three > dead > 0
    assert clv_win_probability(3.0, 3.0) == pytest.approx(0.0)
    assert clv_win_probability(2.5, 3.0) < 0


def test_ties_reported_separately_from_beat_rate():
    bets = [
        grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.5),  # beat
        grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.0),  # tie
        grade_single_bet("AWAY", 3.0, -110, 24, 20, closing_line=-3.5),   # lose
        grade_single_bet("AWAY", 6.5, -110, 20, 24, closing_line=-6.5),   # tie
    ]
    assert [b["clv_result"] for b in bets] == ["BEAT", "TIE", "LOSE", "TIE"]
    m = compute_metrics(bets, "test")
    # Beat, tie and lose are shares of all bets with a close (R2), so they sum to 100
    assert m["clv_beat_pct"] == pytest.approx(25.0)
    assert m["clv_tie_pct"] == pytest.approx(50.0)
    assert m["clv_lose_pct"] == pytest.approx(25.0)


def test_clv_prefers_consensus_close():
    assert closing_reference_line({"closing_line": -3.0, "closing_consensus_line": -3.5}) == -3.5
    assert closing_reference_line({"closing_line": -3.0, "closing_consensus_line": float("nan")}) == -3.0
    assert closing_reference_line({"closing_line": -3.0}) == -3.0
    assert closing_reference_line({"closing_line": "", "closing_consensus_line": ""}) is None


def _devig(a, b):
    from src.odds_utils import american_to_implied_prob, remove_vig

    return remove_vig(american_to_implied_prob(a), american_to_implied_prob(b))[0]


def test_clv_fair_zero_when_line_and_prices_unchanged():
    assert clv_fair_probability("HOME", -3.0, -110, -110, -3.0, -110, -110) == pytest.approx(0.0, abs=1e-6)
    assert clv_fair_probability("AWAY", 3.5, -125, 105, -3.5, -125, 105) == pytest.approx(0.0, abs=1e-6)
    assert clv_fair_probability("HOME", -7.0, -105, -115, -7.0, -105, -115, total_line=41.5) == pytest.approx(0.0, abs=1e-6)


def test_clv_fair_equals_price_move_when_line_unchanged():
    got = clv_fair_probability("HOME", -3.0, -110, -110, -3.0, -125, 105)
    assert got == pytest.approx(_devig(-125, 105) - 0.5, abs=1e-6)
    got_away = clv_fair_probability("AWAY", 3.0, -110, -110, -3.0, -125, 105)
    assert got_away == pytest.approx((1 - _devig(-125, 105)) - 0.5, abs=1e-6)
    assert got_away < 0


def test_clv_fair_positive_when_line_moves_toward_pick():
    assert clv_fair_probability("HOME", -3.0, -110, -110, -4.0, -110, -110) > 0
    assert clv_fair_probability("AWAY", 3.5, -110, -110, -2.5, -110, -110) > 0
    assert clv_fair_probability("HOME", -3.0, -110, -110, -2.0, -110, -110) < 0


def test_clv_fair_none_when_closing_price_missing():
    assert clv_fair_probability("HOME", -3.0, -110, -110, -3.5, None, -110) is None
    assert clv_fair_probability("HOME", -3.0, -110, -110, -3.5, -110, float("nan")) is None
    assert clv_fair_probability("HOME", -3.0, -110, -110, None, -110, -110) is None
    assert clv_fair_probability("HOME", -3.0, None, -110, -3.5, -110, -110) is None


def test_metrics_average_fair_clv_and_fall_back_to_line_only():
    priced = grade_single_bet(
        "HOME", -3.0, -110, 24, 20, closing_line=-3.0,
        open_home_odds=-110, open_away_odds=-110,
        close_single_line=-3.0, close_home_odds=-125, close_away_odds=105,
    )
    unpriced = grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.5)
    assert priced["clv_fair_prob"] == pytest.approx(_devig(-125, 105) - 0.5, abs=1e-6)
    assert unpriced["clv_fair_prob"] == pytest.approx(unpriced["clv_prob"])
    m = compute_metrics([priced, unpriced], "test")
    expected = (priced["clv_fair_prob"] + unpriced["clv_prob"]) / 2 * 100
    assert m["avg_clv_fair_prob_pct"] == pytest.approx(expected, abs=0.01)
    assert m["clv_fair_priced"] == 1
    assert m["clv_fair_se_pct"] > 0
    assert compute_metrics([priced], "one")["clv_fair_se_pct"] == 0.0
