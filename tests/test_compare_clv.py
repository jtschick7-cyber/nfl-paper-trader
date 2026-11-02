import pandas as pd
import pytest

from src.compare_models import clv_win_probability, compute_metrics, grade_single_bet


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
    assert m["clv_beat_pct"] == pytest.approx(50.0)
    assert m["clv_tie_pct"] == pytest.approx(50.0)
