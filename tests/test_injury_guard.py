"""
Unit tests for Injury and Outlier Safeguard Audits.
"""

import pytest
from src.injury_guard import (
    MarketIntegrityAlert,
    audit_candidate_pick,
    check_line_movement,
    check_model_discrepancy,
)


def test_normal_line_movement():
    # 0.5 point move is normal market noise
    res = check_line_movement(open_line=-3.0, current_line=-3.5, pick_side="HOME")
    assert res is None


def test_high_line_movement_against_pick():
    # Line moves 2.5 points against HOME pick (opened -3.5, now -1.0) -> steam on away
    res = check_line_movement(open_line=-3.5, current_line=-1.0, pick_side="HOME")
    assert res is not None
    assert res["flag"] == "REVERSE_LINE_MOVEMENT"
    assert res["pts_against"] == 2.5
    assert "AGAINST HOME" in res["message"]


def test_model_discrepancy():
    # 2.0 point discrepancy is normal
    normal = check_model_discrepancy(mu_home_margin=5.5, market_line=-3.5, max_diff=7.0)
    assert normal is None

    # 10.0 point discrepancy is suspicious outlier
    outlier = check_model_discrepancy(mu_home_margin=14.0, market_line=-3.5, max_diff=7.0)
    assert outlier is not None
    assert outlier["flag"] == "SUSPICIOUS_OUTLIER_MARGIN"
    assert outlier["diff"] > 7.0


def test_audit_candidate_pick_proceed_vs_suppress():
    # Valid normal pick
    proceed_audit = audit_candidate_pick(
        pick_team="GB",
        pick_side="HOME",
        open_line=-3.0,
        current_line=-3.0,
        mu_home_margin=4.5,
        edge=0.025,
        max_auto_edge=0.07,
    )
    assert proceed_audit.recommendation == "PROCEED"
    assert proceed_audit.is_valid is True

    # Steam against pick (RLM) -> SUPPRESS
    suppress_audit = audit_candidate_pick(
        pick_team="GB",
        pick_side="HOME",
        open_line=-3.5,
        current_line=-1.0,
        mu_home_margin=4.5,
        edge=0.025,
        max_auto_edge=0.07,
    )
    assert suppress_audit.recommendation == "SUPPRESS"
    assert suppress_audit.is_valid is False
    assert any("REVERSE_LINE_MOVEMENT" in f for f in suppress_audit.flags)


def test_pickem_current_line_is_not_replaced_by_open_line():
    # Opened -9, now a pick'em 0.0: a model at +8 is 8 pts off the pick'em, only 1 off the open.
    alert = audit_candidate_pick(pick_team="KC", pick_side="HOME", open_line=-9.0, current_line=0.0,
                                 mu_home_margin=8.0, edge=0.03, rlm_threshold=99.0)
    assert any("SUSPICIOUS_OUTLIER" in f for f in alert.flags)


def test_raw_margin_triggers_the_outlier_guard_but_a_blended_one_cannot():
    # beta ~0.02 shrinks a 10-pt raw disagreement to 0.2 pts, so the guard must see the raw margin.
    raw_mu, line, beta = 13.0, -3.0, 0.021
    blended = -line + beta * (raw_mu + line)
    assert check_model_discrepancy(blended, line) is None
    assert check_model_discrepancy(raw_mu, line) is not None
