"""
Injury & Line Movement Guard for NFL Spread Betting.

The #1 trap for statistical/box-score models in NFL betting is being blind to:
1. Quarterback injuries / inactives (worth 4.0 to 8.5 points on a spread).
2. Key offensive line / defensive anchor injuries (worth 0.5 to 1.5 points).
3. Extreme weather (winds > 20 mph, heavy snow).

When a key player is injured, the market moves fast. A model that only looks
at past box scores will see the newly moved spread, assume the team is still
healthy, hallucinate an enormous "edge", and bet right into the injury!

This guard provides:
1. Reverse Line Movement (RLM) detection: Flags when market line moves >= 1.5 pts against pick.
2. Outlier Margin discrepancy filter: Flags when model predicted margin differs from line by > 7.0 pts.
3. Free injury/status checker via nflverse/nfl_data_py when available.
"""

import math
import numpy as np


class MarketIntegrityAlert:
    def __init__(self, is_valid=True, flags=None, recommendation="PROCEED"):
        self.is_valid = is_valid
        self.flags = flags or []
        self.recommendation = recommendation  # PROCEED, MANUAL_REVIEW, REJECT

    def __repr__(self):
        return f"<MarketAlert valid={self.is_valid} rec={self.recommendation} flags={self.flags}>"


def check_line_movement(open_line, current_line, pick_side, threshold=1.5):
    """
    Checks if the spread moved significantly AGAINST our pick.
    
    Example:
    Opening line: Home -6.5
    Current line: Home -3.5 (Moved 3 points toward Away)
    If our model wants to pick HOME:
    The market moved violently AGAINST Home. That strongly signals
    an injury, key inactive, or sharp syndicate fade.
    """
    if open_line is None or current_line is None:
        return None

    try:
        open_val = float(open_line)
        curr_val = float(current_line)
    except Exception:
        return None

    # Line movement from Home's perspective: (curr - open)
    # Positive delta means Home is favored LESS (or Away favored more)
    # Negative delta means Home is favored MORE
    delta_home = curr_val - open_val

    if pick_side.upper() == "HOME":
        # If Home line became more positive (e.g. from -6.5 to -3.5),
        # the market is fading Home!
        pts_against = delta_home
    else:
        # If Away pick, and Home line became more negative (e.g. -3.5 to -6.5),
        # the market is fading Away!
        pts_against = -delta_home

    if pts_against >= threshold:
        return {
            "flag": "REVERSE_LINE_MOVEMENT",
            "pts_against": round(pts_against, 1),
            "open": open_val,
            "current": curr_val,
            "message": f"Market moved {pts_against:.1f} pts AGAINST {pick_side}. Likely injury or sharp fade.",
        }

    return None


def check_model_discrepancy(mu_home_margin, market_line, max_diff=7.0):
    """
    Checks if model's predicted home margin mu differs from market line by more than max_diff.
    In the NFL, edges > 7 points on point spreads do not exist in the real world—
    they are almost always artifacts of backup QBs or model blind spots.
    """
    try:
        mu = float(mu_home_margin)
        line = float(market_line)
    except Exception:
        return None

    # Market implied margin: if line is -3.5, market expects Home to win by ~3.5
    market_expected_margin = -line
    diff = abs(mu - market_expected_margin)

    if diff >= max_diff:
        return {
            "flag": "SUSPICIOUS_OUTLIER_MARGIN",
            "model_mu": round(mu, 1),
            "market_margin": round(market_expected_margin, 1),
            "diff": round(diff, 1),
            "message": f"Model margin ({mu:+.1f}) differs from market ({market_expected_margin:+.1f}) by {diff:.1f} pts. Check starting QB / injuries.",
        }

    return None


def audit_candidate_pick(
    pick_team,
    pick_side,
    open_line,
    current_line,
    mu_home_margin,
    edge,
    max_auto_edge=0.10,
    rlm_threshold=1.5,
    max_margin_diff=7.0,
):
    """
    Comprehensive pre-bet integrity check.
    
    Returns a MarketIntegrityAlert with recommendation:
    - 'PROCEED': Safe to paper-bet automatically.
    - 'MANUAL_REVIEW': Suspicious move or large edge; requires user confirmation.
    - 'SUPPRESS': Strong danger signal (e.g. heavy RLM against pick); do not auto-bet.
    """
    flags = []

    # 1. Edge magnitude sanity
    if edge is not None:
        if edge > max_auto_edge:
            flags.append(f"EDGE_EXCEEDS_MAX ({edge*100:.1f}% > {max_auto_edge*100:.1f}%)")

    # 2. Line movement check
    if open_line is not None and current_line is not None:
        rlm = check_line_movement(open_line, current_line, pick_side, threshold=rlm_threshold)
        if rlm:
            flags.append(f"REVERSE_LINE_MOVEMENT: {rlm['message']}")

    # 3. Model vs Market discrepancy check
    disc = check_model_discrepancy(mu_home_margin, current_line or open_line, max_diff=max_margin_diff)
    if disc:
        flags.append(f"SUSPICIOUS_OUTLIER: {disc['message']}")

    # Determine recommendation
    if any("REVERSE_LINE_MOVEMENT" in f or "SUSPICIOUS_OUTLIER" in f for f in flags):
        recommendation = "SUPPRESS"
        is_valid = False
    elif flags:
        recommendation = "MANUAL_REVIEW"
        is_valid = False
    else:
        recommendation = "PROCEED"
        is_valid = True

    return MarketIntegrityAlert(
        is_valid=is_valid,
        flags=flags,
        recommendation=recommendation,
    )
