"""The shadow models' per-game decision rule, shared by the loggers and the backtest (R4).

Pure, no I/O. Given a raw model margin and the market (home line, both prices, optional
total), it blends toward the market by beta, converts to key-number-aware cover
probabilities, calibrates, and measures each side's edge against the break-even
probability of the price taken (vig included). The integrity guard is not applied here.
"""

import math

from src.config import MIN_EDGE_THRESHOLD
from src.key_numbers import calculate_spread_probabilities
from src.market_blend import blend_margin, market_mu
from src.model_v2_core import calibrate_probability
from src.odds_utils import american_to_implied_prob

PICK = "PICK"
PASS = "PASS"


def apply_calibrator(calibrator, p):
    """(method, params) tuples go through calibrate_probability; sklearn-style objects use
    predict_proba (identity if that fails); anything else is the identity."""
    if isinstance(calibrator, tuple):
        return float(calibrate_probability(calibrator, p))
    if hasattr(calibrator, "predict_proba"):
        try:
            return float(calibrator.predict_proba([[p]])[0][1])
        except Exception:
            return p
    return p


def shadow_decision(raw_mu, home_line, home_odds, away_odds, *, sigma, market_beta,
                    calibrator, total_line=None, min_edge=MIN_EDGE_THRESHOLD):
    """
    Decision for one game. market_beta None means no blend (v2). Returns a dict with
    mu_blend, p_home_discrete, p_push, sigma_adj, p_home_cal, p_away_cal, edge_home,
    edge_away, best_side, best_line, best_odds, best_model_p, best_edge, raw_lean_pts,
    and decision (PICK if best_edge >= min_edge, else PASS).
    """
    raw_mu = float(raw_mu)
    home_line = float(home_line)
    mu = blend_margin(raw_mu, home_line, market_beta) if market_beta is not None else raw_mu

    spread_res = calculate_spread_probabilities(mu, sigma, home_line, total_line=total_line)
    p_home_discrete = float(spread_res["cond_home_win"])
    p_push = float(spread_res["p_push"])
    sigma_adj = float(spread_res.get("sigma", sigma))

    p_home_cal = apply_calibrator(calibrator, p_home_discrete)
    p_away_cal = 1.0 - p_home_cal

    edge_home = p_home_cal - american_to_implied_prob(home_odds)
    edge_away = p_away_cal - american_to_implied_prob(away_odds)

    if edge_home >= edge_away:
        best_side, best_line, best_odds, best_model_p, best_edge = "HOME", home_line, home_odds, p_home_cal, edge_home
    else:
        best_side, best_line, best_odds, best_model_p, best_edge = "AWAY", -home_line, away_odds, p_away_cal, edge_away

    decision = PICK if (math.isfinite(best_edge) and best_edge >= min_edge) else PASS

    return {
        "mu_blend": mu,
        "p_home_discrete": p_home_discrete,
        "p_push": p_push,
        "sigma_adj": sigma_adj,
        "p_home_cal": p_home_cal,
        "p_away_cal": p_away_cal,
        "edge_home": edge_home,
        "edge_away": edge_away,
        "best_side": best_side,
        "best_line": best_line,
        "best_odds": best_odds,
        "best_model_p": best_model_p,
        "best_edge": best_edge,
        "raw_lean_pts": raw_mu - market_mu(home_line),
        "decision": decision,
    }
