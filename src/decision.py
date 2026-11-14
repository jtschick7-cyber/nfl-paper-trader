"""The shadow models' per-game decision rule, shared by the loggers and the backtest (R4).

Pure, no I/O. Given a raw model margin and the market (home line, both prices, optional
total), it blends toward the market by beta, converts to key-number-aware cover
probabilities, calibrates, and measures each side's edge against the break-even
probability of the price taken (vig included). The integrity guard is not applied here.

Two anchors (R5). Without `anchor` (v2, and v3/v4 artifacts before R5) the blend centres
on -home_line and a calibrator follows. With anchor="price" it centres on m*, the mean
that reproduces the de-vigged price at the offered line (src/price_anchor.py), and the
calibrator must be the identity: zero disagreement then gives the market's probability.
"""

import math

from src.config import MIN_EDGE_THRESHOLD
from src.key_numbers import calculate_spread_probabilities
from src.market_blend import blend_margin, blend_margin_anchored, market_mu
from src.model_v2_core import calibrate_probability
from src.odds_utils import american_to_implied_prob
from src.price_anchor import anchor_mean, devig_home

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


def _edge(p, odds):
    """p minus the break-even probability of `odds`; NaN when there is no usable price."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(o) or o == 0:
        return float("nan")
    return p - american_to_implied_prob(o)


def shadow_decision(raw_mu, home_line, home_odds, away_odds, *, sigma, market_beta,
                    calibrator, total_line=None, min_edge=MIN_EDGE_THRESHOLD,
                    anchor=None, anchor_variant=None):
    """
    Decision for one game. market_beta None means no blend (v2). Returns a dict with
    mu_blend, p_home_discrete, p_push, sigma_adj, p_home_cal, p_away_cal, edge_home,
    edge_away, best_side, best_line, best_odds, best_model_p, best_edge, raw_lean_pts,
    anchor_mean, q_home, price_assumed (the last three None/False on the line anchor),
    and decision (PICK if best_edge >= min_edge, else PASS).

    anchor="price": mu = m* + beta * d, d against m* (anchor_variant "A", the default)
    or against -home_line ("B"). A missing price is taken as -110 both sides and flagged.
    """
    raw_mu = float(raw_mu)
    home_line = float(home_line)
    m_star, q_home, price_assumed = None, None, False
    if anchor is None:
        mu = blend_margin(raw_mu, home_line, market_beta) if market_beta is not None else raw_mu
    elif anchor == "price":
        if market_beta is None:
            raise ValueError("price anchor needs market_beta")
        if not (isinstance(calibrator, tuple) and calibrator[0] == "identity"):
            raise ValueError(f"price anchor takes no fitted calibrator, got {calibrator!r}")
        variant = anchor_variant or "A"
        if variant not in ("A", "B"):
            raise ValueError(f"unknown anchor variant {variant!r}")
        q_home, price_assumed = devig_home(home_odds, away_odds)
        m_star = anchor_mean(home_line, q_home, sigma, total_line)
        mu = blend_margin_anchored(raw_mu, m_star, market_beta, None if variant == "A" else market_mu(home_line))
    else:
        raise ValueError(f"unknown anchor {anchor!r}")

    spread_res = calculate_spread_probabilities(mu, sigma, home_line, total_line=total_line)
    p_home_discrete = float(spread_res["cond_home_win"])
    p_push = float(spread_res["p_push"])
    sigma_adj = float(spread_res.get("sigma", sigma))

    p_home_cal = apply_calibrator(calibrator, p_home_discrete)
    p_away_cal = 1.0 - p_home_cal

    if anchor is None:  # unchanged pre-R5 arithmetic
        edge_home = p_home_cal - american_to_implied_prob(home_odds)
        edge_away = p_away_cal - american_to_implied_prob(away_odds)
    else:
        edge_home = _edge(p_home_cal, home_odds)
        edge_away = _edge(p_away_cal, away_odds)

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
        "anchor_mean": m_star,
        "q_home": q_home,
        "price_assumed": price_assumed,
        "decision": decision,
    }
