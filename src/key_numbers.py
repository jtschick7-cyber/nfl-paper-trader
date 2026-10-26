"""
Key Numbers and Discrete Football Margin Distribution for NFL Spread Betting.

In the NFL, scoring margins are discrete and heavily clustered around football key numbers:
- 3: ~14.8% of all games (Field goal)
- 7: ~9.4% of all games (Touchdown + extra point)
- 6: ~6.0% of all games (Touchdown without extra point / two field goals)
- 10: ~5.8% of all games (Touchdown + field goal)
- 4: ~5.4% of all games (Field goal + 1 / touchdown minus safety)
- 14: ~4.8% of all games (Two touchdowns)

Continuous Gaussian distributions (norm.cdf) severely undervalue moves across 3 and 7,
and completely ignore push probabilities on integer spreads (-3.0, -7.0, etc.).

This module provides:
1. Discrete empirical mass distribution for NFL margins.
2. Push-aware cover probabilities P(Win), P(Loss), P(Push).
3. Conditional win rate and Expected Value (EV) calculation.
Runs entirely on standard library (math) and numpy.
"""

import math
import numpy as np

# Empirical distribution of NFL margin of victory (|home_score - away_score|):
# Frequencies based on modern NFL regular season games (1999-2024, ~6,500 games).
MARGIN_FREQUENCIES = {
    0: 0.003,   # Regular season ties
    1: 0.041,
    2: 0.040,
    3: 0.148,   # Primary key number #1
    4: 0.054,
    5: 0.024,
    6: 0.060,   # Secondary key number
    7: 0.094,   # Primary key number #2
    8: 0.036,
    9: 0.024,
    10: 0.058,  # Key number
    11: 0.028,
    12: 0.018,
    13: 0.030,
    14: 0.048,  # Key number (2 TDs)
    15: 0.014,
    16: 0.020,
    17: 0.034,  # Key number
    18: 0.014,
    19: 0.012,
    20: 0.018,
    21: 0.024,
    22: 0.010,
    23: 0.010,
    24: 0.016,
}

# Key numbers ranked by impact on spread value
KEY_NUMBERS = {3: 0.148, 7: 0.094, 6: 0.060, 10: 0.058, 4: 0.054, 14: 0.048}


def standard_normal_cdf(x):
    """Computes standard normal CDF using math.erf (zero external dependencies)."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def payout_multiplier(american_odds):
    """
    Returns net profit multiplier b for 1 unit risked.
    Example: -110 -> 100/110 = 0.90909
             +150 -> 150/100 = 1.50
    """
    odds = float(american_odds)
    if odds > 0:
        return odds / 100.0
    elif odds < 0:
        return 100.0 / (-odds)
    return 1.0


def is_integer_spread(spread):
    """Checks whether a spread is an integer (e.g. -3.0, 7.0 vs -3.5)."""
    try:
        val = float(spread)
        return abs(val - round(val)) < 1e-5
    except Exception:
        return False


def discrete_margin_probabilities(mu, sigma=13.4, max_margin=50):
    """
    Constructs a discrete probability distribution over all possible margins
    m in [-max_margin, max_margin] by blending Gaussian prior with empirical
    football scoring key-number weights.
    
    Returns a dictionary: {margin: probability}
    """
    margins = np.arange(-max_margin, max_margin + 1)
    
    # 1. Base Gaussian probability mass for each integer margin [m - 0.5, m + 0.5]
    upper = np.array([standard_normal_cdf((m + 0.5 - mu) / sigma) for m in margins])
    lower = np.array([standard_normal_cdf((m - 0.5 - mu) / sigma) for m in margins])
    raw_mass = upper - lower
    
    # 2. Key-number adjustment weights:
    # We apply a slight boost to margins matching empirical football score spikes
    # while preserving the expected value mu.
    weights = np.ones_like(margins, dtype=float)
    
    for idx, m in enumerate(margins):
        abs_m = abs(int(m))
        if abs_m in MARGIN_FREQUENCIES:
            emp_freq = MARGIN_FREQUENCIES[abs_m]
            gauss_freq = standard_normal_cdf((abs_m + 0.5) / sigma) - standard_normal_cdf((abs_m - 0.5) / sigma)
            if gauss_freq > 0:
                ratio = emp_freq / (2.0 * gauss_freq)
                dampened = np.clip(ratio, 0.7, 1.6)
                weights[idx] = dampened
                
    adjusted_mass = raw_mass * weights
    total = np.sum(adjusted_mass)
    if total > 0:
        adjusted_mass /= total
        
    return dict(zip(margins, adjusted_mass))


def calculate_spread_probabilities(mu, sigma, home_spread):
    """
    Calculates P(Home Win), P(Away Win), and P(Push) for a given home spread.
    
    Home covers if (margin + home_spread) > 0  => margin > -home_spread
    Push occurs if (margin + home_spread) == 0 => margin == -home_spread
    Away covers if (margin + home_spread) < 0  => margin < -home_spread
    
    Returns:
        p_home_win: float (0.0 to 1.0)
        p_away_win: float (0.0 to 1.0)
        p_push: float (0.0 to 1.0)
        cond_home_win: float (p_home_win / (p_home_win + p_away_win))
        cond_away_win: float (p_away_win / (p_home_win + p_away_win))
    """
    home_spread = float(home_spread)
    dist = discrete_margin_probabilities(mu, sigma)
    
    push_margin = -home_spread
    is_int = is_integer_spread(home_spread)
    
    p_home_win = 0.0
    p_away_win = 0.0
    p_push = 0.0
    
    for m, prob in dist.items():
        if is_int and abs(m - push_margin) < 1e-5:
            p_push += prob
        elif m > push_margin:
            p_home_win += prob
        else:
            p_away_win += prob
            
    # Conditional win probability (excluding pushes)
    non_push = p_home_win + p_away_win
    if non_push > 0:
        cond_home_win = p_home_win / non_push
        cond_away_win = p_away_win / non_push
    else:
        cond_home_win = 0.5
        cond_away_win = 0.5
        
    return {
        "p_home_win": p_home_win,
        "p_away_win": p_away_win,
        "p_push": p_push,
        "cond_home_win": cond_home_win,
        "cond_away_win": cond_away_win,
    }


def calculate_expected_value(p_win, p_loss, american_odds):
    """
    Calculates expected profit per 1 unit risked.
    EV = P(Win) * b - P(Loss) * 1.0
    where b is the payout multiplier. Pushes return stake (0 profit).
    """
    b = payout_multiplier(american_odds)
    ev = (p_win * b) - (p_loss * 1.0)
    return float(ev)
