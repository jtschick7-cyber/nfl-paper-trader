"""
Push-Aware Fractional Kelly Staking Engine.

Ported from FreeMargin Engine v2.2 (src/freemargin/decision/staking.py).

Methodology:
Traditional binary Kelly criterion assumes outcomes are binary (Win or Loss):
    f* = (p * b - q) / b
In NFL spread betting with integer spreads (e.g. -3.0, -7.0, +6.0), push probability
p_push is significant (up to 14.8% on key numbers). 

On a push, the bettor's stake is refunded (log growth = 0).
The exact push-aware Kelly fraction solves numerically for the fraction f that
maximizes expected log bankroll growth:
    E[log(1 + f * X)] = p_win * log(1 + f * b) + p_loss * log(1 - f)
where b = decimal_odds - 1.0.

Quarter-Kelly (f * 0.25) is the industry standard for quantitative sports betting,
dramatically reducing drawdown variance and protecting against model estimation noise
while capturing ~75% of maximum compounding growth rate.
"""

import math
from dataclasses import dataclass
import numpy as np

from src.config import BANKROLL_UNITS, FLAT_BET_SIZE
from src.odds_utils import american_to_implied_prob

DEFAULT_MAX_STAKE_UNITS = 3.0  # Max single-bet exposure limit (3% of bankroll)
DEFAULT_MIN_STAKE_UNITS = 0.25  # Minimum bet size to warrant placing
DEFAULT_MAX_SLATE_UNITS = 10.0  # Max total portfolio exposure per weekly slate


def american_to_decimal_odds(american_odds: float) -> float:
    """Converts American odds (-110, +105) to European decimal odds (1.909, 2.05)."""
    odds = float(american_odds)
    if odds > 0:
        return 1.0 + (odds / 100.0)
    elif odds < 0:
        return 1.0 + (100.0 / abs(odds))
    return 1.0


def validate_probabilities(win_prob: float, push_prob: float, loss_prob: float = None):
    """Validates that probabilities are finite and sum to 1.0."""
    w = float(win_prob)
    p = float(push_prob)

    if loss_prob is None:
        l = max(0.0, 1.0 - w - p)
    else:
        l = float(loss_prob)

    for name, val in [("win_prob", w), ("push_prob", p), ("loss_prob", l)]:
        if not math.isfinite(val) or not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be a valid probability in [0, 1], got {val}")

    total = w + p + l
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"Probabilities must sum to 1.0, got sum = {total:.6f} (w={w}, p={p}, l={l})")

    return w, p, l


def full_kelly_fraction(win_prob: float, push_prob: float, loss_prob: float, decimal_odds: float) -> float:
    """
    Maximizes expected log bankroll growth over ternary outcome space {Win, Push, Loss}.
    Uses exact closed-form analytical solution:
        f* = max(0.0, (w * b - l) / (b * (w + l)))
    """
    w, p, l = validate_probabilities(win_prob, push_prob, loss_prob)
    b = float(decimal_odds) - 1.0

    if b <= 0.0 or (w + l) <= 0.0:
        return 0.0

    # If expected value is non-positive, Kelly fraction is zero (no edge)
    if w * b <= l:
        return 0.0

    f = (w * b - l) / (b * (w + l))
    return float(np.clip(f, 0.0, 1.0))


def compute_push_aware_stake(
    win_prob: float,
    push_prob: float,
    american_odds: float,
    bankroll: float = BANKROLL_UNITS,
    kelly_multiplier: float = 0.25,
    max_stake: float = DEFAULT_MAX_STAKE_UNITS,
    min_stake: float = DEFAULT_MIN_STAKE_UNITS,
) -> dict:
    """
    Calculates the recommended stake in units using Push-Aware Quarter-Kelly sizing.

    Returns dict with full_kelly_pct, quarter_kelly_pct, recommended_units, and flat_units.
    """
    dec_odds = american_to_decimal_odds(american_odds)
    loss_prob = max(0.0, 1.0 - win_prob - push_prob)

    full_f = full_kelly_fraction(win_prob, push_prob, loss_prob, dec_odds)
    scaled_f = full_f * kelly_multiplier

    raw_units = scaled_f * bankroll

    # Apply risk bounds
    if raw_units < min_stake:
        recommended_units = 0.0
    else:
        recommended_units = round(min(raw_units, max_stake), 2)

    return {
        "win_prob": round(win_prob, 4),
        "push_prob": round(push_prob, 4),
        "loss_prob": round(loss_prob, 4),
        "decimal_odds": round(dec_odds, 4),
        "full_kelly_fraction": round(full_f, 4),
        "quarter_kelly_fraction": round(scaled_f, 4),
        "recommended_units": recommended_units,
        "flat_units": FLAT_BET_SIZE,
        "is_bounded_by_max": bool(raw_units >= max_stake),
    }


def cap_slate_stakes(slate_bets: list, max_slate_exposure: float = DEFAULT_MAX_SLATE_UNITS) -> list:
    """
    Applies a portfolio-level simultaneous Kelly exposure constraint to a weekly slate.
    If the sum of recommended units across all bets on a slate exceeds max_slate_exposure,
    each bet's stake is scaled proportionally:
        scale = max_slate_exposure / total_recommended
        capped_units = round(units * scale, 2)
    Prevents portfolio overconcentration while preserving relative edge sizing.
    """
    if not slate_bets:
        return []

    total_units = sum(float(b.get("recommended_units", 0.0)) for b in slate_bets)

    if total_units <= max_slate_exposure or total_units <= 0:
        return slate_bets

    scale_factor = max_slate_exposure / total_units
    capped_slate = []

    for b in slate_bets:
        b_copy = dict(b)
        orig_units = float(b.get("recommended_units", 0.0))
        capped_units = round(orig_units * scale_factor, 2)
        b_copy["recommended_units"] = capped_units
        b_copy["slate_capped"] = True
        b_copy["orig_units"] = orig_units
        capped_slate.append(b_copy)

    return capped_slate


def main():
    print("=" * 70)
    print("Push-Aware Quarter-Kelly Staking Engine (FreeMargin Port)")
    print("=" * 70)

    # Scenarios for Week 8 picks
    scenarios = [
        {
            "name": "LV Away -1.5 @ +105 (High Conviction, Key Edge)",
            "win_prob": 0.528,
            "push_prob": 0.025,
            "odds": 105.0,
        },
        {
            "name": "GB Home -3.0 @ -105 (Key Number 3, High Push Probability)",
            "win_prob": 0.518,
            "push_prob": 0.085,
            "odds": -105.0,
        },
        {
            "name": "LA Home -7.0 @ -105 (Key Number 7, Standard Push Probability)",
            "win_prob": 0.522,
            "push_prob": 0.062,
            "odds": -105.0,
        },
        {
            "name": "MIN Away +2.5 @ +100 (Thin Edge)",
            "win_prob": 0.510,
            "push_prob": 0.020,
            "odds": 100.0,
        },
        {
            "name": "Marginal Play (Edge = +0.5%)",
            "win_prob": 0.505,
            "push_prob": 0.050,
            "odds": -110.0,
        },
    ]

    print(f"{'Pick / Scenario':<42} | {'Win %':<6} | {'Push %':<6} | {'Full K':<6} | {'QK Units':<8} | {'Flat Units'}")
    print("-" * 85)

    for s in scenarios:
        res = compute_push_aware_stake(
            win_prob=s["win_prob"],
            push_prob=s["push_prob"],
            american_odds=s["odds"],
            bankroll=100.0,
            kelly_multiplier=0.25,
        )
        print(
            f"{s['name'][:42]:<42} | {res['win_prob']*100:>5.1f}% | {res['push_prob']*100:>5.1f}% | "
            f"{res['full_kelly_fraction']*100:>5.1f}% | {res['recommended_units']:>6.2f}u  | {res['flat_units']:>5.1f}u"
        )

    print("=" * 85)
    print("Verification complete.")


if __name__ == "__main__":
    main()
