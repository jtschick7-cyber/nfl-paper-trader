"""Game-total pricing (P2 spec, notes/implementation-plan/P2-topdown-v2.md; P3 imports it too).

The total points T is an integer drawn from a discretised Normal(mu, sigma) with a continuity
correction. No key-number reweighting: the most common totals each hold 3.4-3.9% of games,
close to the Normal's 3.0% at the mode.

- total_probabilities(mu, line): (p_over, p_under, p_push); x.5 lines have no push.
- devig_two_way(odds_a, odds_b): multiplicative de-vig, -110/-110 assumed on a missing price.
- total_implied_mean(line, q_over): the mu at which the no-push over probability is q_over.
- price_total_offer(mu, side, line, odds): (p_win, p_push, EV per 1u risked).
- total_quotes(event): each book's two-sided `totals` quote in one Odds API event.
"""

import math

from src.key_numbers import calculate_expected_value
from src.price_anchor import devig_home

# SD of (total - closing total_line), 2012-2025 REG, n = 3,663: 13.24
TOTAL_SIGMA = 13.2


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def total_probabilities(mu, line, sigma=TOTAL_SIGMA):
    """(p_over, p_under, p_push) of an integer total ~ discretised Normal(mu, sigma)."""
    line = float(line)
    if abs(line - round(line)) < 1e-9:
        hi = _cdf((line + 0.5 - mu) / sigma)
        lo = _cdf((line - 0.5 - mu) / sigma)
        return float(1.0 - hi), float(lo), float(hi - lo)
    cut = _cdf((math.floor(line) + 0.5 - mu) / sigma)  # = (line - mu) / sigma on x.5 lines
    return float(1.0 - cut), float(cut), 0.0


def devig_two_way(odds_a, odds_b):
    """(q_a, price_assumed): the same multiplicative de-vig as price_anchor.devig_home."""
    return devig_home(odds_a, odds_b)


def total_implied_mean(line, q_over, sigma=TOTAL_SIGMA):
    """The mu at which p_over / (p_over + p_under) = q_over, by bisection on
    [line - 30, line + 30] to 1e-6 (the conditional over probability rises with mu)."""
    line = float(line)
    lo, hi = line - 30.0, line + 30.0
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        p_over, p_under, _ = total_probabilities(mid, line, sigma)
        if p_over / (p_over + p_under) < q_over:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def price_total_offer(mu, side, line, odds, sigma=TOTAL_SIGMA):
    """(p_win, p_push, ev) of a 1u ticket on `side` ("OVER"/"UNDER") at `line` and `odds`."""
    p_over, p_under, p_push = total_probabilities(mu, line, sigma)
    p_win, p_loss = (p_over, p_under) if side == "OVER" else (p_under, p_over)
    return p_win, p_push, calculate_expected_value(p_win, p_loss, odds)


def total_quotes(event):
    """One dict per book with a two-sided `totals` quote: book, line, over_odds, under_odds,
    last_update. A book whose Over and Under points differ (or are missing) is skipped; a
    missing price is kept as None."""
    out = []
    for b in event.get("bookmakers", []) or []:
        tm = next((m for m in b.get("markets", []) or [] if m.get("key") == "totals"), None)
        if not tm:
            continue
        o = {x.get("name"): x for x in tm.get("outcomes", []) or []}
        if "Over" not in o or "Under" not in o:
            continue
        po, pu = _f(o["Over"].get("point")), _f(o["Under"].get("point"))
        if po is None or pu is None or abs(po - pu) > 1e-9:
            continue
        out.append({"book": b.get("key"), "line": po, "over_odds": _f(o["Over"].get("price")),
                    "under_odds": _f(o["Under"].get("price")),
                    "last_update": tm.get("last_update") or b.get("last_update")})
    return out
