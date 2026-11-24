"""Player-receptions pricing (P4 spec, notes/implementation-plan/P4-prop-scanner.md).

Receptions R are Poisson(lam) (frozen). For a line L: P(over) = P(R > floor(L)),
P(under) = P(R < L), and at an integer L, P(R = L) is a push. The Negative Binomial with
r = NB_SENSITIVITY_R (same mean) is a reported sensitivity only, never used for flags.

- normalize_name(s): the player key (with event_id).
- prop_quotes(response): one row per (book, player, line) from an /events/{id}/odds response.
- prop_probabilities(lam, line, r=None): (p_over, p_under, p_push).
- implied_lambda(line, q_over, r=None): bisection on [LAMBDA_LO, LAMBDA_HI] to LAMBDA_TOL.
- price_prop_offer(lam, side, line, odds, r=None): (p_win, p_push, EV per 1u risked).

The de-vig is total_pricing.devig_two_way and the EV is key_numbers.calculate_expected_value;
neither is redefined here.
"""

import math
import re

from scipy.stats import nbinom, poisson

from src.key_numbers import calculate_expected_value

# ---- frozen (spec 2026-11-24) ---------------------------------------------------------------
NB_SENSITIVITY_R = 8.0
LAMBDA_LO = 0.05
LAMBDA_HI = 20.0
LAMBDA_TOL = 1e-6
# ---------------------------------------------------------------------------------------------

MARKET = "player_receptions"
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_STRIP = re.compile(r"[.'’]")


def normalize_name(s):
    """Lower case; strip '.', "'" and '’'; drop a trailing jr/sr/ii/iii/iv/v; collapse whitespace."""
    parts = _STRIP.sub("", str(s).lower()).split()
    if len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _odds(v):
    x = _num(v)
    if x is None:
        return None
    return int(x) if x == int(x) else x


def prop_quotes(response):
    """[{book, player, player_name, line, over_odds, under_odds, last_update}], one row per
    (book, normalized player, line). Either price may be None (one-sided quote)."""
    rows = {}
    for bm in response.get("bookmakers") or []:
        book = bm.get("key")
        for m in bm.get("markets") or []:
            if m.get("key") != MARKET:
                continue
            upd = m.get("last_update") or bm.get("last_update")
            for o in m.get("outcomes") or []:
                name, side = o.get("description"), str(o.get("name", "")).lower()
                line, price = _num(o.get("point")), _odds(o.get("price"))
                if not name or line is None or price is None or side not in ("over", "under"):
                    continue
                key = (book, normalize_name(name), line)
                r = rows.setdefault(key, {"book": book, "player": key[1], "player_name": name, "line": line,
                                          "over_odds": None, "under_odds": None, "last_update": upd})
                r[f"{side}_odds"] = price
    return list(rows.values())


def _dist(lam, r):
    if r is None:
        return poisson(lam)
    return nbinom(r, r / (r + lam))


def prop_probabilities(lam, line, r=None):
    """(p_over, p_under, p_push) of R ~ Poisson(lam), or NB(mean lam, size r) when r is given."""
    line = float(line)
    d = _dist(lam, r)
    fl = math.floor(line)
    p_over = float(d.sf(fl))
    if abs(line - round(line)) < 1e-9:
        p_push = float(d.pmf(fl))
        return p_over, float(d.cdf(fl - 1)), p_push
    return p_over, float(d.cdf(fl)), 0.0


def implied_lambda(line, q_over, r=None):
    """The lam at which p_over / (p_over + p_under) = q_over (rises with lam), by bisection."""
    lo, hi = LAMBDA_LO, LAMBDA_HI
    while hi - lo > LAMBDA_TOL:
        mid = 0.5 * (lo + hi)
        p_over, p_under, _ = prop_probabilities(mid, line, r)
        if p_over / (p_over + p_under) < q_over:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def price_prop_offer(lam, side, line, odds, r=None):
    """(p_win, p_push, ev) of a 1u ticket on `side` ("OVER"/"UNDER") at `line` and `odds`."""
    p_over, p_under, p_push = prop_probabilities(lam, line, r)
    p_win, p_loss = (p_over, p_under) if side == "OVER" else (p_under, p_over)
    return p_win, p_push, calculate_expected_value(p_win, p_loss, odds)
