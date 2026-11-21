"""Multi-book consensus pricer and off-market (stale) line detector (audit Phase 3, section 3.1).

Model-free, top-down. For one event in an Odds API response:

1. Every book with a two-sided spread gives an implied mean home margin
   m*_book = anchor_mean(home_line, q_book, sigma, total_line), q_book its de-vigged home
   cover probability (src/price_anchor.py). That puts books hanging different numbers on
   one scale, which averaging fair probabilities across numbers does not.
2. m*_consensus is a weighted median of m*_book after dropping books more than
   OUTLIER_PTS from the plain median. Lower-vig books weigh more (BOOK_WEIGHTS); the book
   being priced is left out of its own consensus, so a stale book cannot vote for itself.
3. Each book's offer (side, line, price) is priced against m*_consensus with the
   key-number engine: p_win, p_push, EV per 1u risked. An offer with EV >= STALE_EV is
   flagged; `key_cross` marks an offer that sits on the better side of a key number
   (3, 7, 6, 10, 4) relative to the consensus line.

Nothing here touches the Sheet or the decision rule; src/topdown_replay.py replays it over
the odds-data archive.
"""

import math

import numpy as np

from src.key_numbers import calculate_expected_value, calculate_spread_probabilities
from src.odds_utils import TEAM_ABBR_MAP
from src.price_anchor import anchor_mean, devig_home

# The committed v3 artifact's blended sigma (models/metadata_v3.json), rounded.
DEFAULT_SIGMA = 12.8
# US-region books in the archive. Low-vig / sharper books count double in the consensus.
BOOK_WEIGHTS = {"pinnacle": 3.0, "lowvig": 2.0, "betonlineag": 2.0, "circasports": 2.0}
DEFAULT_WEIGHT = 1.0
OUTLIER_PTS = 1.5
MIN_CONSENSUS_BOOKS = 3
STALE_EV = 0.01
KEY_NUMBERS = (3, 7, 6, 10, 4)


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def book_quotes(event):
    """Two-sided spread quotes of every book in one Odds API event: list of dicts with
    book, home_line, home_odds, away_line, away_odds, total_line, last_update."""
    raw_h, raw_a = event.get("home_team"), event.get("away_team")
    out = []
    for b in event.get("bookmakers", []) or []:
        mkts = {m.get("key"): m for m in b.get("markets", []) or []}
        sp = mkts.get("spreads")
        if not sp:
            continue
        o = {x.get("name"): x for x in sp.get("outcomes", []) or []}
        if raw_h not in o or raw_a not in o:
            continue
        hl, ho = _f(o[raw_h].get("point")), _f(o[raw_h].get("price"))
        al, ao = _f(o[raw_a].get("point")), _f(o[raw_a].get("price"))
        if None in (hl, ho, al, ao) or ho == 0 or ao == 0 or abs(hl + al) > 1e-6:
            continue
        total = None
        tm = mkts.get("totals")
        if tm:
            pts = [_f(x.get("point")) for x in tm.get("outcomes", []) or []]
            pts = [p for p in pts if p is not None]
            total = pts[0] if pts else None
        out.append({"book": b.get("key"), "home_line": hl, "home_odds": ho, "away_line": al, "away_odds": ao,
                    "total_line": total, "last_update": sp.get("last_update") or b.get("last_update")})
    return out


def implied_mean(quote, sigma=DEFAULT_SIGMA, total_line=None):
    q, _ = devig_home(quote["home_odds"], quote["away_odds"])
    return anchor_mean(quote["home_line"], q, sigma, total_line)


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    half = c[-1] / 2.0
    i = int(np.searchsorted(c, half))
    # exact half between two values: average them (keeps symmetric cases symmetric)
    if i < len(v) - 1 and abs(c[i] - half) < 1e-12:
        return float((v[i] + v[i + 1]) / 2.0)
    return float(v[i])


def consensus_mean(means, exclude=None, outlier_pts=OUTLIER_PTS, min_books=MIN_CONSENSUS_BOOKS):
    """Weighted median of {book: m*_book}, books further than outlier_pts from the plain
    median dropped, `exclude` left out. None with fewer than min_books books left."""
    items = [(b, m) for b, m in means.items() if b != exclude and m is not None]
    if len(items) < min_books:
        return None
    med = float(np.median([m for _, m in items]))
    kept = [(b, m) for b, m in items if abs(m - med) <= outlier_pts]
    if len(kept) < min_books:
        return None
    return weighted_median([m for _, m in kept], [BOOK_WEIGHTS.get(b, DEFAULT_WEIGHT) for b, _ in kept])


def price_offer(mu, side, home_line, odds, sigma=DEFAULT_SIGMA, total_line=None):
    """(p_win, p_push, ev) of a 1u ticket on `side` at `home_line` (home perspective)
    and `odds`, when the home margin has mean `mu`."""
    sp = calculate_spread_probabilities(mu, sigma, home_line, total_line=total_line)
    p_win = sp["p_home_win"] if side == "HOME" else sp["p_away_win"]
    p_loss = sp["p_away_win"] if side == "HOME" else sp["p_home_win"]
    return p_win, sp["p_push"], calculate_expected_value(p_win, p_loss, odds)


def crosses_key(side, offer_home_line, consensus_home_line, keys=KEY_NUMBERS):
    """The first key number (by KEY_NUMBERS order) among the home margins the offer wins and
    the consensus line does not, or None. HOME at line h wins margins m > -h, so a higher
    offer line adds margins in (-offer, -consensus]; AWAY adds [-consensus, -offer)."""
    o, c = -float(offer_home_line), -float(consensus_home_line)
    for k in keys:
        for m in (k, -k):
            if (side == "HOME" and o < m <= c) or (side == "AWAY" and c <= m < o):
                return k
    return None


def evaluate_event(event, sigma=DEFAULT_SIGMA, stale_ev=STALE_EV):
    """One row per (book, side) of an event: the offer, the leave-one-out consensus mean
    and line, p_win / p_push / EV against it, and the stale and key-number flags."""
    quotes = book_quotes(event)
    if not quotes:
        return []
    totals = [q["total_line"] for q in quotes if q["total_line"] is not None]
    total = float(np.median(totals)) if totals else None
    means = {q["book"]: implied_mean(q, sigma, total) for q in quotes}
    cons_line = float(np.median([q["home_line"] for q in quotes]))
    home = TEAM_ABBR_MAP.get(event.get("home_team"), event.get("home_team"))
    away = TEAM_ABBR_MAP.get(event.get("away_team"), event.get("away_team"))
    rows = []
    for q in quotes:
        m = consensus_mean(means, exclude=q["book"])
        if m is None:
            continue
        for side, odds in (("HOME", q["home_odds"]), ("AWAY", q["away_odds"])):
            p_win, p_push, ev = price_offer(m, side, q["home_line"], odds, sigma, total)
            rows.append({
                "event_id": event.get("id"), "commence_time": event.get("commence_time"),
                "home_team": home, "away_team": away, "book": q["book"], "side": side,
                "home_line": q["home_line"], "odds": odds, "book_mean": means[q["book"]],
                "consensus_mean": m, "consensus_home_line": cons_line, "total_line": total,
                "p_win": p_win, "p_push": p_push, "ev": ev, "stale": ev >= stale_ev,
                "key_cross": crosses_key(side, q["home_line"], cons_line),
                "n_books": len(quotes),
            })
    return rows
