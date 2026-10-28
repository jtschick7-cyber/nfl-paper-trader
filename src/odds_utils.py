import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from src.config import (
    ODDS_API_KEY,
    SPORT_KEY,
    REGIONS,
    MARKETS,
    PRIMARY_BOOK,
    FALLBACK_BOOK,
)

CACHE_DIR = "data/odds_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ODDS_CACHE_FILE = os.path.join(CACHE_DIR, "odds_latest.json")

# Map The Odds API full team names to nflverse abbreviations.
# Note: Rams must be mapped to "LA" to match nflverse schedule.
TEAM_ABBR_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

BOOK_PRIORITY = [
    PRIMARY_BOOK,
    FALLBACK_BOOK,
    "betmgm",
    "caesars",
    "pointsbet",
    "betrivers",
    "fanatics",
    "williamhill_us",
    "bovada",
]


def american_to_implied_prob(odds):
    """Converts American odds to implied probability."""
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def remove_vig(prob_home, prob_away):
    """Removes bookmaker margin (vig) from two-way market."""
    total_prob = prob_home + prob_away
    if total_prob == 0:
        return 0.5, 0.5
    return prob_home / total_prob, prob_away / total_prob


def choose_bookmaker(bookmakers):
    """Picks the highest priority available bookmaker."""
    for wanted_key in BOOK_PRIORITY:
        for book in bookmakers:
            if book.get("key") == wanted_key:
                return book
    return bookmakers[0] if bookmakers else None


def fetch_odds_api(max_age_seconds=12 * 3600, force_refresh=False):
    """
    Fetches raw JSON from Odds API with age-aware caching.
    - Opening capture uses a long cache (e.g. 12h).
    - Closing capture uses a short cache (e.g. 300s) or force_refresh.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    latest_cache_file = os.path.join(CACHE_DIR, "odds_latest.json")
    today = datetime.now().strftime("%Y-%m-%d")
    dated_cache_file = os.path.join(CACHE_DIR, f"odds_raw_{today}.json")

    # Check Cache (use rolling latest cache file so UTC midnight rollover does not invalidate recent fetches)
    target_cache = latest_cache_file if os.path.exists(latest_cache_file) else (dated_cache_file if os.path.exists(dated_cache_file) else None)
    if not force_refresh and target_cache and os.path.exists(target_cache):
        file_age = time.time() - os.path.getmtime(target_cache)
        if file_age < max_age_seconds:
            print("Using cached Odds API data.")
            with open(target_cache, "r") as f:
                return json.load(f)

    # Fetch Fresh
    print("Fetching fresh data from Odds API...")
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY is missing.")

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        with open(latest_cache_file, "w") as f:
            json.dump(data, f)
        with open(dated_cache_file, "w") as f:
            json.dump(data, f)

        remaining = response.headers.get("x-requests-remaining", "unknown")
        print(f"Remaining API requests: {remaining}")
        return data
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return None


def american_to_decimal_odds(american_odds):
    """Converts American odds (-110, +105) to European decimal multiplier (1.909, 2.05)."""
    odds = float(american_odds)
    if odds > 0:
        return 1.0 + (odds / 100.0)
    elif odds < 0:
        return 1.0 + (100.0 / abs(odds))
    return 1.0


def parse_spreads(api_data):
    """
    Extracts spreads and converts team names to nflverse abbreviations.
    Integrates multi-book consensus vig-free fair probabilities and best-line shopping
    across all available sportsbooks in the Odds API response.
    """
    results = []
    if not api_data:
        return pd.DataFrame(results)

    for event in api_data:
        game_id = event.get("id")
        commence_time = event.get("commence_time")

        raw_home = event.get("home_team")
        raw_away = event.get("away_team")

        home_team = TEAM_ABBR_MAP.get(raw_home, raw_home)
        away_team = TEAM_ABBR_MAP.get(raw_away, raw_away)

        all_books = event.get("bookmakers", [])
        if not all_books:
            continue

        target_book = choose_bookmaker(all_books)
        if not target_book:
            continue

        # Extract primary book spreads market
        spreads_market = next(
            (m for m in target_book.get("markets", []) if m.get("key") == "spreads"),
            None,
        )
        if not spreads_market:
            continue

        outcomes = {o["name"]: o for o in spreads_market.get("outcomes", [])}
        if raw_home not in outcomes or raw_away not in outcomes:
            continue

        # Multi-Book Analysis: scan all sportsbooks for consensus pricing and best line
        book_spreads = []
        fair_h_list = []
        fair_a_list = []
        best_home_opt = None  # (score, title, line, price)
        best_away_opt = None  # (score, title, line, price)

        for b in all_books:
            b_title = b.get("title", b.get("key", "Unknown"))
            b_mkt = next((m for m in b.get("markets", []) if m.get("key") == "spreads"), None)
            if not b_mkt:
                continue

            b_outs = {o["name"]: o for o in b_mkt.get("outcomes", [])}
            if raw_home not in b_outs or raw_away not in b_outs:
                continue

            try:
                bh_point = float(b_outs[raw_home]["point"])
                bh_price = float(b_outs[raw_home]["price"])
                ba_point = float(b_outs[raw_away]["point"])
                ba_price = float(b_outs[raw_away]["price"])

                # Book-level vig-free fair probability
                p_h_imp = american_to_implied_prob(bh_price)
                p_a_imp = american_to_implied_prob(ba_price)
                f_h, f_a = remove_vig(p_h_imp, p_a_imp)
                book_spreads.append(bh_point)
                fair_h_list.append(f_h)
                fair_a_list.append(f_a)

                # Line shopping: higher spread point is better; if tied, higher decimal price is better
                h_score = (bh_point, american_to_decimal_odds(bh_price))
                if best_home_opt is None or h_score > best_home_opt[0]:
                    best_home_opt = (h_score, b_title, bh_point, bh_price)

                a_score = (ba_point, american_to_decimal_odds(ba_price))
                if best_away_opt is None or a_score > best_away_opt[0]:
                    best_away_opt = (a_score, b_title, ba_point, ba_price)
            except Exception:
                continue

        # Filter consensus pricing to books matching target spread to avoid mixing probabilities across different numbers
        target_h_spread = float(outcomes[raw_home]["point"])
        matching_fair_h = [fh for sp, fh in zip(book_spreads, fair_h_list) if abs(sp - target_h_spread) < 1e-4]
        matching_fair_a = [fa for sp, fa in zip(book_spreads, fair_a_list) if abs(sp - target_h_spread) < 1e-4]

        consensus_fair_h = float(np.mean(matching_fair_h)) if matching_fair_h else (float(np.mean(fair_h_list)) if fair_h_list else american_to_implied_prob(outcomes[raw_home]["price"]))
        consensus_fair_a = float(np.mean(matching_fair_a)) if matching_fair_a else (float(np.mean(fair_a_list)) if fair_a_list else american_to_implied_prob(outcomes[raw_away]["price"]))

        tot_c = consensus_fair_h + consensus_fair_a
        if tot_c > 0:
            consensus_fair_h /= tot_c
            consensus_fair_a /= tot_c

        results.append({
            "game_id": game_id,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "home_spread": outcomes[raw_home]["point"],
            "home_odds": outcomes[raw_home]["price"],
            "away_spread": outcomes[raw_away]["point"],
            "away_odds": outcomes[raw_away]["price"],
            "bookmaker": target_book.get("title", ""),
            # Multi-Book Consensus & Best-Line Shopping
            "consensus_fair_home": round(consensus_fair_h, 4),
            "consensus_fair_away": round(consensus_fair_a, 4),
            "books_count": len(fair_h_list),
            "best_home_book": best_home_opt[1] if best_home_opt else target_book.get("title", ""),
            "best_home_line": best_home_opt[2] if best_home_opt else outcomes[raw_home]["point"],
            "best_home_odds": best_home_opt[3] if best_home_opt else outcomes[raw_home]["price"],
            "best_away_book": best_away_opt[1] if best_away_opt else target_book.get("title", ""),
            "best_away_line": best_away_opt[2] if best_away_opt else outcomes[raw_away]["point"],
            "best_away_odds": best_away_opt[3] if best_away_opt else outcomes[raw_away]["price"],
        })

    return pd.DataFrame(results)


def team_to_abbr(name: str) -> str:
    """Converts full team name or city alias to standardized nflverse abbreviation."""
    if not name:
        return ""
    clean = str(name).strip()
    return TEAM_ABBR_MAP.get(clean, clean).upper().strip()


def get_cached_live_home_spread(home_team: str, away_team: str):
    """
    Looks up latest live home spread from data/odds_cache/odds_latest.json
    for active RLM auditing. Returns float or None.
    Selects preferred bookmaker matching BOOK_PRIORITY.
    """
    if not os.path.exists(ODDS_CACHE_FILE):
        return None

    try:
        with open(ODDS_CACHE_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None

        h = team_to_abbr(home_team)
        a = team_to_abbr(away_team)

        for g in data:
            raw_h = g.get("home_team", "")
            raw_a = g.get("away_team", "")
            norm_h = team_to_abbr(raw_h)
            norm_a = team_to_abbr(raw_a)

            if norm_h == h and norm_a == a:
                bookmakers = g.get("bookmakers", [])
                if not bookmakers:
                    continue

                preferred_books = []
                target_book = choose_bookmaker(bookmakers)
                if target_book:
                    preferred_books.append(target_book)
                for b in bookmakers:
                    if b not in preferred_books:
                        preferred_books.append(b)

                for b in preferred_books:
                    for m in b.get("markets", []):
                        if m.get("key") == "spreads":
                            for out in m.get("outcomes", []):
                                out_name = out.get("name", "")
                                if out_name == raw_h or team_to_abbr(out_name) == h:
                                    try:
                                        return float(out.get("point"))
                                    except (TypeError, ValueError):
                                        continue
    except Exception:
        pass

    return None
