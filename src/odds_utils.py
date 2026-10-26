import os
import json
import time
import requests
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
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(CACHE_DIR, f"odds_raw_{today}.json")

    # Check Cache
    if not force_refresh and os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < max_age_seconds:
            print("Using cached Odds API data.")
            with open(cache_file, "r") as f:
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

        with open(cache_file, "w") as f:
            json.dump(data, f)

        remaining = response.headers.get("x-requests-remaining", "unknown")
        print(f"Remaining API requests: {remaining}")
        return data
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return None


def parse_spreads(api_data):
    """Extracts spreads and converts team names to nflverse abbreviations."""
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

        target_book = choose_bookmaker(event.get("bookmakers", []))
        if not target_book:
            continue

        spreads_market = next(
            (m for m in target_book.get("markets", []) if m.get("key") == "spreads"),
            None,
        )
        if not spreads_market:
            continue

        outcomes = {o["name"]: o for o in spreads_market.get("outcomes", [])}

        if raw_home in outcomes and raw_away in outcomes:
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
            })

    return pd.DataFrame(results)
