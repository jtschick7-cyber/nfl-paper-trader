import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from src.config import ODDS_API_KEY, SPORT_KEY, REGIONS, MARKETS, PRIMARY_BOOK, FALLBACK_BOOK, LOCAL_TZ

CACHE_DIR = "data/odds_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def american_to_implied_prob(odds):
    """Converts American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def remove_vig(prob_home, prob_away):
    """Removes bookmaker margin (vig) from two-way market."""
    total_prob = prob_home + prob_away
    if total_prob == 0: return 0.5, 0.5
    return prob_home / total_prob, prob_away / total_prob

def fetch_odds_api():
    """Fetches raw JSON from Odds API, utilizing cache if < 12 hours old."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(CACHE_DIR, f"odds_raw_{today}.json")
    
    # Check Cache
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < (12 * 3600): # 12 hours
            print("Using cached Odds API data.")
            with open(cache_file, 'r') as f:
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
        "dateFormat": "iso"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Cache raw response
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
        print(f"Remaining API requests: {response.headers.get('x-requests-remaining')}")
        return data
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return None

def parse_spreads(api_data):
    """Extracts spreads for DraftKings/FanDuel from raw API JSON."""
    results = []
    if not api_data: return results

    for event in api_data:
        game_id = event.get('id')
        commence_time = event.get('commence_time')
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        
        # Find Bookmakers
        bookmakers = event.get('bookmakers', [])
        target_book = next((b for b in bookmakers if b['key'] == PRIMARY_BOOK), None)
        if not target_book:
            target_book = next((b for b in bookmakers if b['key'] == FALLBACK_BOOK), None)
        if not target_book and bookmakers:
            target_book = bookmakers[0] # Fallback to first available
            
        if not target_book:
            continue
            
        # Find Spreads Market
        spreads_market = next((m for m in target_book['markets'] if m['key'] == 'spreads'), None)
        if not spreads_market:
            continue
            
        outcomes = {o['name']: o for o in spreads_market['outcomes']}
        
        if home_team in outcomes and away_team in outcomes:
            results.append({
                'game_id': game_id,
                'commence_time': commence_time,
                'home_team': home_team,
                'away_team': away_team,
                'home_spread': outcomes[home_team]['point'],
                'home_odds': outcomes[home_team]['price'],
                'away_spread': outcomes[away_team]['point'],
                'away_odds': outcomes[away_team]['price'],
                'bookmaker': target_book['title']
            })
            
    return pd.DataFrame(results)