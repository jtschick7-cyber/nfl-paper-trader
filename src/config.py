import os

ODDS_API_KEY = os.getenv("ODDS_API_KEY")

SPORT_KEY = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "spreads"

PRIMARY_BOOK = "draftkings"
FALLBACK_BOOK = "fanduel"

GSHEET_CREDENTIALS = os.getenv("GSHEET_CREDENTIALS")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
MASTER_SHEET_NAME = "Master_Bets"

SEASON = int(os.getenv("SEASON", "2026"))

BANKROLL_UNITS = float(os.getenv("BANKROLL_UNITS", "100"))
FLAT_BET_SIZE = float(os.getenv("FLAT_BET_SIZE", "1.0"))
MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", "0.02"))
MAX_AUTO_EDGE = float(os.getenv("MAX_AUTO_EDGE", "0.10"))

LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")
