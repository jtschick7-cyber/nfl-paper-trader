import os

# API & Bookmakers
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SPORT_KEY = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "spreads"
PRIMARY_BOOK = "draftkings"
FALLBACK_BOOK = "fanduel"

# Google Sheets
GSHEET_CREDENTIALS = os.getenv("GSHEET_CREDENTIALS") # Base64 encoded JSON
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
MASTER_SHEET_NAME = "Master_Bets"

# Betting Parameters
BANKROLL_UNITS = 100
FLAT_BET_SIZE = 1.0
MIN_EDGE_THRESHOLD = 0.02 # 2%

# Timezone
LOCAL_TZ = "America/New_York"
SEASON = 2026
