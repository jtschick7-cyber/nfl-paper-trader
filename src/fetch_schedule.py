from datetime import datetime, time as dtime
import pytz
import pandas as pd
import nfl_data_py as nfl
from src.sheet_utils import get_sheet_as_df, get_master_sheet
from src.config import SEASON, LOCAL_TZ

def kickoff_iso(game):
    """Build a true UTC kickoff timestamp from nflverse gameday + gametime (Eastern)."""
    gameday = pd.to_datetime(game['gameday']).date()
    gt = game.get('gametime')
    if pd.notna(gt) and str(gt).strip():
        t = pd.to_datetime(str(gt).strip()).time()
    else:
        t = dtime(13, 0)  # fallback: 1:00 PM Eastern
    local = pytz.timezone(LOCAL_TZ).localize(datetime.combine(gameday, t))
    return local.astimezone(pytz.utc).isoformat()

def sync_schedule():
    print(f"Fetching NFL schedule for {SEASON}...")
    sched = nfl.import_schedules([SEASON])
    print(f"nflverse returned {len(sched)} games for season {SEASON}.")

    sheet_df = get_sheet_as_df()
    if 'game_id' in sheet_df.columns:
        existing_ids = set(sheet_df['game_id'].astype(str))
    else:
        existing_ids = set()

    ws = get_master_sheet()
    headers = ws.row_values(1)

    new_rows = []
    for _, game in sched.iterrows():
        gsis_id = str(game['game_id'])
        if gsis_id in existing_ids:
            continue
        row = {
            'season': int(game['season']),
            'week': int(game['week']),
            'game_id': gsis_id,
            'kickoff_time': kickoff_iso(game),
            'home_team': str(game['home_team']).upper(),
            'away_team': str(game['away_team']).upper(),
            'status': 'PENDING',
        }
        new_rows.append([row.get(h, '') for h in headers])

    if new_rows:
        print(f"Adding {len(new_rows)} new games to sheet.")
        ws.append_rows(new_rows, value_input_option='USER_ENTERED')
    else:
        print("No new games to add.")

if __name__ == "__main__":
    sync_schedule()