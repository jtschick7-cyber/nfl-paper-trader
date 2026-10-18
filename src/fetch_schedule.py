import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import pytz
from src.sheet_utils import get_sheet_as_df, get_master_sheet
from src.config import SEASON

def sync_schedule():
    print(f"Fetching NFL schedule for {SEASON}...")
    sched = nfl.import_schedules([SEASON])
    
    sheet_df = get_sheet_as_df()
    existing_ids = set(sheet_df['game_id'].astype(str))
    ws = get_master_sheet()
    
    new_rows = []
    for _, game in sched.iterrows():
        gsis_id = str(game['game_id'])
        if gsis_id not in existing_ids:
            # Convert gameday to UTC ISO string
            # nflverse gameday is usually YYYY-MM-DD, we need a time. 
            # Default to 17:00 UTC (1PM ET) if exact time missing, but schedule usually has exact time.
            kickoff = pd.to_datetime(game['gameday']).tz_localize('UTC')
            
            new_rows.append({
                'season': SEASON,
                'week': game['week'],
                'game_id': gsis_id,
                'kickoff_time': kickoff.isoformat(),
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'status': 'PENDING'
            })
            
    if new_rows:
        print(f"Adding {len(new_rows)} new games to sheet.")
        # Note: Append rows requires list of lists in exact column order
        headers = ws.row_values(1)
        rows_to_append = []
        for r in new_rows:
            row_data = [r.get(h, '') for h in headers]
            rows_to_append.append(row_data)
        ws.append_rows(rows_to_append, value_input_option='USER_ENTERED')
    else:
        print("Schedule is up to date.")

if __name__ == "__main__":
    sync_schedule()