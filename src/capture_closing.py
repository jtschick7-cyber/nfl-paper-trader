from datetime import datetime, timedelta
import pytz
import pandas as pd
from src.sheet_utils import get_sheet_as_df, get_master_sheet
from src.odds_utils import fetch_odds_api, parse_spreads
import time

def capture_closing():
    print("Running Closing Line Monitor...")
    sheet_df = get_sheet_as_df()
    now_utc = datetime.now(pytz.utc)
    
    ws = get_master_sheet()
    headers = ws.row_values(1)
    
    # Only fetch odds if we actually have a game in the window
    needs_closing = []
    for idx, row in sheet_df.iterrows():
        if row['status'] in ['PENDING', 'OPEN_CAPTURED'] and pd.isna(row.get('closing_line')):
            kickoff = pd.to_datetime(row['kickoff_time'], utc=True)
            time_to_kickoff = kickoff - now_utc
            
            if timedelta(minutes=40) <= time_to_kickoff <= timedelta(minutes=60):
                needs_closing.append(row)
                
    if not needs_closing:
        print("No games in closing line window.")
        return
        
    print(f"Fetching closing lines for {len(needs_closing)} games...")
    odds_data = fetch_odds_api()
    spreads = parse_spreads(odds_data)
    
    for row in needs_closing:
        match = spreads[
            (spreads['home_team'].str.lower().str.replace(' ', '') == str(row['home_team']).lower().replace(' ', '')) &
            (spreads['away_team'].str.lower().str.replace(' ', '') == str(row['away_team']).lower().replace(' ', ''))
        ]
        if not match.empty:
            odds_row = match.iloc[0]
            sheet_idx = sheet_df.index[sheet_df['game_id'] == row['game_id']].tolist()[0] + 2
            
            ws.update_cell(sheet_idx, headers.index('closing_line') + 1, odds_row['home_spread'] if odds_row['home_team'] == row['home_team'] else odds_row['away_spread'])
            ws.update_cell(sheet_idx, headers.index('closing_odds') + 1, odds_row['home_odds'] if odds_row['home_team'] == row['home_team'] else odds_row['away_odds'])
            ws.update_cell(sheet_idx, headers.index('closing_captured_at') + 1, datetime.now(pytz.utc).isoformat())
            ws.update_cell(sheet_idx, headers.index('status') + 1, 'CLOSE_CAPTURED')
            print(f"Captured closing line for {row['game_id']}")

if __name__ == "__main__":
    capture_closing()