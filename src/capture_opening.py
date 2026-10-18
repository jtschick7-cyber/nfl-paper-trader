from src.odds_utils import fetch_odds_api, parse_spreads
from src.sheet_utils import get_sheet_as_df, get_master_sheet
from datetime import datetime
import pytz

def capture_opening():
    print("Running Tuesday Opening Capture...")
    sheet_df = get_sheet_as_df()
    odds_data = fetch_odds_api()
    spreads = parse_spreads(odds_data)
    
    if spreads.empty:
        print("No odds data available.")
        return

    ws = get_master_sheet()
    headers = ws.row_values(1)
    updates = []
    
    # Match odds to sheet rows by home/away teams and week
    for idx, row in sheet_df.iterrows():
        if row['status'] == 'PENDING' and pd.isna(row.get('opening_line')):
            # Find matching odds
            match = spreads[
                (spreads['home_team'].str.lower().str.replace(' ', '') == str(row['home_team']).lower().replace(' ', '')) &
                (spreads['away_team'].str.lower().str.replace(' ', '') == str(row['away_team']).lower().replace(' ', ''))
            ]
            
            if not match.empty:
                odds_row = match.iloc[0]
                updates.append({
                    'row': idx + 2, # +1 for 0-index, +1 for header
                    'opening_line': odds_row['home_spread'] if odds_row['home_team'] == row['home_team'] else odds_row['away_spread'],
                    'opening_odds': odds_row['home_odds'] if odds_row['home_team'] == row['home_team'] else odds_row['away_odds'],
                    'opening_captured_at': datetime.now(pytz.utc).isoformat(),
                    'status': 'OPEN_CAPTURED'
                })
                
    for upd in updates:
        ws.update_cell(upd['row'], headers.index('opening_line') + 1, upd['opening_line'])
        ws.update_cell(upd['row'], headers.index('opening_odds') + 1, upd['opening_odds'])
        ws.update_cell(upd['row'], headers.index('opening_captured_at') + 1, upd['opening_captured_at'])
        ws.update_cell(upd['row'], headers.index('status') + 1, upd['status'])
        
    print(f"Captured opening lines for {len(updates)} games.")

if __name__ == "__main__":
    capture_opening()