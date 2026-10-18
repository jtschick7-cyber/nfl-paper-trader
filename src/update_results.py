import nfl_data_py as nfl
import pandas as pd
import numpy as np
from src.sheet_utils import get_sheet_as_df, get_master_sheet
from src.config import SEASON, FLAT_BET_SIZE

def settle_bets():
    print("Updating Results...")
    sheet_df = get_sheet_as_df()
    pbp = nfl.import_pbp([SEASON])
    
    # Aggregate scores by game
    scores = pbp.groupby('game_id').agg({
        'home_score': 'max',
        'away_score': 'max'
    }).reset_index()
    
    ws = get_master_sheet()
    headers = ws.row_values(1)
    
    for idx, row in sheet_df.iterrows():
        if row['status'] == 'CLOSE_CAPTURED' and pd.isna(row.get('pick_result')):
            game_scores = scores[scores['game_id'] == row['game_id']]
            if game_scores.empty: continue
            
            h_score = game_scores.iloc[0]['home_score']
            a_score = game_scores.iloc[0]['away_score']
            
            # Determine pick result
            pick_team = row['pick_team']
            pick_line = float(row['pick_line']) # e.g., -2.5 or +3.5
            
            # Calculate margin from perspective of pick_team
            if pick_team == row['home_team']:
                margin = h_score - a_score
            else:
                margin = a_score - h_score
                
            cover = margin + pick_line
            if cover > 0: pick_result = 'WIN'
            elif cover < 0: pick_result = 'LOSS'
            else: pick_result = 'PUSH'
            
            # Calculate CLV
            closing_line = float(row['closing_line']) if not pd.isna(row['closing_line']) else pick_line
            clv_points = closing_line - pick_line # From bettor perspective
            
            # Update Sheet
            sheet_idx = idx + 2
            ws.update_cell(sheet_idx, headers.index('game_result') + 1, f"{row['home_team']} {h_score}, {row['away_team']} {a_score}")
            ws.update_cell(sheet_idx, headers.index('pick_result') + 1, pick_result)
            ws.update_cell(sheet_idx, headers.index('clv_points') + 1, clv_points)
            ws.update_cell(sheet_idx, headers.index('clv_flag') + 1, clv_points > 0)
            ws.update_cell(sheet_idx, headers.index('roi_units') + 1, FLAT_BET_SIZE if pick_result == 'WIN' else -FLAT_BET_SIZE)
            ws.update_cell(sheet_idx, headers.index('status') + 1, 'SETTLED')
            
            print(f"Settled {row['game_id']}: {pick_result}")

if __name__ == "__main__":
    settle_bets()