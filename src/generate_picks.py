import pandas as pd
import joblib
from src.sheet_utils import get_sheet_as_df, update_cell
from src.odds_utils import american_to_implied_prob, remove_vig
from src.config import MIN_EDGE_THRESHOLD, FLAT_BET_SIZE

def generate_picks():
    print("Generating weekly picks...")
    sheet_df = get_sheet_as_df()
    model = joblib.load("models/nfl_spread_model.pkl")
    
    # Filter for games with opening lines but no pick yet
    candidates = sheet_df[(sheet_df['status'] == 'OPEN_CAPTURED') & (pd.isna(sheet_df['pick_team']))]
    
    picks = []
    for idx, row in candidates.iterrows():
        # ... get features for this game ...
        # model_prob = model.predict_proba(features)[0][1]
        model_prob = 0.55 # Mock
        
        # Calculate no-vig implied prob
        imp_h, imp_a = american_to_implied_prob(row['opening_odds']), american_to_implied_prob(-110) # Simplified
        imp_h_novig, imp_a_novig = remove_vig(imp_h, imp_a)
        
        edge = model_prob - imp_h_novig # Assuming home pick
        
        if edge >= MIN_EDGE_THRESHOLD:
            picks.append({
                'game_id': row['game_id'],
                'pick_team': row['home_team'],
                'edge': edge,
                'model_prob': model_prob
            })
            
    # Sort by edge and take top 5
    picks = sorted(picks, key=lambda x: x['edge'], reverse=True)[:5]
    
    for pick in picks:
        print(f"Pick: {pick['pick_team']} | Edge: {pick['edge']:.3f}")
        # Update sheet with pick details...

if __name__ == "__main__":
    generate_picks()