# src/build_features.py
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os

def build_features(seasons):
    print("Building features from PBP data...")
    pbp = nfl.import_pbp(seasons)
    
    # Filter out non-regular season and non-offense plays
    pbp = pbp[pbp['season_type'] == 'REG']
    pbp = pbp[pbp['play_type'].isin(['pass', 'run'])]
    
    # Calculate EPA per game per team
    game_epa = pbp.groupby(['game_id', 'posteam']).agg({
        'epa': 'mean',
        'success': 'mean'
    }).reset_index()
    
    # Pivot to get home/away
    # ... (simplified for brevity, in reality you merge with schedule to get home/away context)
    # Save to data/features.csv
    os.makedirs("data", exist_ok=True)
    game_epa.to_csv("data/features_raw.csv", index=False)
    print("Features saved to data/features_raw.csv")