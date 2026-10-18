# src/train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    print("Training baseline model...")
    if not os.path.exists("data/features_raw.csv"):
        print("Run build_features.py first!")
        return
        
    df = pd.read_csv("data/features_raw.csv")
    # ... feature engineering (rolling averages, rest days, etc.) ...
    # X = df[features], y = df['covered']
    
    # Mock training for structure
    X = pd.DataFrame({'off_epa_diff': np.random.randn(100), 'def_epa_diff': np.random.randn(100)})
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/nfl_spread_model.pkl")
    print("Model saved to models/nfl_spread_model.pkl")