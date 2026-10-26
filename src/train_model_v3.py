"""
EPA-Enhanced NFL Spread Model (v3).

Features:
- elo_diff: Multi-season Elo rating differential with home-field advantage
- diff_pass_epa: Net passing efficiency differential (Home pass offense vs Away pass defense - Away pass offense vs Home pass defense)
- diff_rush_epa: Net rushing efficiency differential
- diff_success_rate: Net play-by-play consistency differential (% of plays with positive EPA)
- diff_early_down_epa: Net early-down scheme efficiency
- diff_margin: Rolling 4-game margin differential
- diff_rest: Days of rest advantage
- division: Division game indicator (1 or 0)

Target:
- actual_home_margin: home_score - away_score

Probability Engine:
- Discrete Key-Number Margin Distribution (weights 3, 7, 6, 10, etc.)
- Logistic Platt Calibration
"""

import json
import math
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.config import SEASON
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features
from src.key_numbers import calculate_spread_probabilities
from src.model_utils import DIVISION_MAP, _load_sched

MODEL_PATH = "models/nfl_spread_model_v3.pkl"
META_PATH = "models/metadata_v3.json"
EPA_DATA_PATH = "data/team_game_epa.csv"

V3_FEATURES = [
    "elo_diff",
    "diff_pass_epa",
    "diff_rush_epa",
    "diff_success_rate",
    "diff_early_down_epa",
    "diff_margin",
    "diff_rest",
    "division",
]

# Elo Hyperparameters
ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HFA = 48.0  # ~2.0 points of spread value
ELO_REVERSION = 0.33  # 1/3 reversion to mean between seasons


def update_elo(home_elo, away_elo, actual_margin):
    """Updates team Elo ratings based on game outcome and margin of victory."""
    h_adj = home_elo + ELO_HFA
    exp_home = 1.0 / (1.0 + 10.0 ** ((away_elo - h_adj) / 400.0))

    if actual_margin > 0:
        actual_home = 1.0
    elif actual_margin < 0:
        actual_home = 0.0
    else:
        actual_home = 0.5

    # Margin of victory multiplier
    mov = abs(actual_margin)
    mov_mult = math.log(max(mov, 1) + 1.0) * (2.2 / ((home_elo - away_elo) * 0.001 + 2.2))
    shift = ELO_K * mov_mult * (actual_home - exp_home)

    return home_elo + shift, away_elo - shift


def build_v3_training_dataset(seasons, epa_df, sched_df):
    """
    Builds a leak-free point-in-time training dataset combining Elo,
    rolling play-by-play EPA features, and schedule context.
    """
    sched = sched_df[sched_df["season"].isin(seasons)].copy()
    sched = sched.dropna(subset=["home_score", "away_score", "kickoff_utc"])
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True)
    sched = sched.sort_values("kickoff_utc").reset_index(drop=True)

    # Track Elo ratings and team histories dynamically
    elo_ratings = {}
    team_last_date = {}
    team_margins = {}

    rows = []
    current_season = None

    for _, game in sched.iterrows():
        season = int(game["season"])
        home = str(game["home_team"]).upper().strip()
        away = str(game["away_team"]).upper().strip()
        kickoff = game["kickoff_utc"]
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        actual_margin = home_score - away_score
        spread_line = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else 0.0

        # Handle off-season Elo reversion to mean
        if current_season is not None and season != current_season:
            for t in elo_ratings:
                elo_ratings[t] = elo_ratings[t] * (1.0 - ELO_REVERSION) + ELO_INITIAL * ELO_REVERSION
        current_season = season

        h_elo = elo_ratings.get(home, ELO_INITIAL)
        a_elo = elo_ratings.get(away, ELO_INITIAL)
        elo_diff = (h_elo + ELO_HFA) - a_elo

        # Rest days calculation
        h_last = team_last_date.get(home)
        a_last = team_last_date.get(away)
        h_rest = min((kickoff - h_last).days, 21) if h_last else 7
        a_rest = min((kickoff - a_last).days, 21) if a_last else 7
        diff_rest = float(h_rest - a_rest)

        # Rolling margin calculation (last 4 games)
        h_prev_margins = team_margins.get(home, [])
        a_prev_margins = team_margins.get(away, [])
        h_roll_margin = float(np.mean(h_prev_margins[-4:])) if len(h_prev_margins) >= 2 else 0.0
        a_roll_margin = float(np.mean(a_prev_margins[-4:])) if len(a_prev_margins) >= 2 else 0.0
        diff_margin = h_roll_margin - a_roll_margin

        # Division matchup
        h_div = DIVISION_MAP.get(home)
        a_div = DIVISION_MAP.get(away)
        division = 1.0 if h_div and a_div and h_div == a_div else 0.0

        # Point-in-time EPA features
        epa_snap = compute_rolling_epa_snapshot(epa_df, sched, kickoff, window=6, min_periods=2)
        epa_matchup = get_matchup_epa_features(epa_snap, home, away)

        if epa_matchup is not None:
            diff_pass_epa = epa_matchup["diff_pass_epa"]
            diff_rush_epa = epa_matchup["diff_rush_epa"]
            diff_success = epa_matchup["diff_success_rate"]
            diff_early = epa_matchup.get("diff_early_down_epa", 0.0)
        else:
            diff_pass_epa = 0.0
            diff_rush_epa = 0.0
            diff_success = 0.0
            diff_early = 0.0

        # Record feature row
        rows.append({
            "game_id": game["game_id"],
            "season": season,
            "week": game["week"],
            "kickoff_utc": kickoff,
            "home_team": home,
            "away_team": away,
            "home_line": -spread_line,  # Standard convention: negative = favorite
            "actual_home_margin": actual_margin,
            "elo_diff": elo_diff / 25.0,  # Scaled to points of spread
            "diff_pass_epa": diff_pass_epa,
            "diff_rush_epa": diff_rush_epa,
            "diff_success_rate": diff_success,
            "diff_early_down_epa": diff_early,
            "diff_margin": diff_margin,
            "diff_rest": diff_rest,
            "division": division,
        })

        # Update post-game state (leak-free)
        new_h_elo, new_a_elo = update_elo(h_elo, a_elo, actual_margin)
        elo_ratings[home] = new_h_elo
        elo_ratings[away] = new_a_elo

        team_last_date[home] = kickoff
        team_last_date[away] = kickoff

        team_margins.setdefault(home, []).append(actual_margin)
        team_margins.setdefault(away, []).append(-actual_margin)

    return pd.DataFrame(rows)


def train_v3_model(train_seasons=None):
    if train_seasons is None:
        # Default to historical seasons prior to current season
        train_seasons = [2023, 2024, 2025]

    print(f"Loading EPA data from {EPA_DATA_PATH}...")
    if not os.path.exists(EPA_DATA_PATH):
        raise SystemExit(f"EPA data missing. Run: python -m src.build_features")

    epa_df = pd.read_csv(EPA_DATA_PATH)
    sched_df = _load_sched(train_seasons)

    print(f"Building v3 training dataset for seasons {train_seasons}...")
    dataset = build_v3_training_dataset(train_seasons, epa_df, sched_df)
    print(f"Dataset compiled: {len(dataset)} completed games.")

    X = dataset[V3_FEATURES].values
    y = dataset["actual_home_margin"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge Regression for Margin Prediction
    ridge = Ridge(alpha=15.0)
    ridge.fit(X_scaled, y)

    y_pred = ridge.predict(X_scaled)
    residuals = y - y_pred
    sigma = float(np.std(residuals))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(ridge.score(X_scaled, y))

    print(f"\nModel Fit Summary:")
    print(f"  RMSE: {rmse:.2f} points")
    print(f"  R^2:  {r2:.3f}")
    print(f"  Sigma: {sigma:.2f}")

    print("\nFeature Coefficients (Points of Margin per 1 SD change):")
    coef_dict = {}
    for feat, coef in zip(V3_FEATURES, ridge.coef_):
        coef_dict[feat] = round(float(coef), 4)
        print(f"  {feat:20s}: {coef:+.3f} pts")
    print(f"  {'intercept':20s}: {ridge.intercept_:+.3f} pts")

    # Fit Logistic Platt Calibrator
    # Convert predicted margin to discrete cover probability, then calibrate to binary cover
    probs_raw = []
    covers_actual = []

    for idx, row in dataset.iterrows():
        mu_pred = y_pred[idx]
        home_line = row["home_line"]
        actual_margin = row["actual_home_margin"]

        # Only calibrate on non-push games
        if abs(actual_margin + home_line) < 1e-5:
            continue

        p_res = calculate_spread_probabilities(mu_pred, sigma, home_line)
        probs_raw.append(p_res["cond_home_win"])
        covers_actual.append(1 if (actual_margin + home_line) > 0 else 0)

    log_reg = LogisticRegression()
    X_cal = np.array(probs_raw).reshape(-1, 1)
    y_cal = np.array(covers_actual)
    log_reg.fit(X_cal, y_cal)

    calibrator_params = [float(log_reg.intercept_[0]), float(log_reg.coef_[0][0])]
    print(f"\nLogistic Calibrator: intercept={calibrator_params[0]:.3f}, slope={calibrator_params[1]:.3f}")

    # Save artifact
    os.makedirs("models", exist_ok=True)
    artifact = {
        "model_version": "v3_epa_enhanced",
        "feature_columns": V3_FEATURES,
        "scaler": scaler,
        "ridge": ridge,
        "sigma": sigma,
        "calibrator": log_reg,
        "calibrator_params": calibrator_params,
        "trained_seasons": train_seasons,
    }
    joblib.dump(artifact, MODEL_PATH)

    meta = {
        "model_version": "v3_epa_enhanced",
        "feature_columns": V3_FEATURES,
        "sigma": sigma,
        "rmse": rmse,
        "r2": r2,
        "coefficients": coef_dict,
        "intercept": float(ridge.intercept_),
        "calibrator_params": calibrator_params,
        "train_rows": len(dataset),
        "trained_seasons": train_seasons,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifact saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    train_v3_model()
