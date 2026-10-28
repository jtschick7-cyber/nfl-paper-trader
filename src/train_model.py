"""
Model v1 Training Pipeline (Point Differential & Rest Model).

Fits a Ridge regression model on historical games using:
- diff_pf (Points For differential)
- diff_pa (Points Against differential)
- diff_margin (Rolling scoring margin differential)
- diff_rest (Eastern calendar rest days differential)
- division (Divisional game indicator)

Outputs:
- models/nfl_spread_model.pkl
- models/metadata.json
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import joblib

from src.config import SEASON
from src.model_utils import build_training_table, _load_sched

MODEL_PATH = "models/nfl_spread_model.pkl"
METADATA_PATH = "models/metadata.json"
DEFAULT_TRAIN_SEASONS = [2023, 2024, 2025]
FEATURE_COLS = ["diff_pf", "diff_pa", "diff_margin", "diff_rest", "division"]


def train_v1_model(seasons=None, alpha=15.0):
    if seasons is None:
        seasons = DEFAULT_TRAIN_SEASONS

    print("=============================================================")
    print(f"Training Model v1 (Raw Points & Rest) on seasons {seasons}...")
    print("=============================================================")

    sched_df = _load_sched(seasons)
    train_df, feature_cols = build_training_table(seasons, sched=sched_df)

    if train_df.empty:
        raise SystemExit(f"No completed training data found for seasons: {seasons}")

    train_df = train_df.dropna(subset=FEATURE_COLS + ["home_margin"]).reset_index(drop=True)
    X = train_df[FEATURE_COLS].values
    y = train_df["home_margin"].values

    print(f"Dataset compiled: {len(train_df)} training games.")

    # Fit Ridge regression
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    sigma = float(np.std(residuals))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    print(f"Fit Complete: R^2 = {r2:.4f} | RMSE = {rmse:.2f} pts | Sigma = {sigma:.2f} pts")
    print(f"Intercept (HFA): {model.intercept_:+.4f}")
    for col, coef in zip(FEATURE_COLS, model.coef_):
        print(f"  {col:<20s}: {coef:+.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to '{MODEL_PATH}'.")

    meta = {
        "model_version": "v1_points_margin_rest",
        "feature_columns": FEATURE_COLS,
        "feature_cols": FEATURE_COLS,
        "sigma": sigma,
        "rmse": rmse,
        "r2": r2,
        "trained_on_seasons": sorted(seasons),
        "train_rows": len(train_df),
        "trained_for_season": SEASON,
        "coefficients": {col: float(coef) for col, coef in zip(FEATURE_COLS, model.coef_)},
        "intercept": float(model.intercept_),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to '{METADATA_PATH}'.")
    print("=============================================================\n")
    return model, meta


def main():
    parser = argparse.ArgumentParser(description="Train Model v1 Baseline")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SEASONS,
        help="Seasons to train on (default: 2023 2024 2025)",
    )
    parser.add_argument("--alpha", type=float, default=15.0, help="Ridge regularization alpha")
    args = parser.parse_args()
    train_v1_model(seasons=args.seasons, alpha=args.alpha)


if __name__ == "__main__":
    main()
