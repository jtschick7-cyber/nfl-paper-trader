import json
import os

import joblib

from src.model_v2_core import (
    FEATURES,
    HISTORY_SEASONS,
    collect_calibration_evidence,
    load_canonical_games,
    build_features_from_games,
    train_final_artifact,
)


def main():
    print("=============================================================")
    print("Training v2 model artifact")
    print("=============================================================")
    print("History seasons: " + str(HISTORY_SEASONS[0]) + "-" + str(HISTORY_SEASONS[-1]))
    print("Locked holdout: 2025")
    print("Prospective: 2026+")
    print("Features: " + str(FEATURES))
    print("")

    games = load_canonical_games(
        HISTORY_SEASONS,
        exclude_seasons={2025},
    )

    if games.empty:
        raise SystemExit("No historical games loaded.")

    print("Loaded completed historical games: " + str(len(games)))

    games, _, _, _ = build_features_from_games(games)

    print("Building walk-forward calibration evidence...")
    evidence = collect_calibration_evidence(games)

    print("Calibration evidence rows: " + str(len(evidence)))

    print("Training final artifact on 1999-2024...")
    artifact = train_final_artifact(games, evidence)

    os.makedirs("models", exist_ok=True)

    model_path = "models/nfl_spread_model_v2.pkl"
    meta_path = "models/metadata_v2.json"

    joblib.dump(artifact, model_path)

    meta = {
        "feature_columns": artifact["feature_columns"],
        "sigma": artifact["sigma"],
        "calibrator_method": artifact["calibrator_method"],
        "calibrator_params": (
            list(artifact["calibrator_params"])
            if artifact["calibrator_params"] is not None
            else None
        ),
        "trained_on_seasons": artifact["trained_on_seasons"],
        "calibration_evidence_rows": artifact["calibration_evidence_rows"],
        "constants": artifact["constants"],
        "ridge_coefficients": dict(
            zip(
                artifact["feature_columns"],
                [float(x) for x in artifact["ridge"].coef_],
            )
        ),
        "ridge_intercept": float(artifact["ridge"].intercept_),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("")
    print("Saved: " + model_path)
    print("Saved: " + meta_path)
    print("")
    print("Sigma: " + str(round(artifact["sigma"], 4)))
    print("Calibration method: " + str(artifact["calibrator_method"]))
    print("Calibration params: " + str(artifact["calibrator_params"]))
    print("Ridge coefficients: " + str(meta["ridge_coefficients"]))


if __name__ == "__main__":
    main()
