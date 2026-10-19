import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytz
from scipy.stats import norm

from src.config import MAX_AUTO_EDGE, MIN_EDGE_THRESHOLD, SEASON
from src.model_v2_core import (
    HISTORY_SEASONS,
    american_to_implied_prob_scalar,
    build_features_from_games,
    calibrate_probability,
    compute_live_features_from_state,
    load_canonical_games,
    remove_vig_scalar,
    safe_float,
)
from src.sheet_utils import get_sheet_as_df

MODEL_PATH = "models/nfl_spread_model_v2.pkl"
MAX_PICKS = int(os.getenv("MAX_PICKS", "5"))


def has_pick(value):
    if pd.isna(value):
        return False

    s = str(value).strip()

    if s == "":
        return False

    if s.lower() == "nan":
        return False

    return True


def main():
    print("=============================================================")
    print("Generating v2 dry-run picks")
    print("=============================================================")
    print("This script does NOT write to Google Sheets.")
    print("")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            "Missing v2 model artifact: "
            + MODEL_PATH
            + "\nRun: python -m src.train_model_v2"
        )

    artifact = joblib.load(MODEL_PATH)

    feature_columns = artifact["feature_columns"]
    scaler = artifact["scaler"]
    ridge = artifact["ridge"]
    sigma = float(artifact["sigma"])

    calibrator = (
        artifact["calibrator_method"],
        artifact["calibrator_params"],
    )

    print("Model sigma: " + str(round(sigma, 4)))
    print("Calibration method: " + str(calibrator[0]))
    print("Features: " + str(feature_columns))
    print("")

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    now_utc = datetime.now(pytz.utc)

    print("Loading live feature state from completed games...")
    print("History seasons: 1999-2024 plus current season " + str(SEASON))
    print("Locked/excluded season: 2025")
    print("")

    live_seasons = HISTORY_SEASONS + [SEASON]

    games = load_canonical_games(
        live_seasons,
        as_of=now_utc,
        exclude_seasons={2025},
    )

    if games.empty:
        print("No completed games available for live feature state.")
        return

    _, ratings, counts, histories = build_features_from_games(games)

    print("Completed games used for live state: " + str(len(games)))
    print("Teams with Elo state: " + str(len(ratings)))
    print("")

    status = (
        sheet_df.get("status", pd.Series("", index=sheet_df.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )

    opening_line_exists = (
        sheet_df.get("opening_line", pd.Series(np.nan, index=sheet_df.index))
        .notna()
    )

    candidates = sheet_df[(status == "OPEN_CAPTURED") & opening_line_exists]

    if candidates.empty:
        print("No OPEN_CAPTURED games with opening lines found.")
        return

    auto = []
    manual = []

    skipped_past = 0
    skipped_bad_kickoff = 0
    skipped_insufficient_features = 0
    skipped_bad_odds = 0

    for idx, row in candidates.iterrows():
        kickoff = pd.to_datetime(row.get("kickoff_time"), utc=True, errors="coerce")

        if pd.isna(kickoff):
            skipped_bad_kickoff += 1
            continue

        if kickoff <= now_utc:
            skipped_past += 1
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

        if not home or not away or home == "NAN" or away == "NAN":
            continue

        try:
            home_line = float(row.get("opening_line"))
            home_odds = float(row.get("opening_odds"))
        except Exception:
            skipped_bad_odds += 1
            continue

        away_odds = safe_float(row.get("opening_away_odds"), -110.0)
        used_fallback_away_odds = pd.isna(row.get("opening_away_odds")) or str(row.get("opening_away_odds")).strip() == ""

        feats, diag = compute_live_features_from_state(
            ratings,
            counts,
            histories,
            home,
            away,
            kickoff,
        )

        missing_feats = [c for c in feature_columns if pd.isna(feats.get(c))]

        if missing_feats:
            skipped_insufficient_features += 1
            continue

        X = np.array([[feats[c] for c in feature_columns]], dtype=float)

        mu = float(ridge.predict(scaler.transform(X))[0])

        p_home_raw = float(norm.cdf((mu + home_line) / sigma))
        p_home_cal = float(calibrate_probability(calibrator, p_home_raw))
        p_away_cal = 1.0 - p_home_cal

        fair_home, fair_away = remove_vig_scalar(
            american_to_implied_prob_scalar(home_odds),
            american_to_implied_prob_scalar(away_odds),
        )

        edge_home = p_home_cal - fair_home
        edge_away = p_away_cal - fair_away

        if edge_home >= edge_away:
            side = "HOME"
            pick_team = home
            pick_line = home_line
            pick_odds = home_odds
            model_p = p_home_cal
            fair_p = fair_home
            edge = edge_home
        else:
            side = "AWAY"
            pick_team = away
            pick_line = -home_line
            pick_odds = away_odds
            model_p = p_away_cal
            fair_p = fair_away
            edge = edge_away

        if not np.isfinite(edge):
            continue

        existing_v1_pick = ""

        if has_pick(row.get("pick_team")):
            existing_v1_pick = (
                str(row.get("pick_team", "")).strip().upper()
                + " "
                + str(row.get("pick_side", "")).strip().upper()
                + " "
                + str(row.get("pick_line", "")).strip()
                + " @"
                + str(row.get("pick_odds", "")).strip()
            )

        rec = {
            "idx": idx,
            "game_id": row.get("game_id", ""),
            "week": row.get("week", ""),
            "matchup": away + " @ " + home,
            "kickoff_utc": kickoff.isoformat(),
            "pick_team": pick_team,
            "side": side,
            "pick_line": pick_line,
            "pick_odds": pick_odds,
            "model_probability": model_p,
            "fair_probability": fair_p,
            "edge": edge,
            "mu_home_margin": mu,
            "sigma": sigma,
            "home_line": home_line,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "used_fallback_away_odds": used_fallback_away_odds,
            "existing_v1_pick": existing_v1_pick,
            "diag": diag,
        }

        if edge >= MIN_EDGE_THRESHOLD:
            if edge > MAX_AUTO_EDGE:
                manual.append(rec)
            else:
                auto.append(rec)

    auto.sort(key=lambda x: x["edge"], reverse=True)
    manual.sort(key=lambda x: x["edge"], reverse=True)

    print("Skipped past games: " + str(skipped_past))
    print("Skipped bad kickoff times: " + str(skipped_bad_kickoff))
    print("Skipped insufficient features: " + str(skipped_insufficient_features))
    print("Skipped bad odds: " + str(skipped_bad_odds))
    print("")

    if manual:
        print("Manual review only. These v2 edges are above MAX_AUTO_EDGE=" + str(MAX_AUTO_EDGE) + ":")

        for p in manual:
            print(
                "  REVIEW V2: "
                + p["matchup"]
                + " | "
                + p["pick_team"]
                + " "
                + p["side"]
                + " "
                + str(p["pick_line"])
                + " @"
                + str(p["pick_odds"])
                + " | model "
                + str(round(p["model_probability"] * 100, 1))
                + "%"
                + " vs fair "
                + str(round(p["fair_probability"] * 100, 1))
                + "%"
                + " | edge "
                + str(round(p["edge"] * 100, 1))
                + "%"
                + " | mu "
                + str(round(p["mu_home_margin"], 2))
            )

            if p["existing_v1_pick"]:
                print("      existing v1 pick: " + p["existing_v1_pick"])

        print("")

    if not auto:
        print("No v2 auto-band picks cleared MIN_EDGE_THRESHOLD=" + str(MIN_EDGE_THRESHOLD) + ".")
        return

    print("Top v2 dry-run auto-band picks, max " + str(MAX_PICKS) + ":")

    for p in auto[:MAX_PICKS]:
        print(
            "  V2 DRY RUN: "
            + p["matchup"]
            + " | "
            + p["pick_team"]
            + " "
            + p["side"]
            + " "
            + str(p["pick_line"])
            + " @"
            + str(p["pick_odds"])
            + " | model "
            + str(round(p["model_probability"] * 100, 1))
            + "%"
            + " vs fair "
            + str(round(p["fair_probability"] * 100, 1))
            + "%"
            + " | edge "
            + str(round(p["edge"] * 100, 1))
            + "%"
            + " | mu "
            + str(round(p["mu_home_margin"], 2))
        )

        if p["used_fallback_away_odds"]:
            print("      note: away odds fallback -110")

        if p["existing_v1_pick"]:
            print("      existing v1 pick: " + p["existing_v1_pick"])

    print("")
    print("Dry run complete. No Google Sheet writes occurred.")


if __name__ == "__main__":
    main()
