import math
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytz
from scipy.stats import norm

from src.config import MAX_AUTO_EDGE, MIN_EDGE_THRESHOLD, SEASON
from src.injury_guard import audit_candidate_pick
from src.key_numbers import calculate_spread_probabilities
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
from src.v2_sheet_utils import get_or_create_worksheet

MODEL_PATH = "models/nfl_spread_model_v2.pkl"
SHADOW_TAB = "V2_Shadow"
MODEL_VERSION = os.getenv("V2_MODEL_VERSION", "v2_baseline_1999_2024")

HEADERS = [
    "model_version",
    "logged_at_utc",
    "season",
    "week",
    "game_id",
    "matchup",
    "kickoff_utc",
    "sheet_status",
    "home_team",
    "away_team",
    "home_line",
    "home_odds",
    "away_odds",
    "away_fallback_used",
    "elo_diff",
    "diff_pf",
    "diff_pa",
    "diff_margin",
    "diff_rest",
    "division",
    "mu_home_margin",
    "sigma",
    "calibration_method",
    "p_home_raw",
    "p_home_cal",
    "p_away_cal",
    "fair_home",
    "fair_away",
    "edge_home",
    "edge_away",
    "v2_best_team",
    "v2_best_side",
    "v2_best_line",
    "v2_best_odds",
    "v2_model_probability",
    "v2_fair_probability",
    "v2_edge",
    "v2_decision",
    "v1_pick",
    "notes",
]


def clean(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float):
        if not math.isfinite(value):
            return ""

        return round(value, 6)

    if isinstance(value, (np.floating,)):
        return round(float(value), 6)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    return value


def has_pick(value):
    if pd.isna(value):
        return False

    s = str(value).strip()

    if s == "":
        return False

    if s.lower() == "nan":
        return False

    return True


def ensure_headers(ws):
    try:
        first = ws.row_values(1)
    except Exception:
        first = []

    if not first:
        ws.update([HEADERS], "A1")
        return

    if first[0] != HEADERS[0]:
        raise SystemExit(
            "V2_Shadow tab exists but headers do not match expected headers.\n"
            + "Expected first header: "
            + HEADERS[0]
            + "\n"
            + "Found first header: "
            + str(first[0])
        )


def main():
    print("=============================================================")
    print("Logging v2 shadow evaluations")
    print("=============================================================")
    print("This writes only to the " + SHADOW_TAB + " tab.")
    print("It does NOT modify Master_Bets.")
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
    calibration_method = artifact["calibrator_method"]
    calibration_params = artifact["calibrator_params"]

    calibrator = (calibration_method, calibration_params)

    print("Model version: " + MODEL_VERSION)
    print("Sigma: " + str(round(sigma, 4)))
    print("Calibration method: " + str(calibration_method))
    print("Calibration params: " + str(calibration_params))
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

    ws = get_or_create_worksheet(SHADOW_TAB)
    ensure_headers(ws)

    records = ws.get_all_records()

    existing_keys = set()

    for r in records:
        mv = str(r.get("model_version", "")).strip()
        gid = str(r.get("game_id", "")).strip()

        if mv and gid:
            existing_keys.add(mv + "|" + gid)

    print("Existing shadow rows: " + str(len(records)))
    print("")

    new_rows = []

    skipped_past = 0
    skipped_status = 0
    skipped_no_line = 0
    skipped_duplicate = 0
    skipped_bad_team = 0
    skipped_bad_odds = 0
    skipped_missing_features = 0
    evaluated = 0

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status != "OPEN_CAPTURED":
            skipped_status += 1
            continue

        if pd.isna(row.get("opening_line")):
            skipped_no_line += 1
            continue

        kickoff = pd.to_datetime(row.get("kickoff_time"), utc=True, errors="coerce")

        if pd.isna(kickoff):
            skipped_past += 1
            continue

        if kickoff <= now_utc:
            skipped_past += 1
            continue

        game_id = str(row.get("game_id", "")).strip()
        key = MODEL_VERSION + "|" + game_id

        if key in existing_keys:
            skipped_duplicate += 1
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

        if not home or not away or home == "NAN" or away == "NAN":
            skipped_bad_team += 1
            continue

        try:
            home_line = float(row.get("opening_line"))
            home_odds = float(row.get("opening_odds"))
        except Exception:
            skipped_bad_odds += 1
            continue

        away_odds_raw = row.get("opening_away_odds")

        away_fallback_used = (
            pd.isna(away_odds_raw)
            or str(away_odds_raw).strip() == ""
            or str(away_odds_raw).strip().lower() == "nan"
        )

        away_odds = safe_float(away_odds_raw, -110.0)

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
            skipped_missing_features += 1
            continue

        X = np.array([[feats[c] for c in feature_columns]], dtype=float)

        mu = float(ridge.predict(scaler.transform(X))[0])

        # Discrete key-number aware win probability (weights spikes at 3, 7, 6, 10)
        spread_res = calculate_spread_probabilities(mu, sigma, home_line)
        p_home_discrete = float(spread_res["cond_home_win"])

        p_home_cal = float(calibrate_probability(calibrator, p_home_discrete))
        p_away_cal = 1.0 - p_home_cal

        fair_home, fair_away = remove_vig_scalar(
            american_to_implied_prob_scalar(home_odds),
            american_to_implied_prob_scalar(away_odds),
        )

        edge_home = p_home_cal - fair_home
        edge_away = p_away_cal - fair_away

        if edge_home >= edge_away:
            best_side = "HOME"
            best_team = home
            best_line = home_line
            best_odds = home_odds
            best_model_p = p_home_cal
            best_fair_p = fair_home
            best_edge = edge_home
        else:
            best_side = "AWAY"
            best_team = away
            best_line = -home_line
            best_odds = away_odds
            best_model_p = p_away_cal
            best_fair_p = fair_away
            best_edge = edge_away

        if not math.isfinite(best_edge):
            skipped_missing_features += 1
            continue

        alert = audit_candidate_pick(
            pick_team=best_team,
            pick_side=best_side,
            open_line=home_line,
            current_line=home_line,
            mu_home_margin=mu,
            edge=best_edge,
            max_auto_edge=MAX_AUTO_EDGE,
        )

        if best_edge >= MIN_EDGE_THRESHOLD:
            if alert.recommendation == "SUPPRESS":
                decision = "SUPPRESS_DANGER"
            elif alert.recommendation == "MANUAL_REVIEW":
                decision = "MANUAL_REVIEW"
            else:
                decision = "SHADOW_PICK"
        else:
            decision = "PASS"

        v1_pick = ""

        if has_pick(row.get("pick_team")):
            v1_pick = (
                str(row.get("pick_team", "")).strip().upper()
                + " "
                + str(row.get("pick_side", "")).strip().upper()
                + " "
                + str(row.get("pick_line", "")).strip()
                + " @"
                + str(row.get("pick_odds", "")).strip()
            )

        notes = []

        if away_fallback_used:
            notes.append("away_odds_fallback=-110")

        notes.append("calibration=" + str(calibration_method))
        notes.append("sigma=" + str(round(sigma, 3)))

        new_rows.append(
            [
                clean(MODEL_VERSION),
                clean(now_utc.isoformat()),
                clean(row.get("season", SEASON)),
                clean(row.get("week", "")),
                clean(game_id),
                clean(away + " @ " + home),
                clean(kickoff.isoformat()),
                clean(status),
                clean(home),
                clean(away),
                clean(home_line),
                clean(home_odds),
                clean(away_odds),
                clean(away_fallback_used),
                clean(feats.get("elo_diff")),
                clean(feats.get("diff_pf")),
                clean(feats.get("diff_pa")),
                clean(feats.get("diff_margin")),
                clean(feats.get("diff_rest")),
                clean(feats.get("division")),
                clean(mu),
                clean(sigma),
                clean(calibration_method),
                clean(p_home_raw),
                clean(p_home_cal),
                clean(p_away_cal),
                clean(fair_home),
                clean(fair_away),
                clean(edge_home),
                clean(edge_away),
                clean(best_team),
                clean(best_side),
                clean(best_line),
                clean(best_odds),
                clean(best_model_p),
                clean(best_fair_p),
                clean(best_edge),
                clean(decision),
                clean(v1_pick),
                clean("; ".join(notes)),
            ]
        )

        existing_keys.add(key)
        evaluated += 1

    print("Evaluated new future games: " + str(evaluated))
    print("Skipped past games: " + str(skipped_past))
    print("Skipped non-OPEN_CAPTURED: " + str(skipped_status))
    print("Skipped missing opening line: " + str(skipped_no_line))
    print("Skipped duplicates: " + str(skipped_duplicate))
    print("Skipped bad team: " + str(skipped_bad_team))
    print("Skipped bad odds: " + str(skipped_bad_odds))
    print("Skipped missing features: " + str(skipped_missing_features))
    print("")

    if not new_rows:
        print("No new shadow rows to write.")
        return

    needed_rows = len(records) + len(new_rows) + 20

    if ws.row_count < needed_rows:
        ws.resize(rows=needed_rows)

    if ws.col_count < len(HEADERS):
        ws.resize(cols=len(HEADERS))

    ws.append_rows(new_rows, value_input_option="USER_ENTERED")

    print("Wrote " + str(len(new_rows)) + " new row(s) to " + SHADOW_TAB + ".")

    for r in new_rows:
        row_dict = dict(zip(HEADERS, r))
        print(
            "  "
            + str(row_dict["matchup"])
            + " | v2 best: "
            + str(row_dict["v2_best_team"])
            + " "
            + str(row_dict["v2_best_side"])
            + " "
            + str(row_dict["v2_best_line"])
            + " @"
            + str(row_dict["v2_best_odds"])
            + " | edge "
            + str(round(float(row_dict["v2_edge"]) * 100, 2))
            + "%"
            + " | decision "
            + str(row_dict["v2_decision"])
        )


if __name__ == "__main__":
    main()
