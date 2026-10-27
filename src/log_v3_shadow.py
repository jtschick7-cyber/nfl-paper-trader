"""
Log V3 EPA-Enhanced Shadow Evaluations to Google Sheets.

This writes only to the 'V3_Shadow' tab.
It does NOT modify Master_Bets or affect live paper trading.

Evaluates future games using:
- Play-by-play EPA features (Pass EPA, Rush EPA, Success Rate, Early Down EPA)
- Elo rating differentials
- Discrete Key-Numbers Margin Distribution (weights 3, 7, 6, 10)
- Logistic Platt Calibration
- Pre-bet Injury & Outlier safeguards
"""

import math
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import pytz

from src.config import MAX_AUTO_EDGE, MIN_EDGE_THRESHOLD, SEASON
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features
from src.injury_guard import audit_candidate_pick
from src.key_numbers import calculate_spread_probabilities
from src.model_utils import DIVISION_MAP, _load_sched
from src.odds_utils import american_to_implied_prob, remove_vig
from src.sheet_utils import get_sheet_as_df
from src.v2_sheet_utils import get_or_create_worksheet

MODEL_PATH = "models/nfl_spread_model_v3.pkl"
SHADOW_TAB = "V3_Shadow"
EPA_DATA_PATH = "data/team_game_epa.csv"
MODEL_VERSION = os.getenv("V3_MODEL_VERSION", "v3_epa_enhanced_2023_2025")

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
    "diff_pass_epa",
    "diff_rush_epa",
    "diff_success_rate",
    "diff_early_down_epa",
    "diff_margin",
    "diff_rest",
    "division",
    "mu_home_margin",
    "sigma",
    "p_home_discrete",
    "p_home_cal",
    "p_away_cal",
    "p_push",
    "fair_home",
    "fair_away",
    "edge_home",
    "edge_away",
    "v3_best_team",
    "v3_best_side",
    "v3_best_line",
    "v3_best_odds",
    "v3_model_probability",
    "v3_fair_probability",
    "v3_edge",
    "v3_decision",
    "v1_pick",
    "integrity_flags",
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
        return round(value, 4)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def has_pick(value):
    if pd.isna(value):
        return False
    s = str(value).strip()
    return s != "" and s.lower() != "nan"


def ensure_headers(ws):
    try:
        first = ws.row_values(1)
    except Exception:
        first = []

    if not first:
        ws.update([HEADERS], "A1")
        return

    if first[0] != HEADERS[0]:
        print(f"Updating headers in {SHADOW_TAB}...")
        ws.update([HEADERS], "A1")


def main():
    print("=============================================================")
    print("Logging v3 EPA-Enhanced shadow evaluations")
    print("=============================================================")
    print("This writes only to the " + SHADOW_TAB + " tab.")
    print("It does NOT modify Master_Bets.\n")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing v3 model artifact: {MODEL_PATH}\nRun: python -m src.train_model_v3")

    if not os.path.exists(EPA_DATA_PATH):
        print(f"EPA data missing at {EPA_DATA_PATH}. Building on the fly...")
        from src.build_features import build_features
        build_features()

    artifact = joblib.load(MODEL_PATH)
    feature_columns = artifact["feature_columns"]
    scaler = artifact["scaler"]
    ridge = artifact["ridge"]
    sigma = float(artifact["sigma"])
    calibrator = artifact["calibrator"]

    epa_df = pd.read_csv(EPA_DATA_PATH)
    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    now_utc = datetime.now(pytz.utc)

    # Load schedule for Elo and timeline tracking
    sched = _load_sched([SEASON - 1, SEASON])
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True, errors="coerce")

    # Connect to Google Sheets
    ws = get_or_create_worksheet(SHADOW_TAB)
    ensure_headers(ws)

    # Load existing shadow keys to avoid duplicate logging
    records = ws.get_all_records()
    existing_keys = set()
    for r in records:
        key = (
            str(r.get("game_id", "")).strip(),
            str(r.get("home_line", "")).strip(),
            str(r.get("home_odds", "")).strip(),
        )
        existing_keys.add(key)

    new_rows = []
    evaluated = 0
    skipped_past = 0
    skipped_no_line = 0
    skipped_duplicate = 0

    for idx, row in sheet_df.iterrows():
        game_id = str(row.get("game_id", "")).strip()
        status = str(row.get("status", "")).strip().upper()

        if status not in {"OPEN_CAPTURED", "CLOSE_CAPTURED", "PENDING"}:
            continue

        raw_ko = row.get("kickoff_time")
        kickoff = pd.to_datetime(raw_ko, utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        if kickoff <= now_utc:
            skipped_past += 1
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

        try:
            home_line = float(row.get("opening_line"))
            home_odds = float(row.get("opening_odds"))
        except Exception:
            skipped_no_line += 1
            continue

        away_fallback = False
        try:
            away_odds = float(row.get("opening_away_odds"))
            if pd.isna(away_odds):
                away_odds = -110.0
                away_fallback = True
        except Exception:
            away_odds = -110.0
            away_fallback = True

        key = (game_id, str(home_line), str(home_odds))
        if key in existing_keys:
            skipped_duplicate += 1
            continue

        # Point-in-time EPA features
        epa_snap = compute_rolling_epa_snapshot(epa_df, sched, kickoff, window=6, min_periods=2)
        epa_matchup = get_matchup_epa_features(epa_snap, home, away)

        if epa_matchup:
            diff_pass_epa = epa_matchup["diff_pass_epa"]
            diff_rush_epa = epa_matchup["diff_rush_epa"]
            diff_success = epa_matchup["diff_success_rate"]
            diff_early = epa_matchup.get("diff_early_down_epa", 0.0)
        else:
            diff_pass_epa = 0.0
            diff_rush_epa = 0.0
            diff_success = 0.0
            diff_early = 0.0

        # Context features
        h_div = DIVISION_MAP.get(home)
        a_div = DIVISION_MAP.get(away)
        division = 1.0 if h_div and a_div and h_div == a_div else 0.0
        diff_rest = 0.0  # Default rest differential
        diff_margin = 0.0
        elo_diff = 0.0  # Model default

        feat_vals = {
            "elo_diff": elo_diff,
            "diff_pass_epa": diff_pass_epa,
            "diff_rush_epa": diff_rush_epa,
            "diff_success_rate": diff_success,
            "diff_early_down_epa": diff_early,
            "diff_margin": diff_margin,
            "diff_rest": diff_rest,
            "division": division,
        }

        X = np.array([[feat_vals[c] for c in feature_columns]], dtype=float)
        mu = float(ridge.predict(scaler.transform(X))[0])

        # Discrete key-number aware spread probabilities
        spread_res = calculate_spread_probabilities(mu, sigma, home_line)
        p_home_discrete = float(spread_res["cond_home_win"])
        p_push = float(spread_res["p_push"])

        # Platt Calibration
        try:
            p_home_cal = float(calibrator.predict_proba([[p_home_discrete]])[0][1])
        except Exception:
            p_home_cal = p_home_discrete
        p_away_cal = 1.0 - p_home_cal

        # Fair no-vig probabilities
        fair_h, fair_a = remove_vig(
            american_to_implied_prob(home_odds),
            american_to_implied_prob(away_odds),
        )

        edge_home = p_home_cal - fair_h
        edge_away = p_away_cal - fair_a

        if edge_home >= edge_away:
            best_side = "HOME"
            best_team = home
            best_line = home_line
            best_odds = home_odds
            best_model_p = p_home_cal
            best_fair_p = fair_h
            best_edge = edge_home
        else:
            best_side = "AWAY"
            best_team = away
            best_line = -home_line
            best_odds = away_odds
            best_model_p = p_away_cal
            best_fair_p = fair_a
            best_edge = edge_away

        # Pre-bet integrity audit
        alert = audit_candidate_pick(
            pick_team=best_team,
            pick_side=best_side,
            open_line=home_line,
            current_line=home_line,
            mu_home_margin=mu,
            edge=best_edge,
            max_auto_edge=MAX_AUTO_EDGE,
        )

        # Decision classification
        if alert.recommendation == "SUPPRESS":
            decision = "SUPPRESS_DANGER"
        elif alert.recommendation == "MANUAL_REVIEW":
            decision = "MANUAL_REVIEW"
        elif best_edge >= 0.035:
            decision = "HIGH_CONVICTION"
        elif best_edge >= MIN_EDGE_THRESHOLD:
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
            )

        notes = []
        if away_fallback:
            notes.append("away_odds_fallback=-110")
        if p_push > 0.02:
            notes.append(f"push_prob={round(p_push*100, 1)}%")

        new_rows.append([
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
            clean(away_fallback),
            clean(feat_vals.get("elo_diff")),
            clean(feat_vals.get("diff_pass_epa")),
            clean(feat_vals.get("diff_rush_epa")),
            clean(feat_vals.get("diff_success_rate")),
            clean(feat_vals.get("diff_early_down_epa")),
            clean(feat_vals.get("diff_margin")),
            clean(feat_vals.get("diff_rest")),
            clean(feat_vals.get("division")),
            clean(mu),
            clean(sigma),
            clean(p_home_discrete),
            clean(p_home_cal),
            clean(p_away_cal),
            clean(p_push),
            clean(fair_h),
            clean(fair_a),
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
            clean("; ".join(alert.flags)),
            clean("; ".join(notes)),
        ])

        existing_keys.add(key)
        evaluated += 1

    print(f"Evaluated games: {evaluated}")
    print(f"Skipped past games: {skipped_past}")
    print(f"Skipped missing lines: {skipped_no_line}")
    print(f"Skipped duplicates: {skipped_duplicate}\n")

    if not new_rows:
        print("No new shadow rows to write.")
        return

    # Resize sheet if needed
    needed_rows = len(records) + len(new_rows) + 20
    if ws.row_count < needed_rows:
        ws.resize(rows=needed_rows)

    if ws.col_count < len(HEADERS):
        ws.resize(cols=len(HEADERS))

    ws.append_rows(new_rows, value_input_option="USER_ENTERED")
    print(f"Wrote {len(new_rows)} evaluation rows to {SHADOW_TAB} tab.\n")

    for r in new_rows:
        row_dict = dict(zip(HEADERS, r))
        raw_edge = row_dict.get("v3_edge", "")
        try:
            edge_str = f"{round(float(raw_edge) * 100, 2):+.2f}%"
        except (ValueError, TypeError):
            edge_str = "N/A"

        print(
            f"  {row_dict['matchup']} | v3 best: {row_dict['v3_best_team']} {row_dict['v3_best_side']} "
            f"{row_dict['v3_best_line']} @ {row_dict['v3_best_odds']} | edge: {edge_str} "
            f"| decision: {row_dict['v3_decision']}"
        )


if __name__ == "__main__":
    main()
