"""
Log V4 Full Feature Matrix Shadow Evaluations to Google Sheets & Local CSV.

Writes to:
- Google Sheet tab 'V4_Shadow' (if credentials available)
- Local CSV: 'data/v4_shadow_picks.csv'

Evaluates upcoming games using:
1. Play-by-play EPA features (Pass EPA, Rush EPA, Success Rate, Early Down EPA)
2. Empirical Bayes Quarterback Shock (k=60 shrinkage)
3. Full-Team (Offense + Defense) Bayesian-Shrunk Fumble Recovery Luck
4. FiveThirtyEight Elo margin differential & rest differential
5. Discrete Key-Numbers Margin Distribution with Totals-Conditioned Margin Variance (sigma_i)
6. Out-of-Sample Log-Odds Platt Calibration
7. Multi-Book Line Shopping and Pre-bet Safeguards
"""

import math
import os
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import pytz

from src.config import GSHEET_CREDENTIALS, MAX_AUTO_EDGE, MIN_EDGE_THRESHOLD, SEASON, SPREADSHEET_ID
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features
from src.injury_guard import audit_candidate_pick
from src.key_numbers import calculate_spread_probabilities
from src.model_utils import DIVISION_MAP, _load_sched
from src.model_v2_core import ConstrainedRidgeModel, calibrate_probability
from src.odds_utils import (
    american_to_implied_prob,
    best_line_note,
    get_cached_consensus_market,
    get_cached_live_home_spread,
    remove_vig,
)
from src.sheet_utils import get_sheet_as_df
from src.staking import cap_slate_stakes, compute_push_aware_stake
from src.train_model_v3 import ELO_HFA, ELO_INITIAL, ELO_K, ELO_REVERSION, update_elo
from src.train_model_v4 import (
    V4_FEATURES,
    compute_point_in_time_qb_shock,
    compute_point_in_time_fumble_luck,
)
from src.v2_sheet_utils import get_or_create_worksheet

MODEL_PATH = "models/nfl_spread_model_v4.pkl"
SHADOW_TAB = "V4_Shadow"
EPA_DATA_PATH = "data/team_game_epa.csv"
QB_DATA_PATH = "data/team_game_qb.csv"
TURNOVER_DATA_PATH = "data/team_game_turnovers.csv"
LOCAL_CSV_PATH = "data/v4_shadow_picks.csv"
MODEL_VERSION = os.getenv("V4_MODEL_VERSION", "v4_epa_qb_luck_totals_2023_2025")

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
    "total_line",
    "home_odds",
    "away_odds",
    "away_fallback_used",
    "elo_diff",
    "diff_pass_epa",
    "diff_rush_epa",
    "diff_success_rate",
    "diff_early_down_epa",
    "diff_qb_shock",
    "diff_fumble_luck_per_gm",
    "diff_margin",
    "diff_rest",
    "division",
    "mu_home_margin",
    "sigma",
    "sigma_totals_adj",
    "p_home_discrete",
    "p_home_cal",
    "p_away_cal",
    "p_push",
    "fair_home",
    "fair_away",
    "edge_home",
    "edge_away",
    "v4_best_team",
    "v4_best_side",
    "v4_best_line",
    "v4_best_odds",
    "v4_model_probability",
    "v4_fair_probability",
    "v4_edge",
    "v4_stake_units",
    "v4_decision",
    "v1_pick",
    "v1_disagreement",
    "guard_recommendation",
    "guard_notes",
]


def _dedup_key(game_id, home_line, home_odds):
    try:
        hl = f"{float(home_line):.2f}"
    except Exception:
        hl = str(home_line).strip()
    try:
        ho = f"{float(home_odds):.1f}"
    except Exception:
        ho = str(home_odds).strip()
    return (str(game_id).strip(), hl, ho)


def ensure_headers(ws):
    if ws.col_count < len(HEADERS):
        ws.resize(cols=len(HEADERS))
    try:
        first = ws.row_values(1)
    except Exception:
        first = []
    if not first:
        ws.update([HEADERS], "A1")
        return
    if first != HEADERS:
        ws.update([HEADERS], "A1")


def get_point_in_time_context(sched_completed, home_team, away_team, kickoff):
    prior_games = sched_completed[sched_completed["kickoff_utc"] < kickoff].copy()

    elo_ratings = {}
    team_last_date = {}
    team_margins = {}
    current_season = None

    for _, g in prior_games.iterrows():
        s = int(g["season"])
        h = str(g["home_team"]).upper().strip()
        a = str(g["away_team"]).upper().strip()
        h_score = float(g["home_score"])
        a_score = float(g["away_score"])
        margin = h_score - a_score
        ko = g["kickoff_utc"]
        local_day = ko.tz_convert("America/New_York").normalize()

        if current_season is not None and s != current_season:
            for t in elo_ratings:
                elo_ratings[t] = elo_ratings[t] * (1.0 - ELO_REVERSION) + ELO_INITIAL * ELO_REVERSION
        current_season = s

        h_elo = elo_ratings.get(h, ELO_INITIAL)
        a_elo = elo_ratings.get(a, ELO_INITIAL)

        new_h, new_a = update_elo(h_elo, a_elo, margin)
        elo_ratings[h] = new_h
        elo_ratings[a] = new_a
        team_last_date[h] = local_day
        team_last_date[a] = local_day
        team_margins.setdefault(h, []).append(margin)
        team_margins.setdefault(a, []).append(-margin)

    h_elo = elo_ratings.get(home_team, ELO_INITIAL)
    a_elo = elo_ratings.get(away_team, ELO_INITIAL)
    elo_diff = ((h_elo + ELO_HFA) - a_elo) / 25.0

    match_day = kickoff.tz_convert("America/New_York").normalize()
    h_last = team_last_date.get(home_team)
    a_last = team_last_date.get(away_team)
    h_rest = min((match_day - h_last).days, 21) if h_last else 7
    a_rest = min((match_day - a_last).days, 21) if a_last else 7
    diff_rest = float(h_rest - a_rest)

    h_m = team_margins.get(home_team, [])
    a_m = team_margins.get(away_team, [])
    h_roll = np.mean(h_m[-4:]) if len(h_m) >= 2 else 0.0
    a_roll = np.mean(a_m[-4:]) if len(a_m) >= 2 else 0.0
    diff_margin = float(h_roll - a_roll)

    past_gids = prior_games["game_id"].astype(str).tolist()
    return elo_diff, diff_margin, diff_rest, past_gids


def ensure_feature_caches_are_current():
    """Builds or refreshes EPA/QB/Turnover caches if missing or if schedule has newer completed weeks."""
    need_build = False
    for p in (EPA_DATA_PATH, QB_DATA_PATH, TURNOVER_DATA_PATH):
        if not os.path.exists(p):
            need_build = True
            break

    if not need_build:
        try:
            epa_df = pd.read_csv(EPA_DATA_PATH)
            curr_epa = epa_df[epa_df["season"] == SEASON]
            max_epa_week = int(curr_epa["week"].max()) if not curr_epa.empty else 0

            sched = _load_sched([SEASON])
            completed = sched.dropna(subset=["home_score", "away_score"])
            max_sched_week = int(completed["week"].max()) if not completed.empty else 0

            if max_sched_week > max_epa_week:
                # Check if feature cache was created or updated recently (< 6 hours)
                mtime = os.path.getmtime(EPA_DATA_PATH)
                if time.time() - mtime > 6 * 3600:
                    need_build = True
        except Exception as e:
            print(f"Warning checking feature caches freshness: {e}")

    if need_build:
        print("Feature caches missing or stale. Running build_features()...")
        from src.build_features import build_features
        build_features()


def run_v4_shadow_evaluations():
    print("=============================================================")
    print("Running Model v4 Shadow Logger (EPA + QB + Luck + Totals)...")
    print("=============================================================")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model v4 artifact not found at {MODEL_PATH}. Train it first via src.train_model_v4")

    ensure_feature_caches_are_current()

    if not os.path.exists(EPA_DATA_PATH):
        raise SystemExit(f"Missing EPA data at {EPA_DATA_PATH}.")

    artifact = joblib.load(MODEL_PATH)
    ridge = artifact.get("ridge") or artifact.get("ridge_model")
    scaler = artifact["scaler"]
    sigma_base = float(artifact["sigma"])
    feature_columns = artifact["feature_columns"]
    calibrator = artifact.get("calibrator", ("identity", None))

    epa_df = pd.read_csv(EPA_DATA_PATH)
    qb_df = pd.read_csv(QB_DATA_PATH) if os.path.exists(QB_DATA_PATH) else None
    turnover_df = pd.read_csv(TURNOVER_DATA_PATH) if os.path.exists(TURNOVER_DATA_PATH) else None
    from src.qb_features import fetch_pregame_starter_overrides
    starter_overrides = fetch_pregame_starter_overrides(SEASON, qb_df=qb_df)

    has_creds = bool((GSHEET_CREDENTIALS or os.getenv("GSHEET_CREDENTIALS")) and (SPREADSHEET_ID or os.getenv("SPREADSHEET_ID")))
    if not has_creds:
        print("Notice: GSHEET_CREDENTIALS / SPREADSHEET_ID not detected. Exiting shadow evaluation.")
        return

    sheet_df = get_sheet_as_df()
    if sheet_df.empty:
        print("Sheet is empty or unavailable.")
        return

    now_utc = datetime.now(pytz.utc)

    # Load historical schedule for Elo and feature calculations
    sched = _load_sched(list(range(2020, SEASON + 1)))
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True, errors="coerce")
    completed_sched = sched.dropna(subset=["home_score", "away_score", "kickoff_utc"]).sort_values("kickoff_utc").reset_index(drop=True)

    # Connect to Google Sheets if credentials are present
    has_creds = bool(os.getenv("GSHEET_CREDENTIALS") and os.getenv("SPREADSHEET_ID"))
    ws = None
    records = []
    existing_keys = set()
    if has_creds:
        try:
            ws = get_or_create_worksheet(SHADOW_TAB)
            ensure_headers(ws)
            records = ws.get_all_records()
            for r in records:
                key = _dedup_key(
                    r.get("game_id", ""),
                    r.get("home_line", ""),
                    r.get("home_odds", ""),
                )
                existing_keys.add(key)
        except Exception as exc:
            print(f"Warning: Could not connect to Google Sheets tab {SHADOW_TAB}: {exc}")

    # Also load local cache to avoid duplicate rows
    if os.path.exists(LOCAL_CSV_PATH):
        try:
            loc_df = pd.read_csv(LOCAL_CSV_PATH)
            for _, r in loc_df.iterrows():
                key = _dedup_key(
                    r.get("game_id", ""),
                    r.get("home_line", ""),
                    r.get("home_odds", ""),
                )
                existing_keys.add(key)
        except Exception:
            pass

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

        total_line = None
        try:
            if "total_line" in row and pd.notna(row.get("total_line")):
                total_line = float(row.get("total_line"))
        except Exception:
            pass

        if total_line is None:
            sched_match = sched[sched["game_id"] == game_id]
            if not sched_match.empty and "total_line" in sched_match.columns:
                t_val = sched_match["total_line"].values[0]
                if pd.notna(t_val):
                    try:
                        total_line = float(t_val)
                    except Exception:
                        pass

        mkt = get_cached_consensus_market(home, away)
        if total_line is None and mkt is not None and mkt.get("total_line") is not None:
            try:
                total_line = float(mkt["total_line"])
            except Exception:
                pass

        away_fallback = False
        try:
            away_odds = float(row.get("opening_away_odds"))
            if pd.isna(away_odds):
                away_odds = -110.0
                away_fallback = True
        except Exception:
            away_odds = -110.0
            away_fallback = True

        key = _dedup_key(game_id, home_line, home_odds)
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
            h_pass_epa = epa_snap.get(home, {}).get("roll_off_pass_epa", 0.0)
            a_pass_epa = epa_snap.get(away, {}).get("roll_off_pass_epa", 0.0)
        else:
            diff_pass_epa = 0.0
            diff_rush_epa = 0.0
            diff_success = 0.0
            diff_early = 0.0
            h_pass_epa = 0.0
            a_pass_epa = 0.0

        # Context features & historical completed games
        h_div = DIVISION_MAP.get(home)
        a_div = DIVISION_MAP.get(away)
        division = 1.0 if h_div and a_div and h_div == a_div else 0.0
        elo_diff, diff_margin, diff_rest, past_gids = get_point_in_time_context(completed_sched, home, away, kickoff)

        # Model v4 QB Shock & Turnover Luck
        diff_qb_shock = compute_point_in_time_qb_shock(qb_df, past_gids, home, away, h_pass_epa, a_pass_epa, starter_overrides=starter_overrides, window=8)
        diff_fumble_luck = compute_point_in_time_fumble_luck(turnover_df, past_gids, home, away, window=6)

        feat_vals = {
            "elo_diff": elo_diff,
            "diff_pass_epa": diff_pass_epa,
            "diff_rush_epa": diff_rush_epa,
            "diff_success_rate": diff_success,
            "diff_early_down_epa": diff_early,
            "diff_qb_shock": diff_qb_shock,
            "diff_fumble_luck_per_gm": diff_fumble_luck,
            "diff_margin": diff_margin,
            "diff_rest": diff_rest,
            "division": division,
        }

        X = np.array([[feat_vals[c] for c in feature_columns]], dtype=float)
        mu = float(ridge.predict(scaler.transform(X))[0])

        # Discrete key-number spread probabilities conditioned on game total
        spread_res = calculate_spread_probabilities(mu, sigma_base, home_line, total_line=total_line)
        p_home_discrete = float(spread_res["cond_home_win"])
        p_push = float(spread_res["p_push"])
        sigma_adj = float(spread_res.get("sigma", sigma_base))

        # Probability calibration
        p_home_cal = float(calibrate_probability(calibrator, p_home_discrete))
        p_away_cal = 1.0 - p_home_cal

        # Market vig removal
        implied_h = american_to_implied_prob(home_odds)
        implied_a = american_to_implied_prob(away_odds)
        fair_h, fair_a = remove_vig(implied_h, implied_a)
        if mkt is not None:
            try:
                if abs(float(mkt["home_spread"]) - home_line) < 1e-4:
                    fair_h = float(mkt["consensus_fair_home"])
                    fair_a = float(mkt["consensus_fair_away"])
            except Exception:
                pass

        edge_h = p_home_cal - fair_h
        edge_a = p_away_cal - fair_a

        if edge_h >= edge_a:
            best_team = home
            best_side = "HOME"
            best_line = home_line
            best_odds = home_odds
            best_model_prob = p_home_cal
            best_fair_prob = fair_h
            best_edge = edge_h
        else:
            best_team = away
            best_side = "AWAY"
            best_line = -home_line
            best_odds = away_odds
            best_model_prob = p_away_cal
            best_fair_prob = fair_a
            best_edge = edge_a

        # Push-aware Kelly sizing
        uncond_win_prob = best_model_prob * (1.0 - p_push)
        stake_info = compute_push_aware_stake(uncond_win_prob, p_push, best_odds)
        stake_units = float(stake_info["recommended_units"])

        # Audit through injury and line movement guard
        live_spread = get_cached_live_home_spread(home, away)
        curr_l = float(row.get("current_line")) if pd.notna(row.get("current_line")) else (live_spread if live_spread is not None else home_line)
        alert = audit_candidate_pick(
            pick_team=best_team,
            pick_side=best_side,
            open_line=home_line,
            current_line=curr_l,
            mu_home_margin=mu,
            edge=best_edge,
            max_auto_edge=MAX_AUTO_EDGE,
        )

        if alert.recommendation == "SUPPRESS":
            decision = "SUPPRESS_DANGER"
        elif alert.recommendation == "MANUAL_REVIEW":
            decision = "MANUAL_REVIEW"
        elif best_edge >= 0.035:
            decision = "HIGH_CONVICTION"
        elif MIN_EDGE_THRESHOLD <= best_edge <= MAX_AUTO_EDGE:
            decision = "SHADOW_PICK"
        else:
            decision = "PASS"

        v1_p_team = str(row.get("pick_team", "")).strip().upper()
        if decision in {"SHADOW_PICK", "HIGH_CONVICTION"}:
            if not v1_p_team or v1_p_team == "NAN":
                v1_disagree = "V1_NO_PICK"
            elif v1_p_team != best_team:
                v1_disagree = "DISAGREE_OPPOSITE"
            else:
                v1_disagree = "AGREE"
        else:
            v1_disagree = "N/A"

        if best_side == "HOME":
            shop_note = best_line_note(mkt, "HOME", home_line, home_odds)
        else:
            shop_note = best_line_note(mkt, "AWAY", -home_line, away_odds)

        logged_at = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        matchup = f"{away} @ {home}"
        season_val = int(row.get("season", SEASON))
        week_val = int(row.get("week", 1))

        row_data = [
            MODEL_VERSION,
            logged_at,
            season_val,
            week_val,
            game_id,
            matchup,
            raw_ko,
            status,
            home,
            away,
            round(home_line, 1),
            round(total_line, 1) if total_line else "",
            round(home_odds, 0),
            round(away_odds, 0),
            away_fallback,
            round(elo_diff, 4),
            round(diff_pass_epa, 4),
            round(diff_rush_epa, 4),
            round(diff_success, 4),
            round(diff_early, 4),
            round(diff_qb_shock, 4),
            round(diff_fumble_luck, 4),
            round(diff_margin, 2),
            round(diff_rest, 1),
            round(division, 0),
            round(mu, 2),
            round(sigma_base, 2),
            round(sigma_adj, 2),
            round(p_home_discrete, 4),
            round(p_home_cal, 4),
            round(p_away_cal, 4),
            round(p_push, 4),
            round(fair_h, 4),
            round(fair_a, 4),
            round(edge_h, 4),
            round(edge_a, 4),
            best_team,
            best_side,
            round(best_line, 1),
            round(best_odds, 0),
            round(best_model_prob, 4),
            round(best_fair_prob, 4),
            round(best_edge, 4),
            round(stake_units, 2),
            decision,
            v1_p_team if (v1_p_team and v1_p_team != "NAN") else "NONE",
            v1_disagree,
            alert.recommendation,
            "; ".join(
                (list(alert.flags) if alert.flags else [])
                + ([shop_note] if shop_note else [])
            ) or "NONE",
        ]

        new_rows.append(row_data)
        existing_keys.add(key)
        evaluated += 1

    print(f"Evaluated {evaluated} games (Skipped: {skipped_duplicate} duplicate, {skipped_past} past, {skipped_no_line} no-line).")

    if new_rows:
        # Group active pick rows by week and apply cap_slate_stakes per week
        week_idx = HEADERS.index("week")
        stake_idx = HEADERS.index("v4_stake_units")
        decision_idx = HEADERS.index("v4_decision")
        game_id_idx = HEADERS.index("game_id")

        from collections import defaultdict
        picks_by_week = defaultdict(list)
        for i, r in enumerate(new_rows):
            if r[decision_idx] in {"SHADOW_PICK", "HIGH_CONVICTION"}:
                w = r[week_idx]
                picks_by_week[w].append({
                    "row_idx": i,
                    "game_id": r[game_id_idx],
                    "recommended_units": float(r[stake_idx]),
                })

        for w, w_picks in picks_by_week.items():
            capped_w = cap_slate_stakes(w_picks, max_slate_exposure=10.0)
            for b in capped_w:
                new_rows[b["row_idx"]][stake_idx] = round(float(b["recommended_units"]), 2)

        # Save locally to CSV
        os.makedirs(os.path.dirname(LOCAL_CSV_PATH), exist_ok=True)
        new_df = pd.DataFrame(new_rows, columns=HEADERS)
        if os.path.exists(LOCAL_CSV_PATH):
            new_df.to_csv(LOCAL_CSV_PATH, mode="a", header=False, index=False)
        else:
            new_df.to_csv(LOCAL_CSV_PATH, index=False)
        print(f"Saved {len(new_rows)} Model v4 shadow evaluations to '{LOCAL_CSV_PATH}'.")

        # Save to Google Sheets if connected
        if ws is not None:
            try:
                needed_rows = len(records) + len(new_rows) + 20
                if ws.row_count < needed_rows:
                    ws.resize(rows=needed_rows)
                if ws.col_count < len(HEADERS):
                    ws.resize(cols=len(HEADERS))
                ws.append_rows(new_rows, value_input_option="USER_ENTERED")
                print(f"Logged {len(new_rows)} rows to '{SHADOW_TAB}' tab in Google Sheets.")
            except Exception as exc:
                print(f"Warning: Failed to append to Google Sheets tab {SHADOW_TAB}: {exc}")

    print("Model v4 shadow logging complete.")


if __name__ == "__main__":
    run_v4_shadow_evaluations()
