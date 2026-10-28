"""
Model v4: Quarterback-Attributed & Turnover-Regressed NFL Spread Model.

Architecture & Feature Set:
1. elo_diff: Point-in-time Elo rating differential with home-field advantage
2. diff_pass_epa: Net passing efficiency differential (Home pass offense vs Away pass defense - Away pass offense vs Home pass defense)
3. diff_rush_epa: Net rushing efficiency differential
4. diff_success_rate: Net play-by-play consistency differential (% positive EPA plays)
5. diff_early_down_epa: Net 1st/2nd down early-down scheme efficiency differential
6. diff_margin: Rolling 4-game margin differential
7. diff_rest: Days of rest advantage (Eastern calendar days)
8. division: Divisional matchup indicator (1 or 0)
9. diff_qb_shock: Orthogonal QB shock delta between confirmed starter's shrunk EPA/db (k=60)
   and team's rolling 6-game pass EPA/db, scaled to spread points (* 35 * 0.38)
10. diff_fumble_luck_per_gm: Full-team (offense + defense) Bayesian-shrunk net lucky fumble recoveries per game

Probability & Risk Engines:
- Discrete Key-Number Margin Distribution with Totals-Conditioned Variance (sigma_i)
- Out-of-Sample Walk-Forward Log-Odds Platt Calibration (C=1e6, monotonic guard coef > 0)
"""

import json
import math
import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import SEASON
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features
from src.key_numbers import calculate_spread_probabilities, calculate_totals_conditioned_sigma
from src.model_utils import DIVISION_MAP, _load_sched
from src.model_v2_core import ConstrainedRidgeModel, fit_calibrator, calibrate_probability
from src.qb_features import DEFAULT_LEAGUE_QB_PRIOR, EPA_TO_SPREAD_POINTS, shrink_qb_rating
from src.train_model_v3 import ELO_HFA, ELO_INITIAL, ELO_K, ELO_REVERSION, update_elo
from src.turnover_luck import TURNOVER_PRIOR_RECOVERY_RATE, shrink_turnover_rate

MODEL_PATH = "models/nfl_spread_model_v4.pkl"
META_PATH = "models/metadata_v4.json"
EPA_DATA_PATH = "data/team_game_epa.csv"
QB_DATA_PATH = "data/team_game_qb.csv"
TURNOVER_DATA_PATH = "data/team_game_turnovers.csv"

V4_FEATURES = [
    "elo_diff",
    "diff_pass_epa",
    "diff_rush_epa",
    "diff_success_rate",
    "diff_early_down_epa",
    "diff_margin",
    "diff_rest",
    "division",
    "diff_qb_shock",
    "diff_fumble_luck_per_gm",
]


# Bounds for V4_FEATURES:
# All efficiency, margin, elo, rest, qb features must be >= 0
# division is unconstrained
# diff_fumble_luck_per_gm must be <= 0.0 (non-positive constraint: model never rewards positive turnover luck)
V4_BOUNDS_LB = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, 0.0, -np.inf])
V4_BOUNDS_UB = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.0])


def fit_constrained_ridge(X_scaled, y, alpha=15.0, bounds=(V4_BOUNDS_LB, V4_BOUNDS_UB)):
    """
    Fits Ridge regression with box constraints on coefficients using augmented least squares.
    """
    n_samples, n_features = X_scaled.shape
    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    X_aug = np.vstack([X_scaled, np.sqrt(alpha) * np.eye(n_features)])
    y_aug = np.concatenate([y_centered, np.zeros(n_features)])

    lb, ub = bounds
    res = lsq_linear(X_aug, y_aug, bounds=(lb, ub), lsmr_tol="auto")
    return ConstrainedRidgeModel(res.x, y_mean)


def compute_point_in_time_qb_shock(qb_df, past_gids, home_team, away_team, h_pass_epa, a_pass_epa, starter_overrides=None, window=8):
    """
    Computes diff_qb_shock strictly prior to kickoff.
    Identifies each team's active starting QB from past_gids (the team's most recent completed game),
    unless an explicit starter_overrides dict ({team: qb_id_or_name}) is passed.
    Computes their shrunk EPA/db across prior dropbacks strictly in past_gids (up to window games),
    and measures talent differential relative to the team's 6-game pass EPA baseline.
    """
    if qb_df is None or qb_df.empty or not past_gids:
        return 0.0

    recent_qb = qb_df[qb_df["game_id"].isin(past_gids)].sort_values(["season", "week", "game_id"])
    if recent_qb.empty:
        return 0.0

    shocks = {}
    for team, team_pass_epa in [(home_team, h_pass_epa), (away_team, a_pass_epa)]:
        t_rows = recent_qb[recent_qb["posteam"] == team].sort_values(["season", "week", "game_id"])

        starter_id = None
        starter_name = None

        if starter_overrides and team in starter_overrides:
            override_val = str(starter_overrides[team])
            if "qb_id" in recent_qb.columns and override_val in recent_qb["qb_id"].values:
                starter_id = override_val
            else:
                starter_name = override_val

        if not starter_id and not starter_name:
            if t_rows.empty:
                shocks[team] = 0.0
                continue
            last_gid = list(t_rows["game_id"].unique())[-1]
            last_game_qbs = t_rows[t_rows["game_id"] == last_gid].sort_values("dropbacks", ascending=False)
            if last_game_qbs.empty:
                shocks[team] = 0.0
                continue
            s_row = last_game_qbs.iloc[0]
            starter_id = s_row.get("qb_id")
            starter_name = s_row.get("qb_name")

        # Track starter's prior dropbacks strictly in past_gids across up to window games
        if starter_id and starter_id != "UNKNOWN_ID" and "qb_id" in recent_qb.columns:
            starter_rows = recent_qb[recent_qb["qb_id"] == starter_id].sort_values(["season", "week", "game_id"])
        elif starter_name:
            starter_rows = recent_qb[recent_qb["qb_name"] == starter_name].sort_values(["season", "week", "game_id"])
        else:
            starter_rows = pd.DataFrame()

        if starter_rows.empty:
            shocks[team] = 0.0
            continue

        starter_gids = list(starter_rows["game_id"].unique())[-window:]
        starter_sample = starter_rows[starter_rows["game_id"].isin(starter_gids)]

        n_db = int(starter_sample["dropbacks"].sum())
        if "total_epa" in starter_sample.columns:
            tot_epa = float(starter_sample["total_epa"].sum())
        elif "qb_epa_per_dropback" in starter_sample.columns:
            tot_epa = float((starter_sample["dropbacks"] * starter_sample["qb_epa_per_dropback"]).sum())
        elif "epa_per_dropback" in starter_sample.columns:
            tot_epa = float((starter_sample["dropbacks"] * starter_sample["epa_per_dropback"]).sum())
        else:
            tot_epa = 0.0
        raw_epa = (tot_epa / n_db) if n_db > 0 else DEFAULT_LEAGUE_QB_PRIOR
        starter_shrunk_epa = shrink_qb_rating(raw_epa, n_db, prior_epa=DEFAULT_LEAGUE_QB_PRIOR, k=60.0)

        # Scale difference from team's rolling pass EPA to points of spread
        shock_pts = (starter_shrunk_epa - team_pass_epa) * 35.0 * EPA_TO_SPREAD_POINTS
        shocks[team] = shock_pts

    diff_qb_shock = shocks.get(home_team, 0.0) - shocks.get(away_team, 0.0)
    return round(float(diff_qb_shock), 4)


def compute_point_in_time_fumble_luck(turnover_df, past_gids, home_team, away_team, window=6):
    """
    Computes diff_fumble_luck_per_gm strictly prior to kickoff.
    Tracks full-team (offense + defense) fumble recoveries shrunk toward 50.0% prior.
    """
    if turnover_df is None or turnover_df.empty or not past_gids:
        return 0.0

    recent_to = turnover_df[turnover_df["game_id"].isin(past_gids)].sort_values(["season", "week", "game_id"])
    if recent_to.empty:
        return 0.0

    luck_per_gm = {}
    for team in [home_team, away_team]:
        t_rows = recent_to[recent_to["team"] == team].sort_values(["season", "week", "game_id"])
        if t_rows.empty:
            luck_per_gm[team] = 0.0
            continue

        # Last `window` games sorted chronologically
        t_gids = list(t_rows["game_id"].unique())[-window:]
        w_rows = t_rows[t_rows["game_id"].isin(t_gids)]
        n_games = len(t_gids)

        if "total_fumbles_seen" in w_rows.columns:
            tot_fumbles = int(w_rows["total_fumbles_seen"].sum())
            rec_fumbles = int(w_rows["total_fumbles_recovered"].sum())
        else:
            tot_fumbles = int(w_rows["off_fumbles"].sum())
            rec_fumbles = int(w_rows["off_fumbles_recovered"].sum())

        raw_rate = (rec_fumbles / tot_fumbles) if tot_fumbles > 0 else 0.50
        shrunk_rate = shrink_turnover_rate(raw_rate, tot_fumbles, prior=TURNOVER_PRIOR_RECOVERY_RATE, k=4.0)
        net_luck = (shrunk_rate - 0.50) * tot_fumbles
        luck_per_gm[team] = (net_luck / n_games) if n_games > 0 else 0.0

    diff_luck = luck_per_gm.get(home_team, 0.0) - luck_per_gm.get(away_team, 0.0)
    return round(float(diff_luck), 4)


def build_v4_training_dataset(train_seasons, epa_df, sched_df, qb_df=None, turnover_df=None):
    """
    Builds a leak-free point-in-time training dataset combining Elo,
    rolling play-by-play EPA features, QB attribution shock, and turnover luck.
    """
    # Use full historical schedule to warm up Elo and team histories
    sched = sched_df.dropna(subset=["home_score", "away_score", "kickoff_utc"]).copy()
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True)
    sched = sched.sort_values("kickoff_utc").reset_index(drop=True)

    elo_ratings = {}
    team_last_date = {}
    team_margins = {}
    past_completed_gids = []

    rows = []
    current_season = None

    for _, game in sched.iterrows():
        season = int(game["season"])
        gid = str(game["game_id"])
        home = str(game["home_team"]).upper().strip()
        away = str(game["away_team"]).upper().strip()
        kickoff = game["kickoff_utc"]
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        actual_margin = home_score - away_score
        spread_line = float(game["spread_line"]) if pd.notna(game.get("spread_line")) else 0.0
        total_line = float(game["total_line"]) if pd.notna(game.get("total_line")) else None

        # Handle off-season Elo reversion
        if current_season is not None and season != current_season:
            for t in elo_ratings:
                elo_ratings[t] = elo_ratings[t] * (1.0 - ELO_REVERSION) + ELO_INITIAL * ELO_REVERSION
        current_season = season

        h_elo = elo_ratings.get(home, ELO_INITIAL)
        a_elo = elo_ratings.get(away, ELO_INITIAL)
        elo_diff = (h_elo + ELO_HFA) - a_elo

        # Rest days (Eastern calendar days)
        local_day = kickoff.tz_convert("America/New_York").normalize()
        h_last = team_last_date.get(home)
        a_last = team_last_date.get(away)
        h_rest = min((local_day - h_last).days, 21) if h_last else 7
        a_rest = min((local_day - a_last).days, 21) if a_last else 7
        diff_rest = float(h_rest - a_rest)

        # Rolling margin (last 4 games)
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
            h_pass_epa = epa_snap.get(home, {}).get("roll_off_pass_epa", 0.0)
            a_pass_epa = epa_snap.get(away, {}).get("roll_off_pass_epa", 0.0)
        else:
            diff_pass_epa = 0.0
            diff_rush_epa = 0.0
            diff_success = 0.0
            diff_early = 0.0
            h_pass_epa = 0.0
            a_pass_epa = 0.0

        # Model v4 Orthogonal Signals: QB Shock and Fumble Luck
        diff_qb_shock = compute_point_in_time_qb_shock(
            qb_df, past_completed_gids, home, away, h_pass_epa, a_pass_epa, window=8
        )
        diff_fumble_luck = compute_point_in_time_fumble_luck(
            turnover_df, past_completed_gids, home, away, window=6
        )

        # Record target row only for requested training seasons
        if season in train_seasons:
            rows.append({
                "game_id": gid,
                "season": season,
                "week": game["week"],
                "kickoff_utc": kickoff,
                "home_team": home,
                "away_team": away,
                "home_line": -spread_line,  # Standard: negative = favorite
                "total_line": total_line,
                "actual_home_margin": actual_margin,
                "elo_diff": elo_diff / 25.0,
                "diff_pass_epa": diff_pass_epa,
                "diff_rush_epa": diff_rush_epa,
                "diff_success_rate": diff_success,
                "diff_early_down_epa": diff_early,
                "diff_margin": diff_margin,
                "diff_rest": diff_rest,
                "division": division,
                "diff_qb_shock": diff_qb_shock,
                "diff_fumble_luck_per_gm": diff_fumble_luck,
            })

        # Post-game updates (leak-free)
        new_h_elo, new_a_elo = update_elo(h_elo, a_elo, actual_margin)
        elo_ratings[home] = new_h_elo
        elo_ratings[away] = new_a_elo

        team_last_date[home] = local_day
        team_last_date[away] = local_day

        team_margins.setdefault(home, []).append(actual_margin)
        team_margins.setdefault(away, []).append(-actual_margin)
        past_completed_gids.append(gid)

    return pd.DataFrame(rows)


def collect_v4_calibration_evidence(dataset, features, alpha=15.0):
    """
    Collects out-of-sample raw probabilities across historical training folds.
    Uses chronological TimeSeriesSplit with symmetric HOME/AWAY rows (weight=0.5).
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    rows = []
    X_all = dataset[features].values
    y_all = dataset["actual_home_margin"].values

    for train_idx, val_idx in tscv.split(dataset):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val = X_all[val_idx]
        val_df = dataset.iloc[val_idx]

        scaler_f = StandardScaler().fit(X_tr)
        ridge_f = Ridge(alpha=alpha).fit(scaler_f.transform(X_tr), y_tr)

        resid = y_tr - ridge_f.predict(scaler_f.transform(X_tr))
        sigma_f = float(np.std(resid)) if len(resid) > 1 else 13.2

        mu_val = ridge_f.predict(scaler_f.transform(X_val))

        for j, (_, row) in enumerate(val_df.iterrows()):
            home_line = float(row["home_line"])
            act_margin = float(row["actual_home_margin"])
            tot_line = row.get("total_line")
            diff = act_margin + home_line
            push = abs(diff) < 1e-5
            home_win = diff > 0
            away_win = diff < 0

            sp = calculate_spread_probabilities(mu_val[j], sigma_f, home_line, total_line=tot_line)
            p_h = float(sp["cond_home_win"])
            p_a = 1.0 - p_h

            rows.append({
                "season": int(row["season"]),
                "game_id": row["game_id"],
                "side": "HOME",
                "p_raw": p_h,
                "actual_win": bool(home_win),
                "push": bool(push),
                "weight": 0.5,
            })
            rows.append({
                "season": int(row["season"]),
                "game_id": row["game_id"],
                "side": "AWAY",
                "p_raw": p_a,
                "actual_win": bool(away_win),
                "push": bool(push),
                "weight": 0.5,
            })

    return pd.DataFrame(rows)


def train_v4_model(train_seasons=None):
    if train_seasons is None:
        train_seasons = [2023, 2024, 2025]

    print("=============================================================")
    print(f"Training Model v4 (QB-Attributed & Turnover-Regressed)")
    print(f"Target Seasons: {train_seasons}")
    print("=============================================================\n")

    if not os.path.exists(EPA_DATA_PATH):
        raise SystemExit(f"EPA data missing at {EPA_DATA_PATH}. Run: python -m src.build_features")
    if not os.path.exists(QB_DATA_PATH):
        raise SystemExit(f"QB data missing at {QB_DATA_PATH}. Run: python -m src.build_features")
    if not os.path.exists(TURNOVER_DATA_PATH):
        raise SystemExit(f"Turnover data missing at {TURNOVER_DATA_PATH}. Run: python -m src.build_features")

    epa_df = pd.read_csv(EPA_DATA_PATH)
    qb_df = pd.read_csv(QB_DATA_PATH)
    turnover_df = pd.read_csv(TURNOVER_DATA_PATH)

    # Load historical schedules starting from 2020 for Elo warm-up
    all_sched_seasons = sorted(list(set(range(2020, max(train_seasons) + 1))))
    sched_df = _load_sched(all_sched_seasons)

    print("Compiling Model v4 training dataset...")
    dataset = build_v4_training_dataset(train_seasons, epa_df, sched_df, qb_df, turnover_df)
    print(f"Dataset compiled: {len(dataset)} completed games.\n")

    X = dataset[V4_FEATURES].values
    y = dataset["actual_home_margin"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # L2 Ridge Regression for Margin Prediction
    ridge = Ridge(alpha=15.0)
    ridge.fit(X_scaled, y)

    y_pred = ridge.predict(X_scaled)
    residuals = y - y_pred
    base_sigma = float(np.std(residuals))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1.0 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2))

    print("Model Fit Summary:")
    print(f"  RMSE:       {rmse:.2f} points")
    print(f"  R^2:        {r2:.3f}")
    print(f"  Base Sigma: {base_sigma:.2f}")

    print("\nFeature Coefficients (Points of Margin per 1 SD change):")
    coef_dict = {}
    for feat, coef in zip(V4_FEATURES, ridge.coef_):
        coef_dict[feat] = round(float(coef), 4)
        print(f"  {feat:25s}: {coef:+.3f} pts")
    print(f"  {'intercept':25s}: {ridge.intercept_:+.3f} pts")

    # Fit Out-of-Sample Walk-Forward Log-Odds Calibrator
    evidence_df = collect_v4_calibration_evidence(dataset, V4_FEATURES, alpha=15.0)
    calibrator = fit_calibrator(evidence_df)
    calibrator_method, calibrator_params = calibrator
    print(f"\nCalibrator: method={calibrator_method}, params={calibrator_params}")

    # Save artifact
    os.makedirs("models", exist_ok=True)
    artifact = {
        "model_version": "v4_qb_turnover_enhanced",
        "feature_columns": V4_FEATURES,
        "scaler": scaler,
        "ridge": ridge,
        "sigma": base_sigma,
        "calibrator": calibrator,
        "calibrator_method": calibrator_method,
        "calibrator_params": calibrator_params,
        "trained_seasons": train_seasons,
    }
    joblib.dump(artifact, MODEL_PATH)

    meta = {
        "model_version": "v4_qb_turnover_enhanced",
        "feature_columns": V4_FEATURES,
        "sigma": base_sigma,
        "rmse": rmse,
        "r2": r2,
        "coefficients": coef_dict,
        "intercept": float(ridge.intercept_),
        "calibrator_method": calibrator_method,
        "calibrator_params": calibrator_params,
        "train_rows": len(dataset),
        "trained_seasons": train_seasons,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifact saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    train_v4_model()
