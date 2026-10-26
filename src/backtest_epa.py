"""
Walk-Forward Backtester comparing EPA-Enhanced Model (v3) vs. Raw Points Baseline vs. Favorite Baseline.

Zero Lookahead Bias:
- For each test season T, the model is trained strictly on seasons prior to T.
- Play-by-play EPA features and Elo ratings are updated chronologically game-by-game.
- Spreads and odds are evaluated against closing market lines with vig removed.
- Integrates Key Numbers discrete engine and Injury/Outlier safeguards.

Usage:
    python -m src.backtest_epa
    python -m src.backtest_epa --test-seasons 2024 2025
"""

import argparse
import math
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.config import MAX_AUTO_EDGE, MIN_EDGE_THRESHOLD
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features
from src.injury_guard import audit_candidate_pick
from src.key_numbers import calculate_expected_value, calculate_spread_probabilities, payout_multiplier
from src.model_utils import DIVISION_MAP, _load_sched
from src.odds_utils import american_to_implied_prob, remove_vig

EPA_DATA_PATH = "data/team_game_epa.csv"

# Elo constants
ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HFA = 48.0
ELO_REVERSION = 0.33


def update_elo(home_elo, away_elo, actual_margin):
    h_adj = home_elo + ELO_HFA
    exp_home = 1.0 / (1.0 + 10.0 ** ((away_elo - h_adj) / 400.0))
    actual_home = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)
    mov = abs(actual_margin)
    mov_mult = np.log(max(mov, 1) + 1.0) * (2.2 / ((home_elo - away_elo) * 0.001 + 2.2))
    shift = ELO_K * mov_mult * (actual_home - exp_home)
    return home_elo + shift, away_elo - shift


def prepare_season_features(sched_df, epa_df):
    """Compiles all game rows with both Raw Points and EPA features chronologically."""
    sched = sched_df.dropna(subset=["home_score", "away_score", "kickoff_utc"]).copy()
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True)
    sched = sched.sort_values("kickoff_utc").reset_index(drop=True)

    elo_ratings = {}
    team_last_date = {}
    team_margins = {}
    team_pts_for = {}
    team_pts_against = {}

    records = []
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
        home_odds = float(game["home_spread_odds"]) if pd.notna(game.get("home_spread_odds")) else -110.0
        away_odds = float(game["away_spread_odds"]) if pd.notna(game.get("away_spread_odds")) else -110.0

        if current_season is not None and season != current_season:
            for t in elo_ratings:
                elo_ratings[t] = elo_ratings[t] * (1.0 - ELO_REVERSION) + ELO_INITIAL * ELO_REVERSION
        current_season = season

        h_elo = elo_ratings.get(home, ELO_INITIAL)
        a_elo = elo_ratings.get(away, ELO_INITIAL)
        elo_diff = (h_elo + ELO_HFA) - a_elo

        # Rest days
        h_last = team_last_date.get(home)
        a_last = team_last_date.get(away)
        h_rest = min((kickoff - h_last).days, 21) if h_last else 7
        a_rest = min((kickoff - a_last).days, 21) if a_last else 7
        diff_rest = float(h_rest - a_rest)

        # Raw points rolling stats (last 4 games)
        h_pf = team_pts_for.get(home, [])
        a_pf = team_pts_for.get(away, [])
        h_pa = team_pts_against.get(home, [])
        a_pa = team_pts_against.get(away, [])
        h_m = team_margins.get(home, [])
        a_m = team_margins.get(away, [])

        diff_pf = (np.mean(h_pf[-4:]) if len(h_pf) >= 2 else 21.0) - (np.mean(a_pf[-4:]) if len(a_pf) >= 2 else 21.0)
        diff_pa = (np.mean(h_pa[-4:]) if len(h_pa) >= 2 else 21.0) - (np.mean(a_pa[-4:]) if len(a_pa) >= 2 else 21.0)
        diff_margin = (np.mean(h_m[-4:]) if len(h_m) >= 2 else 0.0) - (np.mean(a_m[-4:]) if len(a_m) >= 2 else 0.0)

        h_div = DIVISION_MAP.get(home)
        a_div = DIVISION_MAP.get(away)
        division = 1.0 if h_div and a_div and h_div == a_div else 0.0

        # EPA features
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

        records.append({
            "game_id": game["game_id"],
            "season": season,
            "week": game["week"],
            "kickoff_utc": kickoff,
            "home_team": home,
            "away_team": away,
            "home_line": -spread_line,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "actual_home_margin": actual_margin,
            "elo_diff": elo_diff / 25.0,
            "diff_pf": diff_pf,
            "diff_pa": diff_pa,
            "diff_margin": diff_margin,
            "diff_rest": diff_rest,
            "division": division,
            "diff_pass_epa": diff_pass_epa,
            "diff_rush_epa": diff_rush_epa,
            "diff_success_rate": diff_success,
            "diff_early_down_epa": diff_early,
        })

        # Update historical state
        new_h, new_a = update_elo(h_elo, a_elo, actual_margin)
        elo_ratings[home] = new_h
        elo_ratings[away] = new_a
        team_last_date[home] = kickoff
        team_last_date[away] = kickoff
        team_pts_for.setdefault(home, []).append(home_score)
        team_pts_for.setdefault(away, []).append(away_score)
        team_pts_against.setdefault(home, []).append(away_score)
        team_pts_against.setdefault(away, []).append(home_score)
        team_margins.setdefault(home, []).append(actual_margin)
        team_margins.setdefault(away, []).append(-actual_margin)

    return pd.DataFrame(records)


def run_walk_forward_backtest(test_seasons):
    print("=============================================================")
    print(f"Running Walk-Forward Backtest on seasons: {test_seasons}")
    print("=============================================================\n")

    if not os.path.exists(EPA_DATA_PATH):
        raise SystemExit("Missing EPA data. Run: python -m src.build_features")

    epa_df = pd.read_csv(EPA_DATA_PATH)
    all_seasons = sorted(list(set(test_seasons + [min(test_seasons) - 1])))
    sched_df = _load_sched(all_seasons)

    print("Building full timeline dataset...")
    full_df = prepare_season_features(sched_df, epa_df)

    v3_features = [
        "elo_diff", "diff_pass_epa", "diff_rush_epa", "diff_success_rate",
        "diff_early_down_epa", "diff_margin", "diff_rest", "division"
    ]
    raw_features = ["diff_pf", "diff_pa", "diff_margin", "diff_rest", "division"]

    # Results tracking
    bets_v3 = []
    bets_raw = []
    bets_fav = []

    for test_s in test_seasons:
        print(f"\n--- Testing Season {test_s} (Trained on prior seasons) ---")
        train_df = full_df[full_df["season"] < test_s]
        test_df = full_df[full_df["season"] == test_s]

        if train_df.empty or test_df.empty:
            continue

        # Fit v3 EPA Model
        scaler_v3 = StandardScaler()
        X_tr_v3 = scaler_v3.fit_transform(train_df[v3_features].values)
        y_tr = train_df["actual_home_margin"].values

        model_v3 = Ridge(alpha=15.0)
        model_v3.fit(X_tr_v3, y_tr)
        sigma_v3 = float(np.std(y_tr - model_v3.predict(X_tr_v3)))

        # Fit Platt Calibrator on training fold to prevent overconfidence
        y_tr_pred = model_v3.predict(X_tr_v3)
        cal_X = []
        cal_y = []
        for j, (_, row) in enumerate(train_df.iterrows()):
            act_diff = row["actual_home_margin"] + row["home_line"]
            if abs(act_diff) > 1e-5:
                sp = calculate_spread_probabilities(y_tr_pred[j], sigma_v3, row["home_line"])
                cal_X.append([sp["cond_home_win"]])
                cal_y.append(1 if act_diff > 0 else 0)

        calibrator_v3 = LogisticRegression()
        has_calibrator = False
        if len(cal_X) >= 50:
            calibrator_v3.fit(cal_X, cal_y)
            has_calibrator = True

        # Fit Raw Points Model
        scaler_raw = StandardScaler()
        X_tr_raw = scaler_raw.fit_transform(train_df[raw_features].values)
        model_raw = Ridge(alpha=15.0)
        model_raw.fit(X_tr_raw, y_tr)
        sigma_raw = float(np.std(y_tr - model_raw.predict(X_tr_raw)))

        # Evaluate on Test Games
        X_te_v3 = scaler_v3.transform(test_df[v3_features].values)
        mu_te_v3 = model_v3.predict(X_te_v3)

        X_te_raw = scaler_raw.transform(test_df[raw_features].values)
        mu_te_raw = model_raw.predict(X_te_raw)

        for i, (_, game) in enumerate(test_df.iterrows()):
            home = game["home_team"]
            away = game["away_team"]
            home_line = game["home_line"]
            home_odds = game["home_odds"]
            away_odds = game["away_odds"]
            actual_margin = game["actual_home_margin"]

            # Compute no-vig fair probabilities
            fair_h, fair_a = remove_vig(
                american_to_implied_prob(home_odds),
                american_to_implied_prob(away_odds)
            )

            # --- 1. Evaluate Favorite Baseline ---
            if abs(home_line) > 0:
                fav_side = "HOME" if home_line < 0 else "AWAY"
                fav_line = home_line if fav_side == "HOME" else -home_line
                fav_odds = home_odds if fav_side == "HOME" else away_odds
                fav_diff = actual_margin + home_line if fav_side == "HOME" else -actual_margin + fav_line

                if abs(fav_diff) < 1e-5:
                    fav_res, fav_pnl = "PUSH", 0.0
                elif fav_diff > 0:
                    fav_res, fav_pnl = "WIN", payout_multiplier(fav_odds)
                else:
                    fav_res, fav_pnl = "LOSS", -1.0

                bets_fav.append({"season": test_s, "res": fav_res, "pnl": fav_pnl})

            # --- 2. Evaluate Raw Points Model ---
            mu_raw = mu_te_raw[i]
            # Standard Gaussian without discrete key numbers
            z_raw = (mu_raw + home_line) / sigma_raw
            p_h_raw = 0.5 * (1.0 + math.erf(z_raw / np.sqrt(2.0)))
            p_a_raw = 1.0 - p_h_raw
            edge_h_raw = p_h_raw - fair_h
            edge_a_raw = p_a_raw - fair_a

            if edge_h_raw >= edge_a_raw:
                raw_side = "HOME"
                raw_edge = edge_h_raw
                raw_odds = home_odds
                raw_line = home_line
            else:
                raw_side = "AWAY"
                raw_edge = edge_a_raw
                raw_odds = away_odds
                raw_line = -home_line

            if MIN_EDGE_THRESHOLD <= raw_edge <= MAX_AUTO_EDGE:
                diff_res = actual_margin + home_line if raw_side == "HOME" else -actual_margin + raw_line
                if abs(diff_res) < 1e-5:
                    r_res, r_pnl = "PUSH", 0.0
                elif diff_res > 0:
                    r_res, r_pnl = "WIN", payout_multiplier(raw_odds)
                else:
                    r_res, r_pnl = "LOSS", -1.0
                bets_raw.append({"season": test_s, "res": r_res, "pnl": r_pnl, "edge": raw_edge})

            # --- 3. Evaluate v3 EPA-Enhanced Model ---
            mu_v3 = mu_te_v3[i]
            # Discrete Key-Numbers aware spread probabilities
            prob_dict = calculate_spread_probabilities(mu_v3, sigma_v3, home_line)
            raw_p_h = prob_dict["cond_home_win"]

            if has_calibrator:
                p_h_v3 = float(calibrator_v3.predict_proba([[raw_p_h]])[0][1])
                p_a_v3 = 1.0 - p_h_v3
            else:
                p_h_v3 = raw_p_h
                p_a_v3 = prob_dict["cond_away_win"]

            edge_h_v3 = p_h_v3 - fair_h
            edge_a_v3 = p_a_v3 - fair_a

            if edge_h_v3 >= edge_a_v3:
                v3_side = "HOME"
                v3_edge = edge_h_v3
                v3_odds = home_odds
                v3_line = home_line
                pick_t = home
            else:
                v3_side = "AWAY"
                v3_edge = edge_a_v3
                v3_odds = away_odds
                v3_line = -home_line
                pick_t = away

            # Audit through Injury / Outlier Guard
            alert = audit_candidate_pick(
                pick_team=pick_t,
                pick_side=v3_side,
                open_line=home_line,
                current_line=home_line,
                mu_home_margin=mu_v3,
                edge=v3_edge,
                max_auto_edge=MAX_AUTO_EDGE,
            )

            if MIN_EDGE_THRESHOLD <= v3_edge <= MAX_AUTO_EDGE and alert.recommendation == "PROCEED":
                diff_v3 = actual_margin + home_line if v3_side == "HOME" else -actual_margin + v3_line
                if abs(diff_v3) < 1e-5:
                    v3_res, v3_pnl = "PUSH", 0.0
                elif diff_v3 > 0:
                    v3_res, v3_pnl = "WIN", payout_multiplier(v3_odds)
                else:
                    v3_res, v3_pnl = "LOSS", -1.0
                bets_v3.append({"season": test_s, "res": v3_res, "pnl": v3_pnl, "edge": v3_edge})

    def print_summary(name, bets, min_edge=None):
        if not bets:
            print(f"  {name:30s}: 0 bets placed.")
            return
        df = pd.DataFrame(bets)
        if min_edge is not None and "edge" in df.columns:
            df = df[df["edge"] >= min_edge]
            if df.empty:
                print(f"  {name:30s}: 0 bets placed (>= {min_edge*100:.1f}% edge).")
                return

        wins = (df["res"] == "WIN").sum()
        losses = (df["res"] == "LOSS").sum()
        pushes = (df["res"] == "PUSH").sum()
        decisions = wins + losses
        win_rate = (wins / decisions * 100.0) if decisions > 0 else 0.0
        pnl = df["pnl"].sum()
        roi = (pnl / len(df) * 100.0) if len(df) > 0 else 0.0

        label = f"{name} (>= {min_edge*100:.1f}%)" if min_edge is not None else name
        print(f"  {label:35s}: {len(df):3d} bets | {wins:2d}W - {losses:2d}L - {pushes}P | "
              f"Win Rate: {win_rate:5.1f}% | Total Units: {pnl:+6.2f}u | ROI: {roi:+5.2f}%")

    print("\n=============================================================")
    print("BACKTEST RESULTS COMPARISON (Out-of-Sample Walk-Forward)")
    print("=============================================================")
    print_summary("Favorite Baseline", bets_fav)
    print_summary("Raw Points (v1 Baseline)", bets_raw)
    print_summary("EPA-Enhanced (v3 Calibrated)", bets_v3)
    print_summary("EPA-Enhanced High-Conviction", bets_v3, min_edge=0.035)
    print("=============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward EPA Backtest.")
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Seasons to test on (default: 2024 2025)",
    )
    args = parser.parse_args()
    run_walk_forward_backtest(args.test_seasons)


if __name__ == "__main__":
    main()
