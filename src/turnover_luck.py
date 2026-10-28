"""
Turnover Luck Regression & Shrinkage Engine.

Ported from FreeMargin Engine v2.2 (src/freemargin/features/turnover_luck.py).

Core Quantitative Thesis:
1. Fumble Recoveries are 50/50 Coin Flips:
   - Across NFL history, teams recover approximately 50.0% of all loose-ball fumbles.
   - However, over short 4-game or 6-game windows, a team might recover 80% or 20% of fumbles.
   - Standard box scores and the betting public overreact to this "turnover margin",
     believing a +6 turnover differential reflects team skill rather than fluky bounces.
2. Interception Luck & Regression:
   - High interception rates regress heavily toward league averages.
3. Bayesian Shrinkage toward the 50% League Prior:
   - Observed recovery rates are shrunk toward the 0.50 prior using empirical Bayes:
     shrunk_rate = (n * observed + k * 0.50) / (n + k)  [with k=4.0]
4. Point Spread Regression Value:
   - Each net "lucky" turnover is worth ~3.8 to 4.2 points of EPA/scoring impact.
   - By calculating a team's Net Turnover Luck, we identify "inflated favorites"
     (due for negative regression) and "undervalued underdogs" (due for positive regression).
"""

import math
import numpy as np
import pandas as pd

from src.config import SEASON

TURNOVER_PRIOR_RECOVERY_RATE = 0.50
SHRINKAGE_K_TURNOVER = 4.0
POINTS_PER_TURNOVER_LUCK = 3.8


def parse_pbp_turnovers(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses fumbles, fumbles lost, forced fumbles, and interceptions per team-game.
    """
    if pbp_df.empty:
        return pd.DataFrame()

    required = ["season", "week", "game_id", "posteam", "defteam"]
    for col in required:
        if col not in pbp_df.columns:
            return pd.DataFrame()

    # Filter to scrimmage plays
    plays = pbp_df.copy()
    if "play_type" in plays.columns:
        plays = plays[plays["play_type"].isin(["pass", "run"])].copy()

    # Fill binary columns
    for col in ["fumble", "fumble_lost", "fumble_forced", "interception"]:
        if col in plays.columns:
            plays[col] = pd.to_numeric(plays[col], errors="coerce").fillna(0.0)
        else:
            plays[col] = 0.0

    # Aggregate by offense (posteam)
    off_stats = plays.groupby(["season", "week", "game_id", "posteam"]).agg(
        off_fumbles=("fumble", "sum"),
        off_fumbles_lost=("fumble_lost", "sum"),
        interceptions_thrown=("interception", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})

    # Aggregate by defense (defteam)
    def_stats = plays.groupby(["season", "week", "game_id", "defteam"]).agg(
        def_fumbles_forced=("fumble_forced", "sum"),
        def_interceptions=("interception", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})

    # Merge offensive and defensive turnovers
    merged = pd.merge(off_stats, def_stats, on=["season", "week", "game_id", "team"], how="outer").fillna(0.0)

    # Derived single-game luck metrics
    # Offense recovered = fumbles - fumbles_lost
    merged["off_fumbles_recovered"] = merged["off_fumbles"] - merged["off_fumbles_lost"]
    merged["total_turnovers_committed"] = merged["off_fumbles_lost"] + merged["interceptions_thrown"]
    merged["total_takeaways"] = merged["def_interceptions"]  # fumbles recovered by def is tracked via opponent

    return merged.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def shrink_turnover_rate(observed_rate: float, total_events: int, prior: float = TURNOVER_PRIOR_RECOVERY_RATE, k: float = SHRINKAGE_K_TURNOVER) -> float:
    """
    Shrinks observed fumble recovery rate toward 50.0% prior.
    """
    if total_events <= 0 or pd.isna(observed_rate):
        return float(prior)
    return (total_events * float(observed_rate) + k * float(prior)) / (total_events + k)


def compute_rolling_turnover_luck(turnover_game_df: pd.DataFrame, window: int = 4, k: float = SHRINKAGE_K_TURNOVER) -> pd.DataFrame:
    """
    Computes rolling fumble luck, expected recoveries, and net turnover regression.
    """
    if turnover_game_df.empty:
        return pd.DataFrame()

    df = turnover_game_df.sort_values(["season", "week", "game_id"]).copy()
    records = []

    teams = df["team"].unique()

    for team in teams:
        t_games = df[df["team"] == team].reset_index(drop=True)

        for i in range(len(t_games)):
            curr = t_games.iloc[i]
            season = curr["season"]
            week = curr["week"]
            gid = curr["game_id"]

            # Prior games strictly before this game
            prior = t_games.iloc[max(0, i - window):i]

            if prior.empty:
                records.append({
                    "season": season,
                    "week": week,
                    "game_id": gid,
                    "team": team,
                    "window_games": 0,
                    "total_fumbles": 0,
                    "fumbles_lost": 0,
                    "fumbles_recovered": 0,
                    "raw_recovery_rate": 0.50,
                    "shrunk_recovery_rate": 0.50,
                    "expected_fumbles_lost": 0.0,
                    "net_fumble_luck": 0.0,
                    "net_luck_points": 0.0,
                })
                continue

            n_games = len(prior)
            total_fumbles = int(prior["off_fumbles"].sum())
            fumbles_lost = int(prior["off_fumbles_lost"].sum())
            fumbles_recovered = int(prior["off_fumbles_recovered"].sum())

            # Expected fumbles lost based on 50% league prior
            expected_fumbles_lost = total_fumbles * 0.50
            expected_fumbles_recovered = total_fumbles * 0.50

            # Net Fumble Luck: Positive means the team got lucky (recovered more than 50%)
            # e.g., 6 fumbles, only 1 lost -> recovered 5, expected 3 -> Luck = +2.0 fumbles
            net_fumble_luck = fumbles_recovered - expected_fumbles_recovered

            raw_rate = (fumbles_recovered / total_fumbles) if total_fumbles > 0 else 0.50
            shrunk_rate = shrink_turnover_rate(raw_rate, total_fumbles, prior=TURNOVER_PRIOR_RECOVERY_RATE, k=k)

            # Impact in scoring points (each lucky recovery prevents/gains ~3.8 points)
            net_luck_points = round(net_fumble_luck * POINTS_PER_TURNOVER_LUCK, 2)

            records.append({
                "season": season,
                "week": week,
                "game_id": gid,
                "team": team,
                "window_games": n_games,
                "total_fumbles": total_fumbles,
                "fumbles_lost": fumbles_lost,
                "fumbles_recovered": fumbles_recovered,
                "raw_recovery_rate": round(raw_rate, 3),
                "shrunk_recovery_rate": round(shrunk_rate, 3),
                "expected_fumbles_lost": round(expected_fumbles_lost, 2),
                "net_fumble_luck": round(net_fumble_luck, 2),
                "net_luck_points": net_luck_points,
            })

    return pd.DataFrame(records)


def estimate_turnover_spread_adjustment(home_luck_points: float, away_luck_points: float, regression_factor: float = 0.50) -> float:
    """
    Calculates the point spread regression adjustment based on recent turnover luck.

    If Home was lucky (+7.6 luck points) and Away was unlucky (-3.8 luck points):
    Home is overvalued by the public/market.
    Regression moves the true spread in Away's favor.
    """
    net_luck_diff = float(home_luck_points) - float(away_luck_points)
    # The market already partially bakes in regression; we apply a 50% decay factor
    spread_adj = -(net_luck_diff * regression_factor)
    return round(spread_adj, 2)


def get_matchup_turnover_luck(luck_df: pd.DataFrame, home_team: str, away_team: str, game_id: str = None) -> dict:
    """
    Extracts the turnover luck comparison for an upcoming matchup.
    """
    if luck_df.empty:
        return {}

    home = home_team.upper().strip()
    away = away_team.upper().strip()

    if game_id:
        h_row = luck_df[(luck_df["team"] == home) & (luck_df["game_id"] == game_id)]
        a_row = luck_df[(luck_df["team"] == away) & (luck_df["game_id"] == game_id)]
    else:
        h_row = luck_df[luck_df["team"] == home].tail(1)
        a_row = luck_df[luck_df["team"] == away].tail(1)

    if h_row.empty or a_row.empty:
        return {}

    h_data = h_row.iloc[0].to_dict()
    a_data = a_row.iloc[0].to_dict()

    spread_adj = estimate_turnover_spread_adjustment(h_data["net_luck_points"], a_data["net_luck_points"])

    return {
        "home_team": home,
        "home_net_fumble_luck": h_data["net_fumble_luck"],
        "home_luck_points": h_data["net_luck_points"],
        "home_recovery_rate": h_data["raw_recovery_rate"],
        "away_team": away,
        "away_net_fumble_luck": a_data["net_fumble_luck"],
        "away_luck_points": a_data["net_luck_points"],
        "away_recovery_rate": a_data["raw_recovery_rate"],
        "spread_regression_adjustment": spread_adj,
    }


def main():
    print("=" * 70)
    print("Turnover Luck Regression & Shrinkage Engine (FreeMargin Port)")
    print("=" * 70)

    # Simulated 4-game sample demonstrating lucky vs unlucky teams
    scenarios = [
        {
            "matchup": "KC vs DEN",
            "home": "KC",
            "home_fumbles": 8,
            "home_lost": 1,  # 7 recovered out of 8 (87.5% recovery rate -> Extremely Lucky!)
            "away": "DEN",
            "away_fumbles": 6,
            "away_lost": 5,  # 1 recovered out of 6 (16.7% recovery rate -> Extremely Unlucky!)
        },
        {
            "matchup": "BAL vs CIN",
            "home": "BAL",
            "home_fumbles": 4,
            "home_lost": 2,  # 2 recovered out of 4 (50% -> Neutral)
            "away": "CIN",
            "away_fumbles": 5,
            "away_lost": 3,  # 2 recovered out of 5 (40% -> Slightly Unlucky)
        },
    ]

    for sc in scenarios:
        # Home calculation
        h_rec = sc["home_fumbles"] - sc["home_lost"]
        h_luck = h_rec - (sc["home_fumbles"] * 0.50)
        h_shrunk = shrink_turnover_rate(h_rec / sc["home_fumbles"], sc["home_fumbles"])
        h_pts = h_luck * POINTS_PER_TURNOVER_LUCK

        # Away calculation
        a_rec = sc["away_fumbles"] - sc["away_lost"]
        a_luck = a_rec - (sc["away_fumbles"] * 0.50)
        a_shrunk = shrink_turnover_rate(a_rec / sc["away_fumbles"], sc["away_fumbles"])
        a_pts = a_luck * POINTS_PER_TURNOVER_LUCK

        spread_adj = estimate_turnover_spread_adjustment(h_pts, a_pts)

        print(f"\nMatchup: {sc['matchup']}")
        print(f"  {sc['home']}: {sc['home_fumbles']} fumbles, {sc['home_lost']} lost ({h_rec/sc['home_fumbles']*100:.1f}% rec) | Shrunk: {h_shrunk*100:.1f}% | Net Luck: {h_luck:+.1f} fumbles ({h_pts:+.1f} pts)")
        print(f"  {sc['away']}: {sc['away_fumbles']} fumbles, {sc['away_lost']} lost ({a_rec/sc['away_fumbles']*100:.1f}% rec) | Shrunk: {a_shrunk*100:.1f}% | Net Luck: {a_luck:+.1f} fumbles ({a_pts:+.1f} pts)")
        print(f"  >> Estimated Spread Regression Adjustment: {spread_adj:+.2f} points (Fades {sc['home'] if spread_adj > 0 else sc['away']})")

    print("\nTurnover luck engine verification complete.")


if __name__ == "__main__":
    main()
