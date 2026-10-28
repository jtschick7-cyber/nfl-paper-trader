"""
EPA (Expected Points Added) Feature Engineering Pipeline for NFL Spread Modeling.

Why EPA over raw points:
Raw points per game are noisy and contaminated by:
- Turnover luck (defensive/special teams TDs)
- Garbage time scoring in blowouts
- Extreme red zone variance (field goals vs touchdowns)

Expected Points Added (EPA) measures efficiency on a play-by-play basis against
historical down-and-distance baselines. In sports analytics, Passing EPA/dropback
is ~3-4x more predictive of future point margins than raw points scored.

This module provides:
1. Extraction of team-game offensive and defensive EPA metrics from play-by-play data:
   - Passing EPA/play (dropbacks, passes, sacks, scrambles)
   - Rushing EPA/play (designed runs)
   - Success Rate (% of plays with positive EPA)
   - Early-Down EPA (1st and 2nd down efficiency, free of 3rd down noise)
2. Opponent-aware defensive stats (defensive EPA allowed per pass/rush)
3. Leak-free rolling window snapshot calculation (only past completed games)
4. Matchup differential feature builder for Home vs. Away
"""

import os
import numpy as np
import pandas as pd

EPA_CACHE_DIR = "data/epa_cache"
os.makedirs(EPA_CACHE_DIR, exist_ok=True)


def parse_pbp_to_game_epa(pbp_df):
    """
    Transforms raw play-by-play dataframe into team-game level offensive and defensive metrics.
    
    Returns a DataFrame with one row per team per game:
    ['game_id', 'season', 'week', 'team', 'opponent', 'is_home',
     'off_pass_epa', 'off_rush_epa', 'off_total_epa', 'off_success_rate', 'off_early_down_epa',
     'def_pass_epa', 'def_rush_epa', 'def_total_epa', 'def_success_rate', 'def_early_down_epa']
    """
    if pbp_df is None or pbp_df.empty:
        return pd.DataFrame()

    df = pbp_df.copy()

    # Filter to regular season offensive plays (passes and runs)
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    # Filter out penalties, spikes, kneels, timeouts
    play_types = ["pass", "run"]
    if "play_type" in df.columns:
        df = df[df["play_type"].isin(play_types)]

    # Require valid epa and posteam
    df = df[df["epa"].notna() & df["posteam"].notna() & df["defteam"].notna()]

    if df.empty:
        return pd.DataFrame()

    # Standardize column types
    df["posteam"] = df["posteam"].astype(str).str.upper().str.strip()
    df["defteam"] = df["defteam"].astype(str).str.upper().str.strip()
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["season"] = pd.to_numeric(df.get("season", 0), errors="coerce").fillna(0).astype(int)
    df["week"] = pd.to_numeric(df.get("week", 0), errors="coerce").fillna(0).astype(int)

    # Identify pass vs rush plays
    is_pass = (df.get("pass", 0) == 1) | (df.get("play_type") == "pass")
    is_rush = (df.get("rush", 0) == 1) | (df.get("play_type") == "run")
    is_early_down = df.get("down", 1).isin([1, 2])

    # Compute offensive metrics per game per team
    records = []

    for (game_id, posteam), g in df.groupby(["game_id", "posteam"]):
        defteam = g["defteam"].iloc[0]
        season = g["season"].iloc[0]
        week = g["week"].iloc[0]

        pass_plays = g[is_pass.loc[g.index]]
        rush_plays = g[is_rush.loc[g.index]]
        early_plays = g[is_early_down.loc[g.index]]

        off_pass_epa = float(pass_plays["epa"].mean()) if len(pass_plays) > 0 else 0.0
        off_rush_epa = float(rush_plays["epa"].mean()) if len(rush_plays) > 0 else 0.0
        off_total_epa = float(g["epa"].mean()) if len(g) > 0 else 0.0

        # Success rate: proportion of plays where EPA > 0
        if "success" in g.columns:
            off_success_rate = float(g["success"].mean())
        else:
            off_success_rate = float((g["epa"] > 0).mean())

        off_early_down_epa = float(early_plays["epa"].mean()) if len(early_plays) > 0 else 0.0

        records.append({
            "game_id": game_id,
            "season": season,
            "week": week,
            "team": posteam,
            "opponent": defteam,
            "off_pass_epa": round(off_pass_epa, 4),
            "off_rush_epa": round(off_rush_epa, 4),
            "off_total_epa": round(off_total_epa, 4),
            "off_success_rate": round(off_success_rate, 4),
            "off_early_down_epa": round(off_early_down_epa, 4),
            "plays_count": len(g),
        })

    off_df = pd.DataFrame(records)

    if off_df.empty:
        return pd.DataFrame()

    # Self-merge to attach opponent's defensive metrics
    # A team's defensive EPA allowed is simply the opponent's offensive EPA in that same game!
    def_df = off_df.copy()
    def_cols = {
        "team": "opponent",
        "opponent": "team",
        "off_pass_epa": "def_pass_epa",
        "off_rush_epa": "def_rush_epa",
        "off_total_epa": "def_total_epa",
        "off_success_rate": "def_success_rate",
        "off_early_down_epa": "def_early_down_epa",
    }
    def_df = def_df.rename(columns=def_cols)
    def_df = def_df.drop(columns=["plays_count", "season", "week"], errors="ignore")

    merged = pd.merge(off_df, def_df, on=["game_id", "team", "opponent"], how="left")

    return merged


def compute_rolling_epa_snapshot(game_epa_df, schedule_df, as_of_utc, window=6, min_periods=3):
    """
    Computes a point-in-time rolling average EPA profile for each team
    strictly prior to as_of_utc. Zero future data leakage.
    
    Returns:
        dict: {team_abbr: {
            'roll_off_pass_epa': float,
            'roll_off_rush_epa': float,
            'roll_off_total_epa': float,
            'roll_off_success_rate': float,
            'roll_def_pass_epa': float,
            'roll_def_rush_epa': float,
            'roll_def_total_epa': float,
            'roll_def_success_rate': float,
            'games_played': int
        }}
    """
    if game_epa_df.empty or schedule_df.empty:
        return {}

    # Merge schedule kickoff times to ensure temporal order
    sched_times = schedule_df[["game_id", "kickoff_utc"]].drop_duplicates("game_id").copy()
    sched_times["kickoff_utc"] = pd.to_datetime(sched_times["kickoff_utc"], utc=True, errors="coerce")

    merged = pd.merge(game_epa_df, sched_times, on="game_id", how="inner")
    
    # Filter strictly completed games before as_of_utc
    past_games = merged[merged["kickoff_utc"] < as_of_utc].copy()

    if past_games.empty:
        return {}

    past_games = past_games.sort_values("kickoff_utc")

    if as_of_utc is not None:
        as_of_dt = pd.to_datetime(as_of_utc, utc=True)
        target_season = as_of_dt.year if as_of_dt.month >= 3 else as_of_dt.year - 1
    else:
        target_season = int(game_epa_df["season"].max()) if "season" in game_epa_df.columns else 2026

    snapshot = {}

    epa_metrics = [
        "off_pass_epa", "off_rush_epa", "off_total_epa", "off_success_rate", "off_early_down_epa",
        "def_pass_epa", "def_rush_epa", "def_total_epa", "def_success_rate", "def_early_down_epa",
    ]

    for team, g in past_games.groupby("team"):
        if len(g) < min_periods:
            continue

        recent = g.tail(window).copy()

        # Apply off-season regression: explicitly regress prior-season values 50% toward league prior
        team_stats = {"games_played": len(g)}
        for col in epa_metrics:
            if col in recent.columns:
                is_success = "success" in col.lower()
                league_prior = 0.45 if is_success else 0.0

                raw_vals = recent[col].fillna(league_prior).values
                regressed_vals = np.array([
                    (0.50 * v + 0.50 * league_prior) if int(s) < target_season else v
                    for v, s in zip(raw_vals, recent["season"].values)
                ], dtype=float)

                team_stats[f"roll_{col}"] = float(np.mean(regressed_vals))
            else:
                team_stats[f"roll_{col}"] = 0.45 if "success" in col.lower() else 0.0

        snapshot[team] = team_stats

    return snapshot


def get_matchup_epa_features(snapshot, home_team, away_team):
    """
    Constructs differential matchup features for Home vs Away:
    
    1. diff_pass_epa:
       (Home Pass Offense vs Away Pass Defense) - (Away Pass Offense vs Home Pass Defense)
       = (Home off_pass_epa - Away def_pass_epa) - (Away off_pass_epa - Home def_pass_epa)
       
    2. diff_rush_epa:
       (Home Rush Offense vs Away Rush Defense) - (Away Rush Offense vs Home Rush Defense)
       
    3. diff_success_rate:
       Home net success rate - Away net success rate
       
    4. diff_total_epa:
       Home overall net EPA - Away overall net EPA
    """
    if home_team not in snapshot or away_team not in snapshot:
        return None

    h = snapshot[home_team]
    a = snapshot[away_team]

    # Home Pass attack vs Away Pass defense:
    # A positive number means Home passes better than Away's defense normally allows.
    home_pass_matchup = h.get("roll_off_pass_epa", 0.0) - a.get("roll_def_pass_epa", 0.0)
    away_pass_matchup = a.get("roll_off_pass_epa", 0.0) - h.get("roll_def_pass_epa", 0.0)
    diff_pass_epa = home_pass_matchup - away_pass_matchup

    # Home Rush attack vs Away Rush defense:
    home_rush_matchup = h.get("roll_off_rush_epa", 0.0) - a.get("roll_def_rush_epa", 0.0)
    away_rush_matchup = a.get("roll_off_rush_epa", 0.0) - h.get("roll_def_rush_epa", 0.0)
    diff_rush_epa = home_rush_matchup - away_rush_matchup

    # Net success rate difference:
    home_net_success = h.get("roll_off_success_rate", 0.45) - h.get("roll_def_success_rate", 0.45)
    away_net_success = a.get("roll_off_success_rate", 0.45) - a.get("roll_def_success_rate", 0.45)
    diff_success_rate = home_net_success - away_net_success

    # Total net EPA difference:
    home_net_epa = h.get("roll_off_total_epa", 0.0) - h.get("roll_def_total_epa", 0.0)
    away_net_epa = a.get("roll_off_total_epa", 0.0) - a.get("roll_def_total_epa", 0.0)
    diff_total_epa = home_net_epa - away_net_epa

    # Early-down scheme efficiency difference:
    home_early_matchup = h.get("roll_off_early_down_epa", 0.0) - a.get("roll_def_early_down_epa", 0.0)
    away_early_matchup = a.get("roll_off_early_down_epa", 0.0) - h.get("roll_def_early_down_epa", 0.0)
    diff_early_down_epa = home_early_matchup - away_early_matchup

    return {
        "diff_pass_epa": round(float(diff_pass_epa), 4),
        "diff_rush_epa": round(float(diff_rush_epa), 4),
        "diff_success_rate": round(float(diff_success_rate), 4),
        "diff_early_down_epa": round(float(diff_early_down_epa), 4),
        "diff_total_epa": round(float(diff_total_epa), 4),
        "home_pass_matchup": round(float(home_pass_matchup), 4),
        "away_pass_matchup": round(float(away_pass_matchup), 4),
    }
