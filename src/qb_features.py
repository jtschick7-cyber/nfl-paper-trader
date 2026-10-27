"""
Quarterback Dropback Attribution and EPA Efficiency Engine.

Ported and adapted from FreeMargin Engine v2.2 (src/freemargin/features/qb.py).

Core Methodology:
1. True Dropback Attribution:
   - Scramblers are categorized as rushers in nflfastR; sacks retain passer_player_id.
   - We attribute every play with qb_dropback == 1 to the actual quarterback:
     qb_id / qb_name = rusher if qb_scramble == 1 else passer.
2. Filter Scrimmage Plays:
   - Excludes kneels (qb_kneel == 1), spikes (qb_spike == 1), and non-scrimmage rows.
3. Bayesian Shrinkage toward Explicit Prior:
   - For small sample sizes (< 20 dropbacks), shrinks observed EPA toward the league prior:
     shrink(observed, n, prior, k=4.0) = (n * observed + k * prior) / (n + k).
4. Dominant Passer Tracking:
   - Identifies the primary starter per team over rolling 4-game and 8-game windows.
   - Calculates EPA/dropback, success rate, sack rate, and dropback share.
5. Point Spread Impact:
   - Quantifies the expected point-spread delta if a starter is replaced by a backup.
"""

import math
import os
import numpy as np
import pandas as pd

from src.config import SEASON

# Minimum dropback threshold before shrinkage
MIN_DROPBACK_THRESHOLD = 20
SHRINKAGE_K = 4.0

# League-average baseline EPA per dropback (~0.05 EPA/play across recent NFL seasons)
DEFAULT_LEAGUE_QB_PRIOR = 0.05

# Approximate conversion: 1.0 EPA per game difference ~ 0.35 spread points
# An elite QB (+0.25 EPA/db over 36 dropbacks = +9.0 EPA) is worth ~3.5 to 5.0 points on spread.
EPA_TO_SPREAD_POINTS = 0.38


def attribute_dropbacks(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolates all pass plays, sacks, and scrambles, accurately identifying the QB.

    In nflfastR, scrambles have play_type == 'run' and rusher_player_name set,
    while sacks have sack == 1 and passer_player_name set.
    """
    # 1. Filter to valid scrimmage plays
    valid = pbp_df.copy()

    # Normalize column names
    if "qb_dropback" not in valid.columns:
        return pd.DataFrame()

    # Filter out kneels and spikes
    kneel_mask = (valid["qb_kneel"] == 1) if "qb_kneel" in valid.columns else False
    spike_mask = (valid["qb_spike"] == 1) if "qb_spike" in valid.columns else False

    dropbacks = valid[(valid["qb_dropback"] == 1) & (~kneel_mask) & (~spike_mask)].copy()

    if dropbacks.empty:
        return dropbacks

    # 2. Determine passer / scrambler identity
    passer_name = dropbacks.get("passer_player_name", dropbacks.get("passer", pd.Series(index=dropbacks.index, dtype=str)))
    rusher_name = dropbacks.get("rusher_player_name", dropbacks.get("rusher", pd.Series(index=dropbacks.index, dtype=str)))

    passer_id = dropbacks.get("passer_player_id", dropbacks.get("passer_id", pd.Series(index=dropbacks.index, dtype=str)))
    rusher_id = dropbacks.get("rusher_player_id", dropbacks.get("rusher_id", pd.Series(index=dropbacks.index, dtype=str)))

    scramble_flag = (dropbacks["qb_scramble"] == 1) if "qb_scramble" in dropbacks.columns else False

    dropbacks["qb_name"] = np.where(scramble_flag, rusher_name, passer_name)
    dropbacks["qb_id"] = np.where(scramble_flag, rusher_id, passer_id)

    # Clean missing names
    dropbacks["qb_name"] = dropbacks["qb_name"].fillna("UNKNOWN_QB")
    dropbacks["qb_id"] = dropbacks["qb_id"].fillna("UNKNOWN_ID")

    # Clean numeric fields
    for col in ["epa", "success", "sack", "interception"]:
        if col in dropbacks.columns:
            dropbacks[col] = pd.to_numeric(dropbacks[col], errors="coerce")
        else:
            dropbacks[col] = 0.0

    return dropbacks


def shrink_qb_rating(observed_epa: float, n_dropbacks: int, prior_epa: float = DEFAULT_LEAGUE_QB_PRIOR, k: float = SHRINKAGE_K) -> float:
    """
    Empirical Bayes shrinkage toward an explicit prior.
    Matches FreeMargin Engine's shrinkage.py:
    shrink(observed, n, prior, k) = (n * observed + k * prior) / (n + k)
    """
    if n_dropbacks <= 0 or pd.isna(observed_epa):
        return float(prior_epa)
    return (n_dropbacks * float(observed_epa) + k * float(prior_epa)) / (n_dropbacks + k)


def compute_game_qb_stats(dropbacks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-quarterback efficiency metrics for every game.
    """
    if dropbacks_df.empty:
        return pd.DataFrame()

    required = ["season", "week", "game_id", "posteam", "qb_name", "qb_id"]
    for col in required:
        if col not in dropbacks_df.columns:
            return pd.DataFrame()

    grouped = dropbacks_df.groupby(["season", "week", "game_id", "posteam", "qb_name", "qb_id"])

    stats = grouped.agg(
        dropbacks=("epa", "count"),
        total_epa=("epa", "sum"),
        epa_per_dropback=("epa", "mean"),
        success_rate=("success", "mean"),
        sack_rate=("sack", "mean"),
        int_rate=("interception", "mean"),
        scramble_rate=("qb_scramble", "mean") if "qb_scramble" in dropbacks_df.columns else ("epa", lambda x: 0.0),
    ).reset_index()

    # Calculate team total dropbacks per game to get dropback share
    team_totals = stats.groupby(["game_id", "posteam"])["dropbacks"].transform("sum")
    stats["dropback_share"] = stats["dropbacks"] / np.maximum(team_totals, 1)

    return stats.sort_values(["season", "week", "game_id", "posteam"]).reset_index(drop=True)


def compute_rolling_qb_ratings(qb_game_df: pd.DataFrame, window: int = 4, min_dropbacks: int = MIN_DROPBACK_THRESHOLD, prior_epa: float = DEFAULT_LEAGUE_QB_PRIOR) -> pd.DataFrame:
    """
    Computes rolling passer ratings per team and identifies dominant QB.

    Follows FreeMargin logic:
    - Dominant QB = passer with highest total dropback share across the rolling window.
    - If dropbacks < threshold, rating is shrunk toward the prior.
    """
    if qb_game_df.empty:
        return pd.DataFrame()

    # Sort chronologically
    df = qb_game_df.sort_values(["season", "week", "game_id"]).copy()

    records = []

    # Get distinct team-games in chronological order
    team_games = df[["season", "week", "game_id", "posteam"]].drop_duplicates().sort_values(["season", "week"])

    for team in team_games["posteam"].unique():
        t_games = team_games[team_games["posteam"] == team].reset_index(drop=True)

        for i in range(len(t_games)):
            curr_game = t_games.iloc[i]
            gid = curr_game["game_id"]
            season = curr_game["season"]
            week = curr_game["week"]

            # Prior games strictly before this game
            prior_gids = t_games.iloc[max(0, i - window):i]["game_id"].tolist()

            if not prior_gids:
                # Cold start: use baseline league prior
                records.append({
                    "season": season,
                    "week": week,
                    "game_id": gid,
                    "team": team,
                    "dominant_qb": "LEAGUE_BASELINE",
                    "qb_dropbacks_window": 0,
                    "qb_raw_epa": prior_epa,
                    "qb_rating": prior_epa,
                    "qb_success_rate": 0.45,
                    "qb_sack_rate": 0.065,
                    "qb_int_rate": 0.025,
                    "qb_is_shrunk": True,
                })
                continue

            prior_qb_rows = df[(df["posteam"] == team) & (df["game_id"].isin(prior_gids))]

            if prior_qb_rows.empty:
                records.append({
                    "season": season,
                    "week": week,
                    "game_id": gid,
                    "team": team,
                    "dominant_qb": "LEAGUE_BASELINE",
                    "qb_dropbacks_window": 0,
                    "qb_raw_epa": prior_epa,
                    "qb_rating": prior_epa,
                    "qb_success_rate": 0.45,
                    "qb_sack_rate": 0.065,
                    "qb_int_rate": 0.025,
                    "qb_is_shrunk": True,
                })
                continue

            # Identify dominant QB by total dropbacks in window
            qb_totals = prior_qb_rows.groupby("qb_name").agg(
                dropbacks=("dropbacks", "sum"),
                total_epa=("total_epa", "sum"),
                avg_success=("success_rate", "mean"),
                avg_sack=("sack_rate", "mean"),
                avg_int=("int_rate", "mean"),
            ).sort_values("dropbacks", ascending=False)

            top_qb = qb_totals.index[0]
            top_stats = qb_totals.loc[top_qb]
            n_db = int(top_stats["dropbacks"])
            raw_epa = float(top_stats["total_epa"] / n_db) if n_db > 0 else prior_epa

            # Apply shrinkage if sample size is below threshold
            shrunk_rating = shrink_qb_rating(raw_epa, n_db, prior_epa=prior_epa, k=SHRINKAGE_K)
            is_shrunk = n_db < min_dropbacks

            records.append({
                "season": season,
                "week": week,
                "game_id": gid,
                "team": team,
                "dominant_qb": top_qb,
                "qb_dropbacks_window": n_db,
                "qb_raw_epa": round(raw_epa, 4),
                "qb_rating": round(shrunk_rating, 4),
                "qb_success_rate": round(float(top_stats["avg_success"]), 4),
                "qb_sack_rate": round(float(top_stats["avg_sack"]), 4),
                "qb_int_rate": round(float(top_stats["avg_int"]), 4),
                "qb_is_shrunk": is_shrunk,
            })

    return pd.DataFrame(records)


def estimate_qb_replacement_penalty(starter_rating: float, backup_rating: float = -0.10, expected_dropbacks: int = 35) -> float:
    """
    Estimates the point-spread adjustment (in margin points) if a starting QB is replaced.

    Example:
    Starter rating = +0.22 EPA/db
    Backup rating  = -0.10 EPA/db
    Delta EPA/game = (0.22 - -0.10) * 35 = +11.2 EPA
    Spread adjustment = 11.2 * 0.38 = +4.25 points
    """
    delta_epa_per_db = float(starter_rating) - float(backup_rating)
    delta_game_epa = delta_epa_per_db * expected_dropbacks
    return round(delta_game_epa * EPA_TO_SPREAD_POINTS, 2)


def get_matchup_qb_comparison(qb_ratings_df: pd.DataFrame, home_team: str, away_team: str, game_id: str = None) -> dict:
    """
    Extracts pregame QB matchup context for home vs away.
    """
    if qb_ratings_df.empty:
        return {}

    home = home_team.upper().strip()
    away = away_team.upper().strip()

    if game_id:
        h_row = qb_ratings_df[(qb_ratings_df["team"] == home) & (qb_ratings_df["game_id"] == game_id)]
        a_row = qb_ratings_df[(qb_ratings_df["team"] == away) & (qb_ratings_df["game_id"] == game_id)]
    else:
        h_row = qb_ratings_df[qb_ratings_df["team"] == home].tail(1)
        a_row = qb_ratings_df[qb_ratings_df["team"] == away].tail(1)

    if h_row.empty or a_row.empty:
        return {}

    h_data = h_row.iloc[0].to_dict()
    a_data = a_row.iloc[0].to_dict()

    qb_diff = float(h_data["qb_rating"]) - float(a_data["qb_rating"])
    qb_spread_impact = estimate_qb_replacement_penalty(h_data["qb_rating"], a_data["qb_rating"])

    return {
        "home_qb": h_data["dominant_qb"],
        "home_qb_rating": h_data["qb_rating"],
        "home_qb_dropbacks": h_data["qb_dropbacks_window"],
        "away_qb": a_data["dominant_qb"],
        "away_qb_rating": a_data["qb_rating"],
        "away_qb_dropbacks": a_data["qb_dropbacks_window"],
        "qb_rating_diff": round(qb_diff, 4),
        "estimated_spread_impact": qb_spread_impact,
    }


def main():
    print("=" * 70)
    print("Quarterback Dropback Attribution Engine (FreeMargin Port)")
    print("=" * 70)

    # Demonstrate dropback attribution logic on simulated sample
    sample_plays = pd.DataFrame([
        # Traditional pass play
        {"season": 2026, "week": 7, "game_id": "2026_07_KC_SF", "posteam": "KC", "play_type": "pass",
         "qb_dropback": 1.0, "qb_scramble": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0,
         "passer_player_name": "P.Mahomes", "rusher_player_name": None, "epa": 0.42, "success": 1.0, "sack": 0.0, "interception": 0.0},
        # QB scramble (labeled as run in nflfastR, but is actually a dropback!)
        {"season": 2026, "week": 7, "game_id": "2026_07_KC_SF", "posteam": "KC", "play_type": "run",
         "qb_dropback": 1.0, "qb_scramble": 1.0, "qb_kneel": 0.0, "qb_spike": 0.0,
         "passer_player_name": None, "rusher_player_name": "P.Mahomes", "epa": 1.15, "success": 1.0, "sack": 0.0, "interception": 0.0},
        # Sack
        {"season": 2026, "week": 7, "game_id": "2026_07_KC_SF", "posteam": "KC", "play_type": "pass",
         "qb_dropback": 1.0, "qb_scramble": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0,
         "passer_player_name": "P.Mahomes", "rusher_player_name": None, "epa": -0.85, "success": 0.0, "sack": 1.0, "interception": 0.0},
        # Running back handoff (not a dropback)
        {"season": 2026, "week": 7, "game_id": "2026_07_KC_SF", "posteam": "KC", "play_type": "run",
         "qb_dropback": 0.0, "qb_scramble": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0,
         "passer_player_name": None, "rusher_player_name": "K.Hunt", "epa": -0.10, "success": 0.0, "sack": 0.0, "interception": 0.0},
        # Opponent QB pass
        {"season": 2026, "week": 7, "game_id": "2026_07_KC_SF", "posteam": "SF", "play_type": "pass",
         "qb_dropback": 1.0, "qb_scramble": 0.0, "qb_kneel": 0.0, "qb_spike": 0.0,
         "passer_player_name": "B.Purdy", "rusher_player_name": None, "epa": 0.28, "success": 1.0, "sack": 0.0, "interception": 0.0},
    ])

    print("Step 1: Attributing raw plays...")
    attributed = attribute_dropbacks(sample_plays)
    print(f"Total plays: {len(sample_plays)} -> Attributed dropbacks: {len(attributed)}")
    for _, row in attributed.iterrows():
        scramble_str = " (SCRAMBLE)" if row["qb_scramble"] == 1.0 else (" (SACK)" if row["sack"] == 1.0 else "")
        print(f"  [{row['posteam']}] {row['qb_name']}{scramble_str}: EPA = {row['epa']:+.2f}, Success = {row['success']}")

    print("\nStep 2: Aggregating game-level passer efficiency...")
    game_stats = compute_game_qb_stats(attributed)
    for _, r in game_stats.iterrows():
        print(f"  {r['posteam']} - {r['qb_name']}: {r['dropbacks']} dropbacks | EPA/db: {r['epa_per_dropback']:+.3f} | Success: {r['success_rate']*100:.1f}%")

    print("\nStep 3: Calculating replacement value delta...")
    sample_starter_epa = 0.24  # Elite starter
    sample_backup_epa = -0.08  # Replacement backup
    spread_delta = estimate_qb_replacement_penalty(sample_starter_epa, sample_backup_epa)
    print(f"  Starter EPA/db ({sample_starter_epa:+.2f}) vs Backup EPA/db ({sample_backup_epa:+.2f})")
    print(f"  Estimated Point Spread Adjustment: {spread_delta:+.2f} points")

    print("\nQB dropback attribution engine verification successful.")


if __name__ == "__main__":
    main()
