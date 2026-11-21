"""
Build EPA-per-play and efficiency features from nflverse play-by-play data.

Usage:
    python -m src.build_features
    python -m src.build_features --seasons 2023 2024 2025 2026
"""

import argparse
import os
import numpy as np
import pandas as pd

from src.config import SEASON
from src.epa_features import parse_pbp_to_game_epa
from src.qb_features import attribute_dropbacks, compute_game_qb_stats
from src.turnover_luck import parse_pbp_turnovers

OUTPUT_FILE = "data/team_game_epa.csv"
QB_OUTPUT_FILE = "data/team_game_qb.csv"
TURNOVER_OUTPUT_FILE = "data/team_game_turnovers.csv"


def _atomic_to_csv(df, path):
    """Write to a temp file then rename, so a crash never leaves a half-written CSV."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _reg_game_ids(df):
    if df is None or df.empty or "game_id" not in df.columns:
        return set()
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    return set(df["game_id"].astype(str).str.strip())


def _completed_game_ids(season):
    from src.model_utils import _load_sched
    sched = _load_sched([season])
    comp = sched.dropna(subset=["home_score", "away_score"])
    return set(comp["game_id"].astype(str).str.strip())


def _season_has_completed_games(season):
    try:
        return bool(_completed_game_ids(season))
    except Exception:
        return False


def missing_completed_game_ids(have_game_ids, season, completed_ids=None):
    """Completed regular-season games of `season` that are absent from `have_game_ids`.

    Compares game_id sets, not max week: a cache holding Thursday's game of week N
    still lacks that week's Sunday and Monday games (P0-3).
    """
    if completed_ids is None:
        completed_ids = _completed_game_ids(season)
    have = {str(g).strip() for g in have_game_ids}
    return {str(g).strip() for g in completed_ids} - have


def build_features(seasons=None):
    if seasons is None:
        # Default to previous 3 seasons plus current season
        seasons = [SEASON - 3, SEASON - 2, SEASON - 1, SEASON]

    seasons = sorted({int(s) for s in seasons if s >= 1999})
    print(f"Building EPA features from PBP for seasons: {seasons}...")

    try:
        import nfl_data_py as nfl
    except ImportError:
        raise SystemExit(
            "nfl_data_py is required to download play-by-play data.\n"
            "Install it via: pip install nfl_data_py"
        )

    # Download play-by-play per season with local parquet caching
    pbp_dfs = []
    for s in seasons:
        try:
            cache_pbp = os.path.join("data", "epa_cache", f"pbp_{s}.parquet")
            df_s = None
            if os.path.exists(cache_pbp):
                print(f"  Loading cached play-by-play for {s} from {cache_pbp}...")
                df_s = pd.read_parquet(cache_pbp)
                if s >= SEASON and df_s is not None and not df_s.empty:
                    try:
                        missing = missing_completed_game_ids(_reg_game_ids(df_s), s)
                        if missing:
                            print(f"  PBP cache for {s} lacks {len(missing)} completed game(s) "
                                  f"(e.g. {sorted(missing)[0]}). Re-downloading...")
                            df_s = None
                    except Exception as e:
                        print(f"  Warning checking PBP freshness for {s}: {e}")

            if df_s is None:
                print(f"  Downloading play-by-play for season {s}...")
                if hasattr(nfl, "import_pbp_data"):
                    df_s = nfl.import_pbp_data([s], include_participation=False, downcast=True)
                elif hasattr(nfl, "import_pbp"):
                    df_s = nfl.import_pbp([s])
                else:
                    raise AttributeError("nfl_data_py has neither import_pbp_data nor import_pbp")
                if df_s is not None and not df_s.empty:
                    try:
                        os.makedirs(os.path.dirname(cache_pbp), exist_ok=True)
                        df_s.to_parquet(cache_pbp, index=False)
                    except Exception:
                        pass

            if df_s is not None and not df_s.empty:
                print(f"  Loaded {len(df_s):,} plays for {s}.")
                pbp_dfs.append(df_s)
        except Exception as e:
            print(f"  Warning: Could not load PBP for season {s}: {e}")
            if s >= SEASON and _season_has_completed_games(s):
                # Writing the CSVs without the live season would silently run the models on
                # prior seasons only. Keep the existing CSVs and fail the step instead.
                raise SystemExit(f"PBP for live season {s} failed to load; existing feature CSVs left unchanged.")

    if not pbp_dfs:
        print("No play-by-play data could be loaded.")
        return

    pbp = pd.concat(pbp_dfs, ignore_index=True)

    print(f"Processing {len(pbp):,} plays...")
    game_epa = parse_pbp_to_game_epa(pbp)

    if game_epa.empty:
        print("Failed to parse EPA features.")
        return

    os.makedirs("data", exist_ok=True)
    _atomic_to_csv(game_epa, OUTPUT_FILE)

    print(f"Successfully processed {len(game_epa):,} team-game records.")
    print(f"Saved to: {OUTPUT_FILE}")

    # Extract Quarterback dropback efficiency
    print("\nExtracting Quarterback dropback efficiency...")
    attributed = attribute_dropbacks(pbp)
    game_qb = compute_game_qb_stats(attributed)

    if not game_qb.empty:
        _atomic_to_csv(game_qb, QB_OUTPUT_FILE)
        print(f"Successfully processed {len(game_qb):,} QB-game records.")
        print(f"Saved to: {QB_OUTPUT_FILE}")

    # Extract Turnover Luck metrics
    print("\nExtracting Turnover Luck metrics...")
    game_turnovers = parse_pbp_turnovers(pbp)
    if not game_turnovers.empty:
        _atomic_to_csv(game_turnovers, TURNOVER_OUTPUT_FILE)
        print(f"Successfully processed {len(game_turnovers):,} turnover-game records.")
        print(f"Saved to: {TURNOVER_OUTPUT_FILE}")

    # Display preview
    recent = game_epa[game_epa["season"] == max(seasons)]
    if not recent.empty:
        print("\n=== Top 5 Passing Offenses (Off Pass EPA/play) ===")
        top_pass = (
            recent.groupby("team")["off_pass_epa"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        for team, epa in top_pass.items():
            print(f"  {team:4s}: {epa:+.3f} EPA/dropback")

        print("\n=== Top 5 Pass Defenses (Lowest Def Pass EPA/play Allowed) ===")
        top_def = (
            recent.groupby("team")["def_pass_epa"]
            .mean()
            .sort_values(ascending=True)
            .head(5)
        )
        for team, epa in top_def.items():
            print(f"  {team:4s}: {epa:+.3f} EPA/dropback allowed")

    if not game_qb.empty:
        recent_qb = game_qb[game_qb["season"] == max(seasons)]
        if not recent_qb.empty:
            qb_agg = recent_qb.groupby("qb_name").agg(
                dropbacks=("dropbacks", "sum"),
                total_epa=("total_epa", "sum"),
                success_rate=("success_rate", "mean"),
            )
            qb_agg["epa_per_db"] = qb_agg["total_epa"] / np.maximum(qb_agg["dropbacks"], 1)
            qualified = qb_agg[qb_agg["dropbacks"] >= 50].sort_values("epa_per_db", ascending=False).head(5)

            if not qualified.empty:
                print("\n=== Top 5 Starting Quarterbacks (EPA/dropback, min 50 db) ===")
                for qb, row in qualified.iterrows():
                    print(f"  {qb:18s}: {row['epa_per_db']:+.3f} EPA/db ({int(row['dropbacks'])} dropbacks, {row['success_rate']*100:.1f}% success)")


def main():
    parser = argparse.ArgumentParser(description="Build EPA features from nflverse PBP.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="List of seasons to process (e.g. 2023 2024 2025 2026)",
    )
    args = parser.parse_args()
    build_features(args.seasons)


if __name__ == "__main__":
    main()
