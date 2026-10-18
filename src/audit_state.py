from datetime import datetime

import pandas as pd
import pytz
import nfl_data_py as nfl

from src.config import SEASON
from src.sheet_utils import get_sheet_as_df


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


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
    now_utc = datetime.now(pytz.utc)
    print("Current UTC time: " + now_utc.isoformat())

    print("\n=== Loading Google Sheet ===")
    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    print("Sheet rows: " + str(len(sheet_df)))

    if "status" in sheet_df.columns:
        print("\nStatus counts:")
        print(sheet_df["status"].astype(str).str.upper().str.strip().value_counts(dropna=False).to_string())
    else:
        print("Missing status column.")

    print("\n=== Loading nflverse schedule for season " + str(SEASON) + " ===")
    sched = nfl.import_schedules([SEASON])

    print("Total nflverse schedule rows: " + str(len(sched)))
    print("nflverse columns: " + str(list(sched.columns)))

    if "home_score" in sched.columns and "away_score" in sched.columns:
        scored = sched.dropna(subset=["home_score", "away_score"]).copy()
        print("nflverse rows with final scores: " + str(len(scored)))

        if not scored.empty:
            print("\nSample scored nflverse games:")
            show_cols = ["game_id", "week", "home_team", "away_team", "home_score", "away_score"]
            show_cols = [c for c in show_cols if c in scored.columns]
            print(scored[show_cols].head(10).to_string(index=False))
        else:
            print("No scored nflverse games found for season " + str(SEASON) + ".")
    else:
        scored = pd.DataFrame()
        print("nflverse schedule missing home_score/away_score columns.")

    print("\n=== Game ID matching ===")
    if "game_id" in sheet_df.columns:
        sheet_ids = set(sheet_df["game_id"].astype(str).str.strip())
    else:
        sheet_ids = set()
        print("Sheet missing game_id column.")

    if not scored.empty and "game_id" in scored.columns:
        scored_ids = set(scored["game_id"].astype(str).str.strip())
    else:
        scored_ids = set()

    print("Unique sheet game_ids: " + str(len(sheet_ids)))
    print("Unique scored nflverse game_ids: " + str(len(scored_ids)))
    print("Matching game_ids: " + str(len(sheet_ids.intersection(scored_ids))))

    if sheet_ids and scored_ids and len(sheet_ids.intersection(scored_ids)) == 0:
        print("\nWARNING: sheet game_ids do not match nflverse scored game_ids.")
        print("Sample sheet ids: " + str(list(sheet_ids)[:5]))
        print("Sample nflverse ids: " + str(list(scored_ids)[:5]))

    print("\n=== Existing picks in sheet ===")

    if "pick_team" not in sheet_df.columns:
        print("Missing pick_team column.")
        return

    pick_mask = sheet_df["pick_team"].apply(has_pick)
    picks = sheet_df[pick_mask].copy()

    if picks.empty:
        print("No rows currently have pick_team filled in.")
        return

    print("Rows with picks: " + str(len(picks)))

    display_cols = [
        "game_id",
        "week",
        "kickoff_time",
        "home_team",
        "away_team",
        "pick_team",
        "pick_side",
        "pick_line",
        "pick_odds",
        "model_probability",
        "implied_probability_no_vig",
        "edge",
        "opening_line",
        "opening_odds",
        "closing_line",
        "closing_odds",
        "status",
        "notes",
    ]

    display_cols = [c for c in display_cols if c in picks.columns]

    for _, row in picks.iterrows():
        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            timing = "BAD_KICKOFF_TIME"
        elif kickoff <= now_utc:
            timing = "PAST"
        else:
            timing = "FUTURE"

        print("")
        print("MATCHUP: " + str(row.get("away_team", "?")) + " @ " + str(row.get("home_team", "?")))
        print("game_id: " + str(row.get("game_id", "")))
        print("kickoff_time: " + str(row.get("kickoff_time", "")))
        print("timing: " + timing)
        print("status: " + str(row.get("status", "")))
        print("pick: " + str(row.get("pick_team", "")) + " " + str(row.get("pick_side", "")) + " " + str(row.get("pick_line", "")) + " @" + str(row.get("pick_odds", "")))
        print("edge: " + str(row.get("edge", "")))
        print("model_probability: " + str(row.get("model_probability", "")))
        print("implied_probability_no_vig: " + str(row.get("implied_probability_no_vig", "")))
        print("opening_line: " + str(row.get("opening_line", "")))
        print("opening_odds: " + str(row.get("opening_odds", "")))
        print("closing_line: " + str(row.get("closing_line", "")))
        print("closing_odds: " + str(row.get("closing_odds", "")))
        print("notes: " + str(row.get("notes", "")))

    print("\n=== Full pick table ===")
    print(picks[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
