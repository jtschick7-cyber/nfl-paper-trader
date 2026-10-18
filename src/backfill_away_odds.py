import glob
import json
import os
from datetime import datetime

import pandas as pd
import pytz

from src.config import MIN_EDGE_THRESHOLD
from src.odds_utils import american_to_implied_prob, parse_spreads, remove_vig
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

CLEAR_LOW_EDGE = os.getenv("CLEAR_LOW_EDGE", "").strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def has_pick(value):
    if pd.isna(value):
        return False

    s = str(value).strip()

    if s == "":
        return False

    if s.lower() == "nan":
        return False

    return True


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def latest_cache_spreads():
    files = glob.glob("data/odds_cache/odds_raw_*.json")

    if not files:
        raise SystemExit("No odds cache files found. Run capture_opening or capture_closing first.")

    latest = max(files, key=os.path.getmtime)

    print("Using cache file: " + latest)

    with open(latest, "r") as f:
        data = json.load(f)

    spreads = parse_spreads(data)

    if spreads.empty:
        raise SystemExit("No spreads parsed from cache file.")

    return spreads


def main():
    print("Backfilling away odds and recalculating existing picks...")

    spreads = latest_cache_spreads()

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    if "opening_away_odds" not in col:
        print("Missing column: opening_away_odds")
        print("Run: python -m src.ensure_away_odds_columns")
        return

    now_utc = datetime.now(pytz.utc)

    updates = {}

    def add_update(row_num, col_name, value):
        if col_name not in col:
            return
        updates[(row_num, col[col_name])] = value

    backfilled_opening_away = 0
    recalculated_picks = 0
    cleared_picks = 0
    low_edge_reports = []

    for idx, row in sheet_df.iterrows():
        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

        if not home or not away or home == "NAN" or away == "NAN":
            continue

        mask = (spreads["home_team"] == home) & (spreads["away_team"] == away)
        match = spreads[mask]

        if match.empty:
            continue

        odds_row = match.iloc[0]
        row_num = idx + 2

        opening_line = row.get("opening_line")

        if not pd.isna(opening_line):
            current_opening_away = row.get("opening_away_odds")

            if pd.isna(current_opening_away) or str(current_opening_away).strip() == "":
                add_update(row_num, "opening_away_odds", odds_row["away_odds"])
                backfilled_opening_away += 1

        if not has_pick(row.get("pick_team")):
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff <= now_utc:
            continue

        try:
            model_p = float(row.get("model_probability"))
            home_spread = float(opening_line)
            home_odds = float(row.get("opening_odds"))
        except Exception:
            continue

        away_odds = safe_float(odds_row["away_odds"], -110.0)

        fair_home, fair_away = remove_vig(
            american_to_implied_prob(home_odds),
            american_to_implied_prob(away_odds),
        )

        pick_team = str(row.get("pick_team", "")).strip().upper()
        pick_side = str(row.get("pick_side", "")).strip().upper()

        if not pick_side:
            if pick_team == home:
                pick_side = "HOME"
            elif pick_team == away:
                pick_side = "AWAY"
            else:
                continue

        if pick_side == "HOME":
            fair_p = fair_home
            new_pick_line = home_spread
            new_pick_odds = home_odds
        elif pick_side == "AWAY":
            fair_p = fair_away
            new_pick_line = -home_spread
            new_pick_odds = away_odds
        else:
            continue

        edge = model_p - fair_p

        base_note = ""

        if not pd.isna(row.get("notes")):
            base_note = str(row.get("notes"))

        if "away_odds_backfilled" not in base_note:
            note = base_note + "; away_odds_backfilled" if base_note else "away_odds_backfilled"
        else:
            note = base_note

        matchup = away + " @ " + home

        if edge < MIN_EDGE_THRESHOLD:
            low_edge_reports.append(
                matchup
                + " | "
                + pick_team
                + " "
                + pick_side
                + " "
                + str(new_pick_line)
                + " @"
                + str(new_pick_odds)
                + " | model "
                + str(round(model_p * 100, 1))
                + "%"
                + " vs fair "
                + str(round(fair_p * 100, 1))
                + "%"
                + " | edge "
                + str(round(edge * 100, 1))
                + "%"
            )

            if CLEAR_LOW_EDGE:
                for c in [
                    "pick_team",
                    "pick_side",
                    "pick_line",
                    "pick_odds",
                    "model_probability",
                    "implied_probability_no_vig",
                    "edge",
                ]:
                    add_update(row_num, c, "")

                if not pd.isna(row.get("closing_line")):
                    new_status = "CLOSE_CAPTURED"
                else:
                    new_status = "OPEN_CAPTURED"

                add_update(row_num, "status", new_status)

                clear_note = note + "; CLEARED_LOW_EDGE_AFTER_AWAY_ODDS_RECALC"
                add_update(row_num, "notes", clear_note)

                cleared_picks += 1
                continue

        add_update(row_num, "pick_line", new_pick_line)
        add_update(row_num, "pick_odds", new_pick_odds)
        add_update(row_num, "implied_probability_no_vig", round(fair_p, 4))
        add_update(row_num, "edge", round(edge, 4))
        add_update(row_num, "notes", note)

        recalculated_picks += 1

    update_list = [
        {"row": r, "col": c, "value": v}
        for (r, c), v in updates.items()
    ]

    batch_update_cells(ws, update_list)

    print("Backfilled opening_away_odds for " + str(backfilled_opening_away) + " row(s).")
    print("Recalculated " + str(recalculated_picks) + " existing pick(s).")

    if low_edge_reports:
        print("")
        print("Picks below MIN_EDGE_THRESHOLD after away-odds recalculation:")

        for item in low_edge_reports:
            print("  LOW EDGE: " + item)

        print("")

        if CLEAR_LOW_EDGE:
            print("Cleared " + str(cleared_picks) + " low-edge future pick(s).")
        else:
            print("To clear these future low-edge picks, run:")
            print("  CLEAR_LOW_EDGE=1 python -m src.backfill_away_odds")
    else:
        print("No existing picks fell below the edge threshold after recalculation.")


if __name__ == "__main__":
    main()
