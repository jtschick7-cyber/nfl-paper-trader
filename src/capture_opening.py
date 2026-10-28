from datetime import datetime

import pandas as pd
import pytz

from src.odds_utils import fetch_odds_api, parse_spreads
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def capture_opening():
    print("Running Tuesday Opening Capture...")

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    odds_data = fetch_odds_api()
    spreads = parse_spreads(odds_data)

    if spreads.empty:
        raw_event_count = len(odds_data) if odds_data else 0
        print("No spreads parsed. Raw API events returned: " + str(raw_event_count))

        if odds_data:
            first_event = odds_data[0]
            away = first_event.get("away_team", "")
            home = first_event.get("home_team", "")
            print("First event: " + str(away) + " @ " + str(home))

            books = first_event.get("bookmakers", [])[:3]
            for book in books:
                market_keys = [m.get("key") for m in book.get("markets", [])]
                print("Book: " + str(book.get("key")) + " | Markets: " + str(market_keys))

        return

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    required = [
        "opening_line",
        "opening_odds",
        "opening_captured_at",
        "status",
    ]

    missing = [c for c in required if c not in col]

    if missing:
        print("Missing required columns: " + str(missing))
        return

    now_utc = datetime.now(pytz.utc)

    updates = []
    games_updated = 0

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status != "PENDING":
            continue

        if not pd.isna(row.get("opening_line")):
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff <= now_utc:
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

        mask = (spreads["home_team"] == home) & (spreads["away_team"] == away)
        match = spreads[mask]

        if match.empty:
            continue

        odds_row = match.iloc[0]
        row_num = idx + 2

        updates.append({
            "row": row_num,
            "col": col["opening_line"],
            "value": odds_row["home_spread"],
        })

        updates.append({
            "row": row_num,
            "col": col["opening_odds"],
            "value": odds_row["home_odds"],
        })

        if "opening_away_odds" in col:
            updates.append({
                "row": row_num,
                "col": col["opening_away_odds"],
                "value": odds_row["away_odds"],
            })

        if "total_line" in col and pd.notna(odds_row.get("total_line")):
            updates.append({
                "row": row_num,
                "col": col["total_line"],
                "value": odds_row["total_line"],
            })

        updates.append({
            "row": row_num,
            "col": col["opening_captured_at"],
            "value": datetime.now(pytz.utc).isoformat(),
        })

        updates.append({
            "row": row_num,
            "col": col["status"],
            "value": "OPEN_CAPTURED",
        })

        games_updated += 1

    batch_update_cells(ws, updates)

    print("Captured opening lines for " + str(games_updated) + " future games.")


if __name__ == "__main__":
    capture_opening()
