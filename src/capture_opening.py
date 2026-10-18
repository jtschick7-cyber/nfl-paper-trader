from datetime import datetime, timedelta

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

    col_opening_line = headers.index("opening_line") + 1
    col_opening_odds = headers.index("opening_odds") + 1
    col_captured_at = headers.index("opening_captured_at") + 1
    col_status = headers.index("status") + 1

    now_utc = datetime.now(pytz.utc)

    updates = []
    games_updated = 0

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status != "PENDING":
            continue

        opening_line = row.get("opening_line")
        if not pd.isna(opening_line):
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff <= now_utc:
            continue

        sheet_home = str(row.get("home_team", "")).strip().upper()
        sheet_away = str(row.get("away_team", "")).strip().upper()

        mask_home = spreads["home_team"] == sheet_home
        mask_away = spreads["away_team"] == sheet_away
        match = spreads[mask_home & mask_away]

        if match.empty:
            continue

        odds_row = match.iloc[0]
        sheet_row_num = idx + 2

        updates.append({
            "row": sheet_row_num,
            "col": col_opening_line,
            "value": odds_row["home_spread"],
        })

        updates.append({
            "row": sheet_row_num,
            "col": col_opening_odds,
            "value": odds_row["home_odds"],
        })

        updates.append({
            "row": sheet_row_num,
            "col": col_captured_at,
            "value": datetime.now(pytz.utc).isoformat(),
        })

        updates.append({
            "row": sheet_row_num,
            "col": col_status,
            "value": "OPEN_CAPTURED",
        })

        games_updated += 1

    batch_update_cells(ws, updates)

    print("Captured opening lines for " + str(games_updated) + " future games.")


if __name__ == "__main__":
    capture_opening()
