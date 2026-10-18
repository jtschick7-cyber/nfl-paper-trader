from datetime import datetime

import pandas as pd
import pytz

from src.odds_utils import fetch_odds_api, parse_spreads
from src.sheet_utils import get_sheet_as_df, get_master_sheet, to_native


def capture_opening():
    print("Running Tuesday Opening Capture...")

    sheet_df = get_sheet_as_df()
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

    updates_count = 0

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status != "PENDING":
            continue

        opening_line = row.get("opening_line")
        if not pd.isna(opening_line):
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

        safe_home_spread = to_native(odds_row["home_spread"])
        safe_home_odds = to_native(odds_row["home_odds"])
        safe_captured_at = datetime.now(pytz.utc).isoformat()

        ws.update_cell(
            sheet_row_num,
            col_opening_line,
            safe_home_spread,
        )

        ws.update_cell(
            sheet_row_num,
            col_opening_odds,
            safe_home_odds,
        )

        ws.update_cell(
            sheet_row_num,
            col_captured_at,
            safe_captured_at,
        )

        ws.update_cell(
            sheet_row_num,
            col_status,
            "OPEN_CAPTURED",
        )

        updates_count += 1

    print("Captured opening lines for " + str(updates_count) + " games.")


if __name__ == "__main__":
    capture_opening()
