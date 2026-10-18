import os
from datetime import datetime

import pandas as pd
import pytz

from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def has_existing_pick(value):
    if pd.isna(value):
        return False

    s = str(value).strip()

    if s == "":
        return False

    if s.lower() == "nan":
        return False

    return True


def main():
    if os.getenv("CLEAR_PAST_PICKS") != "1":
        print("This is a one-time cleanup script.")
        print("Run it with:")
        print("")
        print("  CLEAR_PAST_PICKS=1 python -m src.clear_past_picks")
        print("")
        print("It will clear pick fields for unsettled games whose kickoff time is already in the past.")
        return

    print("Clearing past unsettled picks...")

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    now_utc = datetime.now(pytz.utc)

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    pick_cols = [
        "pick_team",
        "pick_side",
        "pick_line",
        "pick_odds",
        "model_probability",
        "implied_probability_no_vig",
        "edge",
    ]

    updates = []
    cleared_count = 0

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status in {"SETTLED", "RESULT_UPDATED", "ERROR"}:
            continue

        if not has_existing_pick(row.get("pick_team")):
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff > now_utc:
            continue

        row_num = idx + 2

        for c in pick_cols:
            if c in col:
                updates.append({
                    "row": row_num,
                    "col": col[c],
                    "value": "",
                })

        if "notes" in col:
            updates.append({
                "row": row_num,
                "col": col["notes"],
                "value": "CLEARED_PAST_PICK",
            })

        if "status" in col:
            opening_line = row.get("opening_line")

            if pd.isna(opening_line):
                new_status = "PENDING"
            else:
                new_status = "OPEN_CAPTURED"

            updates.append({
                "row": row_num,
                "col": col["status"],
                "value": new_status,
            })

        cleared_count += 1

    batch_update_cells(ws, updates)

    print("Cleared past unsettled picks for " + str(cleared_count) + " game(s).")
    print("Now run:")
    print("")
    print("  python -m src.update_results")
    print("  python -m src.generate_picks")


if __name__ == "__main__":
    main()
