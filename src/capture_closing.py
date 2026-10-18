import os
from datetime import datetime, timedelta

import pandas as pd
import pytz

from src.odds_utils import fetch_odds_api, parse_spreads
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

DRY_RUN_CLOSE = os.getenv("DRY_RUN_CLOSE", "").strip().lower() in {"1", "true", "yes", "y"}


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def row_needs_closing(row, now_utc):
    status = str(row.get("status", "")).strip().upper()

    if status not in {"PENDING", "OPEN_CAPTURED", "CLOSE_CAPTURED"}:
        return False

    if not pd.isna(row.get("closing_line")):
        return False

    kickoff = parse_kickoff(row.get("kickoff_time"))

    if pd.isna(kickoff):
        return False

    delta = kickoff - now_utc

    return timedelta(minutes=40) <= delta <= timedelta(minutes=60)


def select_dry_run_target(sheet_df, now_utc):
    candidates = []

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status not in {"PENDING", "OPEN_CAPTURED", "CLOSE_CAPTURED"}:
            continue

        if not pd.isna(row.get("closing_line")):
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff > now_utc - timedelta(hours=12):
            candidates.append((kickoff, idx, row))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])

    return [(candidates[0][1], candidates[0][2])]


def capture_closing():
    print("Running closing line monitor...")

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    now_utc = datetime.now(pytz.utc)

    normal_targets = []

    for idx, row in sheet_df.iterrows():
        if row_needs_closing(row, now_utc):
            normal_targets.append((idx, row))

    dry_targets = []

    if DRY_RUN_CLOSE:
        if normal_targets:
            dry_targets = normal_targets
        else:
            dry_targets = select_dry_run_target(sheet_df, now_utc)

            if dry_targets:
                print("DRY_RUN_CLOSE enabled. Selecting nearest upcoming game without writing to sheet.")

    if not normal_targets and not dry_targets:
        print("No games need closing capture.")
        return

    force_refresh = DRY_RUN_CLOSE and not normal_targets

    odds_data = fetch_odds_api(max_age_seconds=300, force_refresh=force_refresh)
    spreads = parse_spreads(odds_data)

    if spreads.empty:
        print("No spreads parsed from Odds API.")
        return

    if DRY_RUN_CLOSE:
        print("Dry run only. The sheet will not be updated.")

        for idx, row in dry_targets:
            home = str(row.get("home_team", "")).strip().upper()
            away = str(row.get("away_team", "")).strip().upper()

            mask = (spreads["home_team"] == home) & (spreads["away_team"] == away)
            match = spreads[mask]

            if match.empty:
                print("Would not capture: no odds match for " + away + " @ " + home + ".")
            else:
                odds_row = match.iloc[0]

                print(
                    "Would capture closing for "
                    + away
                    + " @ "
                    + home
                    + ": home spread "
                    + str(odds_row["home_spread"])
                    + ", home odds "
                    + str(odds_row["home_odds"])
                    + ", away odds "
                    + str(odds_row["away_odds"])
                    + ", book "
                    + str(odds_row["bookmaker"])
                    + "."
                )

        return

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    required = [
        "closing_line",
        "closing_odds",
        "closing_captured_at",
        "status",
    ]

    missing = [c for c in required if c not in col]

    if missing:
        print("Missing required columns: " + str(missing))
        return

    updates = []
    games_updated = 0

    for idx, row in normal_targets:
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
            "col": col["closing_line"],
            "value": odds_row["home_spread"],
        })

        updates.append({
            "row": row_num,
            "col": col["closing_odds"],
            "value": odds_row["home_odds"],
        })

        if "closing_away_odds" in col:
            updates.append({
                "row": row_num,
                "col": col["closing_away_odds"],
                "value": odds_row["away_odds"],
            })

        updates.append({
            "row": row_num,
            "col": col["closing_captured_at"],
            "value": datetime.now(pytz.utc).isoformat(),
        })

        updates.append({
            "row": row_num,
            "col": col["status"],
            "value": "CLOSE_CAPTURED",
        })

        games_updated += 1

    batch_update_cells(ws, updates)

    print("Captured closing lines for " + str(games_updated) + " games.")


if __name__ == "__main__":
    capture_closing()
