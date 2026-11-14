from datetime import datetime

import pandas as pd
import pytz

from src.config import SEASON
from src.decision_window import decision_guard, print_missing_results, row_is_decidable
from src.model_utils import _load_sched
from src.odds_utils import fetch_odds_api, parse_spreads
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

# Decision-time capture needs a fresh quote, not the 12-hour default cache.
ODDS_MAX_AGE_SECONDS = 1800


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _cell(col, updates, row_num, name, value):
    if name in col:
        updates.append({"row": row_num, "col": col[name], "value": value})


def plan_opening_capture(sheet_df, spreads, col, schedule_df, now_utc):
    """Cell updates for opening capture. Pure: sheet rows, parsed spreads, header map, schedule, now.

    Only decidable games (current decision week, inside the horizon) are touched, and
    nothing at all when a finished game has no result yet.
    """
    decision_week, missing = decision_guard(sheet_df, schedule_df, now_utc)
    summary = {
        "decision_week": decision_week,
        "missing_results": missing,
        "captured": 0,
        "not_decidable": 0,
        "live_market": 0,
    }
    if missing:
        return [], summary

    updates = []
    now_iso = pd.Timestamp(now_utc).isoformat()

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()
        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff) or kickoff <= now_utc:
            continue

        if not row_is_decidable(row, now_utc, decision_week):
            summary["not_decidable"] += 1
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

        mask = (spreads["home_team"] == home) & (spreads["away_team"] == away)
        match = spreads[mask]

        if match.empty:
            continue

        odds_row = match.iloc[0]
        row_num = idx + 2

        if status == "OPEN_CAPTURED":
            if "total_line" in col and pd.isna(row.get("total_line")) and pd.notna(odds_row.get("total_line")):
                _cell(col, updates, row_num, "total_line", odds_row["total_line"])
            continue

        if status != "PENDING":
            continue

        if not pd.isna(row.get("opening_line")):
            continue

        mkt_lu = parse_kickoff(odds_row.get("market_last_update"))
        if pd.notna(mkt_lu) and mkt_lu > kickoff:
            print(f"Skipping live market for {away} @ {home}: market_last_update {mkt_lu.isoformat()} is after kickoff.")
            summary["live_market"] += 1
            continue

        _cell(col, updates, row_num, "opening_line", odds_row["home_spread"])
        _cell(col, updates, row_num, "opening_odds", odds_row["home_odds"])
        _cell(col, updates, row_num, "opening_away_odds", odds_row["away_odds"])
        if pd.notna(odds_row.get("total_line")):
            _cell(col, updates, row_num, "total_line", odds_row["total_line"])
        _cell(col, updates, row_num, "opening_captured_at", now_iso)
        _cell(col, updates, row_num, "opening_book", odds_row.get("bookmaker", ""))
        _cell(col, updates, row_num, "opening_quote_at", odds_row.get("market_last_update", ""))
        _cell(col, updates, row_num, "decision_regime", "in_week")
        _cell(col, updates, row_num, "status", "OPEN_CAPTURED")

        summary["captured"] += 1

    return updates, summary


def capture_opening():
    print("Running Tuesday Opening Capture...")

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    schedule_df = _load_sched([SEASON])
    now_utc = datetime.now(pytz.utc)

    # Check results before spending an Odds API call.
    decision_week, missing = decision_guard(sheet_df, schedule_df, now_utc)
    if missing:
        print_missing_results(missing)
        return
    print("Decision week: " + str(decision_week))

    odds_data = fetch_odds_api(max_age_seconds=ODDS_MAX_AGE_SECONDS)
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
        "week",
    ]

    missing_cols = [c for c in required if c not in col]

    if missing_cols:
        print("Missing required columns: " + str(missing_cols))
        return

    updates, summary = plan_opening_capture(sheet_df, spreads, col, schedule_df, datetime.now(pytz.utc))

    if summary["missing_results"]:
        print_missing_results(summary["missing_results"])
        return

    batch_update_cells(ws, updates)

    print("Skipped " + str(summary["not_decidable"]) + " future game(s) outside decision week " + str(decision_week) + ".")
    if summary["live_market"]:
        print("Skipped " + str(summary["live_market"]) + " game(s) with a live market.")
    print("Captured opening lines for " + str(summary["captured"]) + " decidable games.")


if __name__ == "__main__":
    capture_opening()
