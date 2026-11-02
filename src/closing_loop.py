"""
Self-pacing closing line loop for GitHub Actions.

GitHub drops most dense cron triggers (observed: ~5 of ~100 scheduled runs per
gameday fire, hours apart), so a single run must stay alive and wake itself at
fixed leads before each kickoff instead of relying on the cron cadence.

Each iteration runs capture_closing(), then sleeps until the next
(kickoff - lead) target within the lookahead horizon. Exits once no target fits
before the runtime deadline, leaving time for the workflow to save the odds cache.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import pytz

from src.capture_closing import PREGAME_STATUSES, capture_closing, parse_kickoff
from src.sheet_utils import get_sheet_as_df

# Minutes before kickoff to wake: one post-inactives capture, one near-final capture.
WAKE_LEADS_MINUTES = [int(x) for x in os.getenv("CLOSING_WAKE_LEADS", "45,10").split(",") if x.strip()]
MAX_RUNTIME_MINUTES = float(os.getenv("CLOSING_LOOP_MAX_MINUTES", "320"))
MIN_SLEEP_SECONDS = 30


def upcoming_kickoffs(sheet_df, now_utc):
    """Distinct kickoff times of pregame rows that are still in the future."""
    if sheet_df is None or sheet_df.empty:
        return []
    kickoffs = set()
    for _, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()
        if status not in PREGAME_STATUSES:
            continue
        kickoff = parse_kickoff(row.get("kickoff_time"))
        if pd.isna(kickoff) or kickoff <= now_utc:
            continue
        kickoffs.add(kickoff)
    return sorted(kickoffs)


def next_wake_time(kickoffs, now_utc, deadline_utc, leads_minutes=WAKE_LEADS_MINUTES):
    """Earliest (kickoff - lead) strictly after now and no later than the deadline, else None."""
    earliest = now_utc + timedelta(seconds=MIN_SLEEP_SECONDS)
    targets = [
        k - timedelta(minutes=lead)
        for k in kickoffs
        for lead in leads_minutes
        if earliest <= k - timedelta(minutes=lead) <= deadline_utc
    ]
    return min(targets) if targets else None


def run_loop():
    start = datetime.now(pytz.utc)
    deadline = start + timedelta(minutes=MAX_RUNTIME_MINUTES)

    while True:
        capture_closing()

        now = datetime.now(pytz.utc)
        wake = next_wake_time(upcoming_kickoffs(get_sheet_as_df(), now), now, deadline)
        if wake is None:
            print("No closing targets before the runtime deadline. Exiting loop.")
            return

        print(f"Sleeping until {wake.isoformat()} (next closing target).")
        time.sleep(max(0.0, (wake - datetime.now(pytz.utc)).total_seconds()))


if __name__ == "__main__":
    run_loop()
