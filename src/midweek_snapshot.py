"""
Mid-week game-odds snapshot (owner-approved 2026-11-24, P2 owner question).

One extra full-slate /odds snapshot on Wednesday, Thursday and Saturday from 17:30 UTC, so the
P2 top-down gate (src/topdown_v2.py, 80 in-week offers on 35 games) can fill this season. The
request is the existing game-odds request (src/odds_utils.fetch_odds_api: regions=us,
markets=spreads,totals, no bookmakers filter), about 2 credits. The file is written as
odds_<YYYYMMDDTHHMMSSZ>.json, exactly like _save_timestamped_snapshot, and
scripts/push_odds_snapshots.sh archives it into odds-data snapshots/YYYY/MM/, where
src.topdown_v2 and src.topdown_replay (Phase 3) read it unchanged.

Self-gating (the Worker dispatches this on every Mon/Wed-Sat 13:30/17:30/21:30 wake, and a
GitHub backup cron runs late):
  - only Wed/Thu/Sat (UTC), and only at or after 17:25 UTC (the 17:30 wake, or a later one if
    that wake failed);
  - at most one snapshot a day: no call if the odds-data archive or the local output dir holds
    any snapshot taken since 17:25 UTC today (a closing-loop snapshot then counts too);
  - the ledger is read before any paid call; if it cannot be read, nothing is called.

Credit guards: at most MAX_CALLS_PER_RUN paid calls (a second only to retry a network or 5xx
error), and never below QUOTA_FLOOR remaining (the free /events call reports the quota; while
the quota is unknown, at most one call), the same floor as src/props/capture_props.py.

Exit 0 when nothing was due or the snapshot was written; 1 when a ledger, quota or capture call
failed or a guard blocked a due snapshot.

CLI: python -m src.midweek_snapshot
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from src.config import MARKETS, REGIONS, SPORT_KEY
from src.load_odds_archive import DEFAULT_REF, list_snapshots, snapshot_time

OUT_DIR = "data/midweek_cache/snapshots"
WEEKDAYS = (2, 3, 5)                 # Wed, Thu, Sat (datetime.weekday)
WINDOW_START_UTC = (17, 25)          # the 17:30 wake, with 5 min of clock tolerance
MAX_CALLS_PER_RUN = int(os.getenv("MIDWEEK_MAX_CALLS", "2"))
QUOTA_FLOOR = int(os.getenv("MIDWEEK_QUOTA_FLOOR", "80"))
RETRY_SLEEP_SECONDS = 20

API = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}"
_LOCAL_NAME = re.compile(r"^odds_(\d{8}T\d{6}Z)\.json$")


def _utcnow():
    return datetime.now(timezone.utc)


def odds_request(api_key):
    """(url, params) of the existing game-odds call in src/odds_utils.fetch_odds_api."""
    return f"{API}/odds/", {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }


def estimated_cost():
    """The Odds API bills markets x regions per /odds call."""
    return len([m for m in MARKETS.split(",") if m.strip()]) * len([r for r in REGIONS.split(",") if r.strip()])


# Gate and ledger ---------------------------------------------------------------------

def window_start(now):
    h, m = WINDOW_START_UTC
    return now.astimezone(timezone.utc).replace(hour=h, minute=m, second=0, microsecond=0)


def gate_reason(now):
    """None if today's snapshot may be taken now, else why not."""
    now = now.astimezone(timezone.utc)
    if now.weekday() not in WEEKDAYS:
        return f"weekday {now:%a} is not Wed/Thu/Sat"
    if now < window_start(now):
        return f"{now:%H:%M} UTC is before the {WINDOW_START_UTC[0]:02d}:{WINDOW_START_UTC[1]:02d} window"
    return None


def taken_today(times, now):
    """True if any snapshot time falls between today's window start and now (inclusive)."""
    start = window_start(now)
    return any(start <= t <= now for t in times)


def load_ledger(source=DEFAULT_REF, repo="."):
    """Snapshot times listed in the odds-data archive (dir or git ref)."""
    return [t for t, _ in list_snapshots(source, repo)]


def refresh_ledger():
    """Fetches the odds-data tip, then lists it. Raises if the archive cannot be read."""
    subprocess.run(["git", "fetch", "--quiet", "--depth=1", "origin",
                    "+refs/heads/odds-data:refs/remotes/origin/odds-data"],
                   check=True, capture_output=True)
    return load_ledger(DEFAULT_REF)


def local_times(out_dir):
    if not os.path.isdir(out_dir):
        return []
    out = []
    for name in os.listdir(out_dir):
        if _LOCAL_NAME.match(name):
            out.append(snapshot_time(name + ".gz"))
    return out


# Calls -------------------------------------------------------------------------------

def _int_header(headers, name):
    try:
        return int(float(headers.get(name)))
    except (TypeError, ValueError):
        return None


def _redact(text, api_key):
    return str(text).replace(api_key, "***") if api_key else str(text)


def fetch_quota(api_key):
    """Requests remaining from the free /events call (None if the header is absent). Raises on error."""
    resp = requests.get(f"{API}/events", params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    resp.raise_for_status()
    return _int_header(resp.headers, "x-requests-remaining")


def _block_reason(calls, remaining, quota_known, cost, max_calls, floor):
    if calls >= max_calls:
        return f"per-run cap ({calls}/{max_calls} calls)"
    if not quota_known:
        return "quota unknown after one call" if calls >= 1 else None
    if remaining - cost < floor:
        return f"quota floor ({remaining} left, cost {cost}, floor {floor})"
    return None


def run(*, api_key, now=None, out_dir=OUT_DIR, ledger_fn=refresh_ledger, sleep_fn=time.sleep,
        now_fn=_utcnow, max_calls=MAX_CALLS_PER_RUN, floor=QUOTA_FLOOR):
    """Takes today's snapshot if it is due and allowed. Returns the process exit code."""
    given = now
    now = now or now_fn()
    reason = gate_reason(now)
    if reason:
        print(f"No mid-week snapshot due: {reason}.")
        return 0

    try:
        times = list(ledger_fn())
    except Exception as e:  # noqa: BLE001 - without the ledger a second paid call could go out
        print(f"Ledger read failed, calling nothing: {type(e).__name__}: {_redact(e, api_key)}")
        return 1
    if taken_today(times + local_times(out_dir), now):
        print(f"A snapshot since {window_start(now):%H:%M} UTC today is already archived; nothing to do.")
        return 0

    try:
        remaining = fetch_quota(api_key)
    except Exception as e:  # noqa: BLE001
        print(f"Quota check (/events) failed, calling nothing: {type(e).__name__}: {_redact(e, api_key)}")
        return 1
    quota_known = remaining is not None
    cost = estimated_cost()
    url, params = odds_request(api_key)
    print(f"Quota remaining {remaining}, estimated cost {cost}, floor {floor}, cap {max_calls} call(s).")

    calls = 0
    while True:
        blocked = _block_reason(calls, remaining, quota_known, cost, max_calls, floor)
        if blocked:
            print(f"Mid-week snapshot not taken: {blocked}.")
            return 1
        if calls:
            sleep_fn(RETRY_SLEEP_SECONDS)
        calls += 1
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"Odds call {calls} failed: {type(e).__name__}: {_redact(e, api_key)}")
            if quota_known:
                remaining -= cost   # assume it was billed
            continue
        left = _int_header(resp.headers, "x-requests-remaining")
        if left is not None:
            remaining, quota_known = left, True
        elif quota_known:
            remaining -= cost
        if resp.status_code >= 500 or resp.status_code == 429:
            print(f"Odds call {calls}: HTTP {resp.status_code}; retrying if the guards allow.")
            continue
        if resp.status_code >= 400:
            print(f"Odds call {calls}: HTTP {resp.status_code}; not retried.")
            return 1
        try:
            data = resp.json()
        except ValueError:
            data = None
        if not isinstance(data, list):
            print("Odds response is not a list of events; nothing written.")
            return 1
        captured = (given or now_fn()).astimezone(timezone.utc)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"odds_{captured.strftime('%Y%m%dT%H%M%SZ')}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Wrote {path}: {len(data)} event(s), cost {resp.headers.get('x-requests-last')}, "
              f"remaining {remaining}.")
        return 0


def main():
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY is missing.")
        return 1
    return run(api_key=api_key)


if __name__ == "__main__":
    sys.exit(main())
