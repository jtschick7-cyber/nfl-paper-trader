"""
Self-pacing capture of player_receptions odds for every NFL game (task P1).

Two snapshots per game, archived on `odds-data` under props/ by scripts/push_prop_snapshots.sh:
  open:  min(kickoff - 4h, first Friday 21:00 UTC at or after kickoff - 4 days)
  close: kickoff - 25 min (due from kickoff - 40 to kickoff - 5)

The game list comes from the free /events endpoint. The archive plus local files are the ledger,
so a late or duplicate run catches up and never pays twice for the same (phase, event). Like
src/closing_loop.py, one run stays alive and sleeps to its next target within the deadline.

Credit guards (checked before every call): at most MAX_CREDITS_PER_RUN per run, earliest kickoffs
first; never below QUOTA_FLOOR remaining; while the quota is unknown, at most one call.

CLI: python -m src.props.capture_props [--probe]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from src.props import PROPS_CACHE_DIR
from src.props.load_prop_archive import DEFAULT_REF, list_prop_snapshots, parse_prop_name

MARKET = "player_receptions"
_DEFAULT_BOOKS = ("pinnacle,draftkings,fanduel,betmgm,williamhill_us,"
                  "betrivers,betonlineag,bovada,betus,mybookieag")
BOOKMAKERS = [b.strip() for b in os.getenv("PROPS_BOOKMAKERS", _DEFAULT_BOOKS).split(",") if b.strip()]
CLOSE_LEAD_MIN = int(os.getenv("PROPS_CLOSE_LEAD_MIN", "25"))
CLOSE_WINDOW_MIN = (40, 5)   # close is due from kickoff-40 to kickoff-5
OPEN_LATEST_MIN = 120        # an open is no longer due after kickoff-2h
OPEN_LEAD_HOURS = 4
HORIZON_DAYS = 6
MAX_CREDITS_PER_RUN = int(os.getenv("PROPS_MAX_CREDITS", "20"))
QUOTA_FLOOR = int(os.getenv("PROPS_QUOTA_FLOOR", "80"))
MAX_RUNTIME_MINUTES = float(os.getenv("PROPS_MAX_MINUTES", "320"))   # workflow timeout 345
MIN_SLEEP_SECONDS = 30

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
_EVENT_ID = re.compile(r"[A-Za-z0-9-]+")

assert 1 <= len(BOOKMAKERS) <= 10, "The Odds API bills each 10 bookmakers as one region."


def _utcnow():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def _parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)


# Targets and due rules ---------------------------------------------------------------

def _first_friday_2100_at_or_after(t):
    days = (4 - t.weekday()) % 7
    f = (t + timedelta(days=days)).replace(hour=21, minute=0, second=0, microsecond=0)
    return f if f >= t else f + timedelta(days=7)


def open_target(kickoff):
    """After Friday's injury reports, or kickoff - 4h when that comes first (TNF, holidays)."""
    return min(kickoff - timedelta(hours=OPEN_LEAD_HOURS),
               _first_friday_2100_at_or_after(kickoff - timedelta(days=4)))


def close_target(kickoff):
    return kickoff - timedelta(minutes=CLOSE_LEAD_MIN)


def _open_due(k, now):
    return open_target(k) <= now <= k - timedelta(minutes=OPEN_LATEST_MIN)


def _close_due(k, now):
    return k - timedelta(minutes=CLOSE_WINDOW_MIN[0]) <= now <= k - timedelta(minutes=CLOSE_WINDOW_MIN[1])


def upcoming(events, now):
    """[(kickoff, event_id)] for events kicking off after now and within the horizon."""
    out = []
    for ev in events or []:
        eid = str(ev.get("id", ""))
        if not _EVENT_ID.fullmatch(eid):
            print(f"Skipping event with unusable id {eid!r}.")
            continue
        try:
            k = _parse_time(ev["commence_time"])
        except (KeyError, ValueError):
            continue
        if now < k <= now + timedelta(days=HORIZON_DAYS):
            out.append((k, eid))
    return sorted(out)


def due_items(events, ledger, now):
    """Captures due now and not yet in the ledger: [{phase, event_id, kickoff, target_at}]."""
    items = []
    for k, eid in upcoming(events, now):
        if ("open", eid) not in ledger and _open_due(k, now):
            items.append({"phase": "open", "event_id": eid, "kickoff": k, "target_at": open_target(k)})
        if ("close", eid) not in ledger and _close_due(k, now):
            items.append({"phase": "close", "event_id": eid, "kickoff": k, "target_at": close_target(k)})
    return items


def next_wake(events, ledger, now, deadline):
    """Earliest future open/close target not yet captured and no later than the deadline, else None."""
    earliest = now + timedelta(seconds=MIN_SLEEP_SECONDS)
    targets = []
    for k, eid in upcoming(events, now):
        if ("open", eid) not in ledger:
            targets.append(open_target(k))
        if ("close", eid) not in ledger:
            targets.append(close_target(k))
    targets = [t for t in targets if earliest <= t <= deadline]
    return min(targets) if targets else None


# Ledger ------------------------------------------------------------------------------

def load_ledger(archive_source, local_dir, repo="."):
    """{(phase, event_id)} from the archive (dir or git ref) plus local raw files."""
    ledger = set()
    for _, path in list_prop_snapshots(archive_source, repo):
        p = parse_prop_name(path)
        ledger.add((p["phase"], p["event_id"]))
    if os.path.isdir(local_dir):
        for name in os.listdir(local_dir):
            p = parse_prop_name(name)
            if p:
                ledger.add((p["phase"], p["event_id"]))
    return ledger


def refresh_ledger(local_dir=PROPS_CACHE_DIR):
    """Fetches the archive branch tip, then reads the ledger. Raises if the archive is unreadable."""
    subprocess.run(["git", "fetch", "--quiet", "--depth=1", "origin",
                    "+refs/heads/odds-data:refs/remotes/origin/odds-data"],
                   check=True, capture_output=True)
    return load_ledger(DEFAULT_REF, local_dir)


def push_snapshots():
    result = subprocess.run(["bash", "scripts/push_prop_snapshots.sh"])
    if result.returncode != 0:
        print("Push of prop snapshots failed; the workflow's final step retries it.")


# Credits -----------------------------------------------------------------------------

def _int_header(headers, name):
    try:
        return int(float(headers.get(name)))
    except (TypeError, ValueError):
        return None


class CreditBudget:
    """Per-run cap and quota floor. Costs count 1 per call until x-requests-last is known."""

    def __init__(self, cap=MAX_CREDITS_PER_RUN, floor=QUOTA_FLOOR):
        self.cap, self.floor = cap, floor
        self.spent = 0
        self.remaining = None
        self.used = None
        self.last = None
        self.calls_without_quota = 0

    def observe(self, headers):
        """Reads quota headers from any response (the events call reports them at no cost)."""
        remaining = _int_header(headers, "x-requests-remaining")
        used = _int_header(headers, "x-requests-used")
        if remaining is not None:
            self.remaining = remaining
        if used is not None:
            self.used = used

    def next_cost(self):
        return max(1, self.last or 0)

    def check(self):
        """None if a capture call may go out now, else the reason it may not."""
        cost = self.next_cost()
        if self.spent + cost > self.cap:
            return f"per-run cap ({self.spent}/{self.cap} spent)"
        if self.remaining is None:
            return "quota unknown after one call" if self.calls_without_quota >= 1 else None
        if self.remaining - cost < self.floor:
            return f"quota floor ({self.remaining} left, floor {self.floor})"
        return None

    def record_call(self, headers):
        if self.remaining is None:
            self.calls_without_quota += 1
        self.observe(headers)
        last = _int_header(headers, "x-requests-last")
        if last is not None:
            self.last = last
        self.spent += last if last is not None else 1


# Capture -----------------------------------------------------------------------------

def _redact(text, api_key):
    return str(text).replace(api_key, "***") if api_key else str(text)


def fetch_events(api_key, budget=None):
    """The free /events list, or raises."""
    resp = requests.get(f"{API}/events", params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    if budget is not None:
        budget.observe(resp.headers)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("events response is not a list")
    return data


def snapshot_name(phase, event_id, captured_at):
    return f"props_{phase}_{event_id}_{captured_at.strftime('%Y%m%dT%H%M%SZ')}.json"


def capture_items(items, budget, api_key, out_dir, now):
    """Captures items in kickoff order while the budget allows.

    Returns (written paths, skipped items each with a "reason")."""
    written, skipped = [], []
    os.makedirs(out_dir, exist_ok=True)
    for item in sorted(items, key=lambda i: (i["kickoff"], i["event_id"], i["phase"])):
        reason = budget.check()
        if reason:
            skipped.append({**item, "reason": reason})
            continue
        url = f"{API}/events/{item['event_id']}/odds"
        params = {"apiKey": api_key, "bookmakers": ",".join(BOOKMAKERS), "markets": MARKET,
                  "oddsFormat": "american", "dateFormat": "iso"}
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            budget.record_call({})
            skipped.append({**item, "reason": f"error: {type(e).__name__}: {_redact(e, api_key)}"})
            continue
        budget.record_call(resp.headers)
        if resp.status_code >= 400:
            skipped.append({**item, "reason": f"error: HTTP {resp.status_code}"})
            continue
        try:
            payload = resp.json()
        except ValueError:
            skipped.append({**item, "reason": "error: response is not JSON"})
            continue
        captured_at = now.replace(microsecond=0)
        doc = {
            "schema": 1,
            "phase": item["phase"],
            "market": MARKET,
            "captured_at": _iso(captured_at),
            "target_at": _iso(item.get("target_at")),
            "kickoff": _iso(item["kickoff"]),
            "minutes_to_kickoff": round((item["kickoff"] - now).total_seconds() / 60, 1),
            "bookmakers_requested": list(BOOKMAKERS),
            "quota": {"used": _int_header(resp.headers, "x-requests-used"),
                      "remaining": _int_header(resp.headers, "x-requests-remaining"),
                      "last": _int_header(resp.headers, "x-requests-last")},
            "response": payload,
        }
        path = os.path.join(out_dir, snapshot_name(item["phase"], item["event_id"], captured_at))
        with open(path, "w") as f:
            json.dump(doc, f)
        books = payload.get("bookmakers", []) if isinstance(payload, dict) else []
        print(f"Captured {item['phase']} {item['event_id']} (kickoff {_iso(item['kickoff'])}): "
              f"{len(books)} book(s), cost {doc['quota']['last']}, remaining {doc['quota']['remaining']}.")
        written.append(path)
    for s in skipped:
        print(f"Skipped {s['phase']} {s['event_id']} (kickoff {_iso(s['kickoff'])}): {s['reason']}.")
    return written, skipped


# Loop --------------------------------------------------------------------------------

def run_loop(probe=False, *, api_key, out_dir=PROPS_CACHE_DIR, now_fn=_utcnow, sleep_fn=time.sleep,
             ledger_fn=None, push_fn=push_snapshots, max_minutes=MAX_RUNTIME_MINUTES,
             cap=MAX_CREDITS_PER_RUN, floor=QUOTA_FLOOR):
    """Returns the process exit code: 1 if any events/ledger read or capture failed, or a due
    capture was skipped by a credit guard."""
    ledger_fn = ledger_fn or (lambda: refresh_ledger(out_dir))
    start = now_fn()
    deadline = start + timedelta(minutes=max_minutes)
    budget = CreditBudget(cap, floor)
    events, failed = None, False

    while True:
        now = now_fn()
        try:
            events = fetch_events(api_key, budget)
        except Exception as e:  # noqa: BLE001 - logged, retried at the next wake
            print(f"Events call failed: {type(e).__name__}: {_redact(e, api_key)}")
            failed = True
        if events is None:
            return 1

        if probe:
            ups = upcoming(events, now)
            if not ups:
                print("Probe: no upcoming event.")
                return 1
            k, eid = ups[0]
            written, skipped = capture_items([{"phase": "probe", "event_id": eid, "kickoff": k,
                                               "target_at": None}], budget, api_key, out_dir, now)
            if written:
                push_fn()
            return 0 if written else 1

        try:
            ledger = ledger_fn()
        except Exception as e:  # noqa: BLE001 - without the ledger a capture could be paid twice
            print(f"Ledger read failed, capturing nothing this wake: {type(e).__name__}: {e}")
            failed, ledger, items = True, None, []
        else:
            items = due_items(events, ledger, now)

        if items:
            written, skipped = capture_items(items, budget, api_key, out_dir, now)
            failed = failed or bool(skipped)
            if written:
                push_fn()
                ledger |= {(p["phase"], p["event_id"]) for p in map(parse_prop_name, written)}
        blocked = budget.check()
        if blocked:
            print(f"No further captures allowed this run ({blocked}). Spent {budget.spent} credit(s). Exiting.")
            return 1 if failed else 0

        wake = next_wake(events, ledger if ledger is not None else set(), now_fn(), deadline)
        if wake is None:
            print(f"No prop targets before the runtime deadline. Spent {budget.spent} credit(s). Exiting.")
            return 1 if failed else 0
        print(f"Sleeping until {_iso(wake)} (next prop target).")
        sleep_fn(max(0.0, (wake - now_fn()).total_seconds()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--probe", action="store_true",
                        help="capture the earliest upcoming event once as phase 'probe' (1 credit)")
    args = parser.parse_args(argv)
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY is missing.")
        return 1
    print(f"Props capture: market {MARKET}, books {','.join(BOOKMAKERS)}, cap {MAX_CREDITS_PER_RUN}, "
          f"floor {QUOTA_FLOOR}, probe {args.probe}.")
    return run_loop(args.probe, api_key=api_key)


if __name__ == "__main__":
    sys.exit(main())
