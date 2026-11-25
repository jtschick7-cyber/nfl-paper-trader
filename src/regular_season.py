"""
Regular-season cutoff for the props capture and mid-week snapshots (owner decision 5, 2026-11-25).

The P2 and P4 gates are graded on regular-season games only, so src/props/capture_props.py and
src/midweek_snapshot.py spend nothing on an event that kicks off after its season's last
regular-season kickoff. The cutoff is the latest `game_type == "REG"` kickoff in the nflverse
schedule (data/nflverse_cache/games.csv, written by src.model_utils._load_sched) for the event's
season (season = year if month >= 3, else year - 1). A new season switches capture back on as
soon as games.csv holds its schedule.

Fail closed: a missing or unreadable games.csv raises ScheduleUnavailable, and an event whose
season has no REG rows is skipped with a reason; callers then spend nothing.

nflverse lists `gametime` in US Eastern time. Week 18 kickoffs are often placeholders (13:00 ET)
until the league sets the final slate, so a game up to CUTOFF_GRACE after the listed cutoff still
counts. The first playoff game is always at least 5 days after the last regular-season one.

CLI (CI download step): python -m src.regular_season --fetch
"""

import argparse
import csv
import os
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

GAMES_PATH = os.path.join("data", "nflverse_cache", "games.csv")
CUTOFF_GRACE = timedelta(hours=24)
_ET = ZoneInfo("America/New_York")
_REQUIRED = ("season", "game_type", "gameday")


class ScheduleUnavailable(Exception):
    """The nflverse schedule cannot be read, so no cutoff is known."""


def season_of(dt):
    dt = dt.astimezone(timezone.utc)
    return dt.year if dt.month >= 3 else dt.year - 1


def _kickoff_utc(gameday, gametime):
    day = datetime.strptime(gameday.strip(), "%Y-%m-%d").date()
    try:
        t = datetime.strptime(gametime.strip(), "%H:%M").time()
    except (AttributeError, ValueError):
        t = time(23, 59)   # unknown time: the end of the listed day
    return datetime.combine(day, t, tzinfo=_ET).astimezone(timezone.utc)


def load_reg_cutoffs(path=GAMES_PATH):
    """{season: latest REG kickoff (UTC)} from games.csv. Raises ScheduleUnavailable."""
    if not os.path.exists(path):
        raise ScheduleUnavailable(f"{path} not found")
    cutoffs = {}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise ScheduleUnavailable(f"{path} lacks column(s) {', '.join(missing)}")
            for row in reader:
                if str(row.get("game_type", "")).strip().upper() != "REG":
                    continue
                try:
                    season = int(float(row["season"]))
                    k = _kickoff_utc(row["gameday"], row.get("gametime"))
                except (TypeError, ValueError):
                    continue
                if season not in cutoffs or k > cutoffs[season]:
                    cutoffs[season] = k
    except ScheduleUnavailable:
        raise
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        raise ScheduleUnavailable(f"{path} unreadable: {type(e).__name__}: {e}") from e
    return cutoffs


def skip_reason(kickoff, cutoffs):
    """None if the kickoff falls inside its season's regular season, else why it is skipped."""
    season = season_of(kickoff)
    cutoff = cutoffs.get(season)
    if cutoff is None:
        return f"no regular-season schedule for {season} in {GAMES_PATH}"
    if kickoff > cutoff + CUTOFF_GRACE:
        return (f"after the {season} regular season (last REG kickoff "
                f"{cutoff.strftime('%Y-%m-%dT%H:%MZ')})")
    return None


def is_missing_season(reason):
    return bool(reason) and reason.startswith("no regular-season schedule")


def fetch(now=None):
    """Downloads the current season's schedule into games.csv (src.model_utils._load_sched).
    Never raises: the capture step fails closed if the file is still unusable."""
    season = season_of(now or datetime.now(timezone.utc))
    try:
        from src.model_utils import _load_sched
        _load_sched([season])
    except Exception as e:  # noqa: BLE001 - reported; the capture step then fails closed
        print(f"Schedule download for {season} failed: {type(e).__name__}: {e}")
    try:
        cutoff = load_reg_cutoffs().get(season)
    except ScheduleUnavailable as e:
        print(f"Schedule unavailable after download: {e}")
        return 0
    if cutoff is None:
        print(f"{GAMES_PATH} has no {season} regular-season games yet.")
    else:
        print(f"{season} last regular-season kickoff {cutoff.strftime('%Y-%m-%dT%H:%MZ')} "
              f"(grace {CUTOFF_GRACE}).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regular-season cutoff from nflverse games.csv")
    parser.add_argument("--fetch", action="store_true", help="download the current season's schedule first")
    args = parser.parse_args(argv)
    if args.fetch:
        return fetch()
    try:
        cutoffs = load_reg_cutoffs()
    except ScheduleUnavailable as e:
        print(f"Schedule unavailable: {e}")
        return 1
    for season in sorted(cutoffs)[-3:]:
        print(f"{season}: {cutoffs[season].strftime('%Y-%m-%dT%H:%MZ')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
