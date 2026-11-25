"""
Regular-season filter for the props capture and mid-week snapshots (owner decision 5, 2026-11-25).

The P2 and P4 gates are graded on regular-season games only, so src/props/capture_props.py and
src/midweek_snapshot.py spend nothing on playoff or preseason games. The rule, by kickoff month
(UTC):

  - September to December: always regular season. The schedule is not loaded and cannot block
    these games; the playoffs never start before January.
  - January to August: the event must fall inside its season's regular-season window from the
    nflverse schedule (data/nflverse_cache/games.csv, written by src.model_utils._load_sched):
    the first `game_type == "REG"` kickoff minus CUTOFF_GRACE to the last one plus CUTOFF_GRACE.
    Season = year if month >= 3, else year - 1. A new season's January games count as soon as
    games.csv holds its schedule; August preseason games are skipped.

Fail closed, for January to August only: if games.csv is missing or unreadable, or holds no REG
rows for the event's season, the event is skipped with a reason that is_fail_closed() flags, and
the caller exits 1 after spending nothing on it.

nflverse lists `gametime` in US Eastern time. Week 18 kickoffs are often placeholders (13:00 ET)
until the league sets the final slate, so a game up to CUTOFF_GRACE after the listed last kickoff
still counts. The first playoff game is always at least 5 days after the last regular-season one,
and the last preseason game at least 5 days before the first.

CLI (CI download step): python -m src.regular_season --fetch
"""

import argparse
import csv
import os
import sys
from collections import namedtuple
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

GAMES_PATH = os.path.join("data", "nflverse_cache", "games.csv")
CUTOFF_GRACE = timedelta(hours=24)
ALWAYS_REGULAR_MONTHS = range(9, 13)    # Sep-Dec (UTC): no playoff or preseason game
_ET = ZoneInfo("America/New_York")
_REQUIRED = ("season", "game_type", "gameday")
_FAIL_CLOSED_PREFIXES = ("no regular-season schedule", "regular-season schedule unavailable")

Window = namedtuple("Window", "first last")   # first and last REG kickoff of a season (UTC)


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
    """{season: Window(first, last REG kickoff, UTC)} from games.csv. Raises ScheduleUnavailable."""
    if not os.path.exists(path):
        raise ScheduleUnavailable(f"{path} not found")
    windows = {}
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
                w = windows.get(season)
                windows[season] = Window(min(w.first, k), max(w.last, k)) if w else Window(k, k)
    except ScheduleUnavailable:
        raise
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        raise ScheduleUnavailable(f"{path} unreadable: {type(e).__name__}: {e}") from e
    return windows


class LazyCutoffs:
    """Loads the schedule on first use only, so September-December runs never touch it."""

    def __init__(self, load_fn=load_reg_cutoffs):
        self._load_fn = load_fn
        self._loaded = False
        self.table = None
        self.error = None

    def get(self):
        """The {season: Window} table, or None if it could not be loaded (see .error)."""
        if not self._loaded:
            self._loaded = True
            try:
                self.table = self._load_fn()
            except ScheduleUnavailable as e:
                self.error = e
        return self.table


def _fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def skip_reason(kickoff, cutoffs):
    """None if the kickoff is a regular-season game, else why it is skipped.

    `cutoffs` is a {season: Window} table, a LazyCutoffs, or None (schedule unavailable)."""
    kickoff = kickoff.astimezone(timezone.utc)
    if kickoff.month in ALWAYS_REGULAR_MONTHS:
        return None
    error = None
    if isinstance(cutoffs, LazyCutoffs):
        error = cutoffs.error if cutoffs.get() is None else None
        cutoffs = cutoffs.table
    if cutoffs is None:
        return f"regular-season schedule unavailable ({error or 'not loaded'})"
    season = season_of(kickoff)
    w = cutoffs.get(season)
    if w is None:
        return f"no regular-season schedule for {season} in {GAMES_PATH}"
    if kickoff < w.first - CUTOFF_GRACE:
        return f"before the {season} regular season (first REG kickoff {_fmt(w.first)})"
    if kickoff > w.last + CUTOFF_GRACE:
        return f"after the {season} regular season (last REG kickoff {_fmt(w.last)})"
    return None


def is_fail_closed(reason):
    """True if the skip is for want of a schedule (exit 1), not a playoff or preseason game."""
    return bool(reason) and reason.startswith(_FAIL_CLOSED_PREFIXES)


def fetch(now=None):
    """Downloads the current season's schedule into games.csv (src.model_utils._load_sched).
    Never raises and always returns 0: September-December capture does not need the file, and
    January-August capture fails closed without it."""
    season = season_of(now or datetime.now(timezone.utc))
    try:
        from src.model_utils import _load_sched
        _load_sched([season])
    except Exception as e:  # noqa: BLE001 - reported; capture decides what it needs
        print(f"Schedule download for {season} failed: {type(e).__name__}: {e}")
    try:
        w = load_reg_cutoffs().get(season)
    except ScheduleUnavailable as e:
        print(f"Schedule unavailable after download ({e}); Sep-Dec games still count, "
              f"Jan-Aug games will be skipped.")
        return 0
    if w is None:
        print(f"{GAMES_PATH} has no {season} regular-season games yet.")
    else:
        print(f"{season} regular season {_fmt(w.first)} to {_fmt(w.last)} (grace {CUTOFF_GRACE}).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regular-season window from nflverse games.csv")
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
        print(f"{season}: {_fmt(cutoffs[season].first)} to {_fmt(cutoffs[season].last)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
