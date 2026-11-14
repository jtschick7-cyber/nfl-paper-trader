import numpy as np
import pandas as pd

from src.decision_window import (
    current_decision_week,
    is_decidable,
    missing_results,
    previous_game_day,
    rest_days,
)


def ts(s):
    return pd.Timestamp(s, tz="UTC")


# Synthetic weeks 11-13 (UTC kickoffs). Thursday 20:15 ET is 01:15 UTC on Friday.
GAMES = [
    (11, ts("2026-11-20 01:15")),  # Thu night
    (11, ts("2026-11-22 18:00")),  # Sun
    (11, ts("2026-11-24 01:15")),  # Mon night
    (12, ts("2026-11-27 01:15")),
    (12, ts("2026-11-29 18:00")),
    (12, ts("2026-12-01 01:15")),
    (13, ts("2026-12-04 01:15")),
    (13, ts("2026-12-06 18:00")),
]


# --- current_decision_week -------------------------------------------------

def test_tuesday_after_mnf_returns_next_week():
    assert current_decision_week(GAMES, ts("2026-11-24 15:17")) == 12


def test_thursday_before_kickoff_returns_this_week():
    assert current_decision_week(GAMES, ts("2026-11-26 18:17")) == 12


def test_thursday_after_kickoff_returns_this_week():
    assert current_decision_week(GAMES, ts("2026-11-27 03:00")) == 12


def test_late_thursday_run_under_seven_days_from_next_thursday_stays_this_week():
    now = ts("2026-12-04 01:15") - pd.Timedelta(days=6.97)
    assert now > ts("2026-11-27 01:15")  # after this week's Thursday kickoff
    assert current_decision_week(GAMES, now) == 12


def test_no_future_games_returns_none():
    assert current_decision_week(GAMES, ts("2027-01-01")) is None


def test_ignores_unparseable_rows():
    games = [(np.nan, ts("2026-11-20")), (11, pd.NaT), ("12", ts("2026-11-27"))]
    assert current_decision_week(games, ts("2026-11-18")) == 12


# --- is_decidable -----------------------------------------------------------

def test_next_week_game_not_decidable_even_inside_horizon():
    now = ts("2026-12-04 01:15") - pd.Timedelta(days=6.97)
    week = current_decision_week(GAMES, now)
    assert not is_decidable(13, ts("2026-12-04 01:15"), now, week)


def test_this_weeks_monday_game_decidable_on_tuesday():
    now = ts("2026-11-17 15:17")
    week = current_decision_week(GAMES, now)
    assert week == 11
    assert is_decidable(11, ts("2026-11-24 01:15"), now, week)


def test_past_game_not_decidable():
    now = ts("2026-11-22 20:00")
    assert not is_decidable(11, ts("2026-11-22 18:00"), now, 11)


def test_horizon_backstop():
    now = ts("2026-11-10 00:00")
    assert not is_decidable(11, ts("2026-11-22 18:00"), now, 11)


# --- missing_results -------------------------------------------------------

def _sched(rows):
    return pd.DataFrame(rows, columns=["game_id", "season", "home_team", "away_team", "home_score", "away_score", "kickoff_utc"])


def test_missing_results_respects_grace():
    now = ts("2026-11-23 00:00")
    sched = _sched([
        ("old_no_score", 2026, "A", "B", np.nan, np.nan, now - pd.Timedelta(hours=6)),
        ("recent_no_score", 2026, "C", "D", np.nan, np.nan, now - pd.Timedelta(hours=3)),
        ("old_scored", 2026, "E", "F", 20, 17, now - pd.Timedelta(hours=30)),
        ("future", 2026, "G", "H", np.nan, np.nan, now + pd.Timedelta(hours=30)),
    ])
    assert missing_results(sched, now) == ["old_no_score"]


def test_missing_results_one_score_missing_counts():
    now = ts("2026-11-23 00:00")
    sched = _sched([("half", 2026, "A", "B", 20, np.nan, now - pd.Timedelta(days=2))])
    assert missing_results(sched, now) == ["half"]


# --- previous_game_day ------------------------------------------------------

# Real 2026 rows (gametimes ET; Nov 1 2026 is the DST change). Week 8 has no
# scores yet, as in the stale schedule data the review found.
REAL = _sched([
    ("2026_07_NYG_HOU", 2026, "HOU", "NYG", 27, 21, ts("2026-10-25 17:00")),
    ("2026_06_LAC_KC", 2026, "KC", "LAC", 6, 22, ts("2026-10-18 20:25")),
    ("2026_08_LAC_LA", 2026, "LA", "LAC", np.nan, np.nan, ts("2026-11-01 21:05")),
    ("2026_09_HOU_LAC", 2026, "LAC", "HOU", np.nan, np.nan, ts("2026-11-08 21:05")),
    ("2026_06_BUF_LV", 2026, "LV", "BUF", 33, 30, ts("2026-10-18 20:25")),
    ("2026_07_IND_MIN", 2026, "MIN", "IND", 10, 16, ts("2026-10-25 17:00")),
    ("2026_08_BAL_BUF", 2026, "BUF", "BAL", np.nan, np.nan, ts("2026-11-01 18:00")),
    ("2026_08_MIN_DET", 2026, "DET", "MIN", np.nan, np.nan, ts("2026-11-01 18:00")),
    ("2026_09_BUF_MIN", 2026, "MIN", "BUF", np.nan, np.nan, ts("2026-11-10 01:15")),
])


def _rest(sched, home, away, kickoff):
    return (
        rest_days(previous_game_day(sched, home, kickoff), kickoff),
        rest_days(previous_game_day(sched, away, kickoff), kickoff),
    )


def test_review_case_hou_at_lac():
    assert _rest(REAL, "LAC", "HOU", ts("2026-11-08 21:05")) == (7, 14)


def test_review_case_buf_at_min():
    assert _rest(REAL, "MIN", "BUF", ts("2026-11-10 01:15")) == (8, 8)


def test_uses_scheduled_previous_game_without_score():
    prev = previous_game_day(REAL, "LAC", ts("2026-11-08 21:05"))
    assert prev == pd.Timestamp("2026-11-01", tz="America/New_York")


def test_bye_gives_fourteen_days():
    assert rest_days(previous_game_day(REAL, "HOU", ts("2026-11-08 21:05")), ts("2026-11-08 21:05")) == 14


def test_thursday_night_turnarounds_use_eastern_days():
    sched = _sched([
        ("s1", 2026, "A", "B", np.nan, np.nan, ts("2026-11-08 18:00")),  # Sun 13:00 ET
        ("t1", 2026, "A", "C", np.nan, np.nan, ts("2026-11-13 01:15")),  # Thu 11-12 20:15 ET
        ("s2", 2026, "A", "D", np.nan, np.nan, ts("2026-11-22 18:00")),  # Sun 11-22
    ])
    # Sunday to Thursday night: 4 days.
    assert rest_days(previous_game_day(sched, "A", ts("2026-11-13 01:15")), ts("2026-11-13 01:15")) == 4
    # Thursday night to the Sunday after next: 10 Eastern days (UTC dates would say 9).
    assert rest_days(previous_game_day(sched, "A", ts("2026-11-22 18:00")), ts("2026-11-22 18:00")) == 10


def test_no_previous_game_this_season_returns_none():
    sched = _sched([
        ("last_year", 2025, "A", "B", 10, 7, ts("2026-01-04 18:00")),
        ("opener", 2026, "A", "C", np.nan, np.nan, ts("2026-09-13 17:00")),
    ])
    assert previous_game_day(sched, "A", ts("2026-09-13 17:00")) is None
    assert previous_game_day(sched, "A", ts("2026-09-13 17:00"), season=2026) is None
    assert rest_days(None, ts("2026-09-13 17:00")) is None
