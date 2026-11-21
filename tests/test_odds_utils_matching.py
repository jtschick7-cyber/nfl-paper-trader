"""P1-3: odds rows match a game by team pair AND kickoff date."""

import pandas as pd

from src.odds_utils import match_event_rows

SPREADS = pd.DataFrame({
    "home_team": ["KC", "KC", "BUF"],
    "away_team": ["DEN", "DEN", "MIA"],
    "home_spread": [-7.0, -3.0, -2.5],
    "commence_time": ["2026-12-06T18:00:00Z", "2026-12-20T18:00:00Z", "2026-12-06T18:00:00Z"],
})


def test_rematch_two_weeks_later_is_not_matched():
    m = match_event_rows(SPREADS, "KC", "DEN", pd.Timestamp("2026-12-20T18:00:00Z"))
    assert list(m["home_spread"]) == [-3.0]
    m = match_event_rows(SPREADS, "KC", "DEN", pd.Timestamp("2026-12-06T18:00:00Z"))
    assert list(m["home_spread"]) == [-7.0]


def test_no_event_within_window_is_no_match():
    assert match_event_rows(SPREADS, "KC", "DEN", pd.Timestamp("2027-01-10T18:00:00Z")).empty


def test_missing_kickoff_falls_back_to_team_pair():
    assert len(match_event_rows(SPREADS, "KC", "DEN", None)) == 2
    assert match_event_rows(pd.DataFrame(), "KC", "DEN", None).empty
