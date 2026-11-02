from datetime import datetime, timedelta

import pandas as pd
import pytz

from src.closing_loop import next_wake_time, upcoming_kickoffs

NOW = datetime(2026, 11, 1, 15, 0, tzinfo=pytz.utc)
DEADLINE = NOW + timedelta(hours=5)


def test_upcoming_kickoffs_filters_status_and_past_games():
    df = pd.DataFrame([
        {"status": "OPEN_CAPTURED", "kickoff_time": "2026-11-01T18:00:00Z"},
        {"status": "OPEN_CAPTURED", "kickoff_time": "2026-11-01T18:00:00Z"},
        {"status": "SETTLED", "kickoff_time": "2026-11-01T21:05:00Z"},
        {"status": "PENDING", "kickoff_time": "2026-11-01T14:00:00Z"},
        {"status": "CLOSE_CAPTURED", "kickoff_time": "2026-11-01T21:25:00Z"},
    ])
    kicks = upcoming_kickoffs(df, NOW)
    assert [k.hour for k in kicks] == [18, 21]
    assert len(kicks) == 2


def test_next_wake_picks_earliest_lead_before_kickoff():
    kicks = [pd.Timestamp("2026-11-01T18:00:00Z"), pd.Timestamp("2026-11-01T21:25:00Z")]
    wake = next_wake_time(kicks, NOW, DEADLINE, leads_minutes=[45, 10])
    assert wake == pd.Timestamp("2026-11-01T17:15:00Z")

    after_first = pd.Timestamp("2026-11-01T17:16:00Z").to_pydatetime()
    wake2 = next_wake_time(kicks, after_first, DEADLINE, leads_minutes=[45, 10])
    assert wake2 == pd.Timestamp("2026-11-01T17:50:00Z")


def test_next_wake_respects_deadline():
    kicks = [pd.Timestamp("2026-11-02T01:20:00Z")]
    assert next_wake_time(kicks, NOW, DEADLINE, leads_minutes=[45, 10]) is None
