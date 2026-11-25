from datetime import datetime, timedelta, timezone

import pytest

from src import regular_season as rs


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


HEADER = "game_id,season,game_type,week,gameday,gametime,away_team,home_team\n"


def write_games(tmp_path, rows):
    p = tmp_path / "games.csv"
    p.write_text(HEADER + "".join(r + "\n" for r in rows))
    return str(p)


def test_season_of_uses_march_boundary():
    assert rs.season_of(utc(2026, 9, 10, 0, 20)) == 2026
    assert rs.season_of(utc(2027, 1, 10, 18, 0)) == 2026
    assert rs.season_of(utc(2027, 2, 14, 23, 30)) == 2026     # Super Bowl
    assert rs.season_of(utc(2027, 3, 1, 0, 0)) == 2027


def test_cutoff_is_latest_reg_kickoff_per_season_in_utc(tmp_path):
    path = write_games(tmp_path, [
        "2025_18_A_B,2025,REG,18,2026-01-04,20:20,A,B",
        "2025_18_C_D,2025,REG,18,2026-01-04,13:00,C,D",
        "2025_19_E_F,2025,WC,19,2026-01-10,16:30,E,F",
        "2026_18_A_B,2026,REG,18,2027-01-10,13:00,A,B",
        "2026_17_C_D,2026,REG,17,2027-01-03,16:25,C,D",
    ])
    cutoffs = rs.load_reg_cutoffs(path)
    assert cutoffs == {2025: utc(2026, 1, 5, 1, 20),    # 20:20 ET
                       2026: utc(2027, 1, 10, 18, 0)}   # 13:00 ET


def test_missing_gametime_counts_as_end_of_day(tmp_path):
    path = write_games(tmp_path, ["2026_18_A_B,2026,REG,18,2027-01-10,,A,B"])
    assert rs.load_reg_cutoffs(path) == {2026: utc(2027, 1, 11, 4, 59)}   # 23:59 ET


def test_missing_or_unreadable_schedule_raises(tmp_path):
    with pytest.raises(rs.ScheduleUnavailable, match="not found"):
        rs.load_reg_cutoffs(str(tmp_path / "nope.csv"))
    bad = tmp_path / "games.csv"
    bad.write_text("season,week\n2026,1\n")
    with pytest.raises(rs.ScheduleUnavailable, match="column"):
        rs.load_reg_cutoffs(str(bad))


def test_skip_reason():
    cutoffs = {2026: utc(2027, 1, 10, 18, 0)}
    assert rs.skip_reason(utc(2027, 1, 10, 18, 0), cutoffs) is None
    # a week-18 game flexed later than nflverse's placeholder time still counts
    assert rs.skip_reason(utc(2027, 1, 11, 1, 20), cutoffs) is None
    assert "after the 2026 regular season" in rs.skip_reason(utc(2027, 1, 16, 21, 30), cutoffs)
    assert "no regular-season schedule for 2027" in rs.skip_reason(utc(2027, 9, 10, 0, 20), cutoffs)
    # the grace is far shorter than the gap to the first playoff game (6 days)
    assert rs.CUTOFF_GRACE < timedelta(days=5)


def test_real_cache_has_a_2026_cutoff_before_wild_card_weekend():
    import os
    if not os.path.exists(rs.GAMES_PATH):
        pytest.skip("games.csv not downloaded")
    cutoffs = rs.load_reg_cutoffs()
    if 2026 not in cutoffs:
        pytest.skip("games.csv lacks the 2026 schedule")
    assert utc(2027, 1, 3) < cutoffs[2026] < utc(2027, 1, 13)
