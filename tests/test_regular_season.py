from datetime import datetime, timedelta, timezone

import pytest

from src import regular_season as rs


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


HEADER = "game_id,season,game_type,week,gameday,gametime,away_team,home_team\n"
FIRST_REG_2026 = utc(2026, 9, 11, 0, 20)
LAST_REG_2026 = utc(2027, 1, 10, 18, 0)
CUTOFFS = {2026: rs.Window(FIRST_REG_2026, LAST_REG_2026)}


def write_games(tmp_path, rows):
    p = tmp_path / "games.csv"
    p.write_text(HEADER + "".join(r + "\n" for r in rows))
    return str(p)


def test_season_of_uses_march_boundary():
    assert rs.season_of(utc(2026, 9, 10, 0, 20)) == 2026
    assert rs.season_of(utc(2027, 1, 10, 18, 0)) == 2026
    assert rs.season_of(utc(2027, 2, 14, 23, 30)) == 2026     # Super Bowl
    assert rs.season_of(utc(2027, 3, 1, 0, 0)) == 2027


def test_window_is_first_and_last_reg_kickoff_per_season_in_utc(tmp_path):
    path = write_games(tmp_path, [
        "2025_18_A_B,2025,REG,18,2026-01-04,20:20,A,B",
        "2025_18_C_D,2025,REG,18,2026-01-04,13:00,C,D",
        "2025_01_C_D,2025,REG,1,2025-09-04,20:20,C,D",
        "2025_19_E_F,2025,WC,19,2026-01-10,16:30,E,F",
        "2026_18_A_B,2026,REG,18,2027-01-10,13:00,A,B",
        "2026_01_C_D,2026,REG,1,2026-09-10,20:20,C,D",
    ])
    assert rs.load_reg_cutoffs(path) == {
        2025: rs.Window(utc(2025, 9, 5, 0, 20), utc(2026, 1, 5, 1, 20)),    # 20:20 ET
        2026: rs.Window(utc(2026, 9, 11, 0, 20), utc(2027, 1, 10, 18, 0)),  # 13:00 ET
    }


def test_missing_gametime_counts_as_end_of_day(tmp_path):
    path = write_games(tmp_path, ["2026_18_A_B,2026,REG,18,2027-01-10,,A,B"])
    assert rs.load_reg_cutoffs(path)[2026].last == utc(2027, 1, 11, 4, 59)   # 23:59 ET


def test_missing_or_unreadable_schedule_raises(tmp_path):
    with pytest.raises(rs.ScheduleUnavailable, match="not found"):
        rs.load_reg_cutoffs(str(tmp_path / "nope.csv"))
    bad = tmp_path / "games.csv"
    bad.write_text("season,week\n2026,1\n")
    with pytest.raises(rs.ScheduleUnavailable, match="column"):
        rs.load_reg_cutoffs(str(bad))


def test_skip_reason_upper_bound():
    assert rs.skip_reason(LAST_REG_2026, CUTOFFS) is None
    # a week-18 game flexed later than nflverse's placeholder time still counts
    assert rs.skip_reason(LAST_REG_2026 + timedelta(hours=7, minutes=20), CUTOFFS) is None
    assert rs.skip_reason(LAST_REG_2026 + rs.CUTOFF_GRACE, CUTOFFS) is None
    assert "after the 2026 regular season" in rs.skip_reason(utc(2027, 1, 16, 21, 30), CUTOFFS)
    # the grace is far shorter than the gap to the first playoff game (6 days)
    assert rs.CUTOFF_GRACE < timedelta(days=5)


def test_skip_reason_lower_bound_skips_preseason():
    with_2027 = {**CUTOFFS, 2027: rs.Window(utc(2027, 9, 10, 0, 20), utc(2028, 1, 9, 21, 25))}
    reason = rs.skip_reason(utc(2027, 8, 27, 0, 0), with_2027)
    assert "before the 2027 regular season" in reason and not rs.is_fail_closed(reason)


def test_september_to_december_never_needs_the_schedule():
    def boom():
        raise AssertionError("schedule must not be loaded for Sep-Dec kickoffs")

    lazy = rs.LazyCutoffs(boom)
    for k in (utc(2026, 9, 11, 0, 20), utc(2026, 11, 26, 17, 30), utc(2026, 12, 31, 23, 59)):
        assert rs.skip_reason(k, lazy) is None
        assert rs.skip_reason(k, None) is None


def test_january_to_august_fails_closed_without_a_schedule():
    def missing():
        raise rs.ScheduleUnavailable("data/nflverse_cache/games.csv not found")

    lazy = rs.LazyCutoffs(missing)
    reason = rs.skip_reason(utc(2027, 1, 16, 21, 30), lazy)
    assert rs.is_fail_closed(reason) and "games.csv not found" in reason
    assert rs.is_fail_closed(rs.skip_reason(utc(2027, 9, 12, 17, 0) - timedelta(days=30), CUTOFFS))
    # loaded once, however many events ask
    calls = []
    lazy = rs.LazyCutoffs(lambda: calls.append(1) or CUTOFFS)
    rs.skip_reason(utc(2027, 1, 3), lazy)
    rs.skip_reason(utc(2027, 1, 16), lazy)
    assert calls == [1]


def test_real_cache_has_a_2026_cutoff_before_wild_card_weekend():
    import os
    if not os.path.exists(rs.GAMES_PATH):
        pytest.skip("games.csv not downloaded")
    cutoffs = rs.load_reg_cutoffs()
    if 2026 not in cutoffs:
        pytest.skip("games.csv lacks the 2026 schedule")
    assert utc(2026, 9, 1) < cutoffs[2026].first < utc(2026, 9, 15)
    assert utc(2027, 1, 3) < cutoffs[2026].last < utc(2027, 1, 13)
