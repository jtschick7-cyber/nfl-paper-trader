"""P3 totals kill test: team-game stats and leak-free game features (spec tests 1-4)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.totals import features as F


# --- synthetic data -----------------------------------------------------------------

def _play(game_id, season, week, posteam, defteam, drive, qtr, gsr, play_type="pass",
          wp=0.5, pass_oe=10.0, epa=0.1, yardline=60.0, touchdown=0, td_team=None, play_id=None):
    return {
        "game_id": game_id, "season": season, "season_type": "REG", "week": week,
        "posteam": posteam, "defteam": defteam, "drive": float(drive), "qtr": float(qtr),
        "game_seconds_remaining": float(gsr), "play_type": play_type, "wp": wp,
        "pass_oe": pass_oe, "epa": epa, "pass": 1 if play_type == "pass" else 0,
        "rush": 1 if play_type == "run" else 0, "qb_dropback": 1 if play_type == "pass" else 0,
        "qb_scramble": 0, "down": 1.0, "yardline_100": yardline, "touchdown": touchdown,
        "td_team": td_team, "play_id": play_id,
    }


def _game_pbp(game_id, season, week, home, away, epa_home=0.1, epa_away=0.0, gap_home=30):
    """Two drives per team: pass/run plays at a steady pace in Q1-Q2, neutral wp."""
    rows = []
    gsr = 3600.0
    drive = 1
    for _ in range(2):
        for team, opp, epa, gap in ((home, away, epa_home, gap_home), (away, home, epa_away, 35)):
            for k, pt in enumerate(("pass", "run", "pass", "run")):
                rows.append(_play(game_id, season, week, team, opp, drive, 1 if gsr > 2700 else 2, gsr,
                                  play_type=pt, epa=epa, yardline=60 - 15 * k))
                gsr -= gap
            drive += 1
    for i, r in enumerate(rows):
        r["play_id"] = float(i + 1)
    return rows


ROUND_ROBIN = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C"),
               ("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]


def _schedule(season=2015):
    games = []
    for i, (home, away) in enumerate(ROUND_ROBIN):
        week = i // 2 + 1
        games.append({
            "game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "game_type": "REG",
            "week": week, "gameday": f"{season}-09-{7 + 7 * (week - 1):02d}" if week <= 3 else
            f"{season}-10-{7 * (week - 3):02d}", "gametime": "13:00",
            "home_team": home, "away_team": away, "home_score": 20.0 + i, "away_score": 17.0,
            "roof": "outdoors", "wind": 5.0, "temp": 60.0,
            "total_line": 44.5, "over_odds": -110.0, "under_odds": -110.0,
        })
    return pd.DataFrame(games)


def _pbp_for(sched, overrides=None):
    overrides = overrides or {}
    rows = []
    for g in sched.itertuples():
        kw = overrides.get(g.game_id, {})
        rows.extend(_game_pbp(g.game_id, g.season, g.week, g.home_team, g.away_team, **kw))
    return pd.DataFrame(rows)


def _features(sched, pbp):
    games = F.prepare_games(sched)
    tg = F.team_game_stats(pbp, games)
    return F.game_frame(games, F.rolling_team_features(tg))


TARGET = "2015_05_C_A"  # week 5: A and C have 4 prior games each


# --- 1. leak-free --------------------------------------------------------------------

def test_features_use_only_earlier_games():
    sched = _schedule()
    base = _features(sched, _pbp_for(sched)).set_index("game_id")
    cols = [c for c in base.columns if c.endswith("_r")]
    assert base.loc[TARGET, "home_n_prior"] == 4 and base.loc[TARGET, "away_n_prior"] == 4

    earlier = _features(sched, _pbp_for(sched, {"2015_02_C_A": {"epa_home": 0.9, "gap_home": 20}})).set_index("game_id")
    assert not np.allclose(base.loc[TARGET, cols].astype(float), earlier.loc[TARGET, cols].astype(float))

    for later_id in ("2015_06_D_A", TARGET):  # a later game, and the target game itself
        later = _features(sched, _pbp_for(sched, {later_id: {"epa_home": 0.9, "gap_home": 20}})).set_index("game_id")
        pd.testing.assert_series_equal(base.loc[TARGET, cols].astype(float), later.loc[TARGET, cols].astype(float))


def test_fewer_than_min_games_is_nan_then_training_league_mean():
    sched = _schedule()
    games = F.prepare_games(sched)
    tg = F.team_game_stats(_pbp_for(sched), games)
    gf = F.game_frame(games, F.rolling_team_features(tg)).set_index("game_id")
    first = "2015_01_B_A"
    assert gf.loc[first, "home_n_prior"] == 0
    assert math.isnan(gf.loc[first, "home_off_pass_epa_r"])
    fills = F.training_fills(tg, games, [2015])
    X = F.design_matrix(gf.reset_index(), fills)
    # every team-game in the training season has the same synthetic sec/play; first game gets it too
    assert X.loc[0, "pace"] == pytest.approx(2 * fills["neutral_sec_per_play"])
    assert list(X.columns) == F.FEATURES


def test_window_is_last_eight_games_across_seasons():
    tg = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(12)], "team": "A",
        "season": [2014] * 6 + [2015] * 6,
        "kickoff_utc": pd.date_range("2014-09-07", periods=12, freq="7D", tz="UTC"),
        **{s: np.arange(12, dtype=float) for s in F.TEAM_STATS},
    })
    r = F.rolling_team_features(tg).set_index("game_id")
    assert r.loc["g11", "points_for_r"] == pytest.approx(np.mean(np.arange(3, 11)))
    assert r.loc["g11", "rz_trips_r"] == pytest.approx(np.sum(np.arange(3, 11)))  # counts are summed
    assert math.isnan(r.loc["g2", "points_for_r"]) and r.loc["g3", "points_for_r"] == pytest.approx(1.0)


# --- 2. neutral seconds per play ------------------------------------------------------

def test_neutral_sec_per_play_and_pass_oe():
    g, s, w = "2015_01_B_A", 2015, 1
    rows = [
        _play(g, s, w, "A", "B", 1, 1, 3600, pass_oe=10),
        _play(g, s, w, "A", "B", 1, 1, 3570, pass_oe=20),
        _play(g, s, w, "A", "B", 1, 1, 3530, pass_oe=30),        # gaps 30, 40
        _play(g, s, w, "A", "B", 1, 1, 3500, wp=0.9, pass_oe=99),  # wp out of range: both its gaps drop
        _play(g, s, w, "A", "B", 1, 1, 3480, pass_oe=40),
        _play(g, s, w, "B", "A", 2, 1, 3400),                    # other team
        _play(g, s, w, "A", "B", 3, 1, 3000, pass_oe=50),
        _play(g, s, w, "A", "B", 3, 1, 2980, pass_oe=60),        # gap 20; no gap across drives
        _play(g, s, w, "A", "B", 4, 4, 600, pass_oe=99),
        _play(g, s, w, "A", "B", 4, 4, 590, pass_oe=99),         # Q4: excluded
        _play(g, s, w, "A", "B", 3, 1, 2990, play_type="no_play", pass_oe=99),  # not an offensive play
    ]
    pbp = pd.DataFrame(rows)
    pbp["play_id"] = np.arange(1, len(pbp) + 1, dtype=float)
    pbp = pbp.sort_values("game_seconds_remaining", ascending=False)
    pbp["play_id"] = np.arange(1, len(pbp) + 1, dtype=float)
    out = F.neutral_pace(pbp).set_index(["game_id", "team"])
    assert out.loc[(g, "A"), "neutral_sec_per_play"] == pytest.approx((30 + 40 + 20) / 3)
    assert out.loc[(g, "A"), "neutral_pass_oe"] == pytest.approx((10 + 20 + 30 + 40 + 50 + 60) / 6)


# --- red zone -------------------------------------------------------------------------

def test_red_zone_trips_and_tds():
    g, s, w = "2015_01_B_A", 2015, 1
    rows = [
        _play(g, s, w, "A", "B", 1, 1, 3500, yardline=30),
        _play(g, s, w, "A", "B", 1, 1, 3470, yardline=15),
        _play(g, s, w, "A", "B", 1, 1, 3440, yardline=5, touchdown=1, td_team="A"),   # trip, TD
        _play(g, s, w, "A", "B", 2, 1, 3000, yardline=18),
        _play(g, s, w, "A", "B", 2, 1, 2970, play_type="field_goal", yardline=18),    # trip, FG
        _play(g, s, w, "A", "B", 3, 2, 2500, yardline=40),
        _play(g, s, w, "A", "B", 3, 2, 2470, yardline=25, touchdown=1, td_team="A"),  # no trip (TD from 25)
        _play(g, s, w, "A", "B", 4, 2, 2000, yardline=10, touchdown=1, td_team="B"),  # trip, pick-six
        _play(g, s, w, "A", "B", 5, 2, 1900, play_type="kickoff", yardline=15),       # kickoff: not a snap
    ]
    out = F.red_zone(pd.DataFrame(rows)).set_index(["game_id", "team"])
    assert out.loc[(g, "A"), "rz_trips"] == 3
    assert out.loc[(g, "A"), "rz_tds"] == 1


# --- 3. red-zone shrink ---------------------------------------------------------------

def test_rz_shrink_formula():
    assert F.rz_shrunk_rate(3, 5, 0.5) == pytest.approx((3 + 20 * 0.5) / (5 + 20))
    assert F.rz_shrunk_rate(0, 0, 0.57) == pytest.approx(0.57)
    assert F.RZ_PSEUDO_TRIPS == 20


# --- 4. weather -----------------------------------------------------------------------

def _wx(rows):
    return pd.DataFrame(rows, columns=["season", "roof", "wind", "temp"])


def test_indoor_gives_zero_wind():
    out = F.weather_features(_wx([(2020, "dome", 20.0, 30.0), (2020, "closed", np.nan, np.nan)]), wind_median=7.0)
    assert list(out["is_indoor"]) == [1, 1]
    assert list(out["wind"]) == [0.0, 0.0]
    assert list(out["high_wind"]) == [0, 0]
    assert list(out["wind_missing"]) == [0, 0]
    assert list(out["cold"]) == [0, 0]


def test_unknown_outdoor_wind_gets_median_and_flag():
    out = F.weather_features(_wx([(2020, "outdoors", np.nan, 25.0), (2020, "open", 16.0, 40.0),
                                  (2020, "outdoors", 15.0, 32.0)]), wind_median=7.0)
    assert list(out["is_indoor"]) == [0, 0, 0]
    assert list(out["wind"]) == [7.0, 16.0, 15.0]
    assert list(out["wind_missing"]) == [1, 0, 0]
    assert list(out["high_wind"]) == [0, 1, 1]
    assert list(out["cold"]) == [1, 0, 1]


def test_wind_median_uses_training_seasons_only():
    games = _wx([(2018, "outdoors", 4.0, 60), (2018, "outdoors", 6.0, 60), (2019, "outdoors", 8.0, 60),
                 (2019, "dome", 0.0, 70), (2020, "outdoors", 40.0, 60), (2020, "outdoors", 40.0, 60)])
    assert F.training_wind_median(games, [2018, 2019]) == pytest.approx(6.0)  # dome row ignored
    assert F.training_wind_median(games, [2018, 2019, 2020]) == pytest.approx(8.0)
