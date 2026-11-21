"""
Unit tests for EPA feature calculations and off-season Bayesian regression.
"""

import pandas as pd
import pytest
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features, parse_pbp_to_game_epa


def test_week1_offseason_epa_regression():
    # 6 games played in prior season (2024)
    epa_rows = []
    sched_rows = []

    for w in range(1, 7):
        gid = f"2024_{w:02d}_GB_MIN"
        ko = f"2024-10-{w:02d} 18:00:00+00:00"
        epa_rows.append({
            "game_id": gid,
            "season": 2024,
            "week": w,
            "team": "GB",
            "off_pass_epa": 0.30,  # Prior season pass EPA = +0.30
            "off_rush_epa": 0.10,
            "off_total_epa": 0.40,
            "off_success_rate": 0.55,  # Prior season success rate = 55%
            "off_early_down_epa": 0.20,
            "def_pass_epa": -0.10,
            "def_rush_epa": -0.05,
            "def_total_epa": -0.15,
            "def_success_rate": 0.40,
            "def_early_down_epa": -0.05,
        })
        sched_rows.append({
            "game_id": gid,
            "kickoff_utc": ko,
        })

    epa_df = pd.DataFrame(epa_rows)
    sched_df = pd.DataFrame(sched_rows)

    # In Week 1 of 2025 (Sept 7, 2025), team has 0 games in 2025
    as_of = pd.to_datetime("2025-09-07 17:00:00+00:00")
    snap = compute_rolling_epa_snapshot(epa_df, sched_df, as_of, window=6, min_periods=2)

    assert "GB" in snap
    gb_stats = snap["GB"]

    # pass_epa (0.30) should be shrunk 50% toward 0.0 -> 0.15
    assert pytest.approx(gb_stats["roll_off_pass_epa"], 1e-4) == 0.15

    # success_rate (0.55) should be shrunk 50% toward league prior 0.45 -> 0.50 * 0.55 + 0.50 * 0.45 = 0.50
    assert pytest.approx(gb_stats["roll_off_success_rate"], 1e-4) == 0.50


def test_get_matchup_epa_features():
    snapshot = {
        "KC": {
            "roll_off_pass_epa": 0.20,
            "roll_def_pass_epa": -0.10,
            "roll_off_rush_epa": 0.05,
            "roll_def_rush_epa": -0.05,
            "roll_off_success_rate": 0.50,
            "roll_def_success_rate": 0.42,
            "roll_off_early_down_epa": 0.12,
            "roll_def_early_down_epa": -0.08,
        },
        "DEN": {
            "roll_off_pass_epa": -0.05,
            "roll_def_pass_epa": 0.05,
            "roll_off_rush_epa": -0.02,
            "roll_def_rush_epa": 0.02,
            "roll_off_success_rate": 0.43,
            "roll_def_success_rate": 0.48,
            "roll_off_early_down_epa": -0.04,
            "roll_def_early_down_epa": 0.04,
        },
    }

    matchup = get_matchup_epa_features(snapshot, "KC", "DEN")
    assert matchup is not None
    # diff_pass_epa: (KC off - DEN def) - (DEN off - KC def) = (0.20 - 0.05) - (-0.05 - (-0.10)) = 0.15 - 0.05 = 0.10
    assert pytest.approx(matchup["diff_pass_epa"], 1e-4) == 0.10
    # diff_early_down_epa should be present
    assert "diff_early_down_epa" in matchup


def _kc_den_snapshot():
    return {
        "KC": {"roll_off_pass_epa": 0.20, "roll_def_pass_epa": -0.10, "roll_off_rush_epa": 0.05,
               "roll_def_rush_epa": -0.05, "roll_off_success_rate": 0.50, "roll_def_success_rate": 0.42,
               "roll_off_early_down_epa": 0.12, "roll_def_early_down_epa": -0.08},
        "DEN": {"roll_off_pass_epa": -0.05, "roll_def_pass_epa": 0.05, "roll_off_rush_epa": -0.02,
                "roll_def_rush_epa": 0.02, "roll_off_success_rate": 0.43, "roll_def_success_rate": 0.48,
                "roll_off_early_down_epa": -0.04, "roll_def_early_down_epa": 0.04},
    }


def test_fixed_matchup_has_the_right_sign():
    m = get_matchup_epa_features(_kc_den_snapshot(), "KC", "DEN", version="fixed")
    # (KC off - KC def) - (DEN off - DEN def) = (0.20 + 0.10) - (-0.05 - 0.05) = 0.40
    assert m["diff_pass_epa"] == pytest.approx(0.40, abs=1e-4)
    assert m["diff_off_pass"] == pytest.approx(0.25, abs=1e-4)
    assert m["diff_def_pass"] == pytest.approx(0.15, abs=1e-4)
    # Success rate was already right in the legacy formula: both versions agree
    legacy = get_matchup_epa_features(_kc_den_snapshot(), "KC", "DEN")
    assert m["diff_success_rate"] == pytest.approx(legacy["diff_success_rate"], abs=1e-4)


@pytest.mark.parametrize("metric,key", [("pass_epa", "diff_pass_epa"), ("rush_epa", "diff_rush_epa"),
                                        ("early_down_epa", "diff_early_down_epa")])
def test_better_home_defence_raises_the_fixed_differential(metric, key):
    snap = _kc_den_snapshot()
    base = get_matchup_epa_features(snap, "KC", "DEN", version="fixed")[key]
    snap["KC"][f"roll_def_{metric}"] -= 0.10  # KC allows less: a better defence
    assert get_matchup_epa_features(snap, "KC", "DEN", version="fixed")[key] > base
    snap = _kc_den_snapshot()
    snap["DEN"][f"roll_def_{metric}"] -= 0.10  # the away defence improves
    assert get_matchup_epa_features(snap, "KC", "DEN", version="fixed")[key] < base


def test_unknown_matchup_version_raises():
    with pytest.raises(ValueError):
        get_matchup_epa_features(_kc_den_snapshot(), "KC", "DEN", version="nope")


def _pbp(rows):
    base = dict(season=2025, week=1, season_type="REG", game_id="G", posteam="KC", defteam="DEN",
                down=1, qb_dropback=0, qb_scramble=0, success=0)
    return pd.DataFrame([dict(base, **r) for r in rows])


def test_dropback_split_counts_a_scramble_as_a_pass_only():
    pbp = _pbp([
        dict(play_type="pass", **{"pass": 1, "rush": 0}, qb_dropback=1, epa=0.2),
        dict(play_type="run", **{"pass": 0, "rush": 1}, epa=-0.1),               # designed run
        dict(play_type="run", **{"pass": 1, "rush": 0}, qb_dropback=1, qb_scramble=1, epa=1.0),
    ])
    kc = lambda df: df[df["team"] == "KC"].iloc[0]
    legacy = kc(parse_pbp_to_game_epa(pbp))
    fixed = kc(parse_pbp_to_game_epa(pbp, play_split="dropback"))
    assert legacy["off_rush_epa"] == pytest.approx(0.45)  # scramble double-counted
    assert fixed["off_rush_epa"] == pytest.approx(-0.1)
    assert fixed["off_pass_epa"] == pytest.approx(0.6)


def test_opponent_adjusted_epa_snapshot():
    def build(def_epa_by_opp):
        epa_rows, sched_rows = [], []
        n = 0
        for opp, d in def_epa_by_opp.items():
            for w in range(1, 4):
                n += 1
                gid = f"2026_{n:02d}_{opp}"
                ko = f"2026-09-{n:02d} 18:00:00+00:00"
                sched_rows.append({"game_id": gid, "kickoff_utc": ko})
                base = {"season": 2026, "week": w, "off_pass_epa": 0.0, "off_rush_epa": 0.0,
                        "off_total_epa": 0.0, "off_success_rate": 0.45, "off_early_down_epa": 0.0,
                        "def_pass_epa": 0.0, "def_rush_epa": 0.0, "def_total_epa": 0.0,
                        "def_success_rate": 0.45, "def_early_down_epa": 0.0}
                epa_rows.append({**base, "game_id": gid, "team": "TEAM", "opponent": opp, "off_pass_epa": 0.10})
                epa_rows.append({**base, "game_id": gid, "team": opp, "opponent": "TEAM", "def_pass_epa": d})
        return pd.DataFrame(epa_rows), pd.DataFrame(sched_rows)

    as_of = pd.Timestamp("2026-12-01", tz="UTC")
    epa_a, sched_a = build({"STRONG": -0.20})
    epa_b, sched_b = build({"WEAK": 0.20})
    snap_a = compute_rolling_epa_snapshot(epa_a, sched_a, as_of, min_periods=3)
    snap_b = compute_rolling_epa_snapshot(epa_b, sched_b, as_of, min_periods=3)
    base_a = compute_rolling_epa_snapshot(epa_a, sched_a, as_of, min_periods=3, opponent_adjust=False)
    base_b = compute_rolling_epa_snapshot(epa_b, sched_b, as_of, min_periods=3, opponent_adjust=False)
    assert base_a["STRONG"]["roll_def_pass_epa"] == pytest.approx(-0.20)
    assert base_b["WEAK"]["roll_def_pass_epa"] == pytest.approx(0.20)
    assert base_a["TEAM"]["roll_off_pass_epa"] == pytest.approx(0.10)
    assert snap_a["TEAM"]["roll_off_pass_epa"] > snap_b["TEAM"]["roll_off_pass_epa"]
    assert snap_a["TEAM"]["roll_off_pass_epa"] == pytest.approx(0.10 + 0.35 * 0.20)
