"""
Unit tests for EPA feature calculations and off-season Bayesian regression.
"""

import pandas as pd
import pytest
from src.epa_features import compute_rolling_epa_snapshot, get_matchup_epa_features


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
