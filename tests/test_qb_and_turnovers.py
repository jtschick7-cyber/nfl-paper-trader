"""
Unit tests for QB Dropback Attribution and Turnover Luck Shrinkage.
"""

import pandas as pd
import pytest
from src.qb_features import (
    SHRINKAGE_K,
    DEFAULT_LEAGUE_QB_PRIOR,
    EPA_TO_SPREAD_POINTS,
    estimate_qb_replacement_penalty,
    get_matchup_qb_shock,
    shrink_qb_rating,
)
from src.turnover_luck import (
    TURNOVER_PRIOR_RECOVERY_RATE,
    SHRINKAGE_K_TURNOVER,
    estimate_turnover_spread_adjustment,
    get_matchup_turnover_features,
    shrink_turnover_rate,
)


def test_qb_shrinkage_k():
    # Shrinkage K must be set to 60.0 to reflect NFL dropback variance ratio
    assert SHRINKAGE_K == 60.0


def test_empirical_bayes_qb_shrinkage():
    league_prior = 0.05
    # Small sample (10 dropbacks) with +0.50 EPA should shrink heavily towards prior
    small_sample_shrunk = shrink_qb_rating(
        observed_epa=0.50,
        n_dropbacks=10,
        prior_epa=league_prior,
        k=60.0,
    )
    # Expected: (10*0.50 + 60*0.05) / 70 = (5.0 + 3.0) / 70 = 8.0 / 70 = 0.1143
    assert pytest.approx(small_sample_shrunk, 1e-4) == (10 * 0.50 + 60 * 0.05) / 70.0
    assert small_sample_shrunk < 0.20

    # Large sample (500 dropbacks) with +0.30 EPA should stay close to sample mean (shrunk = 153/560 ~ 0.273)
    large_sample_shrunk = shrink_qb_rating(
        observed_epa=0.30,
        n_dropbacks=500,
        prior_epa=league_prior,
        k=60.0,
    )
    assert pytest.approx(large_sample_shrunk, 1e-4) == (500 * 0.30 + 60 * 0.05) / 560.0
    assert large_sample_shrunk > 0.25


def test_estimate_qb_replacement_penalty():
    # Elite starter (+0.24 EPA/db) vs Backup (-0.10 EPA/db) over 35 dropbacks
    penalty = estimate_qb_replacement_penalty(starter_rating=0.24, backup_rating=-0.10, expected_dropbacks=35)
    assert penalty > 3.0
    assert penalty < 6.0


def test_get_matchup_qb_shock():
    # Home starter (+0.25) above team pass EPA (+0.10)
    # Away starter (-0.05) below team pass EPA (+0.15)
    shock = get_matchup_qb_shock(
        home_qb_rating=0.25,
        home_team_pass_epa=0.10,
        away_qb_rating=-0.05,
        away_team_pass_epa=0.15,
    )
    assert shock["home_qb_shock"] > 0.0
    assert shock["away_qb_shock"] < 0.0
    assert shock["diff_qb_shock"] > 0.0


def test_turnover_luck_shrinkage():
    # 8 recovered out of 10 fumbles (80% observed)
    shrunk_rate = shrink_turnover_rate(observed_rate=0.80, total_events=10, prior=0.50, k=4.0)
    # Expected: (10*0.80 + 4*0.50) / 14 = (8 + 2) / 14 = 10 / 14 = 0.714
    assert pytest.approx(shrunk_rate, 1e-3) == 10.0 / 14.0
    assert shrunk_rate < 0.80
    assert shrunk_rate > 0.50


def test_estimate_turnover_spread_adjustment():
    # Home team was very lucky (+7.6 luck points), Away was unlucky (-3.8 luck points)
    # Net diff = +11.4 points. Fading home with 50% decay factor gives ~ -5.7 points.
    adj = estimate_turnover_spread_adjustment(home_luck_points=7.6, away_luck_points=-3.8)
    assert adj < 0.0
    assert pytest.approx(adj, 1e-2) == -5.70


def test_fit_constrained_ridge_bounds():
    import numpy as np
    from src.train_model_v4 import fit_constrained_ridge, V4_BOUNDS_LB, V4_BOUNDS_UB

    # Synthetic collinear data where standard OLS/Ridge would produce inverted signs
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 10)
    # Feature 4 is diff_early_down_epa, Feature 9 is diff_fumble_luck_per_gm
    # Simulate a target positively correlated with everything except fumble luck
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 1.2 * X[:, 8] - 1.0 * X[:, 9] + np.random.randn(n) * 2.0

    model = fit_constrained_ridge(X, y, alpha=10.0, bounds=(V4_BOUNDS_LB, V4_BOUNDS_UB))
    coefs = model.coef_

    # Verify constraints are strictly respected
    assert coefs[4] >= 0.0  # diff_early_down_epa >= 0.0
    assert coefs[8] >= 0.0  # diff_qb_shock >= 0.0
    assert coefs[9] <= 0.0  # diff_fumble_luck_per_gm <= 0.0 (non-positive constraint)


def test_compute_point_in_time_qb_shock_starter_lookup():
    from src.train_model_v4 import compute_point_in_time_qb_shock

    # Mock QB historical dataframe strictly with past completed games
    qb_data = pd.DataFrame([
        # Past games for Home Team: Love started W1, Willis started W2
        {"game_id": "2024_01_DET_GB", "season": 2024, "week": 1, "posteam": "GB", "qb_id": "qb_love", "qb_name": "J.Love", "dropbacks": 35, "qb_epa_per_dropback": 0.25},
        {"game_id": "2024_02_IND_GB", "season": 2024, "week": 2, "posteam": "GB", "qb_id": "qb_willis", "qb_name": "M.Willis", "dropbacks": 15, "qb_epa_per_dropback": 0.05},
        # Away team starter: Levis started both W1 and W2
        {"game_id": "2024_01_TEN_CHI", "season": 2024, "week": 1, "posteam": "TEN", "qb_id": "qb_levis", "qb_name": "W.Levis", "dropbacks": 32, "qb_epa_per_dropback": -0.15},
        {"game_id": "2024_02_NYJ_TEN", "season": 2024, "week": 2, "posteam": "TEN", "qb_id": "qb_levis", "qb_name": "W.Levis", "dropbacks": 28, "qb_epa_per_dropback": -0.10},
    ])

    past_gids = ["2024_01_DET_GB", "2024_02_IND_GB", "2024_01_TEN_CHI", "2024_02_NYJ_TEN"]

    # 1. With starter_overrides specifying Love returns in W3:
    # Evaluates Love's historical performance strictly from past_gids (W1)
    diff_shock_love = compute_point_in_time_qb_shock(
        qb_df=qb_data,
        past_gids=past_gids,
        home_team="GB",
        away_team="TEN",
        h_pass_epa=0.10,
        a_pass_epa=-0.05,
        starter_overrides={"GB": "qb_love"},
        window=8,
    )
    # Love (+0.25 vs 0.10) - Levis (-0.125 vs -0.05) -> positive shock
    assert diff_shock_love > 0.0

    # 2. Default lookup (no override): selects Willis from last_gid ("2024_02_IND_GB")
    diff_shock_default = compute_point_in_time_qb_shock(
        qb_df=qb_data,
        past_gids=past_gids,
        home_team="GB",
        away_team="TEN",
        h_pass_epa=0.10,
        a_pass_epa=-0.05,
        starter_overrides=None,
        window=8,
    )
    # Willis (+0.05 vs 0.10) is worse than Love (+0.25 vs 0.10)
    assert diff_shock_love > diff_shock_default


def test_fetch_pregame_starter_overrides_manual(tmp_path):
    import json
    from src.qb_features import fetch_pregame_starter_overrides

    path = tmp_path / "qb_overrides.json"
    path.write_text(json.dumps({"GB": "00-0036264", "kc": "Backup QB"}), encoding="utf-8")
    res = fetch_pregame_starter_overrides(2026, qb_df=None, overrides_path=str(path))
    assert res["GB"] == "00-0036264"
    assert res["KC"] == "Backup QB"

    missing = fetch_pregame_starter_overrides(2026, qb_df=None, overrides_path=str(tmp_path / "nope.json"))
    assert missing == {}
