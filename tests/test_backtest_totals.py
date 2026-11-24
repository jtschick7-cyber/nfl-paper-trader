"""P3 totals kill test: walk-forward, metrics and the frozen kill/keep rule (spec tests 5-6)."""

import numpy as np
import pandas as pd
import pytest

from src.totals import backtest_totals as B
from src.totals import features as F


def test_frozen_constants():
    assert B.ALPHAS == (1, 3, 10, 30, 100)
    assert B.N_SPLITS == 5
    assert B.FIRST_TRAIN_SEASON == 2012
    assert B.TEST_SEASONS == tuple(range(2014, 2026))
    assert B.PERIODS == {"2014-2019": (2014, 2019), "2020-2025": (2020, 2025)}
    assert B.DECISION_PERIOD == "2020-2025"
    assert B.SIGMA == 13.2
    assert (B.BET_OVER, B.BET_UNDER) == (0.55, 0.45)
    assert F.WINDOW == 8 and F.MIN_GAMES == 3 and F.HIGH_WIND_MPH == 15 and F.COLD_F == 32


# --- synthetic walk-forward data ------------------------------------------------------

def _synthetic(seasons=range(2012, 2017), per_season=120, market="blind", seed=0):
    """A game frame shaped like features.game_frame plus anchors. The rolled team stats are
    random; the actual total depends on pass_env. market="blind": the line ignores it;
    market="knows": the line equals the true mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for i in range(per_season):
            row = {"game_id": f"{s}_{i:03d}", "season": s,
                   "kickoff_utc": pd.Timestamp(f"{s}-09-01", tz="UTC") + pd.Timedelta(hours=12 * i),
                   "home_n_prior": 8, "away_n_prior": 8,
                   "roof": "outdoors", "wind": float(rng.uniform(0, 20)), "temp": 60.0}
            for side in ("home", "away"):
                for st in F.TEAM_STATS:
                    row[f"{side}_{st}_r"] = float(rng.normal(0, 1))
                row[f"{side}_rz_trips_r"] = 20.0
                row[f"{side}_rz_tds_r"] = 11.0
            rows.append(row)
    gf = pd.DataFrame(rows)
    signal = 4.0 * (gf["home_off_pass_epa_r"] + gf["away_def_pass_epa_r"]
                    + gf["away_off_pass_epa_r"] + gf["home_def_pass_epa_r"])
    true_mu = 45.0 + signal
    gf["actual_total"] = np.round(true_mu + rng.normal(0, 6, len(gf)))
    gf["total_line"] = 45.5 if market == "blind" else np.round(true_mu * 2) / 2 + 0.0
    gf["over_odds"] = -110.0
    gf["under_odds"] = -110.0
    tg = pd.DataFrame({"season": np.repeat(list(seasons), 10), **{st: 0.0 for st in F.TEAM_STATS},
                       "rz_trips": 2.0, "rz_tds": 1.0})
    return B.add_anchor(gf), tg


def test_no_test_season_row_in_training_alpha_or_beta():
    gf, tg = _synthetic()
    preds, folds = B.walk_forward(gf, tg, test_seasons=(2014, 2015, 2016))
    season_of = dict(zip(gf["game_id"], gf["season"]))
    assert [f["season"] for f in folds] == [2014, 2015, 2016]
    for f in folds:
        for key in ("train_ids", "alpha_ids", "beta_ids"):
            assert f[key], key
            assert max(season_of[g] for g in f[key]) == f["season"] - 1
            assert min(season_of[g] for g in f[key]) == 2012
        assert f["beta_ids"] <= f["train_ids"] and f["alpha_ids"] <= f["train_ids"]
        assert f["fills"]["seasons"] == list(range(2012, f["season"]))
    assert set(preds["season"]) == {2014, 2015, 2016}
    assert len(preds) == 3 * 120


def test_alpha_and_beta_do_not_change_when_test_season_changes():
    gf, tg = _synthetic()
    _, base = B.walk_forward(gf, tg, test_seasons=(2015,))
    moved = gf.copy()
    m = moved["season"] >= 2015
    moved.loc[m, "actual_total"] = 99.0
    moved.loc[m, "wind"] = 40.0
    moved.loc[m, "home_off_pass_epa_r"] *= -5
    moved = B.add_anchor(moved)
    tg2 = tg.copy()
    tg2.loc[tg2["season"] >= 2015, "neutral_sec_per_play"] = 1e6
    _, again = B.walk_forward(moved, tg2, test_seasons=(2015,))
    for k in ("alpha", "beta_raw", "beta_se", "beta"):
        assert again[0][k] == pytest.approx(base[0][k]), k
    assert again[0]["fills"] == base[0]["fills"]


def test_beta_near_one_when_market_blind_near_zero_when_market_knows():
    blind, tg = _synthetic(market="blind")
    knows, _ = _synthetic(market="knows")
    f_blind = B.fit_fold(blind[blind["season"] < 2016], tg, list(range(2012, 2016)))
    f_knows = B.fit_fold(knows[knows["season"] < 2016], tg, list(range(2012, 2016)))
    assert f_blind["beta_raw"] == pytest.approx(1.0, abs=0.15)
    assert f_blind["beta"] <= 1.0
    assert abs(f_knows["beta_raw"]) < 3 * f_knows["beta_se"] + 0.05
    assert f_blind["alpha"] in B.ALPHAS


def test_pooled_beta_recovers_known_slope():
    rng = np.random.default_rng(3)
    t_star = rng.uniform(38, 52, 4000)
    d = rng.normal(0, 3, 4000)
    actual = t_star + 0.4 * d + rng.normal(0, 10, 4000)
    beta, se, n = B.no_intercept_ols(t_star + d, t_star, actual)
    assert n == 4000
    assert beta == pytest.approx(0.4, abs=3 * se)
    assert se == pytest.approx(10 / (3 * np.sqrt(4000)), rel=0.1)


def test_anchor_matches_total_pricing():
    from src.total_pricing import devig_two_way, total_implied_mean
    gf = B.add_anchor(pd.DataFrame({"total_line": [44.5, 47.0, np.nan], "over_odds": [-120, -110, -110],
                                    "under_odds": [100, -110, -110]}))
    q, _ = devig_two_way(-120, 100)
    assert gf.loc[0, "q_over"] == pytest.approx(q)
    assert gf.loc[0, "t_star"] == pytest.approx(total_implied_mean(44.5, q, 13.2))
    assert gf.loc[1, "t_star"] == pytest.approx(47.0, abs=1e-5)
    assert np.isnan(gf.loc[2, "t_star"])


# --- metrics --------------------------------------------------------------------------

def _preds(n=400, seed=1):
    rng = np.random.default_rng(seed)
    p = pd.DataFrame({"season": rng.choice([2016, 2022], n), "total_line": 44.5,
                      "q_over": 0.5, "t_star": 44.5, "raw_mu": 44.5 + rng.normal(0, 2, n),
                      "is_indoor": rng.integers(0, 2, n), "wind": rng.uniform(0, 25, n),
                      "over_odds": -110.0, "under_odds": -110.0})
    p["high_wind"] = ((p["is_indoor"] == 0) & (p["wind"] >= 15)).astype(int)
    p["actual_total"] = np.round(p["t_star"] + rng.normal(0, 13, n))
    p["p_over"] = 0.5 + 0.1 * np.tanh(p["raw_mu"] - 44.5)
    return p


def test_log_loss_delta_is_paired_and_push_excluded():
    p = pd.DataFrame({"total_line": [44.0, 44.0, 44.5, 44.5], "actual_total": [44, 50, 40, 50],
                      "p_over": [0.9, 0.6, 0.4, 0.6], "q_over": [0.5, 0.5, 0.5, 0.5]})
    out = B.log_loss_delta(p)
    model = [-np.log(0.6), -np.log(0.6), -np.log(0.6)]
    market = [-np.log(0.5)] * 3
    assert out["n"] == 3
    assert out["delta"] == pytest.approx(np.mean(np.subtract(model, market)))
    assert out["se"] == pytest.approx(0.0, abs=1e-12)


def test_wind_check_and_record():
    p = _preds()
    w = B.mean_residual(p[p["high_wind"] == 1])
    resid = (p["actual_total"] - p["t_star"])[p["high_wind"] == 1]
    assert w["mean"] == pytest.approx(resid.mean())
    assert w["se"] == pytest.approx(resid.std(ddof=1) / np.sqrt(len(resid)))
    rec = B.record(pd.DataFrame({"p_over": [0.6, 0.4, 0.5, 0.56], "total_line": [44.5, 44.5, 44.5, 44.0],
                                 "actual_total": [50, 50, 50, 44], "over_odds": [-110, -110, -110, 120],
                                 "under_odds": [-110, -110, -110, -140]}))
    assert (rec["bets"], rec["wins"], rec["losses"], rec["pushes"]) == (3, 1, 1, 1)
    assert rec["units"] == pytest.approx(100 / 110 - 1.0)


def test_period_summary_has_every_table():
    s = B.period_summary(_preds(), 2020, 2025)
    for key in ("beta_all", "beta_wind", "beta_indoor", "ll_all", "ll_wind", "ll_indoor",
                "resid_wind", "resid_calm", "resid_indoor", "record"):
        assert key in s, key
    assert s["beta_all"]["n"] == int((_preds()["season"] == 2022).sum())


# --- 6. kill/keep ---------------------------------------------------------------------

def _summary(beta=0.0, beta_se=0.1, dll=0.001, wind_mean=0.0, wind_se=1.2, wind_dll=0.001):
    return {"beta_all": {"beta": beta, "se": beta_se, "n": 1500},
            "ll_all": {"delta": dll, "se": 0.002, "n": 1450},
            "resid_wind": {"mean": wind_mean, "se": wind_se, "n": 121},
            "ll_wind": {"delta": wind_dll, "se": 0.01, "n": 118}}


def test_keep_on_beta_path():
    v = B.verdict(_summary(beta=0.30, beta_se=0.19, dll=-0.001))
    assert v["decision"] == "keep" and v["beta_path"] and not v["wind_path"]


def test_keep_on_wind_path():
    v = B.verdict(_summary(wind_mean=-3.0, wind_se=1.2, wind_dll=-0.004))
    assert v["decision"] == "keep" and v["wind_path"] and not v["beta_path"]
    assert B.verdict(_summary(wind_mean=2.6, wind_se=1.2, wind_dll=-0.004))["decision"] == "keep"


def test_kill_otherwise():
    assert B.verdict(_summary())["decision"] == "kill"
    assert B.verdict(_summary(beta=0.28, beta_se=0.19, dll=-0.01))["decision"] == "kill"   # t 1.47
    assert B.verdict(_summary(beta=0.40, beta_se=0.19, dll=0.0))["decision"] == "kill"     # no log-loss gain
    assert B.verdict(_summary(beta=-0.40, beta_se=0.19, dll=-0.01))["decision"] == "kill"  # wrong sign
    assert B.verdict(_summary(wind_mean=-3.0, wind_se=1.2, wind_dll=0.002))["decision"] == "kill"
    assert B.verdict(_summary(wind_mean=-2.3, wind_se=1.2, wind_dll=-0.01))["decision"] == "kill"  # |t| 1.92
