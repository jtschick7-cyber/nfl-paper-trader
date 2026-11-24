"""P5a receptions model: NB PMF and dispersion, walk-forward folds, Stage A metrics and gate
(P5 spec tests 5-7, 9). Spec: notes/implementation-plan/P5-receptions-model.md."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from src.props import backtest_receptions as B
from src.props import train_receptions as T


# --- 5. Negative Binomial -----------------------------------------------------------

@pytest.mark.parametrize("mu,r", [(0.4, 3.0), (3.7, 8.0), (6.2, 1.0), (9.5, 50.0)])
def test_pmf_sums_to_one_and_mean_matches(mu, r):
    pmf = T.nb_pmf(np.array([mu]), r)
    assert pmf.shape == (1, T.PMF_MAX + 2)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    full = T.nb_pmf_full(mu, r, kmax=2000)
    assert full.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(np.arange(len(full)) @ full) == pytest.approx(mu, abs=1e-6)
    assert np.allclose(pmf[0, :T.PMF_MAX + 1], full[:T.PMF_MAX + 1])
    assert pmf[0, -1] == pytest.approx(full[T.PMF_MAX + 1:].sum())


def test_r_at_cap_falls_back_to_poisson():
    mu = np.array([2.0, 5.0])
    a = T.nb_pmf(mu, None)
    b = T.nb_pmf(mu, T.R_MAX)
    assert np.allclose(a, b)
    assert a[1, 3] == pytest.approx(poisson(5.0).pmf(3))


def test_ml_fit_recovers_known_r():
    rng = np.random.default_rng(3)
    mu = rng.uniform(1.5, 7.0, 20000)
    r = 6.0
    y = rng.negative_binomial(r, r / (r + mu))
    est = T.fit_nb_r(y, mu)
    assert est == pytest.approx(r, rel=0.10)


def test_ml_fit_hits_cap_on_poisson_data():
    rng = np.random.default_rng(4)
    mu = rng.uniform(1.5, 7.0, 20000)
    y = rng.poisson(mu)
    assert T.fit_nb_r(y, mu) is None  # reaches R_MAX: use Poisson


def test_p_over_matches_scipy():
    mu = np.array([3.2])
    p = T.p_over(mu, 7.0, B.LINES)
    for j, line in enumerate(B.LINES):
        assert p[0, j] == pytest.approx(nbinom(7.0, 7.0 / 10.2).sf(math.floor(line)))
    q = T.p_over(mu, None, B.LINES)
    assert q[0, 0] == pytest.approx(poisson(3.2).sf(2))


# --- folds --------------------------------------------------------------------------

def test_inner_seasons_need_two_earlier_seasons():
    assert T.inner_seasons([2016, 2017]) == []          # test season 2018: no fold (owner: skip 2018)
    assert T.inner_seasons([2016, 2017, 2018]) == [2018]
    assert T.inner_seasons(list(range(2016, 2025))) == list(range(2018, 2025))


def test_choose_alpha_picks_lowest_inner_deviance():
    scores = {0.001: 1.20, 0.01: 1.10, 0.1: 1.15, 1.0: 1.30}
    assert T.pick_alpha(scores) == 0.01
    assert T.pick_alpha({0.001: 1.0, 0.01: 1.0, 0.1: 2.0, 1.0: 2.0}) == 0.001  # ties: smaller alpha


def test_fit_model_learns_a_signal():
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": rng.normal(size=4000), "b": rng.normal(size=4000)})
    mu = np.exp(1.0 + 0.4 * X["a"])
    y = rng.poisson(mu)
    m = T.fit_model(X, y, alpha=0.001)
    pred = m.predict(X)
    assert np.corrcoef(pred, mu)[0, 1] > 0.98


# --- 6. metrics on hand-computed examples -------------------------------------------

def test_ll_sum_by_hand():
    y = np.array([3])
    p = np.array([[0.8, 0.5, 0.3, 0.1]])
    expected = -(math.log(0.8) + math.log(0.5) + math.log(0.7) + math.log(0.9))
    assert B.ll_sum(y, p)[0] == pytest.approx(expected)
    y2 = np.array([6])  # over every line
    assert B.ll_sum(y2, p)[0] == pytest.approx(-sum(math.log(x) for x in (0.8, 0.5, 0.3, 0.1)))


def test_clustered_delta_by_hand():
    d = pd.DataFrame({"ll_sum_model": [1.0, 2.0, 6.0], "ll_sum_base": [0.0, 0.0, 0.0],
                      "game_id": ["a", "a", "b"]})
    m, se, n = B.delta(d)
    assert m == pytest.approx(3.0) and n == 3
    # residual sums: a = (1-3)+(2-3) = -3, b = +3; sqrt(18 * 2/1) / 3 = 2
    assert se == pytest.approx(2.0)


def test_calibration_rule_by_hand():
    assert B.bin_calibrated(0.54, 0.5, 500)        # bound 2*sqrt(.25/500) = 0.0447
    assert not B.bin_calibrated(0.55, 0.5, 500)
    p = np.array([0.05, 0.1, 0.19, 0.999, 1.0])
    assert list(B.bin_index(p)) == [0, 1, 1, 9, 9]
    rng = np.random.default_rng(1)
    pp = np.full(600, 0.35)
    yy = (rng.uniform(size=600) < 0.35).astype(int)
    tab = B.calibration_table(pp, yy)
    row = tab[tab["n"] > 0].iloc[0]
    assert row["bin"] == 3 and row["n"] == 600 and row["mean_p"] == pytest.approx(0.35)
    assert row["observed"] == pytest.approx(yy.mean())
    assert bool(row["tested"]) and bool(row["calibrated"]) == B.bin_calibrated(yy.mean(), 0.35, 600)
    small = B.calibration_table(np.full(499, 0.35), np.ones(499, dtype=int))
    assert not small["tested"].any() and small["calibrated"].all()


def test_log_score_and_crps_by_hand():
    pmf = np.zeros((2, T.PMF_MAX + 2))
    pmf[0, 0] = 1.0
    pmf[1, 2] = 0.5
    pmf[1, 4] = 0.5
    y = np.array([0, 3])
    ls = B.log_score(y, pmf)
    assert ls[0] == pytest.approx(0.0)
    assert ls[1] > 30  # P(R = 3) = 0 is clipped, not infinite
    c = B.crps(y, pmf)
    assert c[0] == pytest.approx(0.0)
    # F = 0,0,.5,.5,1,... vs 1{3 <= k}: k=2 (.5-0)^2, k=3 (.5-1)^2 -> 0.5
    assert c[1] == pytest.approx(0.5)
    lump = np.zeros((1, T.PMF_MAX + 2))
    lump[0, -1] = 1.0  # all mass at >= 15; y = 17 -> k = 0..14 contribute 0, F = 1 after
    assert B.crps(np.array([17]), lump)[0] == pytest.approx(2.0)  # k = 15, 16: (1 - 0)^2 each
    assert B.log_score(np.array([17]), lump)[0] == pytest.approx(0.0)


# --- 7. gate ------------------------------------------------------------------------

def _passing():
    return {"delta": -0.010, "se": 0.003, "all_bins_calibrated": True, "inj_delta": -0.02,
            "season_deltas": {2020: -0.01, 2021: -0.02, 2022: 0.01, 2023: -0.01, 2024: -0.005, 2025: 0.002}}


def test_gate_passes_when_all_four_hold():
    g = B.gate(_passing())
    assert g["verdict"] == "PASS" and all(g["conditions"].values())


@pytest.mark.parametrize("change,cond", [
    ({"delta": -0.0089}, "delta_3se"),                      # -0.0089 / 0.003 < 3 SE
    ({"delta": 0.001}, "delta_3se"),
    ({"all_bins_calibrated": False}, "calibrated"),
    ({"inj_delta": 0.0}, "injury_subset"),
    ({"season_deltas": {2020: -0.01, 2021: -0.02, 2022: 0.01, 2023: -0.01, 2024: 0.0, 2025: 0.002}},
     "seasons_4_of_6"),
])
def test_gate_fails_on_each_condition(change, cond):
    s = {**_passing(), **change}
    g = B.gate(s)
    assert g["verdict"] == "FAIL"
    assert not g["conditions"][cond]
    assert sum(not v for v in g["conditions"].values()) == 1


def test_gate_fails_without_se():
    s = {**_passing(), "se": float("nan")}
    assert B.gate(s)["verdict"] == "FAIL"


# --- 9. frozen constants ------------------------------------------------------------

def test_frozen_model_and_stage_a_constants():
    assert T.ALPHA_GRID == (0.001, 0.01, 0.1, 1.0)
    assert (T.R_MIN, T.R_MAX) == (1.0, 200.0)
    assert T.PMF_MAX == 14
    assert T.MIN_INNER_EARLIER == 2
    assert B.LINES == (2.5, 3.5, 4.5, 5.5)
    assert B.CAL_MIN_N == 500
    assert B.CAL_BIN_WIDTH == 0.1
    assert B.INJ_SUBSET == 0.10
    assert B.TEST_SEASONS == tuple(range(2018, 2026))
    assert B.GATE_SEASONS == tuple(range(2020, 2026))
    assert B.GATE_SE == 3.0
    assert B.GATE_MIN_SEASONS == 4
