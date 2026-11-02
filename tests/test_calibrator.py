import numpy as np
import pandas as pd

from src.model_v2_core import calibrate_probability, fit_calibrator
from src.odds_utils import american_to_implied_prob


def _evidence(p_raw, wins):
    return pd.DataFrame({"p_raw": p_raw, "actual_win": wins, "push": False, "weight": 1.0})


def test_inverted_evidence_flags_no_edge():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 2000)
    # Outcomes anti-correlated with the raw probability
    wins = rng.uniform(size=2000) < (1.0 - p)
    method, _ = fit_calibrator(_evidence(p, wins))
    assert method == "no_edge"


def test_no_edge_calibrator_never_produces_positive_edge():
    cal = ("no_edge", (0.0, -0.13))
    for p in [0.3, 0.55, 0.75]:
        assert calibrate_probability(cal, p) == 0.5
    assert 0.5 - american_to_implied_prob(-110) < 0


def test_informative_evidence_keeps_positive_slope():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.3, 0.7, 2000)
    wins = rng.uniform(size=2000) < p
    method, params = fit_calibrator(_evidence(p, wins))
    assert method == "logistic"
    assert params[1] > 0
