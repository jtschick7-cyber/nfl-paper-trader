import numpy as np
import pytest

from src.market_blend import blend_margin, fit_market_beta


def test_blend_endpoints():
    assert blend_margin(10.0, -3.0, 0.0) == pytest.approx(3.0)
    assert blend_margin(10.0, -3.0, 1.0) == pytest.approx(10.0)
    assert blend_margin(10.0, -3.0, 0.25) == pytest.approx(4.75)


def test_beta_recovers_true_share_and_clips():
    rng = np.random.default_rng(0)
    n = 4000
    home_line = rng.normal(-2.0, 6.0, n)
    disagreement = rng.normal(0.0, 3.0, n)
    model_mu = -home_line + disagreement
    margin = -home_line + 0.3 * disagreement + rng.normal(0.0, 13.0, n)
    beta, se, used = fit_market_beta(model_mu, home_line, margin)
    assert used == n
    assert beta == pytest.approx(0.3, abs=3 * se)

    # Inverted model information clips to zero (market only)
    beta_neg, _, _ = fit_market_beta(model_mu, home_line, -home_line - disagreement + rng.normal(0, 13, n))
    assert beta_neg == 0.0
