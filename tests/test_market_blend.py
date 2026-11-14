import numpy as np
import pytest

from src.market_blend import blend_margin, blend_margin_anchored, fit_market_beta


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


# --- R5: price-consistent anchor --------------------------------------------------

def test_anchored_blend_endpoints():
    # variant A: disagreement against m*
    assert blend_margin_anchored(10.0, 2.2, 0.0) == pytest.approx(2.2)
    assert blend_margin_anchored(10.0, 2.2, 1.0) == pytest.approx(10.0)
    assert blend_margin_anchored(10.0, 2.2, 0.25) == pytest.approx(2.2 + 0.25 * 7.8)
    # variant B: disagreement against the line, anchor still m*
    assert blend_margin_anchored(10.0, 2.2, 0.0, line_mu=3.0) == pytest.approx(2.2)
    assert blend_margin_anchored(10.0, 2.2, 0.5, line_mu=3.0) == pytest.approx(2.2 + 0.5 * 7.0)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_anchored_beta_recovers_true_share(variant):
    rng = np.random.default_rng(5)
    n = 4000
    home_line = rng.normal(-2.0, 6.0, n)
    m_star = -home_line + rng.normal(0.0, 0.6, n)  # price-consistent mean, off the line
    d = rng.normal(0.0, 3.0, n)
    ref = m_star if variant == "A" else -home_line
    model_mu = ref + d
    margin = m_star + 0.3 * d + rng.normal(0.0, 13.0, n)
    beta, se, used = fit_market_beta(model_mu, home_line, margin, anchor=m_star, variant=variant)
    assert used == n
    assert beta == pytest.approx(0.3, abs=3 * se)


def test_anchored_beta_ignores_rows_with_a_nonfinite_anchor():
    rng = np.random.default_rng(6)
    n = 500
    home_line = rng.normal(0.0, 5.0, n)
    m_star = -home_line.copy()
    m_star[:7] = np.nan
    model_mu = m_star + rng.normal(0.0, 3.0, n)
    beta, se, used = fit_market_beta(model_mu, home_line, -home_line + rng.normal(0, 13, n), anchor=m_star)
    assert used == n - 7
