"""
Market-anchored margin blending.

The closing spread predicts NFL margins better than any of the box-score models
(market RMSE ~12.65 vs ~13.1 for v4 on 2023-2025). Rather than betting raw model
margins against the market, treat the market as the prior and only move off it
by the share of model disagreement that has predicted ATS results out of sample:

    mu_blend = market_mu + beta * (model_mu - market_mu),   market_mu = -home_line

beta is the no-intercept OLS slope of the ATS residual (margin - market_mu) on
the model's disagreement (model_mu - market_mu) across walk-forward predictions,
clipped to [0, 1]. beta = 0 means the model adds nothing beyond the line.
"""

import numpy as np


def market_mu(home_line):
    """Market-implied home margin from a home spread (negative = home favored)."""
    return -float(home_line)


def fit_market_beta(oos_model_mu, home_lines, actual_margins):
    """
    Fits beta on out-of-sample predictions. Returns (beta, beta_se, n).
    """
    mu = np.asarray(oos_model_mu, dtype=float)
    mkt = -np.asarray(home_lines, dtype=float)
    y = np.asarray(actual_margins, dtype=float)

    ok = np.isfinite(mu) & np.isfinite(mkt) & np.isfinite(y)
    d = (mu - mkt)[ok]
    r = (y - mkt)[ok]
    n = int(ok.sum())

    sxx = float(np.sum(d * d))
    if n < 3 or sxx <= 0.0:
        return 0.0, float("nan"), n

    raw_beta = float(np.sum(d * r) / sxx)
    resid = r - raw_beta * d
    beta_se = float(np.sqrt(np.sum(resid ** 2) / (n - 1) / sxx))
    beta = float(min(max(raw_beta, 0.0), 1.0))
    return beta, beta_se, n


def blend_margin(model_mu, home_line, beta):
    """Shrinks the model margin toward the market margin by beta."""
    m = market_mu(home_line)
    return m + float(beta) * (float(model_mu) - m)
