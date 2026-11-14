"""Price-consistent market anchor for v3/v4 (R5).

The line anchor sets the margin distribution's mean to -home_line and ignores the price,
so with no model input a +110 side looks like a 50% proposition. The price anchor uses
m*, the mean at which the key-number engine's no-push home cover probability at the
offered line equals the de-vigged market probability q_home:

    m*  = anchor_mean(home_line, q_home, sigma, total_line)
    mu  = m* + beta * d,   d = model_mu - m*        (variant A)
                           d = model_mu + home_line (variant B: disagreement vs the line)

With d = 0 the model's probability is q_home, so the edge against break-even is minus
about half the vig at any line and price. beta is the shrinkage; there is no logistic
calibrator. fit_calibrator still runs on the walk-forward evidence as a diagnostic.
"""

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.key_numbers import calculate_spread_probabilities
from src.market_blend import blend_margin_anchored, fit_market_beta
from src.model_v2_core import fit_calibrator
from src.odds_utils import american_to_implied_prob, remove_vig

VARIANTS = ("A", "B")
# Chosen by the walk-forward backtest (src/backtest_production.py --anchor all), see the R5 PR.
PRODUCTION_VARIANT = "A"
SIGMA_TOL = 1e-3
MAX_SIGMA_ROUNDS = 4
_EPS = 1e-6


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def devig_home(home_odds, away_odds):
    """(q_home, price_assumed). A missing, zero or non-finite price means -110 both sides."""
    h, a = _finite(home_odds), _finite(away_odds)
    if h is None or a is None or h == 0 or a == 0:
        return 0.5, True
    return float(remove_vig(american_to_implied_prob(h), american_to_implied_prob(a))[0]), False


@lru_cache(maxsize=200_000)
def _anchor_mean_cached(home_line, q_home, sigma, total_line):
    def f(mu):
        return calculate_spread_probabilities(mu, sigma, home_line, total_line)["cond_home_win"] - q_home

    lo, hi = -home_line - 20.0, -home_line + 20.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo >= 0.0:
        return lo
    if f_hi <= 0.0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-10, maxiter=200))


def anchor_mean(home_line, q_home, sigma, total_line=None):
    """
    m*: the mean home margin at which the cover probability at home_line is q_home.
    Same quantity as key_numbers.market_implied_mean (R2), found with Brent's method
    instead of 60 bisection steps, and cached, because training needs it per row.
    """
    total = _finite(total_line)
    return _anchor_mean_cached(float(home_line), float(q_home), float(sigma), total)


def calibrator_slope_se(evidence, intercept, slope):
    """
    Standard error of the logistic calibrator's slope from the weighted Fisher
    information at (intercept, slope). Symmetric HOME/AWAY rows at weight 0.5 count as
    one game each, as in fit_calibrator.
    """
    ev = evidence.dropna(subset=["p_raw", "actual_win"])
    ev = ev[~ev["push"].astype(bool)]
    if ev.empty:
        return float("nan")
    w = pd.to_numeric(ev.get("weight", 1.0), errors="coerce")
    w = np.broadcast_to(np.asarray(w, dtype=float), (len(ev),))
    p = np.clip(ev["p_raw"].to_numpy(dtype=float), _EPS, 1 - _EPS)
    x = np.log(p / (1 - p))
    fitted = 1.0 / (1.0 + np.exp(-(intercept + slope * x)))
    v = w * fitted * (1 - fitted)
    info = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(math.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")


def _evidence(df, mu, sigma, totals):
    rows = []
    for (_, row), m, total in zip(df.iterrows(), mu, totals):
        home_line = float(row["home_line"])
        diff = float(row["actual_home_margin"]) + home_line
        p_h = float(calculate_spread_probabilities(m, sigma, home_line, total_line=total)["cond_home_win"])
        for side, p_raw, win in (("HOME", p_h, diff > 0), ("AWAY", 1.0 - p_h, diff < 0)):
            rows.append({"season": int(row["season"]) if "season" in row else None, "game_id": row.get("game_id"),
                         "side": side, "p_raw": p_raw, "actual_win": bool(win),
                         "push": abs(diff) < 1e-5, "weight": 0.5})
    return pd.DataFrame(rows)


def fit_price_anchor(oos, variant="A", use_total=False, mu_col="oos_mu", total_col="total_line"):
    """
    Fits the price-anchored blend on walk-forward rows. `oos` needs home_line,
    actual_home_margin, home_spread_odds, away_spread_odds, the model margin in `mu_col`
    and, when use_total, the total in `total_col`.

    beta: no-intercept OLS of margin - m* on d, clipped to [0, 1] (fit_market_beta with an
    explicit anchor). sigma: std of margin - mu. m* depends on sigma, so the two are
    iterated from the market residual's std until sigma moves less than SIGMA_TOL.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown anchor variant {variant!r}")
    hl = oos["home_line"].to_numpy(dtype=float)
    y = oos["actual_home_margin"].to_numpy(dtype=float)
    model_mu = oos[mu_col].to_numpy(dtype=float)
    q_flags = [devig_home(h, a) for h, a in zip(oos["home_spread_odds"], oos["away_spread_odds"])]
    q = np.array([f[0] for f in q_flags])
    n_assumed = int(sum(f[1] for f in q_flags))
    totals = [(_finite(t) if use_total else None) for t in (oos[total_col] if use_total else [None] * len(oos))]

    ok = np.isfinite(y) & np.isfinite(hl)
    sigma = float(np.std((y + hl)[ok]))
    rounds = 0
    while True:
        rounds += 1
        m_star = np.array([anchor_mean(h, qq, sigma, t) for h, qq, t in zip(hl, q, totals)])
        beta, beta_se, n_beta = fit_market_beta(model_mu, hl, y, anchor=m_star, variant=variant)
        mu = np.array([blend_margin_anchored(m, ms, beta, None if variant == "A" else -h)
                       for m, ms, h in zip(model_mu, m_star, hl)])
        new_sigma = float(np.std((y - mu)[ok & np.isfinite(mu)]))
        moved = abs(new_sigma - sigma)
        sigma = new_sigma
        if moved < SIGMA_TOL or rounds >= MAX_SIGMA_ROUNDS:
            break

    evidence = _evidence(oos, mu, sigma, totals)
    diag = fit_calibrator(evidence)
    method, params = diag
    slope = float(params[1]) if params else float("nan")
    intercept = float(params[0]) if params else 0.0
    slope_se = calibrator_slope_se(evidence, intercept, slope) if params else float("nan")

    return {
        "anchor": "price",
        "variant": variant,
        "beta": beta,
        "beta_se": beta_se,
        "n_beta": n_beta,
        "sigma": sigma,
        "sigma_rounds": rounds,
        "m_star": m_star,
        "mu": mu,
        "q_home": q,
        "n_price_assumed": n_assumed,
        "calibrator": ("identity", None),
        "calibrator_diagnostic": diag,
        "calibrator_diagnostic_slope": slope,
        "calibrator_diagnostic_se": slope_se,
        "evidence": evidence,
    }


def artifact_fields(fit):
    """The artifact/metadata entries R5 adds (spec: anchor, variant, beta, diagnostic)."""
    method, params = fit["calibrator_diagnostic"]
    return {
        "anchor": "price",
        "anchor_variant": fit["variant"],
        "market_beta": float(fit["beta"]),
        "market_beta_se": float(fit["beta_se"]),
        "blend_rows": int(fit["n_beta"]),
        "blend_sigma": float(fit["sigma"]),
        "anchor_price_assumed_rows": int(fit["n_price_assumed"]),
        "calibrator_diagnostic_method": method,
        "calibrator_diagnostic_params": list(params) if params else None,
        "calibrator_diagnostic_slope": float(fit["calibrator_diagnostic_slope"]),
        "calibrator_diagnostic_se": float(fit["calibrator_diagnostic_se"]),
    }


def attach_market_prices(dataset, sched):
    """Adds nflverse home_spread_odds / away_spread_odds to a training dataset by game_id."""
    cols = ["game_id", "home_spread_odds", "away_spread_odds"]
    prices = sched[cols].drop_duplicates("game_id")
    out = dataset.drop(columns=[c for c in cols[1:] if c in dataset], errors="ignore")
    return out.merge(prices, on="game_id", how="left")
