"""P5 receptions model: Poisson GLM fit, alpha selection, NB dispersion and PMF (P5 spec "Models").

- Model: StandardScaler (fit on the training fold) + sklearn PoissonRegressor, alpha from
  ALPHA_GRID chosen by an inner split that holds out each training season with at least
  MIN_INNER_EARLIER earlier training seasons, training on the earlier seasons only.
- Baseline: mean rec_mean_8.
- Dispersion: NB size r per position group by maximum likelihood on [R_MIN, R_MAX], on the inner
  out-of-fold means, separately for model and baseline. A fit that reaches R_MAX means Poisson.
- PMF: P(R = 0..PMF_MAX) and P(R > PMF_MAX) lumped.

Choices the spec leaves open (fixed before any outcome was computed):
- alpha minimises the mean Poisson deviance pooled over the inner held-out seasons (ties: the
  smaller alpha). The baseline's dispersion uses the same inner held-out rows.
- A fit "reaches" R_MAX when the bounded optimum lies within R_CAP_TOL of it.
- A test season with no inner fold (2018: only 2016-2017 to train on) is skipped (owner, 2026-11-24).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.props import features as F

ALPHA_GRID = (0.001, 0.01, 0.1, 1.0)
R_MIN = 1.0
R_MAX = 200.0
R_CAP_TOL = 0.01
PMF_MAX = 14
MIN_INNER_EARLIER = 2
MAX_ITER = 1000


# --- distributions ------------------------------------------------------------------

def _dist(mu, r):
    mu = np.asarray(mu, dtype=float)
    if r is None or r >= R_MAX:
        return poisson(mu)
    return nbinom(r, r / (r + mu))


def nb_pmf(mu, r):
    """(n, PMF_MAX + 2): P(R = 0..PMF_MAX), then P(R > PMF_MAX)."""
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    d = _dist(mu, r)
    k = np.arange(PMF_MAX + 1).reshape(1, -1)
    body = d.pmf(k)
    tail = d.sf(PMF_MAX).reshape(-1, 1)
    return np.hstack([body, tail])


def nb_pmf_full(mu, r, kmax=500):
    return _dist(float(mu), r).pmf(np.arange(kmax + 1))


def p_over(mu, r, lines):
    """(n, len(lines)) of P(R > line) for half-point lines."""
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    fl = np.floor(np.asarray(lines, dtype=float)).reshape(1, -1)
    return _dist(mu, r).sf(fl)


def _nb_negll(r, y, mu):
    p = r / (r + mu)
    ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(p) + y * np.log1p(-p)
    return -float(ll.sum())


def fit_nb_r(y, mu):
    """ML NB size on [R_MIN, R_MAX]; None (Poisson) when the optimum reaches R_MAX."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    res = minimize_scalar(_nb_negll, bounds=(R_MIN, R_MAX), args=(y, mu), method="bounded",
                          options={"xatol": 1e-4})
    r = float(res.x)
    if r >= R_MAX - R_CAP_TOL or _nb_negll(R_MAX, y, mu) <= _nb_negll(r, y, mu):
        return None
    return r


# --- model --------------------------------------------------------------------------

def fit_model(X, y, alpha):
    m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=MAX_ITER))
    m.fit(X, np.asarray(y, dtype=float))
    return m


def inner_seasons(train_seasons):
    tr = sorted(int(s) for s in train_seasons)
    return [s for s in tr if sum(t < s for t in tr) >= MIN_INNER_EARLIER]


def pick_alpha(scores):
    return min(sorted(scores), key=lambda a: (scores[a], a))


def _fit_predict(uni, games, train_seasons, test_mask, alpha_list):
    pri = F.training_priors(uni, games, train_seasons)
    tr = uni[uni["season"].isin(train_seasons)]
    Xtr = F.design_matrix(tr, pri)
    Xte = F.design_matrix(uni[test_mask], pri)
    return {a: fit_model(Xtr, tr["receptions"], a).predict(Xte) for a in alpha_list}, pri


def fit_fold(uni, games, test_season):
    """Predictions for one test season, or None when it has no inner fold."""
    train = sorted(int(s) for s in uni["season"].unique() if F.FIRST_SEASON <= s < test_season)
    inner = inner_seasons(train)
    if not inner:
        return None
    oof = {a: [] for a in ALPHA_GRID}
    rows = []
    for s in inner:
        mask = uni["season"] == s
        preds, _ = _fit_predict(uni, games, [t for t in train if t < s], mask, ALPHA_GRID)
        for a in ALPHA_GRID:
            oof[a].append(preds[a])
        rows.append(uni.loc[mask, ["receptions", "rec_mean_8", "position_group"]])
    ho = pd.concat(rows)
    scores = {a: float(mean_poisson_deviance(ho["receptions"], np.concatenate(oof[a]))) for a in ALPHA_GRID}
    alpha = pick_alpha(scores)
    mu_oof = np.concatenate(oof[alpha])
    r_model, r_base = {}, {}
    for p in F.POSITIONS:
        m = (ho["position_group"] == p).values
        r_model[p] = fit_nb_r(ho["receptions"].values[m], mu_oof[m]) if m.any() else None
        r_base[p] = fit_nb_r(ho["receptions"].values[m], ho["rec_mean_8"].values[m]) if m.any() else None

    test_mask = uni["season"] == test_season
    preds, pri = _fit_predict(uni, games, train, test_mask, [alpha])
    out = uni.loc[test_mask].copy()
    out["mu_model"] = preds[alpha]
    out["mu_base"] = out["rec_mean_8"].astype(float)
    out["r_model"] = out["position_group"].map(lambda p: r_model.get(p))
    out["r_base"] = out["position_group"].map(lambda p: r_base.get(p))
    out["alpha"] = alpha
    info = {"season": test_season, "train": train, "inner": inner, "alpha": alpha, "inner_deviance": scores,
            "r_model": r_model, "r_base": r_base, "n_train": int(uni["season"].isin(train).sum()),
            "n_test": int(test_mask.sum())}
    return out, info
