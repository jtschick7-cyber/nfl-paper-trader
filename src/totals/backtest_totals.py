"""P3 offline totals kill test: walk-forward ridge, market blend, metrics, verdict.

Spec: notes/implementation-plan/P3-totals-kill-test.md (rules frozen 2026-11-24). Offline only.

    .venv/bin/python -u -m src.totals.backtest_totals [--games PATH] [--pbp-dir DIR] [--out CSV]

For each test season S in 2014..2025: StandardScaler + Ridge on actual_total, trained on
2012..S-1, alpha from {1, 3, 10, 30, 100} by a 5-fold TimeSeriesSplit inside those seasons.
beta is the no-intercept OLS of (actual - t*) on (raw_mu - t*) over the same walk-forward
predictions inside the training seasons, clipped to [0, 1] for the blend. t* is the
price-adjusted closing total. The verdict is decided on 2020-2025 only.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.total_pricing import TOTAL_SIGMA, devig_two_way, total_implied_mean, total_probabilities
from src.totals import features as F

ALPHAS = (1, 3, 10, 30, 100)
N_SPLITS = 5
WARMUP_SEASON = 2010
FIRST_TRAIN_SEASON = 2012
LAST_SEASON = 2025
TEST_SEASONS = tuple(range(2014, LAST_SEASON + 1))
PERIODS = {"2014-2019": (2014, 2019), "2020-2025": (2020, 2025)}
DECISION_PERIOD = "2020-2025"
SIGMA = TOTAL_SIGMA
BET_OVER, BET_UNDER = 0.55, 0.45
KEEP_BETA_T = 1.5
KEEP_WIND_T = 2.0

GAMES_PATH = "data/nflverse_cache/games.csv"
PBP_DIR = "data/epa_cache"
OUT_PATH = "data/totals_backtest_games.csv"


def _finite(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


# --- anchor ---------------------------------------------------------------------------

def add_anchor(gf):
    """q_over (de-vigged closing over probability) and t* (its implied mean) where a line exists."""
    out = gf.copy()
    q, t = [], []
    for line, o, u in zip(out["total_line"], out["over_odds"], out["under_odds"]):
        if not _finite(line):
            q.append(np.nan)
            t.append(np.nan)
            continue
        qo, _ = devig_two_way(o, u)
        q.append(float(qo))
        t.append(total_implied_mean(float(line), float(qo), SIGMA))
    out["q_over"] = q
    out["t_star"] = t
    return out


def no_intercept_ols(raw_mu, t_star, actual):
    """(slope, se, n) of (actual - t*) on (raw_mu - t*), no intercept (market_blend's formula)."""
    mu = np.asarray(raw_mu, dtype=float)
    t = np.asarray(t_star, dtype=float)
    y = np.asarray(actual, dtype=float)
    ok = np.isfinite(mu) & np.isfinite(t) & np.isfinite(y)
    d, r = (mu - t)[ok], (y - t)[ok]
    n = int(ok.sum())
    sxx = float(np.sum(d * d))
    if n < 3 or sxx <= 0.0:
        return float("nan"), float("nan"), n
    b = float(np.sum(d * r) / sxx)
    se = float(np.sqrt(np.sum((r - b * d) ** 2) / (n - 1) / sxx))
    return b, se, n


# --- one fold -------------------------------------------------------------------------

def _fit(X, y, alpha):
    sc = StandardScaler().fit(X)
    return sc, Ridge(alpha=alpha).fit(sc.transform(X), y)


def fit_fold(train, team_games, train_seasons, alphas=ALPHAS, n_splits=N_SPLITS):
    """Fills, alpha, final model and beta, from the training seasons only."""
    train = train.sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    fills = F.training_fills(team_games, train, train_seasons)
    X = F.design_matrix(train, fills).values.astype(float)
    y = train["actual_total"].values.astype(float)
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))

    cv_mse = {}
    for a in alphas:
        sse, n = 0.0, 0
        for tr, va in splits:
            sc, r = _fit(X[tr], y[tr], a)
            err = y[va] - r.predict(sc.transform(X[va]))
            sse += float(np.sum(err ** 2))
            n += len(va)
        cv_mse[a] = sse / n
    alpha = min(alphas, key=lambda a: (cv_mse[a], a))

    oos_idx, oos_mu = [], []
    for tr, va in splits:
        sc, r = _fit(X[tr], y[tr], alpha)
        oos_idx.extend(va)
        oos_mu.extend(r.predict(sc.transform(X[va])))
    oos = train.iloc[oos_idx].copy()
    oos["oos_mu"] = oos_mu
    oos = oos[oos["t_star"].notna()]
    beta_raw, beta_se, n_beta = no_intercept_ols(oos["oos_mu"], oos["t_star"], oos["actual_total"])
    beta = 0.0 if not math.isfinite(beta_raw) else float(min(max(beta_raw, 0.0), 1.0))

    scaler, ridge = _fit(X, y, alpha)
    alpha_ids = set()
    for tr, va in splits:
        alpha_ids |= set(train["game_id"].iloc[tr]) | set(train["game_id"].iloc[va])
    return {
        "fills": fills, "alpha": alpha, "cv_mse": cv_mse, "scaler": scaler, "ridge": ridge,
        "beta_raw": beta_raw, "beta_se": beta_se, "beta": beta, "n_beta": n_beta,
        "train_ids": set(train["game_id"]), "alpha_ids": alpha_ids, "beta_ids": set(oos["game_id"]),
        "coef": dict(zip(F.FEATURES, ridge.coef_)),
    }


def predict(fold, gf):
    X = F.design_matrix(gf, fold["fills"])
    raw = fold["ridge"].predict(fold["scaler"].transform(X.values.astype(float)))
    out = gf.drop(columns=[c for c in F.FEATURES if c in gf.columns]).rename(columns={"wind": "wind_raw"})
    out = out.join(X)
    out["raw_mu"] = raw
    out["beta_fold"] = fold["beta"]
    out["mu_blend"] = out["t_star"] + fold["beta"] * (out["raw_mu"] - out["t_star"])
    p_over = []
    for mu, line in zip(out["mu_blend"], out["total_line"]):
        if not (_finite(mu) and _finite(line)):
            p_over.append(np.nan)
            continue
        po, pu, _ = total_probabilities(float(mu), float(line), SIGMA)
        p_over.append(po / (po + pu))
    out["p_over"] = p_over
    return out


def walk_forward(gf, team_games, test_seasons=TEST_SEASONS, first_train=FIRST_TRAIN_SEASON):
    preds, folds = [], []
    for s in test_seasons:
        train_seasons = list(range(first_train, s))
        train = gf[gf["season"].isin(train_seasons)]
        test = gf[gf["season"] == s]
        fold = fit_fold(train, team_games, train_seasons)
        fold["season"] = s
        fold["n_train"] = len(train)
        fold["n_test"] = int(test["t_star"].notna().sum())
        folds.append(fold)
        preds.append(predict(fold, test))
    return pd.concat(preds, ignore_index=True), folds


# --- metrics --------------------------------------------------------------------------

def _evaluable(p):
    return p[p["t_star"].notna()] if "t_star" in p.columns else p


def log_loss_delta(p):
    """Mean (model - market) over/under log loss, push games excluded, with a paired SE."""
    ok = p["p_over"].notna() & p["q_over"].notna() & p["total_line"].notna() & p["actual_total"].notna()
    q = p[ok & (p["actual_total"] != p["total_line"])]
    y = (q["actual_total"] > q["total_line"]).astype(float).values
    pm = np.clip(q["p_over"].values.astype(float), 1e-12, 1 - 1e-12)
    qm = np.clip(q["q_over"].values.astype(float), 1e-12, 1 - 1e-12)
    ll_m = -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    ll_q = -(y * np.log(qm) + (1 - y) * np.log(1 - qm))
    d = ll_m - ll_q
    n = len(d)
    return {"delta": float(d.mean()) if n else float("nan"),
            "se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "model": float(ll_m.mean()) if n else float("nan"),
            "market": float(ll_q.mean()) if n else float("nan"), "n": n}


def mean_residual(p):
    p = _evaluable(p)
    r = (p["actual_total"] - p["t_star"]).astype(float).values
    n = len(r)
    return {"mean": float(r.mean()) if n else float("nan"),
            "se": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"), "n": n}


def pooled_beta(p):
    p = _evaluable(p)
    b, se, n = no_intercept_ols(p["raw_mu"], p["t_star"], p["actual_total"])
    return {"beta": b, "se": se, "n": n, "t": b / se if se and math.isfinite(se) and se > 0 else float("nan")}


def _profit(odds):
    o = float(odds) if _finite(odds) else -110.0
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def record(p):
    """Descriptive: 1u at the closing price on OVER when p_over >= 0.55, UNDER when <= 0.45."""
    wins = losses = pushes = 0
    units = 0.0
    for po, line, actual, oo, uo in zip(p["p_over"], p["total_line"], p["actual_total"],
                                        p["over_odds"], p["under_odds"]):
        if not (_finite(po) and _finite(line) and _finite(actual)):
            continue
        if po >= BET_OVER:
            side, odds = "OVER", oo
        elif po <= BET_UNDER:
            side, odds = "UNDER", uo
        else:
            continue
        if actual == line:
            pushes += 1
            continue
        won = (actual > line) == (side == "OVER")
        if won:
            wins += 1
            units += _profit(odds)
        else:
            losses += 1
            units -= 1.0
    bets = wins + losses + pushes
    return {"bets": bets, "wins": wins, "losses": losses, "pushes": pushes, "units": units,
            "roi": units / bets if bets else float("nan")}


def _subsets(p):
    outdoor = p["is_indoor"] == 0
    return {
        "all": p,
        "wind": p[outdoor & (p["high_wind"] == 1)],
        "calm": p[outdoor & (p["high_wind"] == 0) & (p.get("wind_missing", 0) == 0)],
        "indoor": p[p["is_indoor"] == 1],
    }


def period_summary(preds, lo, hi):
    p = _evaluable(preds[(preds["season"] >= lo) & (preds["season"] <= hi)])
    sub = _subsets(p)
    return {
        "n_games": len(p),
        "beta_all": pooled_beta(sub["all"]), "beta_wind": pooled_beta(sub["wind"]),
        "beta_indoor": pooled_beta(sub["indoor"]),
        "ll_all": log_loss_delta(sub["all"]), "ll_wind": log_loss_delta(sub["wind"]),
        "ll_indoor": log_loss_delta(sub["indoor"]),
        "resid_wind": mean_residual(sub["wind"]), "resid_calm": mean_residual(sub["calm"]),
        "resid_indoor": mean_residual(sub["indoor"]),
        "record": record(p),
    }


def verdict(summary):
    """The frozen kill/keep rule, applied to the 2020-2025 summary."""
    b = summary["beta_all"]
    beta_t = b["beta"] / b["se"] if _finite(b["beta"]) and _finite(b["se"]) and b["se"] > 0 else float("nan")
    w = summary["resid_wind"]
    wind_t = w["mean"] / w["se"] if _finite(w["mean"]) and _finite(w["se"]) and w["se"] > 0 else float("nan")
    dll, dll_w = summary["ll_all"]["delta"], summary["ll_wind"]["delta"]
    beta_path = bool(_finite(beta_t) and b["beta"] > 0 and beta_t >= KEEP_BETA_T and _finite(dll) and dll < 0)
    wind_path = bool(_finite(wind_t) and abs(wind_t) >= KEEP_WIND_T and _finite(dll_w) and dll_w < 0)
    return {"decision": "keep" if (beta_path or wind_path) else "kill",
            "beta_path": beta_path, "wind_path": wind_path, "beta_t": beta_t, "wind_t": wind_t}


# --- report ---------------------------------------------------------------------------

def _f(v, spec="{:+.3f}"):
    return spec.format(v) if _finite(v) else "n/a"


def report(preds, folds, gf):
    lines = []
    lines.append("## Folds\n")
    lines.append("| Test season | Train games | alpha | beta raw ± SE (n) | beta used | Test games |")
    lines.append("|---|---|---|---|---|---|")
    for f in folds:
        lines.append(f"| {f['season']} | {f['n_train']} | {f['alpha']} | {_f(f['beta_raw'])} ± "
                     f"{_f(f['beta_se'], '{:.3f}')} ({f['n_beta']}) | {f['beta']:.3f} | {f['n_test']} |")
    lines.append("")
    summaries = {}
    for name, (lo, hi) in PERIODS.items():
        s = period_summary(preds, lo, hi)
        summaries[name] = s
        lines.append(f"## {name} ({s['n_games']} games with a line)\n")
        lines.append("Pooled β̂ (no-intercept OLS of actual − t* on raw_mu − t*):\n")
        lines.append("| Subset | n | β̂ | SE | t |")
        lines.append("|---|---|---|---|---|")
        for k, label in (("beta_all", "All"), ("beta_wind", "Outdoor, wind ≥ 15"), ("beta_indoor", "Indoor")):
            b = s[k]
            lines.append(f"| {label} | {b['n']} | {_f(b['beta'])} | {_f(b['se'], '{:.3f}')} | {_f(b['t'], '{:+.2f}')} |")
        lines.append("\nOver/under log loss, pushes excluded (Δ = model − market; negative favours the model):\n")
        lines.append("| Subset | n | Model | Market | Δ | SE | Δ/SE |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, label in (("ll_all", "All"), ("ll_wind", "Outdoor, wind ≥ 15"), ("ll_indoor", "Indoor")):
            v = s[k]
            z = v["delta"] / v["se"] if _finite(v["se"]) and v["se"] > 0 else float("nan")
            lines.append(f"| {label} | {v['n']} | {_f(v['model'], '{:.5f}')} | {_f(v['market'], '{:.5f}')} | "
                         f"{_f(v['delta'], '{:+.5f}')} | {_f(v['se'], '{:.5f}')} | {_f(z, '{:+.2f}')} |")
        lines.append("\nWind check, no model: mean (actual − t*) ± SE:\n")
        lines.append("| Subset | n | Mean | SE | t |")
        lines.append("|---|---|---|---|---|")
        for k, label in (("resid_wind", "Outdoor, wind ≥ 15"), ("resid_calm", "Outdoor, known wind < 15"),
                         ("resid_indoor", "Indoor")):
            v = s[k]
            t = v["mean"] / v["se"] if _finite(v["se"]) and v["se"] > 0 else float("nan")
            lines.append(f"| {label} | {v['n']} | {_f(v['mean'], '{:+.2f}')} | {_f(v['se'], '{:.2f}')} | {_f(t, '{:+.2f}')} |")
        r = s["record"]
        lines.append(f"\nDescriptive only: p_over ≥ {BET_OVER} or ≤ {BET_UNDER} at the closing price: "
                     f"{r['wins']}-{r['losses']}-{r['pushes']} ({r['bets']} bets), {r['units']:+.2f} u, "
                     f"ROI {_f(r['roi'], '{:+.1%}')}.\n")
    v = verdict(summaries[DECISION_PERIOD])
    lines.append(f"## Verdict ({DECISION_PERIOD})\n")
    lines.append(f"- β path (β̂ > 0, t ≥ {KEEP_BETA_T}, Δ log loss < 0): {'met' if v['beta_path'] else 'not met'} "
                 f"(t = {_f(v['beta_t'], '{:+.2f}')}, Δ = {_f(summaries[DECISION_PERIOD]['ll_all']['delta'], '{:+.5f}')}).")
    lines.append(f"- Wind path (|t| ≥ {KEEP_WIND_T} on mean actual − t*, Δ log loss < 0 on the subset): "
                 f"{'met' if v['wind_path'] else 'not met'} (t = {_f(v['wind_t'], '{:+.2f}')}, "
                 f"Δ = {_f(summaries[DECISION_PERIOD]['ll_wind']['delta'], '{:+.5f}')}).")
    lines.append(f"- **Decision: {v['decision'].upper()}**")
    coefs = folds[-1]["coef"]
    lines.append(f"\nLast fold ({folds[-1]['season']}) standardized ridge coefficients: "
                 + ", ".join(f"{k} {c:+.2f}" for k, c in coefs.items()))
    lines.append(f"\nGames: {len(gf)} played REG games {WARMUP_SEASON}-{LAST_SEASON} in the frame; "
                 f"{int(gf['t_star'].isna().sum())} without a line.")
    return "\n".join(lines), summaries, v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", default=GAMES_PATH)
    ap.add_argument("--pbp-dir", default=PBP_DIR)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv)

    print(f"loading {args.games} and PBP {WARMUP_SEASON}-{LAST_SEASON} from {args.pbp_dir}", flush=True)
    sched = pd.read_csv(args.games, low_memory=False)
    gf, tg = F.build_game_frame(sched, args.pbp_dir, WARMUP_SEASON, LAST_SEASON)
    gf = add_anchor(gf)
    print(f"game frame: {len(gf)} games, team-games {len(tg)}", flush=True)
    preds, folds = walk_forward(gf, tg)
    preds.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(preds)} rows)\n", flush=True)
    text, _, _ = report(preds, folds, gf)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
