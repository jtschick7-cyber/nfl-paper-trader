"""Walk-forward backtest of the production shadow decision rule (R4).

    .venv/bin/python -m src.backtest_production            # v3, test seasons 2014-2025
    .venv/bin/python -m src.backtest_production --model v4 # v4 (dataset build is slow)

For each test season S (2014-2025; 2012-2013 are the first training window):

1. The dataset is built once for 2012-2025 (`build_v3_training_dataset`, or v4's builder).
2. Ridge (StandardScaler + constrained ridge, alpha 15) is trained on seasons < S.
3. beta and the calibrator are fitted on walk-forward (TimeSeriesSplit) predictions within
   the training seasons only, the same nested procedure as the trainers. The committed
   artifact's beta is never used for past seasons: that would leak.
4. Each game of S goes through `src.decision.shadow_decision`, the rule the loggers run,
   with nflverse `home_line = -spread_line`, `home_spread_odds`, `away_spread_odds` (and
   `total_line` for v4), and is graded against the final score at the actual price.
5. Zero-disagreement probe: the same game with no model disagreement. Any PICK there
   comes from the price or the line number alone, not from the model. On the line anchor
   that is raw_mu = -home_line; on the price anchor (R5) it is raw_mu = m* (variant A) or
   -home_line (variant B), and the line probe raw_mu = -home_line is kept as well.

Anchors (R5): --anchor line is the pre-R5 rule (blend on -home_line, logistic
calibrator). --anchor price --variant A|B is the price-consistent anchor of
src/price_anchor.py: beta and sigma come from fit_price_anchor, the calibrator is the
identity, and fit_calibrator's slope and its SE are kept per fold as a diagnostic.

Limits:
- nflverse lines are closing lines. This tests the rule against the close, not against
  the in-week opener the live system bets.
- One 2012-2025 game has no spread price; games without a line or both prices are skipped
  for decisions (they stay in training, as in production).
- The integrity guard is not applied (it needs live injury and line-move data).

Writes data/backtest_production_<model>_<anchor>_{games,seasons,calibration,probe}.csv
(<anchor> is line, priceA or priceB). No model files, no Sheet access.
"""

import argparse
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.decision import PICK, shadow_decision
from src.key_numbers import calculate_spread_probabilities
from src.market_blend import blend_margin, fit_market_beta
from src.model_v2_core import fit_calibrator
from src.price_anchor import PRODUCTION_VARIANT, fit_price_anchor

FIRST_SEASON = 2012
LAST_SEASON = 2025
FIRST_TEST_SEASON = 2014
ALPHA = 15.0
N_SPLITS = 5
LOW_INPUT_PTS = 0.5
LONG_PRICE = 105
CALIBRATION_BUCKETS = [0.0, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 1.0]
CALIBRATION_MIN_N = 200  # R5 validation 2: buckets with n >= 200 must sit within 2 SE
DEFAULT_ANCHOR = "price"  # the CLI default is the live v3/v4 rule
DEFAULT_VARIANT = PRODUCTION_VARIANT
OUT_PREFIX = "data/backtest_production"


class ModelSpec:
    def __init__(self, name, features, fit_ridge, use_total):
        self.name = name
        self.features = features
        self.fit_ridge = fit_ridge
        self.use_total = use_total


def v3_spec():
    from src.train_model_v3 import V3_FEATURES, fit_constrained_ridge
    return ModelSpec("v3", V3_FEATURES, fit_constrained_ridge, use_total=False)


def v4_spec():
    from src.train_model_v4 import V4_FEATURES, fit_constrained_ridge
    return ModelSpec("v4", V4_FEATURES, fit_constrained_ridge, use_total=True)


# --- fitting -----------------------------------------------------------------------

def fit_fold(train, spec, alpha=ALPHA, n_splits=N_SPLITS, anchor="line", variant="A"):
    """
    Everything the live artifact holds, fitted on `train` only (chronological order):
    scaler + ridge on all of it; beta, blended sigma and calibrator on its walk-forward
    predictions. Returns a dict that also records which game ids fed each step.
    anchor="price" fits the R5 price-consistent anchor instead (identity calibrator).
    """
    train = train.sort_values("kickoff_utc", kind="stable").reset_index(drop=True)
    X = train[spec.features].values.astype(float)
    y = train["actual_home_margin"].values.astype(float)

    scaler = StandardScaler().fit(X)
    ridge = spec.fit_ridge(scaler.transform(X), y, alpha=alpha)

    oos_idx, oos_mu = [], []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(train):
        sc = StandardScaler().fit(X[tr])
        r = spec.fit_ridge(sc.transform(X[tr]), y[tr], alpha=alpha)
        oos_idx.extend(va)
        oos_mu.extend(r.predict(sc.transform(X[va])))

    oos = train.iloc[oos_idx].copy()
    oos["oos_mu"] = oos_mu
    if anchor == "price":
        fit = fit_price_anchor(oos, variant=variant, use_total=spec.use_total)
        evidence = fit["evidence"]
        return {
            "scaler": scaler, "ridge": ridge, "anchor": "price", "variant": variant,
            "beta": fit["beta"], "beta_se": fit["beta_se"], "n_beta": fit["n_beta"],
            "sigma": fit["sigma"], "calibrator": fit["calibrator"],
            "calibrator_diagnostic": fit["calibrator_diagnostic"],
            "calibrator_diagnostic_slope": fit["calibrator_diagnostic_slope"],
            "calibrator_diagnostic_se": fit["calibrator_diagnostic_se"],
            "n_price_assumed": fit["n_price_assumed"],
            "train_ids": set(train["game_id"]), "blend_ids": set(oos["game_id"]),
            "calibration_ids": set(evidence["game_id"]) if not evidence.empty else set(),
        }
    if anchor != "line":
        raise ValueError(f"unknown anchor {anchor!r}")
    beta, beta_se, n_beta = fit_market_beta(oos["oos_mu"], oos["home_line"], oos["actual_home_margin"])
    oos["blend_mu"] = [blend_margin(m, hl, beta) for m, hl in zip(oos["oos_mu"], oos["home_line"])]
    sigma = float(np.std(oos["actual_home_margin"] - oos["blend_mu"]))

    evidence = []
    for _, row in oos.iterrows():
        home_line = float(row["home_line"])
        diff = float(row["actual_home_margin"]) + home_line
        total = row.get("total_line") if spec.use_total else None
        p_h = float(calculate_spread_probabilities(row["blend_mu"], sigma, home_line, total_line=total)["cond_home_win"])
        for side, p_raw, win in (("HOME", p_h, diff > 0), ("AWAY", 1.0 - p_h, diff < 0)):
            evidence.append({"season": int(row["season"]), "game_id": row["game_id"], "side": side,
                             "p_raw": p_raw, "actual_win": bool(win), "push": abs(diff) < 1e-5, "weight": 0.5})
    evidence = pd.DataFrame(evidence)
    calibrator = fit_calibrator(evidence)

    return {
        "scaler": scaler,
        "ridge": ridge,
        "anchor": "line",
        "variant": None,
        "beta": beta,
        "beta_se": beta_se,
        "n_beta": n_beta,
        "sigma": sigma,
        "calibrator": calibrator,
        "train_ids": set(train["game_id"]),
        "blend_ids": set(oos["game_id"]),
        "calibration_ids": set(evidence["game_id"]) if not evidence.empty else set(),
    }


# --- grading -----------------------------------------------------------------------

def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def grade(side, home_line, odds, home_margin):
    """(result, units) for a 1u ticket on `side` at the home line, priced at `odds`."""
    cover = home_margin + home_line if side == "HOME" else -(home_margin + home_line)
    if cover > 0:
        return "WIN", (odds / 100.0 if odds > 0 else 100.0 / -odds)
    if cover < 0:
        return "LOSS", -1.0
    return "PUSH", 0.0


def decide_season(test, fold, spec):
    """One row per decidable game of a test season: live-rule decision, probe, grade."""
    rows = []
    skipped = 0
    for _, g in test.iterrows():
        spread = _num(g.get("mkt_spread_line"))
        h_odds, a_odds = _num(g.get("home_spread_odds")), _num(g.get("away_spread_odds"))
        if spread is None or h_odds is None or a_odds is None or h_odds == 0 or a_odds == 0:
            skipped += 1
            continue
        home_line = -spread
        total = _num(g.get("mkt_total_line")) if spec.use_total else None
        x = np.array([[float(g[c]) for c in spec.features]])
        raw_mu = float(fold["ridge"].predict(fold["scaler"].transform(x))[0])
        kw = dict(sigma=fold["sigma"], market_beta=fold["beta"], calibrator=fold["calibrator"], total_line=total)
        if fold.get("anchor") == "price":
            kw.update(anchor="price", anchor_variant=fold["variant"])
        d = shadow_decision(raw_mu, home_line, h_odds, a_odds, **kw)
        line_probe = shadow_decision(-home_line, home_line, h_odds, a_odds, **kw)
        if fold.get("anchor") == "price" and fold["variant"] == "A":
            probe = shadow_decision(line_probe["anchor_mean"], home_line, h_odds, a_odds, **kw)
        else:
            probe = line_probe
        margin = float(g["actual_home_margin"])
        result, units = grade(d["best_side"], home_line, d["best_odds"], margin)
        lean = d["raw_lean_pts"]
        lean_result = grade("HOME" if lean > 0 else "AWAY", home_line, -110, margin)[0] if lean != 0 else ""
        rows.append({
            "game_id": g["game_id"], "season": int(g["season"]), "week": g.get("week"),
            "home_line": home_line, "home_odds": h_odds, "away_odds": a_odds, "total_line": total,
            "raw_mu": raw_mu, "raw_lean_pts": lean, "mu_blend": d["mu_blend"],
            "p_home_cal": d["p_home_cal"], "best_side": d["best_side"], "best_odds": d["best_odds"],
            "best_edge": d["best_edge"], "decision": d["decision"],
            "probe_decision": probe["decision"], "probe_side": probe["best_side"],
            "probe_odds": probe["best_odds"], "probe_edge": probe["best_edge"],
            "probe_p_home": probe["p_home_cal"], "q_home": d["q_home"], "anchor_mean": d["anchor_mean"],
            "line_probe_decision": line_probe["decision"],
            "home_margin": margin, "home_cover": (margin + home_line > 0) if margin + home_line != 0 else None,
            "result": result, "units": units, "lean_result": lean_result,
        })
    return pd.DataFrame(rows), skipped


def walk_forward(ds, spec, test_seasons, alpha=ALPHA, n_splits=N_SPLITS, anchor="line", variant="A"):
    """(games_df, folds). folds[i] has the season, the fitted pieces and the ids used."""
    ds = ds.sort_values("kickoff_utc", kind="stable").reset_index(drop=True)
    games, folds = [], []
    for season in test_seasons:
        train = ds[ds["season"] < season]
        test = ds[ds["season"] == season]
        if train.empty or test.empty:
            continue
        fold = fit_fold(train, spec, alpha=alpha, n_splits=n_splits, anchor=anchor, variant=variant)
        g, skipped = decide_season(test, fold, spec)
        fold.update({"season": season, "test_ids": set(test["game_id"]), "skipped": skipped})
        folds.append(fold)
        games.append(g)
    return (pd.concat(games, ignore_index=True) if games else pd.DataFrame()), folds


# --- report ------------------------------------------------------------------------

def _hit(results):
    w = sum(1 for r in results if r == "WIN")
    l = sum(1 for r in results if r == "LOSS")
    n = w + l
    if n == 0:
        return None, None
    p = w / n
    return p, math.sqrt(p * (1 - p) / n)


def summarize(games, label, fold=None):
    bets = games[games["decision"] == PICK] if not games.empty else games
    hit, hit_se = _hit(list(bets["result"])) if len(bets) else (None, None)
    lean, lean_se = _hit(list(games["lean_result"])) if len(games) else (None, None)
    units = float(bets["units"].sum()) if len(bets) else 0.0
    row = {
        "season": label,
        "games": len(games),
        "bets": len(bets),
        "wins": int((bets["result"] == "WIN").sum()) if len(bets) else 0,
        "losses": int((bets["result"] == "LOSS").sum()) if len(bets) else 0,
        "pushes": int((bets["result"] == "PUSH").sum()) if len(bets) else 0,
        "hit_pct": None if hit is None else 100 * hit,
        "hit_se_pct": None if hit_se is None else 100 * hit_se,
        "units": units,
        "roi_pct": (100 * units / len(bets)) if len(bets) else None,
        "low_input_bets": int((bets["raw_lean_pts"].abs() < LOW_INPUT_PTS).sum()) if len(bets) else 0,
        "long_price_bets": int((bets["best_odds"] >= LONG_PRICE).sum()) if len(bets) else 0,
        "probe_picks": int((games["probe_decision"] == PICK).sum()) if len(games) else 0,
        "line_probe_picks": int((games["line_probe_decision"] == PICK).sum())
        if len(games) and "line_probe_decision" in games else 0,
        "lean_hit_pct": None if lean is None else 100 * lean,
        "lean_hit_se_pct": None if lean_se is None else 100 * lean_se,
    }
    if fold is not None:
        cal = fold["calibrator"]
        row.update({
            "beta": fold["beta"], "beta_se": fold["beta_se"], "n_beta": fold["n_beta"],
            "sigma": fold["sigma"], "calibrator": cal[0] if isinstance(cal, tuple) else str(cal),
            "cal_slope": cal[1][1] if isinstance(cal, tuple) and cal[0] == "logistic" and cal[1] else None,
            "skipped_no_price": fold["skipped"],
        })
        if fold.get("anchor") == "price":
            diag = fold["calibrator_diagnostic"]
            row.update({"calibrator": "identity", "cal_slope": None,
                        "diag_method": diag[0],
                        "diag_slope": fold["calibrator_diagnostic_slope"],
                        "diag_slope_se": fold["calibrator_diagnostic_se"]})
    return row


def season_table(games, folds):
    rows = [summarize(games[games["season"] == f["season"]], f["season"], f) for f in folds]
    rows.append(summarize(games, "all"))
    return pd.DataFrame(rows)


def calibration_table(games):
    g = games[games["home_cover"].notna()].copy()
    if g.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_p_home_cal", "home_cover_pct"])
    g["bucket"] = pd.cut(g["p_home_cal"], CALIBRATION_BUCKETS, include_lowest=True)
    out = g.groupby("bucket", observed=True).agg(
        n=("p_home_cal", "size"), mean_p_home_cal=("p_home_cal", "mean"),
        home_cover_pct=("home_cover", lambda s: 100 * s.astype(float).mean()),
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    p = out["mean_p_home_cal"]
    out["se_pct"] = 100 * np.sqrt(p * (1 - p) / out["n"])
    out["z"] = (out["home_cover_pct"] - 100 * p) / out["se_pct"]
    out["within_2se"] = out["z"].abs() <= 2.0
    return out


def fit_metrics(games):
    """Out-of-sample fit of the test-season decisions: cover log loss and Brier on
    p_home_cal (pushes excluded), and RMSE of mu_blend against the final margin."""
    g = games[games["home_cover"].notna()]
    y = g["home_cover"].astype(float).to_numpy()
    p = np.clip(g["p_home_cal"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    err = games["home_margin"].to_numpy(dtype=float) - games["mu_blend"].to_numpy(dtype=float)
    return {
        "n_cover": int(len(g)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(g) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if len(g) else float("nan"),
        "margin_rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
    }


def probe_table(games):
    cols = ["season", "game_id", "home_line", "probe_side", "probe_odds", "probe_edge", "decision", "best_side"]
    if games.empty:
        return pd.DataFrame(columns=cols)
    return games[games["probe_decision"] == PICK][cols].reset_index(drop=True)


def _fmt(v, spec="{:.2f}"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else spec.format(v)


def _calib_label(r):
    method = r.get("calibrator")
    if method is None or (isinstance(method, float) and math.isnan(method)):
        return "-"
    diag = r.get("diag_slope")
    if method == "identity" and diag is not None and not pd.isna(diag):
        return f"identity (diag slope {diag:.3f} ± {r.get('diag_slope_se'):.3f})"
    slope = r.get("cal_slope")
    return f"slope {slope:.3f}" if method == "logistic" and slope is not None and not pd.isna(slope) else str(method)


def print_report(model, seasons, calib, probe, games, label=""):
    print(f"\nPRODUCTION RULE BACKTEST: {model} {label}(walk-forward, closing lines, flat 1u at the actual price)")
    hdr = ["season", "games", "bets", "W-L-P", "hit %", "units", "ROI %", "|lean|<0.5", ">=+105",
           "probe", "lean hit %", "beta", "beta se", "calibrator"]
    print(" | ".join(hdr))
    for _, r in seasons.iterrows():
        print(" | ".join([
            str(r["season"]), str(r["games"]), str(r["bets"]), f"{r['wins']}-{r['losses']}-{r['pushes']}",
            _fmt(r["hit_pct"], "{:.1f}") + ("" if r["hit_se_pct"] is None or pd.isna(r["hit_se_pct"]) else f" ± {r['hit_se_pct']:.1f}"),
            _fmt(r["units"], "{:+.2f}"), _fmt(r["roi_pct"], "{:+.1f}"),
            str(r["low_input_bets"]), str(r["long_price_bets"]), str(r["probe_picks"]),
            _fmt(r["lean_hit_pct"], "{:.1f}") + ("" if pd.isna(r["lean_hit_se_pct"]) else f" ± {r['lean_hit_se_pct']:.1f}"),
            _fmt(r.get("beta"), "{:.4f}"), _fmt(r.get("beta_se"), "{:.4f}"), _calib_label(r),
        ]))

    print("Calibrator 'slope 0.200' is fit_calibrator's fallback when the training evidence is under 500 weighted rows;"
          " 'no_edge' (non-positive slope) sets every probability to 0.5.")
    print("\nCALIBRATION (p_home_cal bucket vs actual home cover rate, pushes excluded)")
    for _, r in calib.iterrows():
        flag = "" if r["n"] < CALIBRATION_MIN_N else ("  ok" if r["within_2se"] else "  OUTSIDE 2 SE")
        print(f"  {r['bucket']:<16} n={int(r['n']):<5} mean p={r['mean_p_home_cal']:.3f}  covered={r['home_cover_pct']:.1f}%"
              f" ± {r['se_pct']:.1f}  z={r['z']:+.2f}{flag}")

    m = fit_metrics(games)
    print(f"\nFIT (test seasons): cover log loss {m['log_loss']:.5f}, Brier {m['brier']:.5f} on {m['n_cover']} games;"
          f" margin RMSE {m['margin_rmse']:.4f}")
    if "q_home" in games and games["q_home"].notna().any():
        gap = (games["probe_p_home"] - games["q_home"]).abs().max()
        print(f"Probe |p_home - q_home| max {gap:.2e}; line-probe (raw_mu = -home_line) picks:"
              f" {int((games['line_probe_decision'] == PICK).sum())}")

    bets = games[games["decision"] == PICK]
    print(f"\nZERO-DISAGREEMENT PROBE: {len(probe)} PICK(s) with raw_mu = -home_line (price or line artifacts)")
    if len(probe):
        tally = probe["probe_odds"].round(0).astype(int).value_counts().sort_index()
        print("  price of each probe pick (odds: count): " + ", ".join(f"{o:+d}: {c}" for o, c in tally.items()))
        overlap = len(set(probe["game_id"]) & set(bets["game_id"]))
        print(f"  {overlap} of {len(bets)} real bets are also probe picks")
    if len(bets):
        tally = bets["best_odds"].round(0).astype(int).value_counts().sort_index()
        print("  price of each real bet (odds: count): " + ", ".join(f"{o:+d}: {c}" for o, c in tally.items()))


def load_dataset(model):
    """Builds the 2012-2025 dataset and attaches the nflverse market columns by game_id."""
    from src.model_utils import _load_sched

    seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))
    epa = pd.read_csv("data/team_game_epa.csv")
    if model == "v4":
        from src.train_model_v4 import build_v4_training_dataset
        sched = _load_sched(list(range(FIRST_SEASON - 2, LAST_SEASON + 1)))
        ds = build_v4_training_dataset(seasons, epa, sched, pd.read_csv("data/team_game_qb.csv"),
                                       pd.read_csv("data/team_game_turnovers.csv"))
    else:
        from src.train_model_v3 import build_v3_training_dataset
        sched = _load_sched(list(range(FIRST_SEASON - 2, LAST_SEASON + 1)))
        ds = build_v3_training_dataset(seasons, epa, sched)
    mkt = sched[["game_id", "spread_line", "home_spread_odds", "away_spread_odds", "total_line"]].rename(
        columns={"spread_line": "mkt_spread_line", "total_line": "mkt_total_line"})
    return ds.merge(mkt.drop_duplicates("game_id"), on="game_id", how="left")


def run_tag(anchor, variant, use_total=True):
    tag = "line" if anchor == "line" else f"price{variant}"
    return tag if use_total else f"{tag}_nototal"


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the production shadow rule")
    parser.add_argument("--model", choices=["v3", "v4"], default="v3")
    parser.add_argument("--first-test-season", type=int, default=FIRST_TEST_SEASON)
    parser.add_argument("--anchor", choices=["line", "price", "all"], default=DEFAULT_ANCHOR,
                        help="line = pre-R5 rule; price = R5 anchor; all = line, price A and price B")
    parser.add_argument("--variant", choices=["A", "B"], default=DEFAULT_VARIANT)
    parser.add_argument("--no-total", action="store_true", help="v4 without totals-conditioned sigma")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    t0 = time.time()
    spec = v4_spec() if args.model == "v4" else v3_spec()
    if args.no_total:
        spec.use_total = False
    ds = load_dataset(args.model)
    print(f"Dataset: {len(ds)} games, built in {time.time() - t0:.0f}s")

    runs = [("line", None), ("price", "A"), ("price", "B")] if args.anchor == "all" else [(args.anchor, args.variant)]
    os.makedirs("data", exist_ok=True)
    for anchor, variant in runs:
        games, folds = walk_forward(ds, spec, list(range(args.first_test_season, LAST_SEASON + 1)),
                                    anchor=anchor, variant=variant or "A")
        seasons = season_table(games, folds)
        calib = calibration_table(games)
        probe = probe_table(games)
        tag = run_tag(anchor, variant, spec.use_total or args.model == "v3")
        print_report(args.model, seasons, calib, probe, games, label=f"[{tag}] ")
        for name, df in (("games", games), ("seasons", seasons), ("calibration", calib), ("probe", probe)):
            df.to_csv(f"{OUT_PREFIX}_{args.model}_{tag}_{name}.csv", index=False)
        print(f"Wrote {OUT_PREFIX}_{args.model}_{tag}_*.csv ({time.time() - t0:.0f}s so far)")


if __name__ == "__main__":
    main()
