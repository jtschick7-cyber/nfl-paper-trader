"""P5 Stage A: walk-forward kill test of the receptions model against the player's own 8-game mean.

    .venv/bin/python -u -m src.props.backtest_receptions [--out CSV]

Spec: notes/implementation-plan/P5-receptions-model.md (rules frozen 2026-11-24). Runs ONCE.
For each test season S in TEST_SEASONS, train on 2016..S-1 and predict S (2018 has no inner fold
and is skipped, owner 2026-11-24). Pooled evaluation and the gate use GATE_SEASONS.

Metrics per player-game: LL_sum = sum over LINES of the binary log loss of 1{R > L} with
p = P(R > L); Delta = mean(LL_sum_model - LL_sum_base) with SE clustered by game_id; the full
discrete log loss -log P(R = r) (r >= 15 lumped) and CRPS beside it. Calibration bins of width
0.1 on the model's P(over) at each line; a bin with n >= CAL_MIN_N is calibrated when
|observed - mean p| <= 2 sqrt(p(1 - p) / n).
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

from src.props import data_sources as D
from src.props import features as F
from src.props import train_receptions as T
from src.topdown_replay import cluster_se
from src.totals.features import load_pbp, neutral_pace, prepare_games

LINES = (2.5, 3.5, 4.5, 5.5)
CAL_MIN_N = 500
CAL_BIN_WIDTH = 0.1
INJ_SUBSET = 0.10
TEST_SEASONS = tuple(range(2018, 2026))
GATE_SEASONS = tuple(range(2020, 2026))
GATE_SE = 3.0
GATE_MIN_SEASONS = 4
LAST_SEASON = 2025
OUT_PATH = "data/receptions_stage_a.csv"
EPS = 1e-15


# --- metrics ------------------------------------------------------------------------

def ll_sum(y, p):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    over = (y > np.asarray(LINES).reshape(1, -1)).astype(float)
    return -(over * np.log(p) + (1 - over) * np.log1p(-p)).sum(axis=1)


def log_score(y, pmf):
    idx = np.minimum(np.asarray(y, dtype=int), T.PMF_MAX + 1)
    return -np.log(np.clip(pmf[np.arange(len(idx)), idx], EPS, None))


def crps(y, pmf):
    """Discrete CRPS: sum over k >= 0 of (F(k) - 1{y <= k})^2, lumped mass at PMF_MAX + 1."""
    y = np.asarray(y, dtype=int)
    cdf = np.cumsum(pmf, axis=1)
    k = np.arange(pmf.shape[1]).reshape(1, -1)
    ind = (y.reshape(-1, 1) <= k).astype(float)
    extra = np.maximum(0, y - (T.PMF_MAX + 2))  # k beyond the lump with F = 1, 1{y <= k} = 0
    return ((cdf - ind) ** 2).sum(axis=1) + extra


def delta(df, a="ll_sum_model", b="ll_sum_base"):
    """(mean of a - b, SE clustered by game_id, n)."""
    d = (df[a] - df[b]).astype(float)
    if len(d) == 0:
        return float("nan"), float("nan"), 0
    return float(d.mean()), cluster_se(d.values, df["game_id"].values), int(len(d))


def bin_index(p):
    return np.clip(np.floor(np.asarray(p, dtype=float) / CAL_BIN_WIDTH + 1e-9), 0, round(1 / CAL_BIN_WIDTH) - 1).astype(int)


def bin_calibrated(observed, mean_p, n):
    return abs(observed - mean_p) <= 2 * math.sqrt(mean_p * (1 - mean_p) / n)


def calibration_table(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    b = bin_index(p)
    rows = []
    for i in range(round(1 / CAL_BIN_WIDTH)):
        m = b == i
        n = int(m.sum())
        mp = float(p[m].mean()) if n else float("nan")
        ob = float(y[m].mean()) if n else float("nan")
        tested = n >= CAL_MIN_N
        rows.append({"bin": i, "lo": round(i * CAL_BIN_WIDTH, 2), "hi": round((i + 1) * CAL_BIN_WIDTH, 2), "n": n,
                     "mean_p": mp, "observed": ob, "tested": tested,
                     "calibrated": bin_calibrated(ob, mp, n) if tested else True})
    return pd.DataFrame(rows)


def gate(s):
    """PASS when all four frozen conditions hold on the pooled GATE_SEASONS summary."""
    d, se = s["delta"], s["se"]
    neg_seasons = sum(1 for yr in GATE_SEASONS if s["season_deltas"].get(yr, float("nan")) < 0)
    cond = {
        "delta_3se": bool(d < 0 and se > 0 and math.isfinite(se) and -d / se >= GATE_SE),
        "calibrated": bool(s["all_bins_calibrated"]),
        "injury_subset": bool(s["inj_delta"] < 0),
        "seasons_4_of_6": bool(neg_seasons >= GATE_MIN_SEASONS),
    }
    return {"verdict": "PASS" if all(cond.values()) else "FAIL", "conditions": cond, "negative_seasons": neg_seasons}


# --- scoring the predictions --------------------------------------------------------

def score(preds, folds):
    """Adds P(over) per line, LL_sum, log score and CRPS for model and baseline."""
    out = []
    for f in folds:
        d = preds[preds["season"] == f["season"]].copy()
        y = d["receptions"].astype(int).values
        for who, mu_col, rr in (("model", "mu_model", f["r_model"]), ("base", "mu_base", f["r_base"])):
            p = np.zeros((len(d), len(LINES)))
            pmf = np.zeros((len(d), T.PMF_MAX + 2))
            for pos in F.POSITIONS:
                m = (d["position_group"] == pos).values
                if m.any():
                    p[m] = T.p_over(d[mu_col].values[m], rr.get(pos), LINES)
                    pmf[m] = T.nb_pmf(d[mu_col].values[m], rr.get(pos))
            for j, line in enumerate(LINES):
                d[f"p_over_{line}_{who}"] = p[:, j]
            d[f"ll_sum_{who}"] = ll_sum(y, p)
            d[f"log_score_{who}"] = log_score(y, pmf)
            d[f"crps_{who}"] = crps(y, pmf)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def summarize(scored):
    g = scored[scored["season"].isin(GATE_SEASONS)]
    dm, se, n = delta(g)
    cal = {line: calibration_table(g[f"p_over_{line}_model"], (g["receptions"] > line).astype(int)) for line in LINES}
    all_cal = all(bool(t["calibrated"].all()) for t in cal.values())
    inj = g[g["missing_same_pos_share"] >= INJ_SUBSET]
    di, sei, ni = delta(inj)
    seasons = {int(s): delta(scored[scored["season"] == s]) for s in sorted(scored["season"].unique())}
    positions = {p: delta(g[g["position_group"] == p]) for p in F.POSITIONS}
    s = {"delta": dm, "se": se, "n": n, "games": int(g["game_id"].nunique()), "all_bins_calibrated": all_cal,
         "inj_delta": di, "inj_se": sei, "inj_n": ni,
         "season_deltas": {k: v[0] for k, v in seasons.items()}, "seasons": seasons, "positions": positions,
         "calibration": cal,
         "log_score": delta(g, "log_score_model", "log_score_base"),
         "crps": delta(g, "crps_model", "crps_base"),
         "means": {c: float(g[c].mean()) for c in ("ll_sum_model", "ll_sum_base", "log_score_model",
                                                    "log_score_base", "crps_model", "crps_base")}}
    s["gate"] = gate(s)
    return s


# --- report -------------------------------------------------------------------------

def _fmt(t):
    m, se, n = t
    return f"{m:+.4f} (SE {se:.4f}, n {n})" if n else "n 0"


def _r(v):
    return "Poisson" if v is None else f"{v:.2f}"


def report(s, folds, skipped, counts, coverage, uni):
    L = []
    gt = s["gate"]
    L.append(f"STAGE A VERDICT: {gt['verdict']}")
    L.append("Gate (pooled " + f"{GATE_SEASONS[0]}-{GATE_SEASONS[-1]}): "
             + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in gt["conditions"].items()))
    z = -s["delta"] / s["se"] if s["se"] and s["se"] > 0 else float("nan")
    L.append(f"\nPrimary: Delta LL_sum = {s['delta']:+.4f}, game-clustered SE {s['se']:.4f} "
             f"(Delta/SE = {-z:+.2f}), n {s['n']} player-games on {s['games']} games")
    L.append(f"Injury subset (missing_same_pos_share >= {INJ_SUBSET}): {_fmt((s['inj_delta'], s['inj_se'], s['inj_n']))}")
    L.append(f"Seasons with Delta < 0 (of {len(GATE_SEASONS)}): {gt['negative_seasons']}")
    L.append("\nPer season (model - baseline LL_sum):")
    for yr, t in s["seasons"].items():
        L.append(f"  {yr}: {_fmt(t)}" + ("" if yr in GATE_SEASONS else "  (not in gate)"))
    for yr in skipped:
        L.append(f"  {yr}: skipped (no inner fold under the frozen alpha rule)")
    L.append("Per position group (pooled gate seasons):")
    for p, t in s["positions"].items():
        L.append(f"  {p}: {_fmt(t)}")
    m = s["means"]
    L.append("\nBeside the primary metric (pooled gate seasons, mean per player-game):")
    L.append(f"  LL_sum      model {m['ll_sum_model']:.4f}  baseline {m['ll_sum_base']:.4f}")
    L.append(f"  log loss    model {m['log_score_model']:.4f}  baseline {m['log_score_base']:.4f}  "
             f"Delta {_fmt(s['log_score'])}")
    L.append(f"  CRPS        model {m['crps_model']:.4f}  baseline {m['crps_base']:.4f}  Delta {_fmt(s['crps'])}")
    L.append("\nCalibration of the model's P(over), pooled gate seasons (bins with n >= 500 are tested):")
    for line, t in s["calibration"].items():
        L.append(f"  line {line}:")
        for r in t[t["n"] > 0].itertuples():
            flag = ("ok" if r.calibrated else "MISCALIBRATED") if r.tested else "untested"
            L.append(f"    [{r.lo:.1f},{r.hi:.1f})  n {r.n:6d}  mean p {r.mean_p:.3f}  observed {r.observed:.3f}  {flag}")
    L.append("\nFolds (alpha by inner mean Poisson deviance; NB r per position group, model / baseline):")
    for f in folds:
        dev = ", ".join(f"{a:g}: {v:.4f}" for a, v in f["inner_deviance"].items())
        rs = "; ".join(f"{p} {_r(f['r_model'][p])} / {_r(f['r_base'][p])}" for p in F.POSITIONS)
        L.append(f"  {f['season']}: train {f['train'][0]}-{f['train'][-1]} (n {f['n_train']}), inner held-out "
                 f"{f['inner'][0]}-{f['inner'][-1]}, alpha {f['alpha']:g} [{dev}], r: {rs}, test n {f['n_test']}")
    L.append("\nCoverage:")
    L.append(f"  REG snap rows {coverage['snap_rows_reg']}, no gsis_id match (dropped) {coverage['snap_rows_unmatched']}"
             f" by season {coverage['unmatched_by_season']}")
    L.append(f"  universe rules (player-game rows, in order): {counts}")
    L.append(f"  universe rows by season: {uni.groupby('season').size().to_dict()}")
    return "\n".join(L)


# --- main ---------------------------------------------------------------------------

def load_inputs(games_path=D.GAMES_PATH, pbp_dir=D.PBP_DIR, first=F.FIRST_SEASON, last=LAST_SEASON):
    sched = pd.read_csv(games_path, low_memory=False)
    games = prepare_games(sched, first, last)
    lines = sched.drop_duplicates("game_id").set_index("game_id")["spread_line"]
    games["spread_line"] = pd.to_numeric(games["game_id"].map(lines), errors="coerce")
    seasons = range(first, last + 1)
    data = D.load_all(seasons)
    pace = neutral_pace(load_pbp(pbp_dir, seasons))
    return games, data, pace


def run(games, data, pace):
    frame = F.build_player_frame(games, data["snaps"], data["stats"], data["injuries"], pace, data["positions"])
    uni, counts = F.apply_universe(frame)
    folds, skipped, preds = [], [], []
    for s in TEST_SEASONS:
        res = T.fit_fold(uni, games, s)
        if res is None:
            skipped.append(s)
            print(f"fold {s}: skipped (no inner fold)", flush=True)
            continue
        p, info = res
        print(f"fold {s}: alpha {info['alpha']:g}, test n {info['n_test']}", flush=True)
        folds.append(info)
        preds.append(p)
    scored = score(pd.concat(preds), folds)
    return scored, folds, skipped, counts, uni


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv)
    games, data, pace = load_inputs()
    print(f"games {len(games)}, snap rows {len(data['snaps'])}, stats rows {len(data['stats'])}, "
          f"pace rows {len(pace)}", flush=True)
    scored, folds, skipped, counts, uni = run(games, data, pace)
    keep = ["season", "week", "game_id", "kickoff_utc", "gsis_id", "team", "opp", "position_group", "receptions",
            "rec_mean_8", "missing_same_pos_share", "alpha", "mu_model", "mu_base", "r_model", "r_base"]
    keep += [c for c in scored.columns if c.startswith(("p_over_", "ll_sum_", "log_score_", "crps_"))]
    scored[keep].to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(scored)} rows)\n", flush=True)
    s = summarize(scored)
    print(report(s, folds, skipped, counts, data["coverage"], uni), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
