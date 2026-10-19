import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.model_utils import DIVISION_MAP, _load_sched


# ---------------------------------------------------------------------
# Protocol boundaries inspired by FreeMargin Engine
# ---------------------------------------------------------------------
# Historical research only.
# 2025 is locked.
# 2026+ is prospective.
# ---------------------------------------------------------------------

HISTORY_START = int(os.getenv("V2_HISTORY_START", "1999"))
HISTORY_END = int(os.getenv("V2_HISTORY_END", "2024"))
HISTORY_SEASONS = list(range(HISTORY_START, HISTORY_END + 1))

DEV_START = int(os.getenv("V2_DEV_START", "2010"))
DEV_END = int(os.getenv("V2_DEV_END", "2017"))

THRESH_START = int(os.getenv("V2_THRESH_START", "2018"))
THRESH_END = int(os.getenv("V2_THRESH_END", "2020"))

VAL_START = int(os.getenv("V2_VAL_START", "2021"))
VAL_END = int(os.getenv("V2_VAL_END", "2024"))

TEST_SEASONS = list(range(DEV_START, VAL_END + 1))

# ---------------------------------------------------------------------
# Simplified v2 model settings
# ---------------------------------------------------------------------

ELO_HFA = float(os.getenv("V2_ELO_HFA", "2.0"))
ELO_K = float(os.getenv("V2_ELO_K", "0.05"))
ELO_RETENTION = float(os.getenv("V2_ELO_RETENTION", "0.5"))

ROLL_WINDOW = int(os.getenv("V2_ROLL_WINDOW", "4"))
MIN_PERIODS = int(os.getenv("V2_MIN_PERIODS", "3"))

RIDGE_ALPHA = float(os.getenv("V2_RIDGE_ALPHA", "10.0"))

SIGMA_MIN = float(os.getenv("V2_SIGMA_MIN", "10.0"))
SIGMA_MAX = float(os.getenv("V2_SIGMA_MAX", "22.0"))

MIN_TRAIN_ROWS = int(os.getenv("V2_MIN_TRAIN_ROWS", "100"))
MIN_CALIBRATION_ROWS = int(os.getenv("V2_MIN_CALIBRATION_ROWS", "150"))

EPSILON = 1e-6

FEATURES = [
    "elo_diff",
    "diff_pf",
    "diff_pa",
    "diff_margin",
    "diff_rest",
    "division",
]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _to_bool_int(value):
    if pd.isna(value):
        return 0

    s = str(value).strip().lower()

    if s in {"1", "true", "yes", "y", "division", "div"}:
        return 1

    if s in {"0", "false", "no", "n", "nan", "none", ""}:
        return 0

    try:
        return int(float(value))
    except Exception:
        return 0


def scope_name(season):
    season = int(season)

    if DEV_START <= season <= DEV_END:
        return "development"

    if THRESH_START <= season <= THRESH_END:
        return "threshold_selection"

    if VAL_START <= season <= VAL_END:
        return "validation"

    return "other"


def american_to_implied_prob_scalar(odds):
    odds = float(odds)

    if odds > 0:
        return 100.0 / (odds + 100.0)

    if odds < 0:
        return (-odds) / ((-odds) + 100.0)

    return 0.5


def remove_vig_scalar(p_home, p_away):
    total = p_home + p_away

    if total <= 0:
        return 0.5, 0.5

    return p_home / total, p_away / total


def profit_units(odds, stake=1.0):
    odds = float(odds)

    if odds > 0:
        return stake * odds / 100.0

    if odds < 0:
        return stake * 100.0 / (-odds)

    return 0.0


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Data loading and canonicalization
# ---------------------------------------------------------------------

def load_canonical_games(seasons, as_of=None, exclude_seasons=None):
    """
    Load completed regular-season games and canonicalize market columns.

    Important convention:
        raw nflverse spread_line behaves like away spread.
        home_line = -spread_line
        away_line = spread_line
    """
    seasons = sorted({int(s) for s in seasons})

    raw = _load_sched(seasons)

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    for col in [
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "div_game",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    df["home_team"] = df["home_team"].astype(str).str.upper().str.strip()
    df["away_team"] = df["away_team"].astype(str).str.upper().str.strip()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["game_id", "season", "home_team", "away_team", "kickoff_utc"])
    df = df[df["home_score"].notna() & df["away_score"].notna()]

    if exclude_seasons:
        excl = {int(x) for x in exclude_seasons}
        df = df[~df["season"].astype(int).isin(excl)]

    ko = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df["kickoff_utc"] = ko
    df = df.dropna(subset=["kickoff_utc"])

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)

        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")

        df = df[df["kickoff_utc"] < as_of_ts]

    df = df.drop_duplicates(subset=["game_id"])

    try:
        local = df["kickoff_utc"].dt.tz_convert("America/New_York")
    except Exception:
        local = df["kickoff_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")

    df["local_day"] = local.dt.normalize()
    df = df.dropna(subset=["local_day"])

    df["actual_home_margin"] = df["home_score"] - df["away_score"]

    df["raw_spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce")

    # FreeMargin-style canonical home sportsbook line.
    df["home_line"] = -df["raw_spread_line"]

    df["home_odds"] = pd.to_numeric(df["home_spread_odds"], errors="coerce")
    df["away_odds"] = pd.to_numeric(df["away_spread_odds"], errors="coerce")

    if "div_game" in df.columns and df["div_game"].notna().any():
        df["division"] = df["div_game"].apply(_to_bool_int)
    else:
        home_div = df["home_team"].map(DIVISION_MAP)
        away_div = df["away_team"].map(DIVISION_MAP)
        df["division"] = (
            home_div.notna()
            & away_div.notna()
            & (home_div == away_div)
        ).astype(int)

    df = df.sort_values(["local_day", "kickoff_utc", "game_id"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Date-safe Elo and rolling features
# ---------------------------------------------------------------------

def build_features_from_games(games):
    """
    Build pregame features from completed games.

    Returns:
        games: DataFrame with per-game features
        ratings: final Elo ratings dictionary
        counts: final Elo game-count dictionary
        histories: final rolling-history dictionary
    """
    df = games.copy()

    if df.empty:
        return df, {}, {}, {}

    df = df.sort_values(["local_day", "kickoff_utc", "game_id"]).reset_index(drop=True)

    ratings = {}
    counts = {}
    histories = {}

    snapshots = {}
    roll_values = {}

    current_season = None

    for day, g in df.groupby("local_day", sort=True):
        season = int(g["season"].iloc[0])

        if current_season is None:
            current_season = season
        elif season > current_season:
            factor = ELO_RETENTION ** (season - current_season)
            ratings = {team: rating * factor for team, rating in ratings.items()}
            current_season = season

        elo_updates = []
        hist_updates = []

        for idx, row in g.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            rh = float(ratings.get(home, 0.0))
            ra = float(ratings.get(away, 0.0))
            nh = int(counts.get(home, 0))
            na = int(counts.get(away, 0))

            snapshots[idx] = (rh, ra, nh, na)

            margin = float(row["actual_home_margin"])
            residual = margin - (ELO_HFA + rh - ra)
            delta = ELO_K * residual

            elo_updates.append((home, away, delta))

            vals = {}

            sides = [
                ("home", home, float(row["home_score"]), float(row["away_score"]), margin),
                ("away", away, float(row["away_score"]), float(row["home_score"]), -margin),
            ]

            for prefix, team, pf, pa, team_margin in sides:
                hist = histories.get(team, [])
                n = len(hist)

                vals[prefix + "_n_prior"] = n

                if n >= MIN_PERIODS:
                    last = hist[-ROLL_WINDOW:]

                    vals[prefix + "_pf"] = float(np.mean([x[1] for x in last]))
                    vals[prefix + "_pa"] = float(np.mean([x[2] for x in last]))
                    vals[prefix + "_margin_roll"] = float(np.mean([x[3] for x in last]))
                    vals[prefix + "_rest"] = float((day - hist[-1][0]).days)
                else:
                    vals[prefix + "_pf"] = np.nan
                    vals[prefix + "_pa"] = np.nan
                    vals[prefix + "_margin_roll"] = np.nan
                    vals[prefix + "_rest"] = np.nan

            roll_values[idx] = vals

            hist_updates.append((home, day, float(row["home_score"]), float(row["away_score"]), margin))
            hist_updates.append((away, day, float(row["away_score"]), float(row["home_score"]), -margin))

        # Apply Elo updates only after all games on this day are snapshotted.
        for home, away, delta in elo_updates:
            ratings[home] = ratings.get(home, 0.0) + delta
            ratings[away] = ratings.get(away, 0.0) - delta

            counts[home] = counts.get(home, 0) + 1
            counts[away] = counts.get(away, 0) + 1

        # Apply rolling histories only after all games on this day are snapshotted.
        for team, d, pf, pa, margin in hist_updates:
            histories.setdefault(team, []).append((d, float(pf), float(pa), float(margin)))

    for idx, (rh, ra, nh, na) in snapshots.items():
        df.loc[idx, "home_elo"] = rh
        df.loc[idx, "away_elo"] = ra
        df.loc[idx, "home_elo_n_prior"] = nh
        df.loc[idx, "away_elo_n_prior"] = na

    for idx, vals in roll_values.items():
        for col, value in vals.items():
            df.loc[idx, col] = value

    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["diff_pf"] = df["home_pf"] - df["away_pf"]
    df["diff_pa"] = df["home_pa"] - df["away_pa"]
    df["diff_margin"] = df["home_margin_roll"] - df["away_margin_roll"]
    df["diff_rest"] = df["home_rest"] - df["away_rest"]

    return df, ratings, counts, histories


def compute_live_features_from_state(ratings, counts, histories, home, away, kickoff_utc, division=None):
    """
    Compute pregame features for a future game using state built from completed games only.
    """
    ko = pd.Timestamp(kickoff_utc)

    if ko.tzinfo is None:
        ko = ko.tz_localize("UTC")
    else:
        ko = ko.tz_convert("UTC")

    try:
        local_day = ko.tz_convert("America/New_York").normalize()
    except Exception:
        local_day = ko.tz_localize("UTC").tz_convert("America/New_York").normalize()

    rh = float(ratings.get(home, 0.0))
    ra = float(ratings.get(away, 0.0))

    def side_stats(team):
        hist = histories.get(team, [])
        n = len(hist)

        if n >= MIN_PERIODS:
            last = hist[-ROLL_WINDOW:]

            pf = float(np.mean([x[1] for x in last]))
            pa = float(np.mean([x[2] for x in last]))
            margin_roll = float(np.mean([x[3] for x in last]))
            rest = float((local_day - hist[-1][0]).days)

            return pf, pa, margin_roll, rest, n

        return np.nan, np.nan, np.nan, np.nan, n

    home_pf, home_pa, home_margin_roll, home_rest, home_n = side_stats(home)
    away_pf, away_pa, away_margin_roll, away_rest, away_n = side_stats(away)

    if division is None:
        home_div = DIVISION_MAP.get(home)
        away_div = DIVISION_MAP.get(away)
        division = 1 if home_div is not None and home_div == away_div else 0

    feats = {
        "elo_diff": rh - ra,
        "diff_pf": home_pf - away_pf,
        "diff_pa": home_pa - away_pa,
        "diff_margin": home_margin_roll - away_margin_roll,
        "diff_rest": home_rest - away_rest,
        "division": float(division),
    }

    diag = {
        "home_elo": rh,
        "away_elo": ra,
        "home_n_prior": home_n,
        "away_n_prior": away_n,
        "local_day": local_day,
    }

    return feats, diag


# ---------------------------------------------------------------------
# Temporal logistic calibration
# ---------------------------------------------------------------------

def fit_calibrator(evidence_df):
    """
    Fit a simple temporal logistic calibrator.

    Evidence columns expected:
        p_raw
        actual_win
        push
        weight optional
    """
    if evidence_df is None or evidence_df.empty:
        return ("identity", None)

    ev = evidence_df.copy()

    required = ["p_raw", "actual_win", "push"]
    missing = [c for c in required if c not in ev.columns]

    if missing:
        return ("identity", None)

    ev = ev.dropna(subset=["p_raw", "actual_win"])
    ev = ev[~ev["push"].astype(bool)]

    if len(ev) < MIN_CALIBRATION_ROWS:
        return ("identity", None)

    y = ev["actual_win"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return ("identity", None)

    p = np.clip(ev["p_raw"].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)

    if "weight" in ev.columns:
        weights = pd.to_numeric(ev["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        weights = np.ones(len(ev), dtype=float)

    try:
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        model.fit(x, y, sample_weight=weights)
    except Exception:
        return ("identity", None)

    intercept = float(model.intercept_[0])
    coef = float(model.coef_[0, 0])

    if not math.isfinite(intercept) or not math.isfinite(coef):
        return ("identity", None)

    # A non-positive slope is not a usable monotonic calibration.
    if coef <= 0.0:
        return ("identity", None)

    return ("logistic", (intercept, coef))


def calibrate_probability(calibrator, p):
    """
    Calibrate a single probability.
    """
    method, params = calibrator

    p = float(p)

    if not math.isfinite(p):
        return np.nan

    if method == "identity":
        return p

    if method == "logistic":
        intercept, coef = params

        pc = min(max(p, EPSILON), 1.0 - EPSILON)
        logit = math.log(pc / (1.0 - pc))
        z = intercept + coef * logit
        z = max(min(z, 700.0), -700.0)

        return 1.0 / (1.0 + math.exp(-z))

    return p


# ---------------------------------------------------------------------
# Walk-forward calibration evidence
# ---------------------------------------------------------------------

def collect_calibration_evidence(games):
    """
    Produce out-of-sample raw probabilities for 2010-2024.

    These rows are used only to fit the final calibrator.
    """
    pool = games.dropna(subset=FEATURES + ["actual_home_margin"]).copy()

    rows = []

    for test_season in TEST_SEASONS:
        train = pool[pool["season"] < test_season].copy()
        test = games[
            (games["season"] == test_season)
            & games["home_line"].notna()
            & games[FEATURES].notna().all(axis=1)
        ].copy()

        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue

        Xtr = train[FEATURES].to_numpy(dtype=float)
        ytr = train["actual_home_margin"].to_numpy(dtype=float)

        scaler = StandardScaler().fit(Xtr)
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(Xtr), ytr)

        resid = ytr - ridge.predict(scaler.transform(Xtr))
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 13.5

        if not math.isfinite(sigma) or sigma <= 0:
            sigma = 13.5

        sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))

        Xt = test[FEATURES].to_numpy(dtype=float)
        mu = ridge.predict(scaler.transform(Xt))

        home_line = test["home_line"].to_numpy(dtype=float)
        actual_margin = test["actual_home_margin"].to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            p_home_raw = norm.cdf((mu + home_line) / sigma)

        p_home_raw = np.clip(p_home_raw, 0.0, 1.0)
        p_away_raw = 1.0 - p_home_raw

        diff = actual_margin + home_line

        push = np.abs(diff) < 1e-9
        home_win = diff > 0
        away_win = diff < 0

        for i in range(len(test)):
            rows.append(
                {
                    "season": int(test.iloc[i]["season"]),
                    "game_id": test.iloc[i]["game_id"],
                    "side": "HOME",
                    "p_raw": float(p_home_raw[i]),
                    "actual_win": bool(home_win[i]),
                    "push": bool(push[i]),
                    "weight": 0.5,
                }
            )

            rows.append(
                {
                    "season": int(test.iloc[i]["season"]),
                    "game_id": test.iloc[i]["game_id"],
                    "side": "AWAY",
                    "p_raw": float(p_away_raw[i]),
                    "actual_win": bool(away_win[i]),
                    "push": bool(push[i]),
                    "weight": 0.5,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Final artifact training
# ---------------------------------------------------------------------

def train_final_artifact(games, evidence_df):
    pool = games.dropna(subset=FEATURES + ["actual_home_margin"]).copy()

    if pool.empty:
        raise SystemExit("No trainable rows for final v2 artifact.")

    X = pool[FEATURES].to_numpy(dtype=float)
    y = pool["actual_home_margin"].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(X), y)

    resid = y - ridge.predict(scaler.transform(X))
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 13.5

    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 13.5

    sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))

    calibrator = fit_calibrator(evidence_df)

    artifact = {
        "feature_columns": FEATURES,
        "scaler": scaler,
        "ridge": ridge,
        "sigma": sigma,
        "calibrator_method": calibrator[0],
        "calibrator_params": calibrator[1],
        "trained_on_seasons": HISTORY_SEASONS,
        "calibration_evidence_rows": int(len(evidence_df)) if evidence_df is not None else 0,
        "constants": {
            "ELO_HFA": ELO_HFA,
            "ELO_K": ELO_K,
            "ELO_RETENTION": ELO_RETENTION,
            "ROLL_WINDOW": ROLL_WINDOW,
            "MIN_PERIODS": MIN_PERIODS,
            "RIDGE_ALPHA": RIDGE_ALPHA,
            "SIGMA_MIN": SIGMA_MIN,
            "SIGMA_MAX": SIGMA_MAX,
            "DEV_START": DEV_START,
            "DEV_END": DEV_END,
            "THRESH_START": THRESH_START,
            "THRESH_END": THRESH_END,
            "VAL_START": VAL_START,
            "VAL_END": VAL_END,
        },
    }

    return artifact
