import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import LOCAL_TZ, SEASON

WINDOW = int(os.getenv("FEATURE_WINDOW", "4"))
MIN_PERIODS = int(os.getenv("MIN_PERIODS", "3"))

DIVISION_MAP = {
    "ARI": "NFC_W",
    "ATL": "NFC_S",
    "BAL": "AFC_N",
    "BUF": "AFC_E",
    "CAR": "NFC_S",
    "CHI": "NFC_N",
    "CIN": "AFC_N",
    "CLE": "AFC_N",
    "DAL": "NFC_E",
    "DEN": "AFC_W",
    "DET": "NFC_N",
    "GB": "NFC_N",
    "HOU": "AFC_S",
    "IND": "AFC_S",
    "JAX": "AFC_S",
    "KC": "AFC_W",
    "LV": "AFC_W",
    "LAC": "AFC_W",
    "LAR": "NFC_W",
    "LA": "NFC_W",
    "MIA": "AFC_E",
    "MIN": "NFC_N",
    "NE": "AFC_E",
    "NO": "NFC_S",
    "NYG": "NFC_E",
    "NYJ": "AFC_E",
    "PHI": "NFC_E",
    "PIT": "AFC_N",
    "SF": "NFC_W",
    "SEA": "NFC_W",
    "TB": "NFC_S",
    "TEN": "AFC_S",
    "WAS": "NFC_E",
}


def _first_col(df, names, required=False, label="column"):
    for name in names:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            "Required "
            + str(label)
            + " not found. Available columns: "
            + str(list(df.columns))
        )

    return None


def _is_regular_value(value):
    if pd.isna(value):
        return False

    s = str(value).upper().replace("_", " ").replace("-", " ")
    return ("REG" in s) or ("REGULAR" in s)


def _to_bool_int(value):
    s = str(value).strip().lower()

    if s in {"1", "true", "yes", "y", "division", "div"}:
        return 1

    if s in {"0", "false", "no", "n", "nan", "none", ""}:
        return 0

    try:
        return int(float(value))
    except Exception:
        return 0


def _to_utc(series):
    dt = pd.to_datetime(series, errors="coerce")

    if getattr(dt.dt, "tz", None) is None:
        try:
            dt = dt.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
        except Exception:
            dt = dt.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    return dt.dt.tz_convert("UTC")


def _make_kickoff_utc(df):
    out = df.copy()
    parts = []

    if "gameday" in out.columns and "gametime" in out.columns:
        day = pd.to_datetime(out["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
        time = out["gametime"].astype(str).str.strip()
        combined = day.astype(str) + " " + time
        parsed = pd.to_datetime(combined, errors="coerce")
        parts.append(_to_utc(parsed))

    if "game_date" in out.columns:
        parts.append(_to_utc(out["game_date"]))

    if parts:
        kickoff = parts[0]
        for part in parts[1:]:
            kickoff = kickoff.fillna(part)
    else:
        kickoff = pd.Series(pd.NaT, index=out.index)

    out["kickoff_utc"] = kickoff
    return out.dropna(subset=["kickoff_utc"])


def _normalize_schedule(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = raw.copy()

    game_col = _first_col(out, ["game_id"], True, "game_id")
    season_col = _first_col(out, ["season"], True, "season")
    week_col = _first_col(out, ["week"], False, "week")
    home_col = _first_col(out, ["home_team"], True, "home_team")
    away_col = _first_col(out, ["away_team"], True, "away_team")
    home_score_col = _first_col(out, ["home_score"], True, "home_score")
    away_score_col = _first_col(out, ["away_score"], True, "away_score")
    game_type_col = _first_col(out, ["game_type", "season_type", "seasontype"], False, "game_type")

    if game_type_col is not None:
        mask = out[game_type_col].apply(_is_regular_value)
        if mask.any():
            out = out[mask].copy()
    elif week_col is not None:
        wk = pd.to_numeric(out[week_col], errors="coerce")
        mask = (wk >= 1) & (wk <= 18)
        if mask.any():
            out = out[mask].copy()

    canon = pd.DataFrame(index=out.index)

    canon["game_id"] = out[game_col].astype(str)
    canon["season"] = pd.to_numeric(out[season_col], errors="coerce")

    if week_col is not None:
        canon["week"] = pd.to_numeric(out[week_col], errors="coerce")
    else:
        canon["week"] = np.nan

    canon["home_team"] = out[home_col].astype(str).str.upper()
    canon["away_team"] = out[away_col].astype(str).str.upper()
    canon["home_score"] = pd.to_numeric(out[home_score_col], errors="coerce")
    canon["away_score"] = pd.to_numeric(out[away_score_col], errors="coerce")

    optional_cols = [
        "gameday",
        "gametime",
        "game_date",
        "div_game",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "total_line",
        "over_odds",
        "under_odds",
        "roof",
        "surface",
        "temp",
        "wind",
        "home_rest",
        "away_rest",
    ]

    for col in optional_cols:
        if col in out.columns:
            canon[col] = out[col]

    canon = _make_kickoff_utc(canon)
    canon = canon.dropna(subset=["game_id", "home_team", "away_team", "season"])

    return canon


def _load_sched(seasons):
    import nfl_data_py as nfl
    raw = nfl.import_schedules(seasons)
    return _normalize_schedule(raw)


def _team_panel(sched):
    if sched.empty:
        return pd.DataFrame()

    home = sched.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = True
    home["pts_for"] = home["home_score"]
    home["pts_against"] = home["away_score"]
    home["margin"] = home["home_score"] - home["away_score"]

    away = sched.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = False
    away["pts_for"] = away["away_score"]
    away["pts_against"] = away["home_score"]
    away["margin"] = away["away_score"] - away["home_score"]

    base_keep = [
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "team",
        "opponent",
        "is_home",
        "pts_for",
        "pts_against",
        "margin",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    optional_keep = ["div_game"]
    keep = [c for c in base_keep + optional_keep if c in home.columns]

    panel = pd.concat([home[keep], away[keep]], ignore_index=True)
    panel = panel.dropna(subset=["team", "opponent", "pts_for", "pts_against", "kickoff_utc"])
    panel = panel.sort_values(["team", "season", "kickoff_utc"]).reset_index(drop=True)

    panel["prev_date"] = panel.groupby(["team", "season"])["kickoff_utc"].shift(1)
    panel["rest_days"] = (panel["kickoff_utc"] - panel["prev_date"]).dt.days

    if "div_game" in panel.columns:
        panel["division"] = panel["div_game"].apply(_to_bool_int)
    else:
        panel["division"] = 0

    return panel

def _add_rolling(panel, include_current):
    out = panel.copy()

    if out.empty:
        out["roll_pts_for"] = np.nan
        out["roll_pts_against"] = np.nan
        out["roll_margin"] = np.nan
        out["roll_games"] = np.nan
        return out

    for col in ["pts_for", "pts_against", "margin"]:
        if include_current:
            s = out[col]
        else:
            s = out.groupby(["team", "season"])[col].shift(1)

        out["roll_" + col] = (
            s.groupby([out["team"], out["season"]])
            .transform(lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).mean())
        )

    out["roll_games"] = out.groupby(["team", "season"]).cumcount() + 1

    if not include_current:
        out["roll_games"] = out["roll_games"] - 1

    return out


def build_training_table(seasons, sched=None):
    feature_cols = ["diff_pf", "diff_pa", "diff_margin", "diff_rest", "division"]

    base_cols = feature_cols + [
        "home_margin",
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
    ]

    if sched is None:
        sched = _load_sched(seasons)

    if sched.empty:
        return pd.DataFrame(columns=base_cols), feature_cols

    completed = sched.dropna(subset=["home_score", "away_score"]).copy()

    if completed.empty:
        return pd.DataFrame(columns=base_cols), feature_cols

    panel = _team_panel(completed)

    if panel.empty:
        return pd.DataFrame(columns=base_cols), feature_cols

    rolled = _add_rolling(panel, include_current=False)

    home = rolled[rolled["is_home"] == True].copy()
    away = rolled[rolled["is_home"] == False].copy()

    if home.empty or away.empty:
        return pd.DataFrame(columns=base_cols), feature_cols

    home = home.rename(
        columns={
            "roll_pts_for": "home_pf",
            "roll_pts_against": "home_pa",
            "roll_margin": "home_margin_roll",
            "rest_days": "home_rest_calc",
        }
    )

    away = away.rename(
        columns={
            "roll_pts_for": "away_pf",
            "roll_pts_against": "away_pa",
            "roll_margin": "away_margin_roll",
            "rest_days": "away_rest_calc",
        }
    )

    home_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_pf",
        "home_pa",
        "home_margin_roll",
        "home_rest_calc",
        "division",
    ]

    away_cols = [
        "game_id",
        "away_pf",
        "away_pa",
        "away_margin_roll",
        "away_rest_calc",
    ]

    home_cols = [c for c in home_cols if c in home.columns]
    away_cols = [c for c in away_cols if c in away.columns]

    train = home[home_cols].merge(away[away_cols], on="game_id", how="inner")

    required_drop = [
        "home_pf",
        "home_pa",
        "home_margin_roll",
        "away_pf",
        "away_pa",
        "away_margin_roll",
        "home_rest_calc",
        "away_rest_calc",
    ]

    required_drop = [c for c in required_drop if c in train.columns]

    if required_drop:
        train = train.dropna(subset=required_drop)

    if train.empty:
        return pd.DataFrame(columns=base_cols), feature_cols

    train["home_margin"] = train["home_score"] - train["away_score"]
    train["diff_pf"] = train["home_pf"] - train["away_pf"]
    train["diff_pa"] = train["home_pa"] - train["away_pa"]
    train["diff_margin"] = train["home_margin_roll"] - train["away_margin_roll"]
    train["diff_rest"] = train["home_rest_calc"] - train["away_rest_calc"]
    train["division"] = train["division"].apply(_to_bool_int)

    out_cols = [c for c in base_cols if c in train.columns]
    train = train[out_cols].copy()

    extra_candidates = [
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "total_line",
        "over_odds",
        "under_odds",
        "roof",
        "surface",
        "temp",
        "wind",
    ]

    extra = [c for c in extra_candidates if c in sched.columns]

    if extra:
        extras = sched[["game_id"] + extra].drop_duplicates("game_id")
        train = train.merge(extras, on="game_id", how="left")

    return train, feature_cols


def compute_snapshot(as_of_utc, seasons=None):
    if seasons is None:
        seasons = [SEASON]

    sched = _load_sched(seasons)

    if sched.empty:
        return {}

    completed = sched[sched["kickoff_utc"] < as_of_utc].copy()
    completed = completed.dropna(subset=["home_score", "away_score"])

    if completed.empty:
        return {}

    panel = _team_panel(completed)

    if panel.empty:
        return {}

    rolled = _add_rolling(panel, include_current=True)

    snap = {}

    for team, g in rolled.groupby("team"):
        g = g.sort_values("kickoff_utc")

        if g.empty:
            continue

        last = g.iloc[-1]

        if (
            pd.isna(last["roll_pts_for"])
            or pd.isna(last["roll_pts_against"])
            or pd.isna(last["roll_margin"])
        ):
            continue

        snap[str(team).upper()] = {
            "pf": float(last["roll_pts_for"]),
            "pa": float(last["roll_pts_against"]),
            "margin": float(last["roll_margin"]),
            "last_date": last["kickoff_utc"],
        }

    return snap


def cover_probs(mu_home_margin, sigma, home_spread):
    if sigma <= 0:
        sigma = 1.0

    try:
        from src.key_numbers import calculate_spread_probabilities
        res = calculate_spread_probabilities(mu_home_margin, sigma, home_spread)
        return float(res["cond_home_win"]), float(res["cond_away_win"])
    except Exception:
        z = (float(mu_home_margin) + float(home_spread)) / float(sigma)
        p_home = float(norm.cdf(z))
        return p_home, 1.0 - p_home
