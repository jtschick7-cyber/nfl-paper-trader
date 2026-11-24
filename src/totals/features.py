"""P3 totals features (spec: notes/implementation-plan/P3-totals-kill-test.md, frozen 2026-11-24).

Pipeline:
    prepare_games(schedule)            REG games, canonical team codes, kickoff_utc, actual_total
    team_game_stats(pbp, games)        one row per team per game (TEAM_STATS)
    rolling_team_features(team_games)  <stat>_r over the team's last 8 earlier games (min 3)
    game_frame(games, rolled)          home_/away_ rolled stats on each game row
    training_fills(...)                league means, league red-zone rate, wind median: training seasons only
    design_matrix(game_frame, fills)   the frozen FEATURES

Choices the spec leaves open (fixed here, before any outcome was computed):
- Offensive plays are play_type pass or run (as in epa_features), regular season only; the
  rolling windows use regular-season games only.
- Neutral pace: a gap between two consecutive offensive plays of the same posteam and drive
  counts when both plays are neutral (Q1-Q3, wp in [0.20, 0.80]); negative gaps (clock errors)
  are dropped.
- A red-zone trip is a drive with a pass, run, field-goal, kneel or spike snap at
  yardline_100 <= 20; it is a TD when a play of that drive has touchdown 1 and td_team = posteam.
- `cold` is 0 when the temperature is unknown. A missing roof counts as outdoor.
"""

import os

import numpy as np
import pandas as pd

from src.epa_features import parse_pbp_to_game_epa
from src.model_utils import _make_kickoff_utc, canonical_team_codes

WINDOW = 8
MIN_GAMES = 3
RZ_PSEUDO_TRIPS = 20
RZ_YARDLINE = 20
NEUTRAL_QTRS = (1, 2, 3)
NEUTRAL_WP = (0.20, 0.80)
HIGH_WIND_MPH = 15
COLD_F = 32
INDOOR_ROOFS = ("dome", "closed")

OFF_PLAY_TYPES = ("pass", "run")
RZ_SNAP_TYPES = ("pass", "run", "field_goal", "qb_kneel", "qb_spike")

MEAN_STATS = ["off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa",
              "neutral_sec_per_play", "neutral_pass_oe", "off_plays", "points_for", "points_against"]
SUM_STATS = ["rz_trips", "rz_tds"]
TEAM_STATS = MEAN_STATS + SUM_STATS

FEATURES = ["pass_env", "rush_env", "pace", "pass_oe_sum", "plays_sum", "rz_td_rate_sum",
            "points_env", "is_indoor", "wind", "wind_missing", "high_wind", "cold"]

PBP_COLUMNS = ["game_id", "play_id", "season", "season_type", "week", "posteam", "defteam", "drive",
               "qtr", "down", "game_seconds_remaining", "wp", "pass_oe", "epa", "pass", "rush",
               "qb_dropback", "qb_scramble", "play_type", "yardline_100", "touchdown", "td_team"]

GAME_COLUMNS = ["game_id", "season", "week", "kickoff_utc", "home_team", "away_team", "roof",
                "wind", "temp", "total_line", "over_odds", "under_odds", "actual_total"]


# --- inputs -------------------------------------------------------------------------

def prepare_games(schedule, first_season=None, last_season=None):
    """Played regular-season games with canonical team codes, kickoff_utc and actual_total."""
    g = schedule.copy()
    if "game_type" in g.columns:
        g = g[g["game_type"] == "REG"]
    if first_season is not None:
        g = g[g["season"] >= first_season]
    if last_season is not None:
        g = g[g["season"] <= last_season]
    g = g[g["home_score"].notna() & g["away_score"].notna()]
    g = canonical_team_codes(g)
    g = _make_kickoff_utc(g)
    g["actual_total"] = g["home_score"].astype(float) + g["away_score"].astype(float)
    for c in ("wind", "temp", "total_line", "over_odds", "under_odds"):
        g[c] = pd.to_numeric(g[c], errors="coerce") if c in g.columns else np.nan
    if "roof" not in g.columns:
        g["roof"] = np.nan
    keep = GAME_COLUMNS + ["home_score", "away_score"]
    return g[keep].sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)


def load_pbp(pbp_dir, seasons):
    frames = []
    for s in seasons:
        path = os.path.join(pbp_dir, f"pbp_{s}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        frames.append(df[[c for c in PBP_COLUMNS if c in df.columns]].copy())
    return pd.concat(frames, ignore_index=True)


# --- team-game stats ----------------------------------------------------------------

def _offensive(pbp):
    df = pbp
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    return df[df["play_type"].isin(OFF_PLAY_TYPES) & df["posteam"].notna()]


def neutral_pace(pbp):
    """(game_id, team, neutral_sec_per_play, neutral_pass_oe) for each team with offensive plays."""
    off = _offensive(pbp)
    off = off[off["drive"].notna()].sort_values(["game_id", "play_id"], kind="stable").copy()
    off["neutral"] = (off["qtr"].isin(NEUTRAL_QTRS)
                      & off["wp"].between(NEUTRAL_WP[0], NEUTRAL_WP[1])).astype(bool)
    grp = off.groupby(["game_id", "posteam", "drive"], sort=False)
    prev_gsr = grp["game_seconds_remaining"].shift(1)
    prev_neutral = grp["neutral"].shift(1)
    gap = prev_gsr - off["game_seconds_remaining"]
    ok = off["neutral"] & (prev_neutral == True) & gap.notna() & (gap >= 0)  # noqa: E712
    off["gap"] = gap.where(ok)
    off["pass_oe_neutral"] = off["pass_oe"].where(off["neutral"])
    out = off.groupby(["game_id", "posteam"]).agg(
        neutral_sec_per_play=("gap", "mean"), neutral_pass_oe=("pass_oe_neutral", "mean")).reset_index()
    return out.rename(columns={"posteam": "team"})


def red_zone(pbp):
    """(game_id, team, rz_trips, rz_tds)."""
    df = pbp
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["posteam"].notna() & df["drive"].notna()].copy()
    df["rz_snap"] = df["play_type"].isin(RZ_SNAP_TYPES) & (df["yardline_100"] <= RZ_YARDLINE)
    df["off_td"] = (df["touchdown"] == 1) & (df["td_team"] == df["posteam"])
    drives = df.groupby(["game_id", "posteam", "drive"]).agg(trip=("rz_snap", "any"), td=("off_td", "any"))
    drives["rz_td"] = drives["trip"] & drives["td"]
    out = drives.groupby(level=["game_id", "posteam"]).agg(rz_trips=("trip", "sum"), rz_tds=("rz_td", "sum"))
    out = out.reset_index().rename(columns={"posteam": "team"})
    out[["rz_trips", "rz_tds"]] = out[["rz_trips", "rz_tds"]].astype(float)
    return out


def team_game_stats(pbp, games):
    """One row per team per game in `games` (TEAM_STATS, NaN where the game has no PBP)."""
    home = games.assign(team=games["home_team"], points_for=games["home_score"], points_against=games["away_score"])
    away = games.assign(team=games["away_team"], points_for=games["away_score"], points_against=games["home_score"])
    base = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "kickoff_utc", "team", "points_for", "points_against"]]
    base[["points_for", "points_against"]] = base[["points_for", "points_against"]].astype(float)

    epa = parse_pbp_to_game_epa(pbp, play_split="dropback")
    if not epa.empty:
        epa = epa[["game_id", "team", "off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa"]]
    else:
        epa = pd.DataFrame(columns=["game_id", "team", "off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa"])
    off = _offensive(pbp)
    plays = off.groupby(["game_id", "posteam"]).size().rename("off_plays").reset_index().rename(columns={"posteam": "team"})
    plays["off_plays"] = plays["off_plays"].astype(float)

    out = base
    for part in (epa, neutral_pace(pbp), plays, red_zone(pbp)):
        out = out.merge(part, on=["game_id", "team"], how="left")
    has_pbp = out["off_plays"].notna()
    for c in SUM_STATS:
        out.loc[has_pbp, c] = out.loc[has_pbp, c].fillna(0.0)
    return out[["game_id", "season", "kickoff_utc", "team"] + TEAM_STATS]


# --- rolling, leak-free -------------------------------------------------------------

def rolling_team_features(team_games, window=WINDOW, min_games=MIN_GAMES):
    """<stat>_r per team-game from that team's previous `window` games (kickoff strictly before),
    carried across seasons; NaN with fewer than `min_games`. Counts (SUM_STATS) are summed."""
    tg = team_games.sort_values(["team", "kickoff_utc", "game_id"], kind="stable").copy()
    grp = tg.groupby("team", sort=False)
    tg["n_prior"] = grp.cumcount()
    for s in TEAM_STATS:
        prev = grp[s].shift(1)
        roll = prev.groupby(tg["team"]).rolling(window, min_periods=min_games)
        vals = roll.sum() if s in SUM_STATS else roll.mean()
        tg[f"{s}_r"] = vals.reset_index(level=0, drop=True)
    return tg


def game_frame(games, rolled):
    """Each game with home_<stat>_r, away_<stat>_r and home_/away_n_prior."""
    cols = ["n_prior"] + [f"{s}_r" for s in TEAM_STATS]
    out = games.copy()
    for side in ("home", "away"):
        part = rolled[["game_id", "team"] + cols].rename(
            columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in cols}})
        out = out.merge(part, on=["game_id", f"{side}_team"], how="left")
    return out


# --- fold-time fills and the design matrix ------------------------------------------

def rz_shrunk_rate(tds, trips, league_rate, k=RZ_PSEUDO_TRIPS):
    return (tds + k * league_rate) / (trips + k)


def _is_indoor(roof):
    return roof.astype(str).str.lower().str.strip().isin(INDOOR_ROOFS)


def training_wind_median(games, train_seasons):
    g = games[games["season"].isin(list(train_seasons))]
    w = pd.to_numeric(g.loc[~_is_indoor(g["roof"]), "wind"], errors="coerce").dropna()
    return float(w.median()) if len(w) else 0.0


def training_fills(team_games, games, train_seasons):
    """Everything a fold fills in, from its training seasons only."""
    seasons = sorted(int(s) for s in train_seasons)
    tg = team_games[team_games["season"].isin(seasons)]
    fills = {"seasons": seasons}
    for s in MEAN_STATS:
        fills[s] = float(tg[s].mean())
    trips = float(tg["rz_trips"].sum())
    fills["league_rz_rate"] = float(tg["rz_tds"].sum()) / trips if trips > 0 else 0.0
    fills["wind_median"] = training_wind_median(games, seasons)
    return fills


def weather_features(df, wind_median):
    indoor = _is_indoor(df["roof"])
    wind_raw = pd.to_numeric(df["wind"], errors="coerce")
    temp = pd.to_numeric(df["temp"], errors="coerce")
    missing = (~indoor) & wind_raw.isna()
    wind = wind_raw.where(~missing, wind_median).where(~indoor, 0.0).astype(float)
    return pd.DataFrame({
        "is_indoor": indoor.astype(int).values,
        "wind": wind.values,
        "wind_missing": missing.astype(int).values,
        "high_wind": ((~indoor) & (wind >= HIGH_WIND_MPH)).astype(int).values,
        "cold": ((~indoor) & (temp <= COLD_F)).astype(int).values,
    }, index=df.index)


def design_matrix(gf, fills):
    """The frozen FEATURES for each row of a game frame."""
    def side(prefix, stat):
        return gf[f"{prefix}_{stat}_r"].astype(float).fillna(fills[stat])

    def rz(prefix):
        trips = gf[f"{prefix}_rz_trips_r"].astype(float)
        tds = gf[f"{prefix}_rz_tds_r"].astype(float)
        rate = rz_shrunk_rate(tds, trips, fills["league_rz_rate"])
        return rate.where(trips.notna() & tds.notna(), fills["league_rz_rate"])

    X = pd.DataFrame(index=gf.index)
    X["pass_env"] = (side("home", "off_pass_epa") + side("away", "def_pass_epa")
                     + side("away", "off_pass_epa") + side("home", "def_pass_epa"))
    X["rush_env"] = (side("home", "off_rush_epa") + side("away", "def_rush_epa")
                     + side("away", "off_rush_epa") + side("home", "def_rush_epa"))
    X["pace"] = side("home", "neutral_sec_per_play") + side("away", "neutral_sec_per_play")
    X["pass_oe_sum"] = side("home", "neutral_pass_oe") + side("away", "neutral_pass_oe")
    X["plays_sum"] = side("home", "off_plays") + side("away", "off_plays")
    X["rz_td_rate_sum"] = rz("home") + rz("away")
    X["points_env"] = (side("home", "points_for") + side("away", "points_for")
                       + side("home", "points_against") + side("away", "points_against"))
    X = X.join(weather_features(gf, fills["wind_median"]))
    return X[FEATURES]


def build_game_frame(schedule, pbp_dir, first_season, last_season):
    """The whole pipeline from files: (game_frame, team_games)."""
    games = prepare_games(schedule, first_season, last_season)
    pbp = load_pbp(pbp_dir, range(first_season, last_season + 1))
    tg = team_game_stats(pbp, games)
    return game_frame(games, rolling_team_features(tg)), tg
