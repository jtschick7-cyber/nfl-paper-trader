"""P5 receptions universe and point-in-time features (spec: notes/implementation-plan/P5-receptions-model.md,
rules frozen 2026-11-24). Pure functions on frames.

Pipeline:
    build_player_frame(games, snaps, stats, injuries, pace, positions)
        one row per regular-season snap-count row, with the raw features (RAW_FEATURES) of the
        WR / TE / RB rows; every window uses only games with kickoff_utc before the row's.
    apply_universe(frame)               the frozen universe, with exclusion counts per rule
    training_priors(universe, games, train_seasons)
                                        EB priors, wind median and fills from training seasons only
    design_matrix(df, priors)           the FEATURES the model sees

Inputs: `games` has game_id, season, week, kickoff_utc, home_team, away_team, roof, wind, temp,
spread_line, total_line (REG games, current team codes); `snaps` is data_sources.join_snaps output
(game_id, gsis_id, team, offense_snaps, offense_pct, position_group); `stats` is stats_player_week
(player_id, game_id, team, receptions, targets, receiving_air_yards, attempts); `injuries` has
gsis_id, season, week, team, report_status; `pace` is src.totals.features.neutral_pace output
(game_id, team, neutral_pass_oe, neutral_sec_per_play); `positions` maps gsis_id -> position_group.

Choices the spec leaves open (fixed here, before any outcome was computed):
- A "played game" is a regular-season snap row with offense_snaps > 0. A player's windows are
  his last 8 (or 4) played games, any team, across seasons; the state as of a kickoff is that of
  his last played game before it. Team targets in a window are his team's targets in those games.
- role_trend: over the 8-game window, mean offense_pct of the last 2 played games minus the
  mean of the 6 before them.
- depth_rank: rank of snap_pct_8 among the team's same-position players as of kickoff. Those are
  players whose last played game before kickoff was for this team, within its last 8 games
  (realised participation in the target game is not used). Ties share the better rank; capped at 3.
- Injured teammates: listed Out or Doubtful on the team's report for that season and week, in
  the position group from players.parquet; each contributes his target_share_8 as of kickoff (0
  when he has fewer than 3 played games).
- qb_out: the player (position group QB) with the most pass attempts in the team's last 8 games
  before kickoff (ties: lower player_id), listed Out or Doubtful that week. 0 with no attempts.
- Opponent features are the opponent's means over its last 8 games (min 3): receptions by its
  opponents' players of the position group, and its opponents' neutral pass_oe.
- EB priors and fills come from the training seasons' universe rows: catch prior = receptions /
  targets, aDOT prior = air yards / targets, per position group; a fill is the feature's mean.
"""

import numpy as np
import pandas as pd

from src.totals.features import HIGH_WIND_MPH, training_wind_median, weather_features

WINDOW = 8
SHORT_WINDOW = 4
MIN_GAMES = 3
MIN_REC_MEAN = 1.5
EB_K = 20
POSITIONS = ("WR", "TE", "RB")
MISSING_STATUSES = ("Out", "Doubtful")
QUESTIONABLE = "Questionable"
FIRST_SEASON = 2016
DEPTH_CAP = 3
TREND_RECENT = 2

ROLE_FEATURES = ["target_share_8", "target_share_4", "snap_pct_8", "snap_pct_4", "targets_per_snap_8",
                 "role_trend", "rec_mean_8"]
EB_SUMS = ["rec_sum_8", "tgt_sum_8", "air_sum_8"]
RAW_FEATURES = (["team_line", "game_total", "team_implied_total"] + ROLE_FEATURES + EB_SUMS
                + ["depth_rank", "missing_same_pos_share", "missing_team_share", "qb_out", "self_questionable",
                   "team_neutral_pass_oe", "team_neutral_sec_per_play", "opp_rec_allowed_pos", "opp_pass_oe_faced"])

FEATURES = ["team_line", "game_total", "team_implied_total", "is_indoor", "wind_mph", "is_high_wind",
            "target_share_8", "target_share_4", "snap_pct_8", "snap_pct_4", "targets_per_snap_8", "role_trend",
            "catch_rate_eb", "adot_eb", "rec_mean_8", "depth_rank", "pos_WR", "pos_TE", "pos_RB",
            "missing_same_pos_share", "missing_team_share", "qb_out", "self_questionable",
            "team_neutral_pass_oe", "team_neutral_sec_per_play", "opp_rec_allowed_pos", "opp_pass_oe_faced"]

_TEAM_ROLL = ["npoe", "nsec", "poe_faced"] + [f"rec_allowed_{p}" for p in POSITIONS]


# --- helpers ------------------------------------------------------------------------

def _utc(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def _pos_map(df, id_col, positions, fallback=None):
    p = df[id_col].map(positions) if positions else pd.Series(np.nan, index=df.index)
    if fallback is not None and fallback in df.columns:
        p = p.fillna(df[fallback])
    return p


def team_games(games):
    """Two rows per game (team, opp, is_home) with k = the team's game index by kickoff."""
    g = games.astype({"season": int, "week": int})
    g["kickoff_utc"] = _utc(g["kickoff_utc"])
    base = ["game_id", "season", "week", "kickoff_utc"]
    home = g[base].assign(team=g["home_team"].values, opp=g["away_team"].values, is_home=True)
    away = g[base].assign(team=g["away_team"].values, opp=g["home_team"].values, is_home=False)
    tg = pd.concat([home, away], ignore_index=True).sort_values(["team", "kickoff_utc", "game_id"], kind="stable")
    tg["k"] = tg.groupby("team").cumcount()
    return tg.reset_index(drop=True)


def _window_pairs(tg, window=WINDOW):
    """(team, game_id, kickoff_utc, prior_game_id) for each team game's last `window` earlier games."""
    a = tg[["team", "game_id", "kickoff_utc", "k"]]
    b = tg[["team", "game_id", "k"]].rename(columns={"game_id": "prior_game_id", "k": "kp"})
    m = a.merge(b, on="team")
    m = m[(m["kp"] < m["k"]) & (m["kp"] >= m["k"] - window)]
    return m[["team", "game_id", "kickoff_utc", "prior_game_id"]].reset_index(drop=True)


# --- player windows -----------------------------------------------------------------

def _post_states(played):
    """Rolling state after each played game (that game included), per player."""
    p = played.sort_values(["gsis_id", "kickoff_utc"], kind="stable").copy()
    g = p.groupby("gsis_id", sort=False)

    def roll(col, w, how="sum"):
        r = g[col].rolling(w, min_periods=1)
        r = r.sum() if how == "sum" else r.mean()
        return r.reset_index(level=0, drop=True)

    p["cnt8"] = g.cumcount().add(1).clip(upper=WINDOW)
    p["cnt4"] = p["cnt8"].clip(upper=SHORT_WINDOW)
    for w in (WINDOW, SHORT_WINDOW):
        for c in ("receptions", "targets", "team_targets", "offense_pct", "offense_snaps", "air_yards"):
            p[f"{c}_{w}"] = roll(c, w)
    recent = roll("offense_pct", TREND_RECENT, "mean")
    p["_pct_lag"] = g["offense_pct"].shift(TREND_RECENT)
    older = (p.groupby("gsis_id", sort=False)["_pct_lag"].rolling(WINDOW - TREND_RECENT, min_periods=1)
             .mean().reset_index(level=0, drop=True))
    ok = p["cnt8"] >= MIN_GAMES
    out = pd.DataFrame({"gsis_id": p["gsis_id"], "team": p["team"], "kickoff_utc": p["kickoff_utc"],
                        "position_group": p["position_group"], "n_prior": p["cnt8"]}, index=p.index)

    def ratio(a, b):
        return (a / b.where(b > 0)).where(ok)

    out["rec_mean_8"] = ratio(p["receptions_8"], p["cnt8"])
    out["target_share_8"] = ratio(p["targets_8"], p["team_targets_8"])
    out["target_share_4"] = ratio(p["targets_4"], p["team_targets_4"])
    out["snap_pct_8"] = ratio(p["offense_pct_8"], p["cnt8"])
    out["snap_pct_4"] = ratio(p["offense_pct_4"], p["cnt4"])
    out["targets_per_snap_8"] = ratio(p["targets_8"], p["offense_snaps_8"])
    out["role_trend"] = (recent - older).where(ok)
    out["rec_sum_8"] = p["receptions_8"].where(ok)
    out["tgt_sum_8"] = p["targets_8"].where(ok)
    out["air_sum_8"] = p["air_yards_8"].where(ok)
    out["next_kickoff"] = g["kickoff_utc"].shift(-1)
    out["game_id"] = p["game_id"]
    return out.reset_index(drop=True)


def _asof(keys, post, cols):
    """For each (gsis_id, kickoff_utc) in keys: `cols` of the player's last post state strictly before."""
    k = keys[["gsis_id", "kickoff_utc"]].copy()
    k["_i"] = np.arange(len(k))
    k = k.sort_values("kickoff_utc", kind="stable")
    right = post[["gsis_id", "kickoff_utc"] + cols].sort_values("kickoff_utc", kind="stable")
    m = pd.merge_asof(k, right, on="kickoff_utc", by="gsis_id", allow_exact_matches=False, direction="backward")
    return m.sort_values("_i")[cols].reset_index(drop=True)


# --- the frame ----------------------------------------------------------------------

def build_player_frame(games, snaps, stats, injuries, pace, positions):
    tg = team_games(games)
    gm = games.copy()
    gm["kickoff_utc"] = _utc(gm["kickoff_utc"])

    st = stats.copy()
    if "season_type" in st.columns:
        st = st[st["season_type"] == "REG"]
    st = st[st["game_id"].isin(set(gm["game_id"]))]
    st["pos"] = _pos_map(st, "player_id", positions, "position_group")
    for c in ("receptions", "targets", "receiving_air_yards", "attempts"):
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0) if c in st.columns else 0.0
    team_tgt = st.groupby(["game_id", "team"])["targets"].sum().rename("team_targets").reset_index()

    sn = snaps.copy()
    if "game_type" in sn.columns:
        sn = sn[sn["game_type"] == "REG"]
    sn = sn.drop(columns=[c for c in ("season", "week", "kickoff_utc") if c in sn.columns])
    sn = sn.merge(tg[["game_id", "team", "season", "week", "kickoff_utc", "opp", "is_home"]],
                  on=["game_id", "team"], how="inner")
    if "position_group" not in sn.columns:
        sn["position_group"] = np.nan
    sn["position_group"] = sn["position_group"].fillna(sn["gsis_id"].map(positions) if positions else np.nan)
    sn["offense_snaps"] = pd.to_numeric(sn["offense_snaps"], errors="coerce").fillna(0.0)
    sn["offense_pct"] = pd.to_numeric(sn["offense_pct"], errors="coerce")
    own = st[["player_id", "game_id", "receptions", "targets", "receiving_air_yards"]].rename(
        columns={"player_id": "gsis_id", "receiving_air_yards": "air_yards"})
    own = own.groupby(["gsis_id", "game_id"], as_index=False).sum()
    sn = sn.merge(own, on=["gsis_id", "game_id"], how="left")
    for c in ("receptions", "targets", "air_yards"):
        sn[c] = sn[c].fillna(0.0)  # played with no stats row: 0 targets, 0 receptions
    sn = sn.merge(team_tgt, on=["game_id", "team"], how="left")
    sn["team_targets"] = sn["team_targets"].fillna(0.0)
    sn = sn.reset_index(drop=True)

    is_pos = sn["position_group"].isin(POSITIONS)
    played = sn[is_pos & (sn["offense_snaps"] > 0)]
    post = _post_states(played)
    pre_cols = ["n_prior"] + ROLE_FEATURES + EB_SUMS
    pre = _asof(sn, post, pre_cols)
    pre.index = sn.index
    pre["n_prior"] = pre["n_prior"].fillna(0).astype(int)
    pre.loc[~is_pos, pre_cols] = np.nan
    out = pd.concat([sn, pre], axis=1)

    # game script
    lines = gm.set_index("game_id")[["spread_line", "total_line", "roof", "wind", "temp"]]
    out = out.join(lines, on="game_id")
    spread = pd.to_numeric(out["spread_line"], errors="coerce")
    out["team_line"] = np.where(out["is_home"], -spread, spread)
    out["game_total"] = pd.to_numeric(out["total_line"], errors="coerce")
    out["team_implied_total"] = (out["game_total"] - out["team_line"]) / 2

    out["depth_rank"] = _depth_rank(out, post, tg, is_pos)
    out = _injury_features(out, post, tg, st, injuries, positions, is_pos)
    out = _team_opp_features(out, tg, st, pace)
    return out


def _depth_rank(out, post, tg, is_pos):
    pairs = _window_pairs(tg)
    cand = pairs.merge(post[["game_id", "team", "gsis_id", "kickoff_utc", "next_kickoff", "position_group",
                             "snap_pct_8"]].rename(columns={"game_id": "prior_game_id", "kickoff_utc": "k_prior"}),
                       on=["team", "prior_game_id"])
    cand = cand[cand["next_kickoff"].isna() | (cand["next_kickoff"] >= cand["kickoff_utc"])]
    cand = cand[["team", "game_id", "position_group", "gsis_id", "snap_pct_8"]].drop_duplicates(["team", "game_id",
                                                                                                "gsis_id"])
    tgt = out.loc[is_pos, ["game_id", "team", "position_group", "gsis_id", "snap_pct_8"]].copy()
    tgt["_i"] = tgt.index
    m = tgt.merge(cand.rename(columns={"gsis_id": "c_id", "snap_pct_8": "c_pct"}),
                  on=["game_id", "team", "position_group"], how="left")
    m["ahead"] = ((m["c_id"] != m["gsis_id"]) & (m["c_pct"] > m["snap_pct_8"])).astype(int)
    ahead = m.groupby("_i")["ahead"].sum()
    rank = (1 + ahead).clip(upper=DEPTH_CAP).astype(float)
    res = pd.Series(np.nan, index=out.index)
    res.loc[rank.index] = rank.values
    return res.where(out["snap_pct_8"].notna())


def _injury_features(out, post, tg, st, injuries, positions, is_pos):
    inj = injuries.copy() if injuries is not None else pd.DataFrame()
    for c in ("gsis_id", "season", "week", "team", "report_status"):
        if c not in inj.columns:
            inj[c] = pd.Series(dtype=object)
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"].isna() | (inj["game_type"] == "REG")]
    inj = inj.dropna(subset=["season", "week", "team", "gsis_id"]).astype({"season": int, "week": int})
    inj = inj.drop_duplicates(["gsis_id", "season", "week", "team"])
    inj = inj.merge(tg[["season", "week", "team", "game_id", "kickoff_utc"]], on=["season", "week", "team"],
                    how="inner")
    inj["pos"] = inj["gsis_id"].map(positions) if positions else np.nan
    miss = inj[inj["report_status"].isin(MISSING_STATUSES)].reset_index(drop=True)
    share = _asof(miss, post, ["target_share_8"])["target_share_8"].fillna(0.0) if len(miss) else pd.Series(dtype=float)
    miss["share"] = share.values if len(miss) else []
    mp = miss[miss["pos"].isin(POSITIONS)]
    pos_sum = mp.groupby(["game_id", "team", "pos"])["share"].sum().rename("_pos_sum")
    team_sum = mp.groupby(["game_id", "team"])["share"].sum().rename("_team_sum")
    self_share = mp.set_index(["game_id", "team", "gsis_id"])["share"].rename("_self")

    o = out.join(pos_sum, on=["game_id", "team", "position_group"]).join(team_sum, on=["game_id", "team"])
    o = o.join(self_share, on=["game_id", "team", "gsis_id"])
    self_ = o["_self"].fillna(0.0)
    same = (o["_pos_sum"].fillna(0.0) - self_).clip(lower=0.0)
    team = (o["_team_sum"].fillna(0.0) - self_).clip(lower=0.0)
    out["missing_same_pos_share"] = same.where(is_pos)
    out["missing_team_share"] = team.where(is_pos)

    q = inj[inj["report_status"] == QUESTIONABLE].set_index(["game_id", "team", "gsis_id"]).index
    key = pd.MultiIndex.from_frame(out[["game_id", "team", "gsis_id"]])
    out["self_questionable"] = pd.Series(key.isin(q).astype(float), index=out.index).where(is_pos)

    # qb_out: last-8 attempts leader listed Out or Doubtful
    pairs = _window_pairs(tg)
    qbs = st[st["pos"] == "QB"][["game_id", "team", "player_id", "attempts"]]
    att = pairs.merge(qbs.rename(columns={"game_id": "prior_game_id"}), on=["team", "prior_game_id"])
    att = att.groupby(["team", "game_id", "player_id"], as_index=False)["attempts"].sum()
    att = att[att["attempts"] > 0].sort_values(["team", "game_id", "attempts", "player_id"],
                                               ascending=[True, True, False, True], kind="stable")
    att = att.drop_duplicates(["team", "game_id"])
    leader = dict(zip(zip(att["game_id"], att["team"]), att["player_id"]))
    lead = [leader.get(k) for k in zip(out["game_id"], out["team"])]
    listed = set(zip(miss["game_id"], miss["team"], miss["gsis_id"]))
    qb_out = [1.0 if isinstance(l, str) and (g, t, l) in listed else 0.0
              for g, t, l in zip(out["game_id"], out["team"], lead)]
    out["qb_out"] = pd.Series(qb_out, index=out.index).where(is_pos)
    return out


def _team_opp_features(out, tg, st, pace):
    t = tg.copy()
    pc = pace[["game_id", "team", "neutral_pass_oe", "neutral_sec_per_play"]] if len(pace) else \
        pd.DataFrame(columns=["game_id", "team", "neutral_pass_oe", "neutral_sec_per_play"])
    t = t.merge(pc.rename(columns={"neutral_pass_oe": "npoe", "neutral_sec_per_play": "nsec"}),
                on=["game_id", "team"], how="left")
    t = t.merge(pc[["game_id", "team", "neutral_pass_oe"]].rename(columns={"team": "opp",
                                                                           "neutral_pass_oe": "poe_faced"}),
                on=["game_id", "opp"], how="left")
    rec = st[st["pos"].isin(POSITIONS)].groupby(["game_id", "team", "pos"])["receptions"].sum().unstack("pos")
    rec = rec.reindex(columns=list(POSITIONS))
    have = set(st["game_id"])
    for p in POSITIONS:
        s = rec[p].rename(f"rec_allowed_{p}") if len(rec) else pd.Series(dtype=float, name=f"rec_allowed_{p}")
        t = t.join(s, on=["game_id", "opp"])
        t[f"rec_allowed_{p}"] = t[f"rec_allowed_{p}"].where(~t["game_id"].isin(have), t[f"rec_allowed_{p}"].fillna(0.0))
    t = t.sort_values(["team", "kickoff_utc", "game_id"], kind="stable")
    g = t.groupby("team", sort=False)
    for c in _TEAM_ROLL:
        t[f"{c}_r"] = (g[c].shift(1).groupby(t["team"]).rolling(WINDOW, min_periods=MIN_GAMES).mean()
                       .reset_index(level=0, drop=True))
    roll = t.set_index(["game_id", "team"])
    is_pos = out["position_group"].isin(POSITIONS)
    own = roll[["npoe_r", "nsec_r"]]
    o = out.join(own, on=["game_id", "team"])
    out["team_neutral_pass_oe"] = o["npoe_r"].where(is_pos)
    out["team_neutral_sec_per_play"] = o["nsec_r"].where(is_pos)
    opp = roll[["poe_faced_r"] + [f"rec_allowed_{p}_r" for p in POSITIONS]]
    oo = out.join(opp, on=["game_id", "opp"])
    allowed = pd.Series(np.nan, index=out.index)
    for p in POSITIONS:
        m = out["position_group"] == p
        allowed[m] = oo.loc[m, f"rec_allowed_{p}_r"]
    out["opp_rec_allowed_pos"] = allowed.where(is_pos)
    out["opp_pass_oe_faced"] = oo["poe_faced_r"].where(is_pos)
    return out


# --- universe -----------------------------------------------------------------------

def apply_universe(frame):
    """(universe rows, counts of rows excluded by each frozen rule, applied in order)."""
    counts = {"snap_rows": int(len(frame))}
    f = frame
    keep = f["position_group"].isin(POSITIONS)
    counts["not_wr_te_rb"] = int((~keep).sum())
    f = f[keep]
    keep = f["offense_snaps"] > 0
    counts["no_offense_snaps"] = int((~keep).sum())
    f = f[keep]
    keep = f["n_prior"].fillna(0) >= MIN_GAMES
    counts["under_min_games"] = int((~keep).sum())
    f = f[keep]
    keep = f["rec_mean_8"] >= MIN_REC_MEAN
    counts["under_min_rec_mean"] = int((~keep).sum())
    f = f[keep]
    counts["universe"] = int(len(f))
    return f.copy(), counts


# --- priors, fills, design ----------------------------------------------------------

def eb_rates(df, priors):
    """(catch_rate_eb, adot_eb) shrunk with EB_K pseudo-targets to the position priors."""
    cp = df["position_group"].map(priors["catch_prior"]).astype(float)
    ap = df["position_group"].map(priors["adot_prior"]).astype(float)
    tg = df["tgt_sum_8"].astype(float)
    cr = (df["rec_sum_8"].astype(float) + EB_K * cp) / (tg + EB_K)
    ad = (df["air_sum_8"].astype(float) + EB_K * ap) / (tg + EB_K)
    return cr, ad


def _raw_design(df, priors):
    X = pd.DataFrame(index=df.index)
    for c in ("team_line", "game_total", "team_implied_total"):
        X[c] = df[c].astype(float)
    w = weather_features(df, priors["wind_median"])
    X["is_indoor"] = w["is_indoor"].astype(float)
    X["wind_mph"] = w["wind"].astype(float)
    X["is_high_wind"] = w["high_wind"].astype(float)
    for c in ROLE_FEATURES[:-1]:
        X[c] = df[c].astype(float)
    X["catch_rate_eb"], X["adot_eb"] = eb_rates(df, priors)
    X["rec_mean_8"] = df["rec_mean_8"].astype(float)
    X["depth_rank"] = df["depth_rank"].astype(float)
    for p in POSITIONS:
        X[f"pos_{p}"] = (df["position_group"] == p).astype(float)
    for c in ("missing_same_pos_share", "missing_team_share", "qb_out", "self_questionable",
              "team_neutral_pass_oe", "team_neutral_sec_per_play", "opp_rec_allowed_pos", "opp_pass_oe_faced"):
        X[c] = df[c].astype(float)
    return X[FEATURES]


def training_priors(universe, games, train_seasons):
    seasons = sorted(int(s) for s in train_seasons)
    tr = universe[universe["season"].isin(seasons)]
    pri = {"seasons": seasons, "catch_prior": {}, "adot_prior": {}}
    for p in POSITIONS:
        g = tr[tr["position_group"] == p]
        t = float(g["targets"].sum())
        pri["catch_prior"][p] = float(g["receptions"].sum()) / t if t > 0 else 0.0
        pri["adot_prior"][p] = float(g["air_yards"].sum()) / t if t > 0 else 0.0
    pri["wind_median"] = training_wind_median(games, seasons)
    raw = _raw_design(tr, pri)
    pri["fill"] = {c: (float(v) if pd.notna(v) else 0.0) for c, v in raw.mean().items()}
    return pri


def design_matrix(df, priors):
    X = _raw_design(df, priors)
    return X.fillna(priors["fill"])[FEATURES]
