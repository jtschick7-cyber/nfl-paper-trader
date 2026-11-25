"""Grading of player_receptions offers against nflverse box scores (P4 spec).

Data: nflverse-data GitHub release assets, read with pandas.read_parquet and cached under
data/nflverse_cache/ (gitignored). Asset names confirmed 2026-11-24:

    https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.parquet
    https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2026.parquet

Used columns: stats `player_id`, `player_display_name`, `team`, `game_id`, `receptions`;
snap counts `pfr_player_id`, `player`, `team`, `game_id`, `offense_snaps`. Both use
`game_id` = "<season>_<week:02>_<AWAY>_<HOME>" with nflverse team codes (LA = Rams), which match
odds_utils.TEAM_ABBR_MAP. nfl_data_py is not used (and not upgraded).

Matching: normalize_name plus team plus game. Props carry no team, so a name is resolved among
the players of both teams in that game, in the stats rows and the snap-count rows. Postseason
rows (stats `season_type`, snaps `game_type` != "REG") are dropped first: the 2026 P2/P4 gates
are regular-season only (owner, 2026-11-25).
- No row of either kind for the game: `pending` (not graded, not listed; box score not out).
- Rows of two different games for one (season, away, home): `ambiguous_game`, listed and left
  ungraded; rows of two games are never summed.
- No player with that name: `unmatched`. Two distinct players with that name in one source, or
  the two sources disagree on the team: `ambiguous`. Both are listed and left ungraded.
- No snap-count row, or 0 offensive snaps: VOID, 0 units, excluded from W-L.
- Offensive snaps but no stats row for the game: nflverse player stats only hold players who
  recorded a stat, so this is usually a real 0 catches. It is graded as 0 (`graded_zero`) when
  the name matches exactly one stats-file player on the same team elsewhere in the season,
  which shows the two files agree on the name. Otherwise `no_stats_row`, listed and ungraded.
- Otherwise WIN / LOSS / PUSH at the offered line and price.
"""

import os
import urllib.request

import numpy as np
import pandas as pd

from src.props.prop_pricing import normalize_name

CACHE_DIR = "data/nflverse_cache"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
ASSETS = {
    "stats": ("stats_player", "stats_player_week_{season}.parquet"),
    "snaps": ("snap_counts", "snap_counts_{season}.parquet"),
}


def _fetch(kind, season, cache_dir=CACHE_DIR, refresh=True):
    tag, name = ASSETS[kind]
    name = name.format(season=season)
    path = os.path.join(cache_dir, name)
    if refresh or not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(f"{RELEASE}/{tag}/{name}", tmp)
            os.replace(tmp, path)
        except Exception as e:  # keep a cached copy when the download fails
            if os.path.exists(tmp):
                os.remove(tmp)
            if not os.path.exists(path):
                raise
            print(f"warning: {name} download failed ({e}); using the cached copy")
    return pd.read_parquet(path)


def load_player_data(season, cache_dir=CACHE_DIR, refresh=True):
    """(stats, snaps) DataFrames for one season, downloaded into the cache unless refresh=False."""
    return _fetch("stats", season, cache_dir, refresh), _fetch("snaps", season, cache_dir, refresh)


def grade_offer(side, line, odds, receptions):
    """(result, units) of a 1u ticket at the offered line and price."""
    diff = float(receptions) - float(line)
    diff = diff if side == "OVER" else -diff
    if diff > 0:
        return "WIN", (odds / 100.0 if odds > 0 else 100.0 / -odds)
    if diff < 0:
        return "LOSS", -1.0
    return "PUSH", 0.0


def _season(kickoff):
    t = pd.Timestamp(kickoff)
    return t.year if t.month >= 3 else t.year - 1


def _game_rows(df, season, away, home):
    if df is None or df.empty:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    gid = df["game_id"].astype(str)
    return df[gid.str.startswith(f"{season}_") & gid.str.endswith(f"_{away}_{home}")
              & df["team"].isin([home, away])]


def _regular_season(df, col):
    """Regular-season rows only, when the type column is present (POST rows never reach grading)."""
    if df is None or col not in df.columns:
        return df
    return df[df[col] == "REG"]


def _resolve(name, st, sn, season_stats):
    """(status, receptions or None) for one normalized name in one game."""
    if st.empty and sn.empty:
        return "pending", None
    if len(set(st["game_id"].astype(str)) | set(sn["game_id"].astype(str))) > 1:
        return "ambiguous_game", None   # two games for one (season, away, home): never summed
    s = st[st["_n"] == name]
    k = sn[sn["_n"] == name]
    if s["player_id"].nunique() > 1 or k["pfr_player_id"].nunique() > 1:
        return "ambiguous", None
    if s.empty and k.empty:
        return "unmatched", None
    if not s.empty and not k.empty and s["team"].iloc[0] != k["team"].iloc[0]:
        return "ambiguous", None
    snaps = float(k["offense_snaps"].fillna(0).sum()) if not k.empty else 0.0
    if snaps <= 0:
        return "void", None
    if s.empty:
        seen = season_stats[(season_stats["_n"] == name) & (season_stats["team"] == k["team"].iloc[0])]
        return ("graded_zero", 0.0) if seen["player_id"].nunique() == 1 else ("no_stats_row", None)
    return "graded", float(s["receptions"].fillna(0).sum())


def attach_results(df, stats, snaps):
    """(df with receptions, result, units, grade_status; unresolved list). `df` needs event_id,
    kickoff, home_team, away_team (nflverse codes), player (normalized), player_name, side, line, odds.
    `unresolved` lists each (event, player) left ungraded as unmatched / ambiguous / ambiguous_game /
    no_stats_row. Only regular-season rows are used when stats carry `season_type` and snaps `game_type`."""
    out = df.copy()
    if out.empty:
        return out.assign(receptions=np.nan, result=None, units=np.nan, grade_status=None), []
    stats = _regular_season(stats, "season_type")
    snaps = _regular_season(snaps, "game_type")
    stats = stats.assign(_n=[normalize_name(x) if isinstance(x, str) else None
                             for x in stats.get("player_display_name", pd.Series(dtype=object))])
    snaps = snaps.assign(_n=[normalize_name(x) if isinstance(x, str) else None
                             for x in snaps.get("player", pd.Series(dtype=object))])
    cache, unresolved = {}, []
    rec, res, units, status = [], [], [], []
    for r in out.itertuples():
        key = (r.event_id, r.player)
        if key not in cache:
            season = _season(r.kickoff)
            st = _game_rows(stats, season, r.away_team, r.home_team)
            sn = _game_rows(snaps, season, r.away_team, r.home_team)
            season_stats = stats[stats["game_id"].astype(str).str.startswith(f"{season}_")] if len(stats) else stats
            cache[key] = _resolve(r.player, st, sn, season_stats)
            if cache[key][0] in ("unmatched", "ambiguous", "ambiguous_game", "no_stats_row"):
                unresolved.append({"event_id": r.event_id, "game": f"{r.away_team} @ {r.home_team}",
                                   "player": r.player, "player_name": r.player_name, "reason": cache[key][0]})
        st_, n = cache[key]
        status.append(st_)
        if st_ in ("graded", "graded_zero"):
            x, u = grade_offer(r.side, r.line, r.odds, n)
            rec.append(n), res.append(x), units.append(u)
        elif st_ == "void":
            rec.append(np.nan), res.append("VOID"), units.append(0.0)
        else:
            rec.append(np.nan), res.append(None), units.append(np.nan)
    out["receptions"], out["result"], out["units"], out["grade_status"] = rec, res, units, status
    return out, unresolved
