"""P5 data: download, cache and ID-join the free nflverse assets (P5 spec "Data").

Assets (nflverse-data GitHub releases, cached under data/nflverse_cache/, gitignored):
    stats_player/stats_player_week_<S>.parquet   player_id (GSIS), team, game_id, receptions, ...
    snap_counts/snap_counts_<S>.parquet          pfr_player_id, game_id, team, offense_snaps, ...
    injuries/injuries_<S>.parquet                gsis_id, team, week, report_status
    players/players.parquet                      gsis_id, pfr_id, position_group

Snap counts use PFR IDs, the rest GSIS IDs: they are joined through players.parquet; a snap
row with no gsis_id match is dropped and counted. Team codes are mapped to the current ones
(OAK/SD/STL -> LV/LAC/LA), as the play-by-play uses them. game_id keeps nflverse's form.
The schedule is data/nflverse_cache/games.csv; PBP is data/epa_cache/pbp_<S>.parquet.
"""

import os
import urllib.request

import pandas as pd

from src.model_utils import LEGACY_TEAM_CODES

CACHE_DIR = "data/nflverse_cache"
GAMES_PATH = "data/nflverse_cache/games.csv"
PBP_DIR = "data/epa_cache"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
SEASON_ASSETS = {
    "stats": ("stats_player", "stats_player_week_{season}.parquet"),
    "snaps": ("snap_counts", "snap_counts_{season}.parquet"),
    "injuries": ("injuries", "injuries_{season}.parquet"),
}
PLAYERS_ASSET = ("players", "players.parquet")

STATS_COLUMNS = ["player_id", "position_group", "season", "week", "season_type", "game_id", "team",
                 "opponent_team", "attempts", "targets", "receptions", "receiving_air_yards"]
SNAP_COLUMNS = ["game_id", "season", "game_type", "week", "pfr_player_id", "team", "offense_snaps",
                "offense_pct"]
INJURY_COLUMNS = ["season", "game_type", "team", "week", "gsis_id", "report_status"]


def _download(tag, name, cache_dir=CACHE_DIR, refresh=False):
    path = os.path.join(cache_dir, name)
    if refresh or not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(f"{RELEASE}/{tag}/{name}", tmp)
        os.replace(tmp, path)
    return path


def canon_team(s):
    return s.astype(str).str.upper().str.strip().replace(LEGACY_TEAM_CODES)


def load_season(kind, season, cache_dir=CACHE_DIR, refresh=False):
    tag, name = SEASON_ASSETS[kind]
    df = pd.read_parquet(_download(tag, name.format(season=season), cache_dir, refresh))
    cols = {"stats": STATS_COLUMNS, "snaps": SNAP_COLUMNS, "injuries": INJURY_COLUMNS}[kind]
    return df[[c for c in cols if c in df.columns]].copy()


def load_players(cache_dir=CACHE_DIR, refresh=False):
    df = pd.read_parquet(_download(*PLAYERS_ASSET, cache_dir=cache_dir, refresh=refresh))
    return df[["gsis_id", "pfr_id", "position_group"]].copy()


def join_snaps(snaps, players):
    """(snaps with gsis_id and position_group, number of rows with no gsis_id match)."""
    ids = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    ids = ids[["pfr_id", "gsis_id", "position_group"]]
    out = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    matched = out["gsis_id"].notna()
    out = out[matched].drop(columns=["pfr_id"]).reset_index(drop=True)
    out["team"] = canon_team(out["team"])
    return out, int((~matched).sum())


def load_all(seasons, cache_dir=CACHE_DIR, refresh=False):
    """Dict of stats, snaps (joined), injuries, positions (gsis_id -> position_group) and
    coverage counts for the given seasons."""
    players = load_players(cache_dir, refresh)
    stats = pd.concat([load_season("stats", s, cache_dir, refresh) for s in seasons], ignore_index=True)
    snaps_raw = pd.concat([load_season("snaps", s, cache_dir, refresh) for s in seasons], ignore_index=True)
    inj = pd.concat([load_season("injuries", s, cache_dir, refresh) for s in seasons], ignore_index=True)
    reg = snaps_raw[snaps_raw["game_type"] == "REG"]
    snaps, unmatched = join_snaps(reg, players)
    stats["team"] = canon_team(stats["team"])
    inj["team"] = canon_team(inj["team"])
    positions = players.dropna(subset=["gsis_id"]).set_index("gsis_id")["position_group"].to_dict()
    coverage = {"snap_rows_reg": int(len(reg)), "snap_rows_unmatched": unmatched,
                "unmatched_by_season": reg.assign(
                    m=~reg["pfr_player_id"].isin(set(players["pfr_id"].dropna())))
                .groupby("season")["m"].sum().astype(int).to_dict()}
    return {"stats": stats, "snaps": snaps, "injuries": inj, "positions": positions, "coverage": coverage}
