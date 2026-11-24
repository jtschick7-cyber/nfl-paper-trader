"""P5a receptions model: universe and point-in-time features (P5 spec tests 1-4, 9).

Spec: notes/implementation-plan/P5-receptions-model.md (rules frozen 2026-11-24).
"""

import numpy as np
import pandas as pd
import pytest

from src.props import data_sources as D
from src.props import features as F


# --- synthetic league ---------------------------------------------------------------

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
ROSTER = {  # suffix -> position group
    "QB1": "QB", "QB2": "QB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "TE1": "TE", "RB1": "RB",
}
PAIRINGS = [(("AAA", "BBB"), ("CCC", "DDD")), (("CCC", "AAA"), ("DDD", "BBB")),
            (("AAA", "DDD"), ("BBB", "CCC"))]


def _league(seasons=(2020, 2021), weeks=10, seed=7):
    """(games, snaps, stats, injuries, pace, positions) for a 4-team league."""
    rng = np.random.default_rng(seed)
    games, snaps, stats, inj, pace = [], [], [], [], []
    positions = {f"{t}_{s}": p for t in TEAMS for s, p in ROSTER.items()}
    for season in seasons:
        start = pd.Timestamp(f"{season}-09-13 17:00", tz="UTC")
        for week in range(1, weeks + 1):
            for gi, (away, home) in enumerate(PAIRINGS[(week - 1) % 3]):
                gid = f"{season}_{week:02d}_{away}_{home}"
                kick = start + pd.Timedelta(days=7 * (week - 1), hours=3 * gi)
                games.append({"game_id": gid, "season": season, "week": week, "kickoff_utc": kick,
                              "home_team": home, "away_team": away, "roof": "outdoors",
                              "wind": float(rng.integers(0, 20)), "temp": 60.0,
                              "spread_line": float(rng.integers(-7, 8)), "total_line": 44.5})
                for team, opp in ((home, away), (away, home)):
                    pace.append({"game_id": gid, "team": team,
                                 "neutral_pass_oe": float(rng.normal(0, 5)),
                                 "neutral_sec_per_play": float(rng.normal(28, 2))})
                    for suffix, pos in ROSTER.items():
                        pid = f"{team}_{suffix}"
                        if suffix == "QB2":
                            continue  # the backup never plays
                        snaps.append({"game_id": gid, "season": season, "week": week, "game_type": "REG",
                                      "gsis_id": pid, "team": team, "position_group": pos,
                                      "offense_snaps": 60.0, "offense_pct": float(rng.uniform(0.3, 1.0))})
                        tg = int(rng.integers(0, 10)) if pos != "QB" else 0
                        stats.append({"player_id": pid, "game_id": gid, "season": season, "week": week,
                                      "season_type": "REG", "team": team, "opponent_team": opp,
                                      "position_group": pos, "targets": tg,
                                      "receptions": int(rng.integers(0, tg + 1)) if tg else 0,
                                      "receiving_air_yards": float(tg * rng.uniform(2, 12)),
                                      "attempts": int(rng.integers(25, 40)) if pos == "QB" else 0})
    return (pd.DataFrame(games), pd.DataFrame(snaps), pd.DataFrame(stats),
            pd.DataFrame(columns=["gsis_id", "season", "week", "team", "game_type", "report_status"]),
            pd.DataFrame(pace), positions)


def _build(league):
    games, snaps, stats, inj, pace, pos = league
    return F.build_player_frame(games, snaps, stats, inj, pace, pos)


def _row(frame, game_id, gsis_id):
    r = frame[(frame["game_id"] == game_id) & (frame["gsis_id"] == gsis_id)]
    assert len(r) == 1
    return r.iloc[0]


# --- 1. leakage ---------------------------------------------------------------------

def test_features_never_use_the_target_game_or_later():
    league = _league()
    games, snaps, stats, inj, pace, pos = league
    target = games[(games["season"] == 2021) & (games["week"] == 3)].iloc[0]
    t = target["kickoff_utc"]
    inj = pd.DataFrame([
        {"gsis_id": f"{target['home_team']}_WR2", "season": 2021, "week": 3, "team": target["home_team"],
         "game_type": "REG", "report_status": "Out"},
        {"gsis_id": f"{target['home_team']}_QB1", "season": 2021, "week": 3, "team": target["home_team"],
         "game_type": "REG", "report_status": "Doubtful"},
        {"gsis_id": f"{target['away_team']}_TE1", "season": 2021, "week": 3, "team": target["away_team"],
         "game_type": "REG", "report_status": "Questionable"},
    ])
    base = F.build_player_frame(games, snaps, stats, inj, pace, pos)

    rng = np.random.default_rng(99)
    late = set(games.loc[games["kickoff_utc"] >= t, "game_id"])
    after = set(games.loc[games["kickoff_utc"] > t, "game_id"])
    s2 = stats.copy()
    m = s2["game_id"].isin(late)
    s2.loc[m, "targets"] = rng.integers(0, 15, m.sum())
    s2.loc[m, "receptions"] = rng.integers(0, 15, m.sum())
    s2.loc[m, "receiving_air_yards"] = rng.uniform(0, 200, m.sum())
    s2.loc[m, "attempts"] = rng.integers(0, 60, m.sum())
    k2 = snaps.copy()
    m = k2["game_id"].isin(late)
    k2.loc[m, "offense_pct"] = rng.uniform(0.01, 1.0, m.sum())
    k2.loc[m, "offense_snaps"] = rng.integers(1, 80, m.sum()).astype(float)
    p2 = pace.copy()
    m = p2["game_id"].isin(late)
    p2.loc[m, "neutral_pass_oe"] = rng.normal(0, 50, m.sum())
    p2.loc[m, "neutral_sec_per_play"] = rng.normal(0, 50, m.sum())
    # later games and their reports disappear entirely
    g2 = games[~games["game_id"].isin(after)]
    s2, k2, p2 = (x[~x["game_id"].isin(after)] for x in (s2, k2, p2))
    inj2 = pd.concat([inj, pd.DataFrame([{"gsis_id": f"{target['home_team']}_WR1", "season": 2021, "week": 4,
                                          "team": target["home_team"], "game_type": "REG",
                                          "report_status": "Out"}])], ignore_index=True)
    altered = F.build_player_frame(g2, k2, s2, inj2, p2, pos)

    cols = F.RAW_FEATURES + ["n_prior"]
    keep = base["kickoff_utc"] <= t
    a = base[keep].set_index(["game_id", "gsis_id"]).sort_index()[cols]
    b = altered[altered["kickoff_utc"] <= t].set_index(["game_id", "gsis_id"]).sort_index()[cols]
    pd.testing.assert_frame_equal(a, b)
    # the target game's rows carry real (non-trivial) team, opponent and injury features
    g = base[(base["game_id"] == target["game_id"]) & base["position_group"].isin(F.POSITIONS)]
    assert g["team_neutral_pass_oe"].notna().all() and g["opp_rec_allowed_pos"].notna().all()
    assert (g["missing_same_pos_share"] > 0).any() and (g["qb_out"] == 1).any()


def test_design_matrix_is_point_in_time_given_fixed_priors():
    league = _league()
    frame = _build(league)
    uni, _ = F.apply_universe(frame)
    priors = F.training_priors(uni, league[0], [2020])
    X = F.design_matrix(uni, priors)
    assert list(X.columns) == F.FEATURES
    assert not X.isna().any().any()
    # a row's design row depends on its own raw features only
    X1 = F.design_matrix(uni.iloc[:5], priors)
    pd.testing.assert_frame_equal(X.iloc[:5], X1)


# --- 2. universe --------------------------------------------------------------------

def test_played_without_stats_row_is_zero_and_zero_snaps_is_out():
    games, snaps, stats, inj, pace, pos = _league()
    gid = games[(games["season"] == 2021) & (games["week"] == 5)]["game_id"].iloc[0]
    home = games.loc[games["game_id"] == gid, "home_team"].iloc[0]
    # WR1 played but recorded no stat: no stats row
    stats = stats[~((stats["game_id"] == gid) & (stats["player_id"] == f"{home}_WR1"))]
    # WR3 has a snap row with 0 offensive snaps, and also had 0 snaps in every earlier game
    snaps = snaps.copy()
    snaps.loc[snaps["gsis_id"] == f"{home}_WR3", "offense_snaps"] = 0.0
    frame = F.build_player_frame(games, snaps, stats, inj, pace, pos)
    r = _row(frame, gid, f"{home}_WR1")
    assert r["receptions"] == 0 and r["targets"] == 0 and r["offense_snaps"] > 0
    uni, counts = F.apply_universe(frame)
    assert not ((uni["gsis_id"] == f"{home}_WR3")).any()
    assert counts["no_offense_snaps"] >= 1
    # 0-snap games are not played games, so they never enter a later window
    assert frame.loc[frame["gsis_id"] == f"{home}_WR3", "n_prior"].fillna(0).max() == 0


def test_universe_rules_and_counts():
    frame = pd.DataFrame({
        "season": [2020] * 6, "game_id": list("abcdef"), "gsis_id": list("uvwxyz"),
        "position_group": ["QB", "WR", "TE", "RB", "WR", None],
        "offense_snaps": [50.0, 0.0, 30.0, 30.0, 40.0, 40.0],
        "n_prior": [8, 8, 2, 8, 8, 8], "rec_mean_8": [0.0, 5.0, 5.0, 1.49, 1.5, 4.0],
    })
    uni, counts = F.apply_universe(frame)
    assert list(uni["gsis_id"]) == ["y"]
    assert counts == {"snap_rows": 6, "not_wr_te_rb": 2, "no_offense_snaps": 1, "under_min_games": 1,
                      "under_min_rec_mean": 1, "universe": 1}


def test_rolling_role_features_by_hand():
    games, snaps, stats, inj, pace, pos = _league(seasons=(2020,), weeks=9)
    frame = F.build_player_frame(games, snaps, stats, inj, pace, pos)
    pid = "AAA_WR1"
    mine = frame[frame["gsis_id"] == pid].sort_values("kickoff_utc")
    last = mine.iloc[-1]
    prior = mine.iloc[:-1].tail(8)
    st = stats.set_index(["player_id", "game_id"])
    team_t = stats.groupby(["game_id", "team"])["targets"].sum()
    rec = [st.loc[(pid, g), "receptions"] for g in prior["game_id"]]
    tgt = [st.loc[(pid, g), "targets"] for g in prior["game_id"]]
    tt = [team_t.loc[(g, "AAA")] for g in prior["game_id"]]
    pct = list(prior["offense_pct"])
    assert last["n_prior"] == 8
    assert last["rec_mean_8"] == pytest.approx(np.mean(rec))
    assert last["target_share_8"] == pytest.approx(sum(tgt) / sum(tt))
    assert last["target_share_4"] == pytest.approx(sum(tgt[-4:]) / sum(tt[-4:]))
    assert last["snap_pct_8"] == pytest.approx(np.mean(pct))
    assert last["snap_pct_4"] == pytest.approx(np.mean(pct[-4:]))
    assert last["targets_per_snap_8"] == pytest.approx(sum(tgt) / (60.0 * 8))
    assert last["role_trend"] == pytest.approx(np.mean(pct[-2:]) - np.mean(pct[:6]))
    assert last["rec_sum_8"] == pytest.approx(sum(rec)) and last["tgt_sum_8"] == pytest.approx(sum(tgt))
    # fewer than MIN_GAMES earlier games: no rolling feature
    third = mine.iloc[2]
    assert third["n_prior"] == 2 and np.isnan(third["rec_mean_8"])


def test_game_script_and_depth_rank():
    games, snaps, stats, inj, pace, pos = _league(seasons=(2020,), weeks=9)
    frame = F.build_player_frame(games, snaps, stats, inj, pace, pos)
    g = games.iloc[-1]
    h = _row(frame, g["game_id"], f"{g['home_team']}_WR1")
    a = _row(frame, g["game_id"], f"{g['away_team']}_WR1")
    assert h["team_line"] == pytest.approx(-g["spread_line"]) and a["team_line"] == pytest.approx(g["spread_line"])
    assert h["game_total"] == 44.5
    assert h["team_implied_total"] == pytest.approx((44.5 + g["spread_line"]) / 2)
    wrs = frame[(frame["game_id"] == g["game_id"]) & (frame["team"] == g["home_team"])
                & (frame["position_group"] == "WR")].sort_values("snap_pct_8", ascending=False)
    assert list(wrs["depth_rank"]) == [1, 2, 3]


# --- 3. injuries --------------------------------------------------------------------

def _inj(rows):
    return pd.DataFrame([{"season": 2020, "game_type": "REG", **r} for r in rows])


def test_missing_share_sums_out_and_doubtful_teammates_only():
    games, snaps, stats, _, pace, pos = _league(seasons=(2020,), weeks=9)
    g = games[games["week"] == 9].iloc[0]
    team = g["home_team"]
    # WR2 Out (did not play), TE1 Doubtful (did not play), WR3 Questionable, WR1 Doubtful but played
    snaps = snaps[~((snaps["game_id"] == g["game_id"]) & snaps["gsis_id"].isin([f"{team}_WR2", f"{team}_TE1"]))]
    stats = stats[~((stats["game_id"] == g["game_id"]) & stats["player_id"].isin([f"{team}_WR2", f"{team}_TE1"]))]
    inj = _inj([{"gsis_id": f"{team}_WR2", "week": 9, "team": team, "report_status": "Out"},
                {"gsis_id": f"{team}_TE1", "week": 9, "team": team, "report_status": "Doubtful"},
                {"gsis_id": f"{team}_WR3", "week": 9, "team": team, "report_status": "Questionable"},
                {"gsis_id": f"{team}_WR1", "week": 9, "team": team, "report_status": "Doubtful"},
                {"gsis_id": f"{team}_RB1", "week": 9, "team": team, "report_status": None}])
    frame = F.build_player_frame(games, snaps, stats, inj, pace, pos)
    # a teammate's share as of kickoff: his share over his last 8 played games before week 9
    mine = frame[(frame["gsis_id"].str.startswith(team)) & (frame["week"] <= 8)]

    def share_asof(pid):
        rows = mine[mine["gsis_id"] == pid].sort_values("kickoff_utc").tail(8)
        st = stats.set_index(["player_id", "game_id"])["targets"]
        tt = stats.groupby(["game_id", "team"])["targets"].sum()
        return (sum(st.loc[(pid, x)] for x in rows["game_id"])
                / sum(tt.loc[(x, team)] for x in rows["game_id"]))

    wr2, te1, wr1 = share_asof(f"{team}_WR2"), share_asof(f"{team}_TE1"), share_asof(f"{team}_WR1")
    w1 = _row(frame, g["game_id"], f"{team}_WR1")
    w3 = _row(frame, g["game_id"], f"{team}_WR3")
    rb = _row(frame, g["game_id"], f"{team}_RB1")
    assert w1["missing_same_pos_share"] == pytest.approx(wr2)           # not himself, not WR3
    assert w3["missing_same_pos_share"] == pytest.approx(wr2 + wr1)     # WR1 Doubtful counts for teammates
    assert w1["missing_team_share"] == pytest.approx(wr2 + te1)
    assert rb["missing_same_pos_share"] == 0
    assert rb["missing_team_share"] == pytest.approx(wr2 + te1 + wr1)
    assert w3["self_questionable"] == 1 and w1["self_questionable"] == 0


def test_qb_out_uses_last_8_attempts_leader():
    games, snaps, stats, _, pace, pos = _league(seasons=(2020,), weeks=9)
    team = "AAA"
    stats = stats.copy()
    # QB1 throws a lot through week 4, then QB2 takes over (weeks 5-8 plus his own snaps)
    q1 = stats["player_id"] == f"{team}_QB1"
    stats.loc[q1 & (stats["week"] <= 4), "attempts"] = 40
    stats.loc[q1 & (stats["week"] > 4), "attempts"] = 0
    extra = stats[q1 & (stats["week"] > 4)].copy()
    extra["player_id"] = f"{team}_QB2"
    extra["attempts"] = 35
    stats = pd.concat([stats, extra], ignore_index=True)
    g9 = games[(games["week"] == 9) & ((games["home_team"] == team) | (games["away_team"] == team))].iloc[0]
    g5 = games[(games["week"] == 5) & ((games["home_team"] == team) | (games["away_team"] == team))].iloc[0]
    inj = _inj([{"gsis_id": f"{team}_QB1", "week": 9, "team": team, "report_status": "Out"},
                {"gsis_id": f"{team}_QB1", "week": 5, "team": team, "report_status": "Out"}])
    frame = F.build_player_frame(games, snaps, stats, inj, pace, pos)
    # last 8 team games before week 9 are weeks 1-8: QB1 4 x 40 = 160, QB2 4 x 35 = 140, so QB1 leads;
    # before week 5 only QB1 has thrown
    assert _row(frame, g9["game_id"], f"{team}_WR1")["qb_out"] == 1
    assert _row(frame, g5["game_id"], f"{team}_WR1")["qb_out"] == 1
    inj_b = _inj([{"gsis_id": f"{team}_QB2", "week": 9, "team": team, "report_status": "Out"}])
    frame_b = F.build_player_frame(games, snaps, stats, inj_b, pace, pos)
    assert _row(frame_b, g9["game_id"], f"{team}_WR1")["qb_out"] == 0
    # with more QB2 volume (4 x 45 = 180) he becomes the leader and his absence sets qb_out
    stats.loc[stats["player_id"] == f"{team}_QB2", "attempts"] = 45
    frame_c = F.build_player_frame(games, snaps, stats, inj_b, pace, pos)
    assert _row(frame_c, g9["game_id"], f"{team}_WR1")["qb_out"] == 1
    inj_q = _inj([{"gsis_id": f"{team}_QB2", "week": 9, "team": team, "report_status": "Questionable"}])
    frame_q = F.build_player_frame(games, snaps, stats, inj_q, pace, pos)
    assert _row(frame_q, g9["game_id"], f"{team}_WR1")["qb_out"] == 0


# --- 4. fills and priors from training seasons only ---------------------------------

def test_priors_and_fills_use_training_seasons_only():
    league = _league()
    games = league[0]
    uni, _ = F.apply_universe(_build(league))
    p1 = F.training_priors(uni, games, [2020])
    u2 = uni.copy()
    m = u2["season"] == 2021
    u2.loc[m, "rec_sum_8"] = u2.loc[m, "rec_sum_8"] * 3 + 7
    u2.loc[m, "tgt_sum_8"] = u2.loc[m, "tgt_sum_8"] + 50
    u2.loc[m, "air_sum_8"] = 1e5
    for c in F.RAW_FEATURES:
        if pd.api.types.is_numeric_dtype(u2[c]):
            u2.loc[m, c] = 999.0
    g2 = games.copy()
    g2.loc[g2["season"] == 2021, "wind"] = 99.0
    p2 = F.training_priors(u2, g2, [2020])
    assert p1 == p2
    # a training-season change does move them
    u3 = uni.copy()
    u3.loc[u3["season"] == 2020, "tgt_sum_8"] += 40
    assert F.training_priors(u3, games, [2020]) != p1


def test_eb_shrinkage_and_fills():
    df = pd.DataFrame({"position_group": ["WR", "TE"], "rec_sum_8": [10.0, 0.0], "tgt_sum_8": [20.0, 0.0],
                       "air_sum_8": [200.0, 0.0]})
    priors = {"catch_prior": {"WR": 0.6, "TE": 0.7, "RB": 0.8}, "adot_prior": {"WR": 11.0, "TE": 7.0, "RB": 1.0}}
    cr, ad = F.eb_rates(df, priors)
    assert cr.iloc[0] == pytest.approx((10 + 20 * 0.6) / (20 + 20))
    assert cr.iloc[1] == pytest.approx(0.7)
    assert ad.iloc[0] == pytest.approx((200 + 20 * 11.0) / (20 + 20))
    assert ad.iloc[1] == pytest.approx(7.0)


# --- data join ----------------------------------------------------------------------

def test_snap_join_drops_and_counts_unmatched():
    snaps = pd.DataFrame({"game_id": ["g1"] * 3, "pfr_player_id": ["p1", "p2", "zz"], "team": ["OAK", "SD", "KC"],
                          "offense_snaps": [10, 20, 30], "offense_pct": [0.1, 0.2, 0.3], "game_type": "REG",
                          "season": 2016, "week": 1})
    players = pd.DataFrame({"gsis_id": ["G1", "G2", "G3"], "pfr_id": ["p1", "p2", None],
                            "position_group": ["WR", "TE", "QB"]})
    out, unmatched = D.join_snaps(snaps, players)
    assert unmatched == 1
    assert list(out["gsis_id"]) == ["G1", "G2"]
    assert list(out["team"]) == ["LV", "LAC"]
    assert list(out["position_group"]) == ["WR", "TE"]


# --- 9. frozen constants ------------------------------------------------------------

def test_frozen_feature_constants():
    assert F.WINDOW == 8
    assert F.SHORT_WINDOW == 4
    assert F.MIN_GAMES == 3
    assert F.MIN_REC_MEAN == 1.5
    assert F.EB_K == 20
    assert F.HIGH_WIND_MPH == 15
    assert F.POSITIONS == ("WR", "TE", "RB")
    assert F.MISSING_STATUSES == ("Out", "Doubtful")
    assert F.FIRST_SEASON == 2016
