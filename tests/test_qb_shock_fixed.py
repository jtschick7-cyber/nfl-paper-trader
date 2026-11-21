"""Audit P0-7: the opt-in fixed v4 QB shock, starter overrides and team codes."""

import os

import numpy as np
import pandas as pd
import pytest

from src.qb_features import (
    BACKUP_QB_PRIOR, DEFAULT_LEAGUE_QB_PRIOR, DROPBACKS_PER_GAME, EPA_TO_SPREAD_POINTS, REPLACEMENT_BACKUP,
    fetch_pregame_starter_overrides, fixed_qb_rating, shrink_qb_rating,
)
from src.train_model_v4 import (
    V4_BOUNDS_LB, V4_BOUNDS_UB, V4_FEATURES, build_v4_training_dataset, compute_point_in_time_qb_shock,
    v4_feature_bounds, v4_feature_list,
)

PTS = DROPBACKS_PER_GAME * EPA_TO_SPREAD_POINTS


def _row(gid, season, week, team, qid, name, db, epa_db, share=1.0):
    return {"game_id": gid, "season": season, "week": week, "posteam": team, "qb_id": qid, "qb_name": name,
            "dropbacks": db, "total_epa": db * epa_db, "dropback_share": share}


def _healthy(team, qid, epa_db, n=6, season=2024):
    return [_row(f"{season}_{w:02d}_{team}", season, w, team, qid, qid, 35, epa_db) for w in range(1, n + 1)]


def _shock(qb, home, away, overrides=None, version="fixed", h_pass=0.0, a_pass=0.0):
    return compute_point_in_time_qb_shock(qb, list(qb["game_id"].unique()), home, away, h_pass, a_pass,
                                          starter_overrides=overrides, window=8, version=version)


def test_healthy_starters_have_zero_fixed_shock_but_legacy_is_biased():
    # Elite home QB (+0.30) and bad away QB (-0.20), each threw every dropback of the last 6 games
    qb = pd.DataFrame(_healthy("KC", "elite", 0.30) + _healthy("TEN", "bad", -0.20))
    assert _shock(qb, "KC", "TEN") == 0.0
    # Legacy compares the k=60-shrunk QB with the unshrunk team pass EPA: the elite QB looks
    # worse than his own team, the bad one better (audit P0-7 item 3)
    legacy = _shock(qb, "KC", "TEN", version="legacy", h_pass=0.30, a_pass=-0.20)
    assert legacy < -1.0


def test_backup_with_no_rows_gets_the_backup_prior_not_zero():
    qb = pd.DataFrame(_healthy("KC", "elite", 0.30) + _healthy("TEN", "avg", 0.05))
    starter = fixed_qb_rating(qb, ("id", "elite"))
    got = _shock(qb, "KC", "TEN", overrides={"KC": "never_played"})
    assert got == pytest.approx(round((BACKUP_QB_PRIOR - starter) * PTS, 4))
    assert got < -3.0
    assert _shock(qb, "KC", "TEN", overrides={"KC": REPLACEMENT_BACKUP}) == got
    # Legacy returns 0 for the same game (item 2)
    assert _shock(qb, "KC", "TEN", overrides={"KC": "never_played"}, version="legacy", h_pass=0.3, a_pass=0.05) == \
        pytest.approx(-round((shrink_qb_rating(0.05, 210, k=60.0) - 0.05) * PTS, 4))


def test_a_backup_with_few_dropbacks_shrinks_toward_the_backup_prior():
    rows = _healthy("KC", "elite", 0.30) + [_row("2023_17_KC", 2023, 17, "KC", "backup", "backup", 10, 0.30, 0.2)]
    qb = pd.DataFrame(rows)
    got = fixed_qb_rating(qb, ("id", "backup"))
    assert got == pytest.approx(shrink_qb_rating(0.30, 10, prior_epa=BACKUP_QB_PRIOR, k=60.0))
    assert got < shrink_qb_rating(0.30, 10, prior_epa=DEFAULT_LEAGUE_QB_PRIOR, k=60.0)
    # A QB who has been a team's main passer keeps the league prior
    assert fixed_qb_rating(qb, ("id", "elite")) == pytest.approx(shrink_qb_rating(0.30, 210, k=60.0))


def test_returning_starter_shock_is_the_share_of_games_he_missed():
    # Starter 5 games, backup started the last one; the starter returns
    rows = _healthy("KC", "elite", 0.30, n=5) + [_row("2024_06_KC", 2024, 6, "KC", "backup", "backup", 35, -0.20)]
    rows += _healthy("TEN", "avg", 0.05)
    qb = pd.DataFrame(rows)
    full = fixed_qb_rating(qb, ("id", "elite")) - fixed_qb_rating(qb, ("id", "backup"))
    returning = _shock(qb, "KC", "TEN", overrides={"KC": "elite"})
    assert returning == pytest.approx(round(full / 6 * PTS, 4))
    # With no override the backup (last game's passer) is assumed to start again
    assert _shock(qb, "KC", "TEN") == pytest.approx(round(-full * 5 / 6 * PTS, 4))


def test_default_version_is_legacy():
    qb = pd.DataFrame(_healthy("KC", "elite", 0.30) + _healthy("TEN", "bad", -0.20))
    gids = list(qb["game_id"].unique())
    a = compute_point_in_time_qb_shock(qb, gids, "KC", "TEN", 0.3, -0.2, window=8)
    b = compute_point_in_time_qb_shock(qb, gids, "KC", "TEN", 0.3, -0.2, window=8, version="legacy")
    assert a == b != 0.0
    with pytest.raises(ValueError):
        compute_point_in_time_qb_shock(qb, gids, "KC", "TEN", 0.3, -0.2, version="nope")


def _sched(home_codes):
    rows = []
    for i, (home, away, hq, aq) in enumerate(home_codes):
        rows.append({"game_id": f"2015_{i + 1:02d}_{away}_{home}", "season": 2015, "week": i + 1,
                     "home_team": home, "away_team": away, "home_score": 24, "away_score": 17,
                     "kickoff_utc": pd.Timestamp("2015-09-13 17:00", tz="UTC") + pd.Timedelta(days=7 * i),
                     "spread_line": 3.0, "total_line": 44.0, "home_qb_id": hq, "away_qb_id": aq})
    return pd.DataFrame(rows)


def test_v4_builder_uses_schedule_starters_and_canonical_codes():
    # Week 3: the Raiders (OAK in the schedule, LV in the QB table) start a backup with no rows
    sched = _sched([("OAK", "KC", "carr", "smith"), ("KC", "OAK", "smith", "carr"), ("OAK", "KC", "nobody", "smith")])
    qb = pd.DataFrame([
        _row(sched.game_id[0], 2015, 1, "LV", "carr", "carr", 35, 0.10), _row(sched.game_id[0], 2015, 1, "KC", "smith", "smith", 35, 0.10),
        _row(sched.game_id[1], 2015, 2, "LV", "carr", "carr", 35, 0.10), _row(sched.game_id[1], 2015, 2, "KC", "smith", "smith", 35, 0.10),
    ])
    epa = pd.DataFrame(columns=["game_id", "season", "week", "team", "opponent"])
    kw = dict(qb_df=qb, turnover_df=None, qb_version="fixed")
    fixed = build_v4_training_dataset([2015], epa, sched, canonical_teams=True, **kw)
    carr = fixed_qb_rating(qb, ("id", "carr"))
    assert fixed.loc[2, "home_team"] == "LV"
    assert fixed.loc[2, "diff_qb_shock"] == pytest.approx(round((BACKUP_QB_PRIOR - carr) * PTS, 4))
    assert fixed.loc[1, "diff_qb_shock"] == 0.0
    # Without canonical codes the OAK rows never meet the LV QB rows: no shock at all
    raw = build_v4_training_dataset([2015], epa, sched, canonical_teams=False, **kw)
    assert raw.loc[2, "home_team"] == "OAK" and raw.loc[2, "diff_qb_shock"] == 0.0
    assert "home_field" in fixed.columns


def test_v4_feature_lists_and_bounds():
    assert v4_feature_list() == V4_FEATURES
    lb, ub = v4_feature_bounds(V4_FEATURES)
    np.testing.assert_array_equal(lb, V4_BOUNDS_LB)
    np.testing.assert_array_equal(ub, V4_BOUNDS_UB)
    split = v4_feature_list(split=True, home_field=True)
    assert "diff_off_pass" in split and split[-1] == "home_field" and "diff_pass_epa" not in split
    lb, ub = v4_feature_bounds(split)
    i = split.index("diff_fumble_luck_per_gm")
    assert ub[i] == 0.0 and lb[split.index("division")] == -np.inf and lb[split.index("home_field")] == 0.0


def test_backtest_feature_sets_per_model():
    from src.backtest_production import feature_set_config
    cfg = feature_set_config("split_neutral_qb_teams", "v4")
    assert cfg["qb_version"] == "fixed" and cfg["canonical_teams"] and cfg["features"][-1] == "home_field"
    assert feature_set_config("legacy", "v4")["features"] == V4_FEATURES
    assert feature_set_config("split_neutral", "v3")["features"][-1] == "home_field"
    with pytest.raises(ValueError):
        feature_set_config("qb", "v3")


# --- live override lookup (fixed version) -------------------------------------------

def _injuries(tmp_path, rows):
    os.makedirs(tmp_path / "data" / "nflverse_cache", exist_ok=True)
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "nflverse_cache" / "injuries.csv", index=False)


def _live_qb():
    return pd.DataFrame([
        _row("2026_10_KC", 2026, 10, "KC", "mahomes", "P.Mahomes", 35, 0.25),
        _row("2025_05_KC", 2025, 5, "KC", "backup1", "C.Backup", 30, -0.05, 0.9),
        _row("2026_10_TEN", 2026, 10, "TEN", "ward", "C.Ward", 35, 0.0),
        _row("2026_09_TEN", 2026, 9, "TEN", "ward", "C.Ward", 30, 0.0, 0.9),
        _row("2026_09_TEN", 2026, 9, "TEN", "tenbk", "T.Backup", 3, 0.0, 0.1),
    ])


def test_fixed_overrides_read_only_the_decision_week(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _injuries(tmp_path, [{"season": 2026, "week": 10, "team": "KC", "position": "QB", "gsis_id": "mahomes",
                          "full_name": "Patrick Mahomes", "report_status": "Doubtful"}])
    # Tuesday before week 11: only last week's report exists, so no automatic override
    assert fetch_pregame_starter_overrides(2026, _live_qb(), overrides_path=None, version="fixed", week=11) == {}
    # Legacy reads the latest week whatever it is
    assert fetch_pregame_starter_overrides(2026, _live_qb(), overrides_path=None) == {}  # no current-season backup
    # The decision week's own report: the backup comes from an earlier season
    got = fetch_pregame_starter_overrides(2026, _live_qb(), overrides_path=None, version="fixed", week=10)
    assert got == {"KC": "backup1"}


def test_fixed_overrides_fall_back_to_replacement_sentinel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _injuries(tmp_path, [{"season": 2026, "week": 11, "team": "TEN", "position": "QB", "gsis_id": "ward",
                          "full_name": "Cam Ward", "report_status": "Out"},
                         {"season": 2026, "week": 11, "team": "TEN", "position": "QB", "gsis_id": "tenbk",
                          "full_name": "T Backup", "report_status": "Out"}])
    got = fetch_pregame_starter_overrides(2026, _live_qb(), overrides_path=None, version="fixed", week=11)
    assert got == {"TEN": REPLACEMENT_BACKUP}
    with pytest.raises(ValueError):
        fetch_pregame_starter_overrides(2026, _live_qb(), overrides_path=None, version="nope")


def test_canonical_team_codes():
    from src.model_utils import canonical_team_codes
    s = pd.DataFrame({"home_team": ["OAK", "SD", "KC"], "away_team": ["STL", "LV", "oak"]})
    out = canonical_team_codes(s)
    assert list(out["home_team"]) == ["LV", "LAC", "KC"] and list(out["away_team"]) == ["LA", "LV", "LV"]
    assert list(s["home_team"]) == ["OAK", "SD", "KC"]
