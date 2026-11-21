"""P0-3: the EPA/PBP cache refreshes on missing completed game_ids, not on max week or file age."""

import os
import time

import pandas as pd
import pytest

import src.build_features as bf
import src.log_v3_shadow as v3
import src.log_v4_shadow as v4

WEEK10 = {"2026_10_TNF", "2026_10_SUN1", "2026_10_SUN2", "2026_10_MNF"}


def test_tnf_only_cache_is_missing_the_rest_of_the_week():
    missing = bf.missing_completed_game_ids({"2026_09_A", "2026_10_TNF"}, 2026,
                                            completed_ids=WEEK10 | {"2026_09_A"})
    assert missing == WEEK10 - {"2026_10_TNF"}


def test_complete_cache_is_current():
    assert bf.missing_completed_game_ids(WEEK10, 2026, completed_ids=WEEK10) == set()


def test_reg_game_ids_ignores_playoffs():
    df = pd.DataFrame({"game_id": ["a", "b"], "season_type": ["REG", "POST"]})
    assert bf._reg_game_ids(df) == {"a"}


def _write_epa(path, gids):
    pd.DataFrame({"game_id": sorted(gids), "season": 2026, "week": 10, "team": "KC"}).to_csv(path, index=False)
    os.utime(path, (time.time(), time.time()))  # as fresh as a git checkout


@pytest.mark.parametrize("mod,fn,path_attr", [
    (v3, "ensure_epa_cache_is_current", "EPA_DATA_PATH"),
    (v4, "ensure_feature_caches_are_current", "EPA_DATA_PATH"),
])
def test_fresh_mtime_still_refreshes_when_games_are_missing(monkeypatch, tmp_path, mod, fn, path_attr):
    epa = tmp_path / "epa.csv"
    _write_epa(epa, {"2026_10_TNF"})
    for attr in ("EPA_DATA_PATH", "QB_DATA_PATH", "TURNOVER_DATA_PATH"):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, str(epa))
    monkeypatch.setattr(mod, "SEASON", 2026)
    monkeypatch.setattr(bf, "_completed_game_ids", lambda season: WEEK10)
    calls = []
    monkeypatch.setattr(bf, "build_features", lambda *a, **k: calls.append(1))
    getattr(mod, fn)()
    assert calls == [1]


@pytest.mark.parametrize("mod,fn", [(v3, "ensure_epa_cache_is_current"), (v4, "ensure_feature_caches_are_current")])
def test_no_refresh_when_every_completed_game_is_cached(monkeypatch, tmp_path, mod, fn):
    epa = tmp_path / "epa.csv"
    _write_epa(epa, WEEK10)
    for attr in ("EPA_DATA_PATH", "QB_DATA_PATH", "TURNOVER_DATA_PATH"):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, str(epa))
    monkeypatch.setattr(mod, "SEASON", 2026)
    monkeypatch.setattr(bf, "_completed_game_ids", lambda season: WEEK10)
    calls = []
    monkeypatch.setattr(bf, "build_features", lambda *a, **k: calls.append(1))
    getattr(mod, fn)()
    assert calls == []


def test_atomic_write_leaves_no_tmp(tmp_path):
    out = tmp_path / "x.csv"
    bf._atomic_to_csv(pd.DataFrame({"a": [1]}), str(out))
    assert out.exists() and not (tmp_path / "x.csv.tmp").exists()


def test_failed_live_season_download_keeps_existing_csvs(monkeypatch, tmp_path):
    import sys, types
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    (tmp_path / "data" / "team_game_epa.csv").write_text("sentinel\n")
    fake = types.ModuleType("nfl_data_py")
    def boom(*a, **k):
        raise RuntimeError("nflverse down")
    fake.import_pbp_data = boom
    monkeypatch.setitem(sys.modules, "nfl_data_py", fake)
    monkeypatch.setattr(bf, "SEASON", 2026)
    monkeypatch.setattr(bf, "_completed_game_ids", lambda season: WEEK10)
    with pytest.raises(SystemExit):
        bf.build_features([2026])
    assert (tmp_path / "data" / "team_game_epa.csv").read_text() == "sentinel\n"
