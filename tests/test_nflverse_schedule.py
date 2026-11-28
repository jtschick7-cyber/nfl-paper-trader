import pandas as pd
import pytest

from src import nflverse_schedule as ns


def _csv(tmp_path, name, seasons):
    p = tmp_path / name
    pd.DataFrame({"game_id": [f"{s}_01_A_B" for s in seasons], "season": seasons,
                  "home_score": [1] * len(seasons)}).to_csv(p, index=False)
    return str(p)


def test_primary_is_nflverse_github_not_the_dead_mirror():
    assert ns.URLS[0].startswith("https://raw.githubusercontent.com/nflverse/nfldata/")
    assert "habitatring" not in ns.URLS[0]


def test_filters_to_requested_seasons(tmp_path):
    src = _csv(tmp_path, "g.csv", [2024, 2025, 2026, 2026])
    out = ns.import_schedules([2026], urls=(src,))
    assert list(out["season"].unique()) == [2026] and len(out) == 2
    assert len(ns.import_schedules(range(2025, 2027), urls=(src,))) == 3


def test_falls_back_when_primary_fails(tmp_path):
    good = _csv(tmp_path, "g.csv", [2026])
    out = ns.import_schedules([2026], urls=(str(tmp_path / "missing.csv"), good))
    assert len(out) == 1


def test_skips_a_source_without_season_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("<html>not found</html>\n")
    good = _csv(tmp_path, "g.csv", [2026])
    assert len(ns.import_schedules([2026], urls=(str(bad), good))) == 1


def test_raises_when_every_source_fails(tmp_path):
    with pytest.raises(RuntimeError, match="No schedule source reachable"):
        ns.import_schedules([2026], urls=(str(tmp_path / "a.csv"), str(tmp_path / "b.csv")))


@pytest.mark.parametrize("years", [2026, (2026,)])
def test_rejects_non_list_input_like_nfl_data_py(years):
    with pytest.raises(ValueError):
        ns.import_schedules(years)


def test_rejects_pre_1999():
    with pytest.raises(ValueError):
        ns.import_schedules([1998])
