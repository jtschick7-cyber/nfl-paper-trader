import numpy as np
import pandas as pd
import pytest

import src.backtest_production as bp
from src.decision import PICK
from src.train_model_v3 import V3_FEATURES


def _synthetic(seasons=(2012, 2013, 2014), per_season=120, seed=0, probe_game=None):
    rng = np.random.default_rng(seed)
    rows = []
    t = pd.Timestamp("2012-09-01", tz="UTC")
    for s in seasons:
        for i in range(per_season):
            feats = {c: float(rng.normal()) for c in V3_FEATURES}
            feats["division"] = float(rng.integers(0, 2))
            home_line = float(rng.choice([-7.0, -3.5, -3.0, -2.5, 1.5, 3.0, 6.5]))
            margin = float(round(-home_line + 2.0 * feats["elo_diff"] + rng.normal(0, 13)))
            rows.append({
                "game_id": f"{s}_{i:03d}", "season": s, "week": 1 + i // 16,
                "kickoff_utc": t + pd.Timedelta(days=(s - 2012) * 365 + i),
                "home_line": home_line, "actual_home_margin": margin,
                "mkt_spread_line": -home_line, "home_spread_odds": -110.0, "away_spread_odds": -110.0,
                "mkt_total_line": 44.5, **feats,
            })
    df = pd.DataFrame(rows)
    if probe_game is not None:
        idx = df.index[df["game_id"] == probe_game][0]
        df.loc[idx, ["mkt_spread_line", "home_line", "home_spread_odds", "away_spread_odds"]] = [2.5, -2.5, 115.0, -135.0]
    return df


def test_no_test_season_row_reaches_its_own_training_beta_or_calibration():
    ds = _synthetic()
    games, folds = bp.walk_forward(ds, bp.v3_spec(), [2013, 2014])
    assert [f["season"] for f in folds] == [2013, 2014]
    for f in folds:
        test_ids = f["test_ids"]
        assert test_ids and all(gid.startswith(str(f["season"])) for gid in test_ids)
        for key in ("train_ids", "blend_ids", "calibration_ids"):
            assert f[key], key
            assert not (f[key] & test_ids), key
            assert all(int(gid[:4]) < f["season"] for gid in f[key]), key
    assert set(games["game_id"]) == set(ds[ds["season"] >= 2013]["game_id"])


def test_zero_disagreement_probe_flags_a_long_price():
    ds = _synthetic(probe_game="2014_050")
    games, folds = bp.walk_forward(ds, bp.v3_spec(), [2014])
    probe = bp.probe_table(games)
    assert "2014_050" in set(probe["game_id"])
    row = probe[probe["game_id"] == "2014_050"].iloc[0]
    assert row["probe_side"] == "HOME" and row["probe_odds"] == 115.0
    # at -110/-110 a zero-disagreement half-point line has no edge
    assert (games[games["game_id"] != "2014_050"]["probe_decision"] != PICK).all()


def test_games_without_a_price_are_skipped_not_graded():
    ds = _synthetic()
    ds.loc[ds["game_id"] == "2014_010", "home_spread_odds"] = np.nan
    games, folds = bp.walk_forward(ds, bp.v3_spec(), [2014])
    assert "2014_010" not in set(games["game_id"])
    assert folds[0]["skipped"] == 1


@pytest.mark.parametrize("side,line,odds,margin,expected", [
    ("HOME", -3.0, -110, 7, ("WIN", 100 / 110)),
    ("HOME", -3.0, -110, 3, ("PUSH", 0.0)),
    ("AWAY", -3.0, 120, 1, ("WIN", 1.2)),
    ("AWAY", -3.0, 120, 10, ("LOSS", -1.0)),
])
def test_grade_at_actual_price(side, line, odds, margin, expected):
    result, units = bp.grade(side, line, odds, margin)
    assert result == expected[0] and units == pytest.approx(expected[1])


def test_season_table_counts_low_input_and_long_price_bets():
    games = pd.DataFrame([
        {"season": 2014, "decision": PICK, "result": "WIN", "units": 1.15, "raw_lean_pts": 0.2,
         "best_odds": 115.0, "probe_decision": PICK, "lean_result": "WIN"},
        {"season": 2014, "decision": PICK, "result": "LOSS", "units": -1.0, "raw_lean_pts": 3.0,
         "best_odds": -110.0, "probe_decision": "PASS", "lean_result": "LOSS"},
        {"season": 2014, "decision": "PASS", "result": "WIN", "units": 0.9, "raw_lean_pts": -1.0,
         "best_odds": -110.0, "probe_decision": "PASS", "lean_result": ""},
    ])
    r = bp.summarize(games, "all")
    assert (r["games"], r["bets"], r["wins"], r["losses"]) == (3, 2, 1, 1)
    assert r["units"] == pytest.approx(0.15) and r["roi_pct"] == pytest.approx(7.5)
    assert r["low_input_bets"] == 1 and r["long_price_bets"] == 1 and r["probe_picks"] == 1
    assert r["hit_pct"] == pytest.approx(50.0)
