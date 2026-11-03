import pandas as pd
import pytest

import src.compare_models as cm
from src.staking import compute_push_aware_stake


def _v3_pick(gid, prob, line="3.5", week=10):
    return {
        "game_id": gid,
        "week": week,
        "v3_decision": "SHADOW_PICK",
        "v3_best_side": "HOME",
        "v3_best_line": line,
        "v3_best_odds": "-110",
        "v3_edge": "0.025",
        "v3_model_probability": prob,
        "p_push": "0.05",
    }


def _expected_stake(prob):
    return compute_push_aware_stake(prob * 0.95, 0.05, -110.0)["recommended_units"]


def test_v3_kelly_columns_use_kelly_stakes_not_flat():
    v3_df = pd.DataFrame([
        _v3_pick("2026_10_DAL_IND", "0.60"),  # wins
        _v3_pick("2026_10_NYJ_BUF", "0.56"),  # loses
    ])
    scores = {
        "2026_10_DAL_IND": {"home_score": 24, "away_score": 21},
        "2026_10_NYJ_BUF": {"home_score": 10, "away_score": 20},
    }
    win_stake, loss_stake = _expected_stake(0.60), _expected_stake(0.56)
    assert win_stake != loss_stake  # otherwise the test cannot tell Kelly from flat

    summary_rows, _, _ = cm.evaluate_all_models(pd.DataFrame(), pd.DataFrame(), v3_df, scores)
    v3 = next(r for r in summary_rows if r["model"].startswith("Model v3 (EPA"))

    expected_k_units = round(cm.profit_units(-110.0, win_stake), 4) - loss_stake
    assert v3["kelly_units"] == pytest.approx(expected_k_units, abs=0.01)
    assert v3["kelly_units"] != pytest.approx(v3["units"], abs=0.01)


def test_v3_kelly_stakes_capped_per_week():
    # 12 max-confidence picks in one week would exceed the 10u slate cap
    v3_df = pd.DataFrame([_v3_pick(f"2026_10_G{i}", "0.75") for i in range(12)])
    stakes = cm.v3_kelly_stake_map(v3_df)
    assert len(stakes) == 12
    assert sum(stakes.values()) <= 10.0 + 1e-6


def test_v3_kelly_stake_map_skips_non_picks():
    row = _v3_pick("2026_10_DAL_IND", "0.60")
    row["v3_decision"] = "PASS"
    assert cm.v3_kelly_stake_map(pd.DataFrame([row])) == {}
    assert cm.v3_kelly_stake_map(pd.DataFrame()) == {}


class _FakeWorksheet:
    def __init__(self):
        self.values = None

    def clear(self):
        pass

    def update(self, values, range_name):
        self.values = values


def test_sheet_guard_row_matches_header_width(monkeypatch):
    ws = _FakeWorksheet()
    monkeypatch.setattr(cm, "get_or_create_worksheet", lambda _name: ws)
    bets = [cm.grade_single_bet("HOME", -3.0, -110, 17, 24, closing_line=-3.5)]
    summary = [cm.compute_metrics(bets, "Model v1 (Master_Bets Baseline)")]
    guard = cm.compute_metrics(bets, "Guard Suppressed Danger Bets", "Hypothetical Placed")

    cm.write_google_sheet_scorecard(summary, guard, pd.DataFrame())

    header_idx = next(i for i, r in enumerate(ws.values) if r and r[0] == "Model")
    header = ws.values[header_idx]
    assert len(header) == 13
    model_row = ws.values[header_idx + 1]
    guard_row = next(r for r in ws.values if r and r[0] == "Injury / RLM Guard Suppressed")
    assert len(model_row) == len(header)
    assert len(guard_row) == len(header)


def test_console_table_shows_clv_tie_pct(capsys):
    bets = [
        cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.5),  # beat
        cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.0),  # tie
    ]
    summary = [cm.compute_metrics(bets, "Model v1 (Master_Bets Baseline)")]
    guard = cm.compute_metrics([], "Guard", "Hypothetical Placed")

    cm.print_terminal_dashboard(summary, guard, pd.DataFrame())

    lines = capsys.readouterr().out.splitlines()
    header = next(l for l in lines if l.startswith("Model / Architecture"))
    row = next(l for l in lines if l.startswith("Model v1"))
    cols = [c.strip() for c in header.split("|")]
    vals = [c.strip() for c in row.split("|")]
    assert "CLV Tie" in cols
    assert vals[cols.index("CLV Tie")] == "50.0%"
    assert len(vals) == len(cols)
