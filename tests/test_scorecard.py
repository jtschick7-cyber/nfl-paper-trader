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
    assert len(header) == len(cm.SCORECARD_HEADER)
    assert list(header[:13]) == list(cm.SCORECARD_HEADER[:13])  # old columns keep their places
    assert {"CLV Lose %", "Avg CLV Fair Prob", "Avg CLV Fair Prob SE"} <= set(header)
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


def test_weekly_clv_counts_positive_weeks_with_a_close():
    bets = [
        cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.5, week=9),   # beat
        cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.0, week=9),   # tie
        cm.grade_single_bet("HOME", -3.5, -110, 24, 20, closing_line=-3.0, week=10),  # lose
        cm.grade_single_bet("AWAY", 7.5, -110, 24, 20, closing_line=-7.0, week=11),   # beat
        cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=None, week=12),  # no close
    ]
    m = cm.compute_metrics(bets, "test")
    assert sorted(m["weekly_clv_prob_pct"]) == [9, 10, 11]
    assert m["weekly_clv_prob_pct"][9] == pytest.approx(bets[0]["clv_prob"] * 50.0, abs=0.01)
    assert m["weekly_clv_prob_pct"][10] < 0 < m["weekly_clv_prob_pct"][11]
    assert m["clv_weeks"] == 3
    assert m["clv_positive_weeks"] == 2
    assert m["clv_positive_week_pct"] == pytest.approx(66.7)

    empty = cm.compute_metrics([], "none")
    assert empty["clv_weeks"] == 0 and empty["weekly_clv_prob_pct"] == {}


def test_bet_week_falls_back_to_game_id():
    assert cm.bet_week({"week": "9"}, "2026_10_DAL_IND") == 9
    assert cm.bet_week({}, "2026_10_DAL_IND") == 10
    assert cm.bet_week({}, "bad") is None


def test_graded_shadow_bets_carry_week():
    row = _v3_pick("2026_10_DAL_IND", "0.60")
    del row["week"]
    scores = {"2026_10_DAL_IND": {"home_score": 24, "away_score": 21}}
    master = pd.DataFrame([{"game_id": "2026_10_DAL_IND", "closing_line": -4.0}])
    summary_rows, _, _ = cm.evaluate_all_models(master, pd.DataFrame(), pd.DataFrame([row]), scores)
    v3 = next(r for r in summary_rows if r["model"].startswith("Model v3 (EPA"))
    assert list(v3["weekly_clv_prob_pct"]) == [10]


def test_sheet_has_weekly_clv_section(monkeypatch):
    ws = _FakeWorksheet()
    monkeypatch.setattr(cm, "get_or_create_worksheet", lambda _name: ws)
    bets = [
        cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.5, week=9),
        cm.grade_single_bet("HOME", -3.5, -110, 24, 20, closing_line=-3.0, week=10),
    ]
    summary = [cm.compute_metrics(bets, "Model v1 (Master_Bets Baseline)")]
    guard = cm.compute_metrics([], "Guard", "Hypothetical Placed")

    cm.write_google_sheet_scorecard(summary, guard, pd.DataFrame())

    title_idx = next(i for i, r in enumerate(ws.values) if r and str(r[0]).startswith("WEEKLY AVG CLV WIN PROB"))
    assert ws.values[title_idx + 1] == ["Model", "Tier", "Weeks +", "Wk 9", "Wk 10"]
    row = ws.values[title_idx + 2]
    assert row[0] == "Model v1 (Master_Bets Baseline)"
    assert row[2] == "1/2 (50%)"
    assert row[3].startswith("+") and row[4].startswith("-")


IN_WEEK_TS = "2026-11-17T16:00:00+00:00"
LEGACY_TS = "2026-10-27T16:00:00+00:00"


def test_parse_logged_at_handles_both_formats():
    a = cm.parse_logged_at("2026-11-03 21:25:00 UTC")
    b = cm.parse_logged_at("2026-11-03T21:25:00.123456+00:00")
    assert a is not None and b is not None
    assert a.year == 2026 and a.hour == 21 and b.minute == 25
    assert cm.parse_logged_at("") is None and cm.parse_logged_at(None) is None
    cutoff = cm.parse_logged_at(cm.SHADOW_LEGACY_CUTOFF_UTC)
    assert cm.parse_logged_at("2026-11-02 23:59:59 UTC") < cutoff <= cm.parse_logged_at("2026-11-03T00:00:00+00:00")


def test_shadow_row_regime_rules():
    regimes = {"G_NEW": "in_week", "G_OLD": "lookahead", "G_BLANK": ""}
    assert cm.shadow_row_regime({"game_id": "G_NEW", "logged_at_utc": IN_WEEK_TS}, regimes) == "in_week"
    assert cm.shadow_row_regime({"game_id": "G_NEW", "logged_at_utc": "2026-11-17 16:00:00 UTC"}, regimes) == "in_week"
    assert cm.shadow_row_regime({"game_id": "G_NEW", "logged_at_utc": LEGACY_TS}, regimes) == "legacy"
    assert cm.shadow_row_regime({"game_id": "G_OLD", "logged_at_utc": IN_WEEK_TS}, regimes) == "legacy"
    assert cm.shadow_row_regime({"game_id": "G_BLANK", "logged_at_utc": IN_WEEK_TS}, regimes) == "legacy"
    assert cm.shadow_row_regime({"game_id": "G_MISSING", "logged_at_utc": IN_WEEK_TS}, regimes) == "legacy"
    assert cm.shadow_row_regime({"game_id": "G_NEW", "logged_at_utc": ""}, regimes) == "legacy"


def _with(row, **kw):
    row = dict(row)
    row.update(kw)
    return row


def test_legacy_pick_and_in_week_pass_land_in_different_blocks():
    master = pd.DataFrame([
        {"game_id": "2026_09_DAL_IND", "decision_regime": "", "closing_line": -4.0},
        {"game_id": "2026_12_NYJ_BUF", "decision_regime": "in_week", "closing_line": -4.0},
    ])
    v3 = pd.DataFrame([
        _with(_v3_pick("2026_09_DAL_IND", "0.60", week=9), logged_at_utc=LEGACY_TS),
        _with(_v3_pick("2026_12_NYJ_BUF", "0.51", week=12), logged_at_utc=IN_WEEK_TS, v3_decision="PASS", v3_edge="0.004"),
    ])
    scores = {
        "2026_09_DAL_IND": {"home_score": 24, "away_score": 21},
        "2026_12_NYJ_BUF": {"home_score": 24, "away_score": 21},
    }
    blocks = cm.evaluate_regimes(master, pd.DataFrame(), v3, scores)
    v3_in = next(r for r in blocks["in_week"][0] if r["model"].startswith("Model v3 (EPA"))
    v3_old = next(r for r in blocks["legacy"][0] if r["model"].startswith("Model v3 (EPA"))
    assert v3_in["bets"] == 0
    assert v3_old["bets"] == 1


def test_in_week_row_is_not_shadowed_by_earlier_legacy_row_for_same_game():
    master = pd.DataFrame([{"game_id": "2026_12_NYJ_BUF", "decision_regime": "in_week", "closing_line": -4.0}])
    v3 = pd.DataFrame([
        _with(_v3_pick("2026_12_NYJ_BUF", "0.51", week=12), logged_at_utc=LEGACY_TS, v3_decision="PASS"),
        _with(_v3_pick("2026_12_NYJ_BUF", "0.60", week=12), logged_at_utc=IN_WEEK_TS),
    ])
    scores = {"2026_12_NYJ_BUF": {"home_score": 24, "away_score": 21}}
    blocks = cm.evaluate_regimes(master, pd.DataFrame(), v3, scores)
    v3_in = next(r for r in blocks["in_week"][0] if r["model"].startswith("Model v3 (EPA"))
    assert v3_in["bets"] == 1


def test_v1_bets_split_by_master_decision_regime():
    def bet(gid, regime):
        return {"game_id": gid, "week": 12, "pick_team": "BUF", "pick_side": "HOME", "pick_line": -3.0,
                "pick_odds": -110, "closing_line": -3.5, "decision_regime": regime}

    master = pd.DataFrame([bet("A", "in_week"), bet("B", "lookahead"), bet("C", "")])
    scores = {g: {"home_score": 24, "away_score": 20} for g in "ABC"}
    blocks = cm.evaluate_regimes(master, pd.DataFrame(), pd.DataFrame(), scores)
    assert blocks["in_week"][0][0]["bets"] == 1
    assert blocks["legacy"][0][0]["bets"] == 2


def test_guard_ignores_suppress_rows_below_edge_threshold():
    master = pd.DataFrame([
        {"game_id": "G1", "decision_regime": "", "closing_line": -4.0},
        {"game_id": "G2", "decision_regime": "", "closing_line": -4.0},
    ])
    v3 = pd.DataFrame([
        _with(_v3_pick("G1", "0.55"), v3_decision="SUPPRESS_DANGER", v3_edge="0.0153", logged_at_utc=LEGACY_TS),
        _with(_v3_pick("G2", "0.56"), v3_decision="SUPPRESS_DANGER", v3_edge="0.0221", logged_at_utc=LEGACY_TS),
    ])
    v4 = pd.DataFrame([{
        "game_id": "G1", "week": 10, "v4_decision": "SUPPRESS_DANGER", "v4_best_side": "HOME",
        "v4_best_line": "3.5", "v4_best_odds": "-110", "v4_edge": "0.0101", "logged_at_utc": "2026-10-27 16:00:00 UTC",
    }])
    scores = {g: {"home_score": 24, "away_score": 21} for g in ("G1", "G2")}
    _, guard, counts = cm.evaluate_all_models(master, pd.DataFrame(), v3, scores, v4_df=v4)
    assert guard["bets"] == 1
    assert counts["suppressed_count"] == 1


def test_sheet_has_separate_legacy_block(monkeypatch):
    ws = _FakeWorksheet()
    monkeypatch.setattr(cm, "get_or_create_worksheet", lambda _name: ws)
    in_week = [cm.compute_metrics([], "Model v1 (Master_Bets Baseline)")]
    legacy_bets = [cm.grade_single_bet("HOME", -3.0, -110, 24, 20, closing_line=-3.5, week=9)]
    legacy = ([cm.compute_metrics(legacy_bets, "Model v1 (Master_Bets Baseline)")],
              cm.compute_metrics([], "Guard", "Hypothetical Placed"))
    guard = cm.compute_metrics([], "Guard", "Hypothetical Placed")

    cm.write_google_sheet_scorecard(in_week, guard, pd.DataFrame(), legacy=legacy)

    firsts = [r[0] if r else "" for r in ws.values]
    legacy_idx = next(i for i, v in enumerate(firsts) if str(v).startswith("LEGACY"))
    weekly_idx = next(i for i, v in enumerate(firsts) if str(v).startswith("WEEKLY AVG CLV"))
    summary_idx = firsts.index("Model")
    assert summary_idx < legacy_idx
    assert ws.values[legacy_idx + 1] == list(cm.SCORECARD_HEADER)
    assert ws.values[legacy_idx + 2][2] == 1  # the legacy bet
    assert ws.values[summary_idx + 1][2] == 0  # in-week block is empty
    # weekly CLV (promotion input) is built from in-week rows only: no week columns
    assert ws.values[weekly_idx + 1] == ["Model", "Tier", "Weeks +"]
