import math

import numpy as np
import pandas as pd

from src.compare_models import evaluate_all_models, first_priced_decision
from src.log_v4_shadow import sheet_safe


def _v3_row(gid, decision, side, line, logged_at):
    return {
        "game_id": gid,
        "logged_at": logged_at,
        "v3_decision": decision,
        "v3_best_side": side,
        "v3_best_line": line,
        "v3_best_odds": "-110" if line else "",
        "v3_edge": "0.025" if line else "",
    }


def test_blank_line_row_does_not_shadow_real_decision():
    # Logged weeks early with no opening line, then the real Tuesday pick, then a later re-evaluation
    df = pd.DataFrame([
        _v3_row("2026_10_DAL_IND", "PASS", "", "", "2026-10-28"),
        _v3_row("2026_10_DAL_IND", "SHADOW_PICK", "HOME", "3.5", "2026-11-03"),
        _v3_row("2026_10_DAL_IND", "PASS", "HOME", "2.5", "2026-11-05"),
    ])
    kept = first_priced_decision(df, "v3_best_line")
    assert len(kept) == 1
    assert kept.iloc[0]["v3_decision"] == "SHADOW_PICK"


def test_scorecard_grades_pick_behind_blank_line_row():
    v3_df = pd.DataFrame([
        _v3_row("2026_10_DAL_IND", "PASS", "", "", "2026-10-28"),
        _v3_row("2026_10_DAL_IND", "SHADOW_PICK", "HOME", "3.5", "2026-11-03"),
    ])
    scores = {"2026_10_DAL_IND": {"home_score": 24, "away_score": 21}}
    summary_rows, _, _ = evaluate_all_models(pd.DataFrame(), pd.DataFrame(), v3_df, scores)
    v3 = next(r for r in summary_rows if r["model"].startswith("Model v3 (EPA"))
    assert v3["bets"] == 1
    assert v3["record"] == "1W - 0L - 0P"


def test_first_priced_decision_handles_empty_and_missing_columns():
    assert first_priced_decision(pd.DataFrame(), "v4_best_line").empty
    assert first_priced_decision(None, "v4_best_line") is None


def test_sheet_safe_blanks_non_finite_values():
    assert sheet_safe(float("nan")) == ""
    assert sheet_safe(math.inf) == ""
    assert sheet_safe(np.float64("nan")) == ""
    assert sheet_safe(np.float64(1.25)) == 1.25 and type(sheet_safe(np.float64(1.25))) is float
    assert sheet_safe(np.int64(3)) == 3 and type(sheet_safe(np.int64(3))) is int
    assert sheet_safe(True) is True
    assert sheet_safe("PASS") == "PASS"
