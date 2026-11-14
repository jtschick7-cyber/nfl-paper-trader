import pytest

import src.backfill_raw_leans as bf
import src.log_v2_shadow as v2
import src.log_v3_shadow as v3
import src.log_v4_shadow as v4
from src.market_blend import blend_margin
from src.raw_lean import RAW_LEAN_HEADERS, lean_side, raw_lean_cells, unblend_margin

V3_BETA = 0.03693757162735288
V4_BETA = 0.042324548578395654

# Live header rows as of the 2026-11-13 dump (before R2's two columns)
V2_LIVE = ['model_version', 'logged_at_utc', 'season', 'week', 'game_id', 'matchup', 'kickoff_utc', 'sheet_status', 'home_team', 'away_team', 'home_line', 'home_odds', 'away_odds', 'away_fallback_used', 'elo_diff', 'diff_pf', 'diff_pa', 'diff_margin', 'diff_rest', 'division', 'mu_home_margin', 'sigma', 'calibration_method', 'p_home_raw', 'p_home_cal', 'p_away_cal', 'fair_home', 'fair_away', 'edge_home', 'edge_away', 'v2_best_team', 'v2_best_side', 'v2_best_line', 'v2_best_odds', 'v2_model_probability', 'v2_fair_probability', 'v2_edge', 'v2_decision', 'v1_pick', 'notes']
V3_LIVE = ['model_version', 'logged_at_utc', 'season', 'week', 'game_id', 'matchup', 'kickoff_utc', 'sheet_status', 'home_team', 'away_team', 'home_line', 'home_odds', 'away_odds', 'away_fallback_used', 'elo_diff', 'diff_pass_epa', 'diff_rush_epa', 'diff_success_rate', 'diff_early_down_epa', 'diff_margin', 'diff_rest', 'division', 'mu_home_margin', 'sigma', 'p_home_discrete', 'p_home_cal', 'p_away_cal', 'p_push', 'fair_home', 'fair_away', 'edge_home', 'edge_away', 'v3_best_team', 'v3_best_side', 'v3_best_line', 'v3_best_odds', 'v3_model_probability', 'v3_fair_probability', 'v3_edge', 'v3_decision', 'v1_pick', 'integrity_flags', 'notes']
V4_LIVE = ['model_version', 'logged_at_utc', 'season', 'week', 'game_id', 'matchup', 'kickoff_utc', 'sheet_status', 'home_team', 'away_team', 'home_line', 'total_line', 'home_odds', 'away_odds', 'away_fallback_used', 'elo_diff', 'diff_pass_epa', 'diff_rush_epa', 'diff_success_rate', 'diff_early_down_epa', 'diff_qb_shock', 'diff_fumble_luck_per_gm', 'diff_margin', 'diff_rest', 'division', 'mu_home_margin', 'sigma', 'sigma_totals_adj', 'p_home_discrete', 'p_home_cal', 'p_away_cal', 'p_push', 'fair_home', 'fair_away', 'edge_home', 'edge_away', 'v4_best_team', 'v4_best_side', 'v4_best_line', 'v4_best_odds', 'v4_model_probability', 'v4_fair_probability', 'v4_edge', 'v4_stake_units', 'v4_decision', 'v1_pick', 'v1_disagreement', 'guard_recommendation', 'guard_notes']


# --- definitions -----------------------------------------------------------

def test_home_lean_sign_and_size():
    # home -3 => market says home by 3; model says home by 5 => lean 2 to HOME
    raw, pts, side = raw_lean_cells(5.0, -3.0)
    assert raw == 5.0 and pts == pytest.approx(2.0) and side == "HOME"


def test_away_lean_sign_and_size():
    raw, pts, side = raw_lean_cells(-1.0, -3.0)
    assert pts == pytest.approx(-4.0) and side == "AWAY"
    # home underdog +6.5, model has home losing by 8
    assert raw_lean_cells(-8.0, 6.5)[1:] == (pytest.approx(-1.5), "AWAY")


def test_zero_lean_has_blank_side():
    assert raw_lean_cells(3.0, -3.0)[1:] == (0.0, "")
    assert lean_side(0.0) == ""


def test_cells_are_rounded_to_four_decimals():
    raw, pts, _ = raw_lean_cells(5.123456789, -3.0)
    assert raw == 5.1235 and pts == 2.1235


@pytest.mark.parametrize("beta", [V3_BETA, V4_BETA])
@pytest.mark.parametrize("raw_mu,home_line", [(6.3, -2.5), (-9.1, 3.0), (0.4, 7.0)])
def test_lean_times_beta_equals_blended_deviation(beta, raw_mu, home_line):
    blended = blend_margin(raw_mu, home_line, beta)
    _, pts, _ = raw_lean_cells(raw_mu, home_line, decimals=12)
    assert pts * beta == pytest.approx(blended - (-home_line), abs=1e-9)


# --- backfill --------------------------------------------------------------

@pytest.mark.parametrize("raw_mu,home_line", [(6.3, -2.5), (-9.1, 3.0), (0.4, 7.0), (-2.77, -1.5)])
def test_unblend_recovers_raw_margin_at_v3_precision(raw_mu, home_line):
    logged = round(blend_margin(raw_mu, home_line, V3_BETA), 4)
    # The logged value is off by at most 0.5e-4, so the raw margin is off by at most 0.5e-4 / beta
    assert unblend_margin(logged, home_line, V3_BETA) == pytest.approx(raw_mu, abs=0.5e-4 / V3_BETA + 1e-12)
    assert 0.5e-4 / V3_BETA < 1.4e-3


def _v3_row(mu, home_line, sigma="12.8145", **kw):
    row = {"mu_home_margin": mu, "home_line": home_line, "sigma": sigma}
    row.update(kw)
    return row


def test_backfill_v3_current_artifact_row():
    logged = round(blend_margin(6.3, -2.5, V3_BETA), 4)
    raw, pts, side = bf.backfill_values("V3_Shadow", _v3_row(logged, -2.5))
    assert raw == pytest.approx(6.3, abs=1.4e-3)
    assert pts == pytest.approx(3.8, abs=1.4e-3) and side == "HOME"


def test_backfill_skips_pre_fix_v3_rows():
    assert bf.backfill_values("V3_Shadow", _v3_row(-1.0063, 4.5, sigma="13.2147")) is None


def test_backfill_skips_rows_with_unknown_artifact():
    row = _v3_row(1.0, -3.0, artifact_id="v3_2012-2025_deadbeef")
    assert bf.backfill_values("V3_Shadow", row, betas={}) is None


def test_backfill_reads_beta_by_artifact_id():
    logged = round(blend_margin(6.3, -2.5, 0.5), 4)
    row = _v3_row(logged, -2.5, sigma="99", artifact_id="v3_x_1234")
    raw, _, _ = bf.backfill_values("V3_Shadow", row, betas={"v3_x_1234": 0.5})
    assert raw == pytest.approx(6.3, abs=1e-3)


def test_backfill_v4_leaves_side_blank_when_ambiguous():
    # Logged to 2 decimals: a deviation that rounds to 0.00 has no recoverable sign
    raw, pts, side = bf.backfill_values("V4_Shadow", {"mu_home_margin": "2.5", "home_line": "-2.5", "sigma": "12.81"})
    assert side == ""
    raw, pts, side = bf.backfill_values("V4_Shadow", {"mu_home_margin": "2.53", "home_line": "-2.5", "sigma": "12.81"})
    assert side == "HOME" and pts == pytest.approx(0.03 / V4_BETA, abs=1e-3)


def test_backfill_v2_raw_is_logged_margin():
    assert bf.backfill_values("V2_Shadow", {"mu_home_margin": "8.066145", "home_line": "-7.5"}) == (
        8.0661, pytest.approx(0.5661), "HOME")


def test_plan_fills_only_blank_cells():
    import pandas as pd

    headers = V3_LIVE + ["artifact_id", "decision_regime"] + RAW_LEAN_HEADERS
    col = {h: i + 1 for i, h in enumerate(headers)}
    df = pd.DataFrame([
        _v3_row("1.0", "-3.0", game_id="A"),
        _v3_row("1.0", "-3.0", game_id="B", raw_mu_home_margin="5", raw_lean_pts="2", raw_lean_side="HOME"),
        _v3_row("-1.0", "4.5", sigma="13.2147", game_id="C"),
    ])
    updates, summary = bf.plan_backfill("V3_Shadow", df, col)
    assert {u["row"] for u in updates} == {2}
    assert {u["col"] for u in updates} == {col[h] for h in RAW_LEAN_HEADERS}
    assert summary["filled"] == 1 and summary["skipped"] == 1 and summary["already_set"] == 1


# --- logger headers --------------------------------------------------------

@pytest.mark.parametrize("mod,live", [(v2, V2_LIVE), (v3, V3_LIVE), (v4, V4_LIVE)])
def test_logger_headers_append_raw_lean_columns_last(mod, live):
    assert mod.HEADERS[-3:] == ["raw_mu_home_margin", "raw_lean_pts", "raw_lean_side"]
    assert mod.HEADERS[:-3] == live + ["artifact_id", "decision_regime"]
    assert len(set(mod.HEADERS)) == len(mod.HEADERS)
