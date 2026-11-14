import numpy as np
import pandas as pd

from src.capture_opening import plan_opening_capture
from src.generate_picks import apply_weekly_cap, build_pick_updates, eligible_candidates

NOW = pd.Timestamp("2026-11-24 15:17", tz="UTC")  # Tuesday after week 11 MNF

SHEET_COLS = [
    "game_id", "season", "week", "home_team", "away_team", "kickoff_time", "status",
    "opening_line", "opening_odds", "opening_away_odds", "opening_captured_at",
    "total_line", "pick_team",
]


def sheet(rows):
    return pd.DataFrame(rows, columns=SHEET_COLS)


def row(gid, week, home, away, kickoff, status="PENDING", line=np.nan, captured=np.nan, pick=np.nan):
    odds = -110 if pd.notna(line) else np.nan
    return [gid, 2026, week, home, away, kickoff, status, line, odds, odds, captured, np.nan, pick]


def schedule(extra=()):
    """Week 11 (all scored) plus the week-12 games on the sheet below."""
    rows = [
        ("w11_a", 2026, 11, "KC", "DEN", 24, 20, pd.Timestamp("2026-11-22 18:00", tz="UTC")),
        ("w11_b", 2026, 11, "BUF", "MIA", 31, 10, pd.Timestamp("2026-11-22 18:00", tz="UTC")),
        ("w11_c", 2026, 11, "NE", "NYJ", 17, 14, pd.Timestamp("2026-11-22 18:00", tz="UTC")),
        ("w11_d", 2026, 11, "LV", "LAC", 9, 27, pd.Timestamp("2026-11-24 01:15", tz="UTC")),
        ("w12_a", 2026, 12, "DEN", "KC", np.nan, np.nan, pd.Timestamp("2026-11-29 18:00", tz="UTC")),
        ("w12_b", 2026, 12, "MIA", "BUF", np.nan, np.nan, pd.Timestamp("2026-11-29 18:00", tz="UTC")),
        ("w12_c", 2026, 12, "NYJ", "NE", np.nan, np.nan, pd.Timestamp("2026-11-29 18:00", tz="UTC")),
        ("w13_a", 2026, 13, "KC", "BUF", np.nan, np.nan, pd.Timestamp("2026-12-06 18:00", tz="UTC")),
    ]
    rows += list(extra)
    return pd.DataFrame(rows, columns=["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score", "kickoff_utc"])


SPREADS = pd.DataFrame([
    {"home_team": h, "away_team": a, "home_spread": -3.0, "home_odds": -110, "away_odds": -110,
     "total_line": 44.5, "bookmaker": "DraftKings", "market_last_update": "2026-11-24T15:00:00Z"}
    for h, a in [("DEN", "KC"), ("MIA", "BUF"), ("NYJ", "NE"), ("KC", "BUF")]
])

COL = {c: i + 1 for i, c in enumerate(SHEET_COLS + ["opening_book", "opening_quote_at", "decision_regime", "pick_side", "pick_line",
                                                    "pick_odds", "model_probability", "implied_probability_no_vig",
                                                    "edge", "notes", "pick_made_at"])}
COL_NAME = {v: k for k, v in COL.items()}


def cells_by_row(updates):
    out = {}
    for u in updates:
        out.setdefault(u["row"], {})[COL_NAME[u["col"]]] = u["value"]
    return out


def base_sheet():
    return sheet([
        row("w11_d", 11, "LV", "LAC", "2026-11-24T01:15:00Z", status="OPEN_CAPTURED", line=3.0, captured="2026-11-10T15:20:00Z"),
        row("w12_a", 12, "DEN", "KC", "2026-11-29T18:00:00Z"),
        row("w12_b", 12, "MIA", "BUF", "2026-11-29T18:00:00Z"),
        row("w13_a", 13, "KC", "BUF", "2026-12-06T18:00:00Z"),
    ])


# --- capture_opening -------------------------------------------------------

def test_capture_writes_current_week_with_book_quote_and_regime():
    updates, summary = plan_opening_capture(base_sheet(), SPREADS, COL, schedule(), NOW)
    by_row = cells_by_row(updates)
    assert set(by_row) == {3, 4}  # week-12 rows (sheet idx 1, 2)
    cells = by_row[3]
    assert cells["status"] == "OPEN_CAPTURED"
    assert cells["opening_line"] == -3.0
    assert cells["opening_book"] == "DraftKings"
    assert cells["opening_quote_at"] == "2026-11-24T15:00:00Z"
    assert cells["decision_regime"] == "in_week"
    assert summary["decision_week"] == 12


def test_capture_writes_nothing_for_later_week_game():
    updates, _ = plan_opening_capture(base_sheet(), SPREADS, COL, schedule(), NOW)
    assert 5 not in cells_by_row(updates)  # week-13 row


def test_capture_writes_nothing_when_a_result_is_missing():
    late = ("w11_e", 2026, 11, "NYG", "DAL", np.nan, np.nan, pd.Timestamp("2026-11-24 01:15", tz="UTC"))
    updates, summary = plan_opening_capture(base_sheet(), SPREADS, COL, schedule([late]), NOW)
    assert updates == []
    assert summary["missing_results"] == ["w11_e"]


def test_capture_skips_live_market():
    spreads = SPREADS.copy()
    spreads.loc[spreads["home_team"] == "DEN", "market_last_update"] = "2026-11-29T18:30:00Z"
    updates, _ = plan_opening_capture(base_sheet(), spreads, COL, schedule(), NOW)
    assert set(cells_by_row(updates)) == {4}


def test_capture_total_line_fill_restricted_to_decidable_rows():
    df = base_sheet()
    df.loc[3, ["status", "opening_line", "opening_odds", "opening_captured_at"]] = ["OPEN_CAPTURED", -1.0, -110, "2026-11-17T15:20:00Z"]
    updates, _ = plan_opening_capture(df, SPREADS, COL, schedule(), NOW)
    assert 5 not in cells_by_row(updates)


# --- generate_picks --------------------------------------------------------

def captured_sheet(minutes_ago):
    stamp = (NOW - pd.Timedelta(minutes=minutes_ago)).isoformat()
    return sheet([
        row("w12_a", 12, "DEN", "KC", "2026-11-29T18:00:00Z", "OPEN_CAPTURED", -3.0, stamp),
        row("w12_b", 12, "MIA", "BUF", "2026-11-29T18:00:00Z", "OPEN_CAPTURED", -3.0, stamp),
        row("w13_a", 13, "KC", "BUF", "2026-12-06T18:00:00Z", "OPEN_CAPTURED", -3.0, stamp),
    ])


def test_candidate_with_91_minute_old_opening_is_skipped():
    cands, counts = eligible_candidates(captured_sheet(91), schedule(), NOW, max_line_age_minutes=90)
    assert cands == []
    assert counts["stale_line"] == 2


def test_fresh_opening_candidates_are_current_week_with_rest():
    cands, counts = eligible_candidates(captured_sheet(20), schedule(), NOW, max_line_age_minutes=90)
    assert [c["idx"] for c in cands] == [0, 1]
    assert counts["not_decidable"] == 1
    assert (cands[0]["home_rest"], cands[0]["away_rest"]) == (7, 7)


def test_candidates_blocked_when_result_missing():
    late = ("w11_e", 2026, 11, "NYG", "DAL", np.nan, np.nan, pd.Timestamp("2026-11-24 01:15", tz="UTC"))
    cands, counts = eligible_candidates(captured_sheet(20), schedule([late]), NOW, max_line_age_minutes=90)
    assert cands == []
    assert counts["missing_results"] == ["w11_e"]


def test_weekly_cap_counts_existing_picks():
    df = sheet(
        [row("p" + str(i), 12, "X", "Y", "2026-11-29T18:00:00Z", "OPEN_CAPTURED", -3.0, NOW.isoformat(), pick="X") for i in range(3)]
        + [row("q" + str(i), 12, "X", "Y", "2026-11-29T18:00:00Z", "OPEN_CAPTURED", -3.0, NOW.isoformat()) for i in range(4)]
        + [row("old", 11, "X", "Y", "2026-11-22T18:00:00Z", "OPEN_CAPTURED", -3.0, NOW.isoformat(), pick="X")]
    )
    scored = [{"idx": 3 + i, "edge": 0.1 - i * 0.01} for i in range(4)]
    picks = apply_weekly_cap(scored, df, week=12, max_picks=5)
    assert [p["idx"] for p in picks] == [3, 4]


def test_weekly_cap_full_week_adds_nothing():
    df = sheet([row("p" + str(i), 12, "X", "Y", "2026-11-29T18:00:00Z", pick="X") for i in range(5)])
    assert apply_weekly_cap([{"idx": 9, "edge": 0.2}], df, week=12, max_picks=5) == []


def test_every_pick_write_includes_pick_made_at():
    picks = [
        {"idx": i, "pick_team": "KC", "side": "AWAY", "pick_line": 3.0, "pick_odds": -110, "model_p": 0.55,
         "fair_p": 0.5, "edge": 0.03, "used_fallback": False, "mu_home_margin": -1.0, "sigma": 13.5}
        for i in range(3)
    ]
    by_row = cells_by_row(build_pick_updates(picks, COL, NOW))
    assert set(by_row) == {2, 3, 4}
    for cells in by_row.values():
        assert cells["pick_made_at"] == NOW.isoformat()
        assert cells["pick_team"] == "KC"
