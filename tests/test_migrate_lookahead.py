import numpy as np
import pandas as pd

from src.migrate_lookahead_openers import NEW_COLUMNS, plan_lookahead_migration

NOW = pd.Timestamp("2026-11-18 12:00", tz="UTC")  # week 11 is the decision week
COLS = ["game_id", "week", "kickoff_time", "status", "opening_line", "opening_odds",
        "opening_away_odds", "opening_captured_at", "pick_team"]
COL = {c: i + 1 for i, c in enumerate(COLS + NEW_COLUMNS)}
NAME = {v: k for k, v in COL.items()}


def test_moves_unpicked_lookahead_openers_and_labels_picked_ones():
    df = pd.DataFrame([
        ["w11", 11, "2026-11-22T18:00:00Z", "OPEN_CAPTURED", -3.0, -110, -110, "2026-11-10T15:20:00Z", np.nan],
        ["w12_open", 12, "2026-11-29T18:00:00Z", "OPEN_CAPTURED", -2.5, -105, -115, "2026-11-17T15:20:00Z", np.nan],
        ["w12_pick", 12, "2026-11-29T18:00:00Z", "OPEN_CAPTURED", 1.0, -110, -110, "2026-11-17T15:20:00Z", "KC"],
        ["w12_pending", 12, "2026-11-29T18:00:00Z", "PENDING", np.nan, np.nan, np.nan, np.nan, np.nan],
    ], columns=COLS)

    updates, summary = plan_lookahead_migration(df, COL, NOW)
    assert summary["decision_week"] == 11
    assert summary["reset"] == ["w12_open"]
    assert summary["labelled"] == ["w12_pick"]

    by_row = {}
    for u in updates:
        by_row.setdefault(u["row"], {})[NAME[u["col"]]] = u["value"]
    assert set(by_row) == {3, 4}
    assert by_row[3] == {
        "lookahead_line": -2.5, "lookahead_odds": -105, "lookahead_away_odds": -115,
        "lookahead_captured_at": "2026-11-17T15:20:00Z",
        "opening_line": "", "opening_odds": "", "opening_away_odds": "", "opening_captured_at": "",
        "status": "PENDING",
    }
    assert by_row[4] == {"decision_regime": "lookahead"}
