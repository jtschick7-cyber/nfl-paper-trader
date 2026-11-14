"""One-off: move look-ahead openers out of the opening columns (R1 migration step).

Needed only if openers for a week later than the current decision week were captured
before R1 deployed. For OPEN_CAPTURED rows with a future kickoff in a later week:

- no pick: copy opening_line/odds/away_odds/captured_at into lookahead_* columns,
  blank the four opening cells, set status PENDING (the next in-week run re-captures);
- pick: leave the row alone and set decision_regime = lookahead.

Dry-run by default; writes only with --apply. Run it inside the sheet-writer
concurrency group (the "Migrate look-ahead openers" workflow), never alongside
another Master_Bets writer.
"""

import argparse
from datetime import datetime

import pandas as pd
import pytz

from src.decision_window import sheet_decision_week
from src.generate_picks import has_existing_pick

OPENING_TO_LOOKAHEAD = {
    "opening_line": "lookahead_line",
    "opening_odds": "lookahead_odds",
    "opening_away_odds": "lookahead_away_odds",
    "opening_captured_at": "lookahead_captured_at",
}
NEW_COLUMNS = list(OPENING_TO_LOOKAHEAD.values()) + ["decision_regime"]


def _blank(value):
    return pd.isna(value) or str(value).strip() == "" or str(value).strip().lower() == "nan"


def plan_lookahead_migration(sheet_df, col, now_utc):
    """(updates, summary). Pure. `col` must already include NEW_COLUMNS."""
    decision_week = sheet_decision_week(sheet_df, now_utc)
    summary = {"decision_week": decision_week, "reset": [], "labelled": []}
    updates = []
    if decision_week is None:
        return updates, summary

    for idx, row in sheet_df.iterrows():
        if str(row.get("status", "")).strip().upper() != "OPEN_CAPTURED":
            continue
        kickoff = pd.to_datetime(row.get("kickoff_time"), utc=True, errors="coerce")
        week = pd.to_numeric(row.get("week"), errors="coerce")
        if pd.isna(kickoff) or kickoff <= now_utc or pd.isna(week) or int(week) <= decision_week:
            continue

        r = idx + 2
        gid = str(row.get("game_id", ""))

        if has_existing_pick(row.get("pick_team")):
            updates.append({"row": r, "col": col["decision_regime"], "value": "lookahead"})
            summary["labelled"].append(gid)
            continue

        for src_name, dst_name in OPENING_TO_LOOKAHEAD.items():
            if src_name not in col:
                continue
            value = row.get(src_name)
            updates.append({"row": r, "col": col[dst_name], "value": "" if _blank(value) else value})
            updates.append({"row": r, "col": col[src_name], "value": ""})
        updates.append({"row": r, "col": col["status"], "value": "PENDING"})
        summary["reset"].append(gid)

    return updates, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write to the Sheet (default: dry-run)")
    args = parser.parse_args()

    from src.sheet_utils import batch_update_cells, ensure_columns, get_master_sheet, get_sheet_as_df

    ws = get_master_sheet()
    sheet_df = get_sheet_as_df(ws)
    headers = ws.row_values(1)

    if args.apply:
        col = ensure_columns(ws, NEW_COLUMNS)
    else:
        col = {h: i + 1 for i, h in enumerate(headers)}
        for name in NEW_COLUMNS:
            if name not in col:
                col[name] = len(col) + 1
                print("Would add header: " + name)

    updates, summary = plan_lookahead_migration(sheet_df, col, datetime.now(pytz.utc))
    print("Decision week: " + str(summary["decision_week"]))
    print("Reset to PENDING (openers moved to lookahead_*): " + str(summary["reset"]))
    print("Left alone, decision_regime=lookahead (already picked): " + str(summary["labelled"]))

    if not args.apply:
        print("Dry run: " + str(len(updates)) + " cell update(s) planned. Re-run with --apply to write.")
        return

    batch_update_cells(ws, updates)
    print("Applied " + str(len(updates)) + " cell update(s).")


if __name__ == "__main__":
    main()
