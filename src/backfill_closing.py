"""
Verified Closing Line Backfill Utility for Master_Bets.

Populates verified pre-kickoff closing spreads/odds for games missed due to
runner scheduling delays (e.g., Week 8 TNF 2026_08_CAR_GB) and optionally
refreshes Model_Scorecard via src.compare_models.
"""

import argparse
from datetime import datetime
import pytz

from src.capture_closing import _is_missing_val, build_closing_row_updates
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

VERIFIED_CLOSING_BACKFILLS = [
    {
        "game_id": "2026_08_CAR_GB",
        "closing_line": -3.0,
        "closing_odds": -105.0,
        "closing_away_odds": -115.0,
        "closing_captured_at": "2026-10-29T22:28:15+00:00",
    },
]


def backfill_closing_records(records=None, force: bool = False, refresh_scorecard: bool = False) -> int:
    """
    Applies one or more verified closing line records to Master_Bets.
    Returns the number of rows updated.
    """
    target_records = records if records is not None else VERIFIED_CLOSING_BACKFILLS
    if not target_records:
        print("No backfill records provided.")
        return 0

    sheet_df = get_sheet_as_df()
    if sheet_df.empty:
        print("Master_Bets sheet is empty.")
        return 0

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    if "closing_away_odds" not in col:
        new_col_idx = len(headers) + 1
        if ws.col_count < new_col_idx:
            ws.add_cols(new_col_idx - ws.col_count)
        batch_update_cells(ws, [{"row": 1, "col": new_col_idx, "value": "closing_away_odds"}])
        col["closing_away_odds"] = new_col_idx

    required = ["closing_line", "closing_odds", "closing_captured_at", "status"]
    missing = [c for c in required if c not in col]
    if missing:
        raise ValueError(f"Master_Bets is missing required columns: {missing}")

    row_by_gid = {}
    for idx, row in sheet_df.iterrows():
        gid = str(row.get("game_id", "")).strip()
        if gid:
            row_by_gid[gid] = (idx, row)

    updates = []
    updated_count = 0

    for rec in target_records:
        gid = str(rec["game_id"]).strip()
        if gid not in row_by_gid:
            print(f"Warning: game_id '{gid}' not found in Master_Bets.")
            continue

        idx, row = row_by_gid[gid]
        if not force and not _is_missing_val(row.get("closing_line")):
            print(f"Skipping {gid}: closing_line already populated ({row.get('closing_line')}).")
            continue

        row_num = idx + 2
        row_updates = build_closing_row_updates(
            row=row,
            row_num=row_num,
            col=col,
            home_spread=float(rec["closing_line"]),
            home_odds=float(rec["closing_odds"]),
            away_odds=float(rec["closing_away_odds"]) if rec.get("closing_away_odds") is not None else None,
            captured_at_iso=str(rec["closing_captured_at"]),
        )
        updates.extend(row_updates)
        updated_count += 1
        print(
            f"Backfilling {gid} (row {row_num}, status={row.get('status')}): "
            f"closing_line={rec['closing_line']}, closing_odds={rec['closing_odds']}, "
            f"closing_away_odds={rec.get('closing_away_odds')}, "
            f"closing_captured_at={rec['closing_captured_at']}"
        )

    if updates:
        batch_update_cells(ws, updates)
        print(f"Successfully backfilled closing lines for {updated_count} game(s).")
    else:
        print("No Master_Bets updates needed.")

    if refresh_scorecard:
        from src.compare_models import main as run_compare_models
        run_compare_models()

    return updated_count


def main():
    parser = argparse.ArgumentParser(description="Backfill missing closing lines in Master_Bets.")
    parser.add_argument("--game-id", help="Target game_id (defaults to verified backfill table if omitted)")
    parser.add_argument("--closing-line", type=float, help="Home closing spread (e.g. -3.0)")
    parser.add_argument("--closing-odds", type=float, default=-110.0, help="Home closing American odds")
    parser.add_argument("--closing-away-odds", type=float, default=-110.0, help="Away closing American odds")
    parser.add_argument("--closing-captured-at", help="ISO-8601 UTC timestamp of pre-kickoff odds snapshot")
    parser.add_argument("--force", action="store_true", help="Overwrite existing closing_line if already present")
    parser.add_argument(
        "--refresh-scorecard",
        action="store_true",
        help="Re-run src.compare_models after backfill to refresh Model_Scorecard",
    )
    args = parser.parse_args()

    if args.game_id:
        if args.closing_line is None:
            parser.error("--closing-line is required when --game-id is specified.")
        captured_at = args.closing_captured_at or datetime.now(pytz.utc).isoformat()
        records = [
            {
                "game_id": args.game_id,
                "closing_line": args.closing_line,
                "closing_odds": args.closing_odds,
                "closing_away_odds": args.closing_away_odds,
                "closing_captured_at": captured_at,
            }
        ]
    else:
        records = VERIFIED_CLOSING_BACKFILLS

    backfill_closing_records(
        records=records,
        force=args.force,
        refresh_scorecard=args.refresh_scorecard,
    )


if __name__ == "__main__":
    main()
