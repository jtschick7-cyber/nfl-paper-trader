from src.sheet_utils import ensure_columns, get_master_sheet

# Master_Bets columns added after the original schema; created here (a sheet-writer job)
# so the long-lived closing loop only ever writes to headers that already exist.
EXTRA_COLUMNS = ["opening_away_odds", "closing_away_odds", "total_line", "closing_consensus_line"]


def main():
    ws = get_master_sheet()
    before = len(ws.row_values(1))
    existing = ensure_columns(ws, EXTRA_COLUMNS)

    added = len(existing) - before
    if added:
        print("Added " + str(added) + " header(s) to Master_Bets.")
    else:
        print("Extra Master_Bets headers already exist.")

    print("Current headers:")
    print(list(existing.keys()))


if __name__ == "__main__":
    main()
