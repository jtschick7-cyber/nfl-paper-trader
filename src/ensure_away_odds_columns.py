from src.sheet_utils import batch_update_cells, get_master_sheet


def main():
    ws = get_master_sheet()
    headers = ws.row_values(1)

    existing = {h: i + 1 for i, h in enumerate(headers)}

    updates = []
    next_col = len(headers) + 1

    for name in ["opening_away_odds", "closing_away_odds"]:
        if name not in existing:
            updates.append({
                "row": 1,
                "col": next_col,
                "value": name,
            })
            existing[name] = next_col
            next_col += 1

    if updates:
        batch_update_cells(ws, updates)
        print("Added " + str(len(updates)) + " header(s) to Master_Bets.")
    else:
        print("Away odds headers already exist.")

    print("Current headers:")
    print(list(existing.keys()))


if __name__ == "__main__":
    main()
