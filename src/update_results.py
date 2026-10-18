import pandas as pd
import nfl_data_py as nfl

from src.config import FLAT_BET_SIZE, SEASON
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def profit_units(odds, stake):
    odds = float(odds)

    if odds > 0:
        return stake * odds / 100.0

    if odds < 0:
        return stake * 100.0 / (-odds)

    return 0.0


def update_results():
    print("Updating results...")

    sheet_df = get_sheet_as_df()
    if sheet_df.empty:
        print("Sheet is empty.")
        return

    sched = nfl.import_schedules([SEASON])
    sched = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()

    scores = {}
    for _, game in sched.iterrows():
        try:
            scores[str(game["game_id"])] = (int(game["home_score"]), int(game["away_score"]))
        except Exception:
            continue

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {header: idx + 1 for idx, header in enumerate(headers)}

    required_cols = [
        "game_result",
        "pick_result",
        "clv_points",
        "clv_flag",
        "roi_units",
        "status",
        "notes",
    ]

    missing = [c for c in required_cols if c not in col]
    if missing:
        print("Missing required columns in sheet: " + str(missing))
        return

    updates = []
    settled_count = 0
    updated_count = 0

    for idx, row in sheet_df.iterrows():
        game_id = str(row.get("game_id", "")).strip()
        if not game_id or game_id.lower() == "nan":
            continue

        if game_id not in scores:
            continue

        status = str(row.get("status", "")).strip().upper()
        if status in {"SETTLED", "RESULT_UPDATED", "ERROR"}:
            continue

        home_score, away_score = scores[game_id]
        row_num = idx + 2

        home_team = str(row.get("home_team", "")).strip().upper()
        away_team = str(row.get("away_team", "")).strip().upper()

        result_text = home_team + " " + str(home_score) + ", " + away_team + " " + str(away_score)
        updates.append({"row": row_num, "col": col["game_result"], "value": result_text})

        pick_team = str(row.get("pick_team", "")).strip().upper()

        if not pick_team or pick_team == "NAN":
            updates.append({"row": row_num, "col": col["status"], "value": "RESULT_UPDATED"})
            updated_count += 1
            continue

        pick_side = str(row.get("pick_side", "")).strip().upper()
        pick_line = safe_float(row.get("pick_line"))
        pick_odds = safe_float(row.get("pick_odds"), -110.0)

        if pick_line is None:
            updates.append({"row": row_num, "col": col["notes"], "value": "MISSING_PICK_LINE"})
            updates.append({"row": row_num, "col": col["status"], "value": "ERROR"})
            continue

        if pick_team == home_team:
            margin = home_score - away_score
            if not pick_side:
                pick_side = "HOME"
        else:
            margin = away_score - home_score
            if not pick_side:
                pick_side = "AWAY"

        cover = margin + pick_line

        if cover > 0:
            pick_result = "WIN"
            roi = profit_units(pick_odds, FLAT_BET_SIZE)
        elif cover < 0:
            pick_result = "LOSS"
            roi = -FLAT_BET_SIZE
        else:
            pick_result = "PUSH"
            roi = 0.0

        closing_line = safe_float(row.get("closing_line"))
        clv_points = ""
        clv_flag = ""

        if closing_line is not None:
            if pick_side == "HOME":
                closing_pick_line = closing_line
            else:
                closing_pick_line = -closing_line

            clv_points = float(pick_line) - float(closing_pick_line)
            clv_flag = bool(clv_points > 0)
        else:
            updates.append({"row": row_num, "col": col["notes"], "value": "MISSING_CLOSING_LINE"})

        updates.append({"row": row_num, "col": col["pick_result"], "value": pick_result})
        updates.append({"row": row_num, "col": col["clv_points"], "value": clv_points})
        updates.append({"row": row_num, "col": col["clv_flag"], "value": clv_flag})
        updates.append({"row": row_num, "col": col["roi_units"], "value": roi})
        updates.append({"row": row_num, "col": col["status"], "value": "SETTLED"})

        settled_count += 1

    batch_update_cells(ws, updates)

    print(
        "Updated "
        + str(updated_count)
        + " no-pick games and settled "
        + str(settled_count)
        + " paper bets."
    )


if __name__ == "__main__":
    update_results()
