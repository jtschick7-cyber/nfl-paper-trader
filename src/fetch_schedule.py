from datetime import datetime, time as dtime
import pytz
import pandas as pd
from src.sheet_utils import get_sheet_as_df, get_master_sheet, batch_update_cells
from src.config import SEASON, LOCAL_TZ

def kickoff_iso(game):
    """Build a true UTC kickoff timestamp from nflverse gameday + gametime (Eastern), DST-safe."""
    gameday = pd.to_datetime(game['gameday']).date()
    gt = game.get('gametime')
    if pd.notna(gt) and str(gt).strip():
        t = pd.to_datetime(str(gt).strip()).time()
    else:
        t = dtime(13, 0)  # fallback: 1:00 PM Eastern
    tz = pytz.timezone(LOCAL_TZ)
    dt_combined = datetime.combine(gameday, t)
    try:
        local = tz.localize(dt_combined, is_dst=None)
    except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
        local = tz.localize(dt_combined, is_dst=False)
    return local.astimezone(pytz.utc).isoformat()

def sync_schedule():
    print(f"Fetching NFL schedule for {SEASON}...")
    try:
        from src.model_utils import _load_sched
        sched = _load_sched([SEASON])
    except Exception:
        import nfl_data_py as nfl
        sched = nfl.import_schedules([SEASON])
    print(f"Schedule provider returned {len(sched)} games for season {SEASON}.")

    sheet_df = get_sheet_as_df()
    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}
    ko_col = col.get("kickoff_time")

    existing_map = {}
    if not sheet_df.empty and 'game_id' in sheet_df.columns:
        for idx, r in sheet_df.iterrows():
            gid = str(r['game_id']).strip()
            existing_map[gid] = {
                "row_num": idx + 2,  # Sheet row is 1-based, header is row 1
                "status": str(r.get("status", "")).strip().upper(),
                "kickoff_time": str(r.get("kickoff_time", "")).strip(),
            }

    new_rows = []
    flex_updates = []

    for _, game in sched.iterrows():
        gsis_id = str(game['game_id']).strip()
        new_ko = kickoff_iso(game)

        if gsis_id in existing_map:
            info = existing_map[gsis_id]
            if info["status"] in {"PENDING", "OPEN_CAPTURED"} and ko_col:
                curr_ko_dt = pd.to_datetime(info["kickoff_time"], utc=True, errors="coerce")
                new_ko_dt = pd.to_datetime(new_ko, utc=True, errors="coerce")
                if pd.notna(curr_ko_dt) and pd.notna(new_ko_dt) and abs((new_ko_dt - curr_ko_dt).total_seconds()) > 60:
                    print(f"Flex schedule change detected for {gsis_id}: {info['kickoff_time']} -> {new_ko}")
                    flex_updates.append({
                        "row": info["row_num"],
                        "col": ko_col,
                        "value": new_ko,
                    })
            continue

        row = {
            'season': int(game['season']),
            'week': int(game['week']),
            'game_id': gsis_id,
            'kickoff_time': new_ko,
            'home_team': str(game['home_team']).upper(),
            'away_team': str(game['away_team']).upper(),
            'status': 'PENDING',
        }
        new_rows.append([row.get(h, '') for h in headers])

    if flex_updates:
        print(f"Applying {len(flex_updates)} flex schedule kickoff updates...")
        batch_update_cells(ws, flex_updates)

    if new_rows:
        print(f"Adding {len(new_rows)} new games to sheet.")
        ws.append_rows(new_rows, value_input_option='USER_ENTERED')
    else:
        print("No new games to add.")

if __name__ == "__main__":
    sync_schedule()