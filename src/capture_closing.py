import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytz

from src.injury_guard import check_line_movement
from src.odds_utils import (
    CACHE_DIR,
    fetch_odds_api,
    find_cached_pregame_closing_spread,
    parse_spreads,
)
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

DRY_RUN_CLOSE = os.getenv("DRY_RUN_CLOSE", "").strip().lower() in {"1", "true", "yes", "y"}
ATTEMPTS_FILE = os.path.join(CACHE_DIR, "closing_attempts.json")

# Pre-kickoff live capture tiers:
# 1. Early safety-net window (240m -> 60m pre-kickoff): captures if closing_line is blank so
#    3-4 hour GitHub Actions cron throttling blackouts never leave a game without a close.
# 2. Prime post-inactives window (60m -> 2m pre-kickoff): captures if blank, OR upgrades an
#    early safety-net capture once after 90-minute NFL inactive lists are priced in.
EARLY_CAPTURE_MAX_MINUTES = int(os.getenv("EARLY_CAPTURE_MAX_MINUTES", "240"))
PRIME_CAPTURE_MAX_MINUTES = int(os.getenv("PRIME_CAPTURE_MAX_MINUTES", "60"))
MIN_LIVE_LEAD_MINUTES = int(os.getenv("MIN_LIVE_LEAD_MINUTES", "2"))

# Post-kickoff / post-settlement cache recovery settings:
MAX_CACHE_PRE_KICKOFF_HOURS = float(os.getenv("MAX_CACHE_PRE_KICKOFF_HOURS", "18.0"))
MAX_CATCHUP_AGE_DAYS = int(os.getenv("MAX_CATCHUP_AGE_DAYS", "14"))
MAX_CLOSING_ATTEMPTS = int(os.getenv("MAX_CLOSING_ATTEMPTS", "6"))

PREGAME_STATUSES = {"PENDING", "OPEN_CAPTURED", "CLOSE_CAPTURED"}
POSTGAME_STATUSES = {"RESULT_UPDATED", "SETTLED"}
CATCHUP_ELIGIBLE_STATUSES = PREGAME_STATUSES | POSTGAME_STATUSES


def load_closing_attempts():
    if os.path.exists(ATTEMPTS_FILE):
        try:
            with open(ATTEMPTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_closing_attempts(attempts):
    try:
        os.makedirs(os.path.dirname(ATTEMPTS_FILE), exist_ok=True)
        with open(ATTEMPTS_FILE, "w") as f:
            json.dump(attempts, f, indent=2)
    except Exception:
        pass


def parse_kickoff(value):
    try:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return pd.NaT
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return pd.NaT
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _is_missing_val(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    s = str(value).strip()
    return s == "" or s.lower() == "nan"


def _safe_float(value):
    if _is_missing_val(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def row_needs_closing(row, now_utc, attempts=None):
    """
    Determines whether a game needs a LIVE pre-kickoff closing capture right now.
    - Captures any missing closing_line within [MIN_LIVE_LEAD_MINUTES, EARLY_CAPTURE_MAX_MINUTES]
      before kickoff (default 2m to 240m).
    - Upgrades an early safety-net capture once during the prime post-inactives window
      [MIN_LIVE_LEAD_MINUTES, PRIME_CAPTURE_MAX_MINUTES] (default 2m to 60m).
    """
    gid = str(row.get("game_id", "")).strip()
    if attempts and attempts.get(gid, 0) >= MAX_CLOSING_ATTEMPTS:
        return False

    status = str(row.get("status", "")).strip().upper()
    if status not in PREGAME_STATUSES:
        return False

    kickoff = parse_kickoff(row.get("kickoff_time"))
    if pd.isna(kickoff):
        return False

    delta = kickoff - now_utc
    min_lead = timedelta(minutes=MIN_LIVE_LEAD_MINUTES)
    max_lead = timedelta(minutes=EARLY_CAPTURE_MAX_MINUTES)
    prime_lead = timedelta(minutes=PRIME_CAPTURE_MAX_MINUTES)

    if not (min_lead <= delta <= max_lead):
        return False

    # Missing closing line: capture immediately (either safety-net or prime window)
    if _is_missing_val(row.get("closing_line")):
        return True

    # Already has a closing line: upgrade once if we are now in the prime window
    # and the existing line was captured earlier than PRIME_CAPTURE_MAX_MINUTES pre-kickoff
    # (or if closing_captured_at is missing/blank).
    if delta <= prime_lead:
        captured_at = parse_kickoff(row.get("closing_captured_at"))
        if pd.isna(captured_at) or (kickoff - captured_at) > prime_lead:
            return True

    return False


def row_needs_cache_catchup(row, now_utc):
    """
    Determines whether a game whose live pre-kickoff window has closed is missing its
    closing_line and should be backfilled from verified pre-kickoff cached snapshots.
    Allows RESULT_UPDATED and SETTLED rows so overnight result runs never lock out CLV.
    """
    status = str(row.get("status", "")).strip().upper()
    if status not in CATCHUP_ELIGIBLE_STATUSES:
        return False

    if not _is_missing_val(row.get("closing_line")):
        return False

    kickoff = parse_kickoff(row.get("kickoff_time"))
    if pd.isna(kickoff):
        return False

    delta = kickoff - now_utc
    if delta >= timedelta(minutes=MIN_LIVE_LEAD_MINUTES):
        return False

    if (now_utc - kickoff) > timedelta(days=MAX_CATCHUP_AGE_DAYS):
        return False

    return True


def build_closing_row_updates(
    row,
    row_num: int,
    col: dict,
    home_spread: float,
    home_odds: float,
    away_odds: float,
    captured_at_iso: str,
):
    """
    Builds cell update dicts for populating closing line data on a Master_Bets row.
    - Preserves RESULT_UPDATED / SETTLED status on post-game catch-up or backfill.
    - Computes clv_points / clv_flag when backfilling an already SETTLED bet.
    - Appends RLM_ALERT idempotently and clears stale MISSING_CLOSING_LINE notes.
    """
    cell_map = {}

    def set_col(col_name, val):
        if col_name in col:
            cell_map[col[col_name]] = val

    set_col("closing_line", float(home_spread))
    set_col("closing_odds", float(home_odds))
    if away_odds is not None and not pd.isna(away_odds):
        set_col("closing_away_odds", float(away_odds))
    set_col("closing_captured_at", str(captured_at_iso))

    status = str(row.get("status", "")).strip().upper()
    if status not in POSTGAME_STATUSES:
        set_col("status", "CLOSE_CAPTURED")

    home_team = str(row.get("home_team", "")).strip().upper()
    away_team = str(row.get("away_team", "")).strip().upper()
    pick_team = str(row.get("pick_team", "")).strip().upper()
    pick_side = str(row.get("pick_side", "")).strip().upper()

    if not pick_side and pick_team and pick_team != "NAN":
        if pick_team == home_team:
            pick_side = "HOME"
        elif pick_team == away_team:
            pick_side = "AWAY"

    raw_notes = "" if _is_missing_val(row.get("notes")) else str(row.get("notes")).strip()
    note_parts = [p.strip() for p in raw_notes.split(";") if p.strip() and p.strip() != "MISSING_CLOSING_LINE"]
    notes_changed = raw_notes != "; ".join(note_parts)

    if pick_side in {"HOME", "AWAY"}:
        rlm = check_line_movement(
            open_line=row.get("opening_line"),
            current_line=home_spread,
            pick_side=pick_side,
        )
        if rlm:
            print(f"RLM ALERT: {row.get('game_id')} {rlm['message']}")
            note_str = f"RLM_ALERT: moved {rlm['pts_against']:.1f}pts against {pick_side}"
            if not any(p.startswith("RLM_ALERT:") for p in note_parts):
                note_parts.append(note_str)
                notes_changed = True

        # If this row was already SETTLED without a closing line, backfill realized CLV too
        pick_line = _safe_float(row.get("pick_line"))
        if status == "SETTLED" and pick_line is not None:
            closing_pick_line = float(home_spread) if pick_side == "HOME" else -float(home_spread)
            clv_points = float(pick_line) - closing_pick_line
            clv_flag = bool(clv_points > 0)
            set_col("clv_points", clv_points)
            set_col("clv_flag", clv_flag)

    if notes_changed:
        set_col("notes", "; ".join(note_parts))

    return [{"row": row_num, "col": c_idx, "value": val} for c_idx, val in cell_map.items()]


def select_dry_run_target(sheet_df, now_utc):
    candidates = []

    for idx, row in sheet_df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        if status not in PREGAME_STATUSES:
            continue

        if not _is_missing_val(row.get("closing_line")):
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff > now_utc - timedelta(hours=12):
            candidates.append((kickoff, idx, row))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])

    return [(candidates[0][1], candidates[0][2])]


def _emit_github_output(key: str, value: str):
    gh_out = os.getenv("GITHUB_OUTPUT")
    if not gh_out:
        return
    try:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    except Exception:
        pass


def capture_closing(cache_dir: str = None):
    print("Running closing line monitor...")

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    now_utc = datetime.now(pytz.utc)
    print(f"Current UTC: {now_utc.isoformat()}")

    attempts = load_closing_attempts()
    catchup_targets = []
    normal_targets = []

    for idx, row in sheet_df.iterrows():
        if row_needs_cache_catchup(row, now_utc):
            catchup_targets.append((idx, row))
        elif row_needs_closing(row, now_utc, attempts=attempts):
            normal_targets.append((idx, row))

    # Step 1: Recover pre-kickoff cached closing lines BEFORE any live API fetch
    # so fresh in-game fetches never overwrite cached pre-game snapshots first.
    resolved_catchups = []
    for idx, row in catchup_targets:
        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()
        kickoff = parse_kickoff(row.get("kickoff_time"))
        cached_mkt = find_cached_pregame_closing_spread(
            home_team=home,
            away_team=away,
            kickoff_utc=kickoff,
            cache_dir=cache_dir,
            max_pre_kickoff_hours=MAX_CACHE_PRE_KICKOFF_HOURS,
        )
        if cached_mkt is not None:
            resolved_catchups.append((idx, row, cached_mkt))
        else:
            print(f"Catch-up skipped for {away} @ {home}: no pre-kickoff snapshot within {MAX_CACHE_PRE_KICKOFF_HOURS:g}h window in cache.")

    dry_targets = []
    if DRY_RUN_CLOSE:
        if normal_targets:
            dry_targets = normal_targets
        elif not resolved_catchups:
            dry_targets = select_dry_run_target(sheet_df, now_utc)
            if dry_targets:
                print("DRY_RUN_CLOSE enabled. Selecting nearest upcoming game without writing to sheet.")

    if not resolved_catchups and not normal_targets and not dry_targets:
        print("No games need closing capture.")
        return

    spreads = pd.DataFrame()
    if normal_targets or dry_targets:
        force_refresh = DRY_RUN_CLOSE and not normal_targets
        try:
            odds_data = fetch_odds_api(max_age_seconds=900, force_refresh=force_refresh)
            spreads = parse_spreads(odds_data)
        except Exception as exc:
            print(f"Warning: Live Odds API fetch unavailable ({exc}). Proceeding with any cached catch-ups.")
            spreads = pd.DataFrame()
        if spreads.empty and not resolved_catchups:
            print("No spreads parsed from Odds API.")
            return

    if DRY_RUN_CLOSE:
        print("Dry run only. The sheet will not be updated.")
        for idx, row, cached_mkt in resolved_catchups:
            home = str(row.get("home_team", "")).strip().upper()
            away = str(row.get("away_team", "")).strip().upper()
            print(
                f"Would backfill pre-kickoff cached closing for {away} @ {home}: "
                f"home spread {cached_mkt['home_spread']}, home odds {cached_mkt['home_odds']}, "
                f"away odds {cached_mkt['away_odds']}, captured_at {cached_mkt['captured_at_utc']} "
                f"(source {cached_mkt.get('cache_source', 'cache')})."
            )

        for idx, row in dry_targets:
            home = str(row.get("home_team", "")).strip().upper()
            away = str(row.get("away_team", "")).strip().upper()
            mask = (spreads["home_team"] == home) & (spreads["away_team"] == away) if not spreads.empty else pd.Series(dtype=bool)
            match = spreads[mask] if not spreads.empty else pd.DataFrame()

            if match.empty:
                print("Would not capture: no odds match for " + away + " @ " + home + ".")
            else:
                odds_row = match.iloc[0]
                print(
                    "Would capture closing for "
                    + away
                    + " @ "
                    + home
                    + ": home spread "
                    + str(odds_row["home_spread"])
                    + ", home odds "
                    + str(odds_row["home_odds"])
                    + ", away odds "
                    + str(odds_row["away_odds"])
                    + ", book "
                    + str(odds_row["bookmaker"])
                    + "."
                )
        return

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    required = [
        "closing_line",
        "closing_odds",
        "closing_captured_at",
        "status",
    ]

    missing = [c for c in required if c not in col]
    if missing:
        print("Missing required columns: " + str(missing))
        return

    updates = []
    games_updated = 0
    postgame_backfilled = 0

    # Apply pre-kickoff cache catch-up updates
    for idx, row, cached_mkt in resolved_catchups:
        gid = str(row.get("game_id", "")).strip()
        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()
        status = str(row.get("status", "")).strip().upper()
        row_num = idx + 2

        if gid in attempts:
            del attempts[gid]

        row_updates = build_closing_row_updates(
            row=row,
            row_num=row_num,
            col=col,
            home_spread=cached_mkt["home_spread"],
            home_odds=cached_mkt["home_odds"],
            away_odds=cached_mkt.get("away_odds"),
            captured_at_iso=cached_mkt["captured_at_utc"],
        )
        updates.extend(row_updates)
        games_updated += 1
        if status in POSTGAME_STATUSES:
            postgame_backfilled += 1
        print(
            f"Recovered pre-kickoff closing line for {away} @ {home} ({gid}) from "
            f"{cached_mkt.get('cache_source', 'cache')} @ {cached_mkt['captured_at_utc']}: "
            f"{home} {cached_mkt['home_spread']:+.1f} ({cached_mkt['home_odds']:g} / {cached_mkt['away_odds']:g})"
        )

    # Apply live pre-kickoff closing updates
    if not spreads.empty:
        for idx, row in normal_targets:
            gid = str(row.get("game_id", "")).strip()
            home = str(row.get("home_team", "")).strip().upper()
            away = str(row.get("away_team", "")).strip().upper()
            kickoff = parse_kickoff(row.get("kickoff_time"))

            mask = (spreads["home_team"] == home) & (spreads["away_team"] == away)
            match = spreads[mask]

            if match.empty:
                attempts[gid] = attempts.get(gid, 0) + 1
                print(f"Unmatched game {away} @ {home} (attempt {attempts[gid]}/{MAX_CLOSING_ATTEMPTS}).")
                continue

            odds_row = match.iloc[0]

            # Guard against early kickoffs / live markets sneaking into pre-game capture
            mkt_lu = parse_kickoff(odds_row.get("market_last_update"))
            if not pd.isna(mkt_lu) and not pd.isna(kickoff) and mkt_lu > kickoff:
                print(f"Skipping live capture for {away} @ {home}: market_last_update ({mkt_lu.isoformat()}) is after kickoff.")
                continue

            if gid in attempts:
                del attempts[gid]

            captured_ts = mkt_lu.isoformat() if not pd.isna(mkt_lu) else datetime.now(pytz.utc).isoformat()
            row_num = idx + 2

            row_updates = build_closing_row_updates(
                row=row,
                row_num=row_num,
                col=col,
                home_spread=odds_row["home_spread"],
                home_odds=odds_row["home_odds"],
                away_odds=odds_row.get("away_odds"),
                captured_at_iso=captured_ts,
            )
            updates.extend(row_updates)
            games_updated += 1

    save_closing_attempts(attempts)
    batch_update_cells(ws, updates)

    if postgame_backfilled > 0:
        _emit_github_output("settled_backfilled", "true")

    print("Captured closing lines for " + str(games_updated) + " games.")


if __name__ == "__main__":
    capture_closing()
