import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytz

from src.config import (
    FLAT_BET_SIZE,
    MAX_AUTO_EDGE,
    MIN_EDGE_THRESHOLD,
    SEASON,
)
from src.decision_window import (
    decision_guard,
    previous_game_day,
    print_missing_results,
    rest_days,
    row_is_decidable,
)
from src.injury_guard import audit_candidate_pick
from src.key_numbers import is_integer_spread
from src.model_utils import DIVISION_MAP, _load_sched, compute_snapshot, cover_probs
from src.odds_utils import american_to_implied_prob, get_cached_live_home_spread, remove_vig
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

MAX_PICKS = int(os.getenv("MAX_PICKS", "5"))  # per week, not per run
PICK_MAX_LINE_AGE_MINUTES = float(os.getenv("PICK_MAX_LINE_AGE_MINUTES", "90"))


def load_model():
    model = joblib.load("models/nfl_spread_model.pkl")

    with open("models/metadata.json") as f:
        meta = json.load(f)

    return model, meta


def parse_kickoff(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def has_existing_pick(value):
    if pd.isna(value):
        return False

    s = str(value).strip()

    if s == "":
        return False

    if s.lower() == "nan":
        return False

    return True


def line_age_minutes(captured_at, now_utc):
    ts = parse_kickoff(captured_at)
    if pd.isna(ts):
        return None
    return (pd.Timestamp(now_utc) - ts).total_seconds() / 60.0


def eligible_candidates(sheet_df, schedule_df, now_utc, max_line_age_minutes=PICK_MAX_LINE_AGE_MINUTES):
    """Rows v1 may pick this run, with v1 rest days. Pure: no model, no I/O.

    A candidate is OPEN_CAPTURED with an opening line, has no pick yet, is decidable,
    and its opening was captured within max_line_age_minutes (an older line means the
    decision moment has passed). Rest is integer Eastern days from each team's latest
    scheduled game this season; a team with none skips the game.
    """
    decision_week, missing = decision_guard(sheet_df, schedule_df, now_utc)
    counts = {
        "decision_week": decision_week,
        "missing_results": missing,
        "existing_pick": 0,
        "past": 0,
        "not_decidable": 0,
        "stale_line": 0,
        "no_previous_game": 0,
    }
    if missing:
        return [], counts

    status = (
        sheet_df.get("status", pd.Series("", index=sheet_df.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )
    opening_line_exists = sheet_df.get("opening_line", pd.Series(np.nan, index=sheet_df.index)).notna()

    out = []
    for idx, row in sheet_df[(status == "OPEN_CAPTURED") & opening_line_exists].iterrows():
        if has_existing_pick(row.get("pick_team")):
            counts["existing_pick"] += 1
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))
        if pd.isna(kickoff):
            continue
        if kickoff <= now_utc:
            counts["past"] += 1
            continue

        if not row_is_decidable(row, now_utc, decision_week):
            counts["not_decidable"] += 1
            continue

        age = line_age_minutes(row.get("opening_captured_at"), now_utc)
        if age is None or age > max_line_age_minutes:
            counts["stale_line"] += 1
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()
        home_rest = rest_days(previous_game_day(schedule_df, home, kickoff, season=SEASON), kickoff)
        away_rest = rest_days(previous_game_day(schedule_df, away, kickoff, season=SEASON), kickoff)
        if home_rest is None or away_rest is None or home_rest < 0 or away_rest < 0:
            counts["no_previous_game"] += 1
            continue

        out.append({
            "idx": idx,
            "row": row,
            "kickoff": kickoff,
            "week": decision_week,
            "home": home,
            "away": away,
            "home_rest": home_rest,
            "away_rest": away_rest,
        })

    return out, counts


def picks_recorded_for_week(sheet_df, week):
    if week is None or "week" not in sheet_df.columns or "pick_team" not in sheet_df.columns:
        return 0
    weeks = pd.to_numeric(sheet_df["week"], errors="coerce")
    has_pick = sheet_df["pick_team"].apply(has_existing_pick)
    return int(((weeks == int(week)) & has_pick).sum())


def apply_weekly_cap(scored, sheet_df, week, max_picks=MAX_PICKS):
    """MAX_PICKS is a weekly cap: allowed this run = max_picks - picks already on the week."""
    allowed = max(0, int(max_picks) - picks_recorded_for_week(sheet_df, week))
    ranked = sorted(scored, key=lambda x: x["edge"], reverse=True)
    return ranked[:allowed]


def build_pick_updates(picks, col, now_utc):
    made_at = pd.Timestamp(now_utc).isoformat()
    updates = []

    for p in picks:
        r = p["idx"] + 2

        note = "flat " + str(FLAT_BET_SIZE) + "u"

        if p["used_fallback"]:
            note += "; away_odds_fallback=-110"

        note += "; mu=" + str(round(p["mu_home_margin"], 2))
        note += "; sigma=" + str(round(p["sigma"], 2))

        updates.append({"row": r, "col": col["pick_team"], "value": p["pick_team"]})
        updates.append({"row": r, "col": col["pick_side"], "value": p["side"]})
        updates.append({"row": r, "col": col["pick_line"], "value": p["pick_line"]})
        updates.append({"row": r, "col": col["pick_odds"], "value": p["pick_odds"]})
        updates.append({"row": r, "col": col["model_probability"], "value": round(p["model_p"], 4)})
        updates.append({"row": r, "col": col["implied_probability_no_vig"], "value": round(p["fair_p"], 4)})
        updates.append({"row": r, "col": col["edge"], "value": round(p["edge"], 4)})
        updates.append({"row": r, "col": col["notes"], "value": note})
        updates.append({"row": r, "col": col["pick_made_at"], "value": made_at})

    return updates


def main():
    print("Generating weekly picks...")

    if not os.path.exists("models/nfl_spread_model.pkl"):
        raise SystemExit(
            "No model found. Run python -m src.build_features and python -m src.train_model first."
        )

    model, meta = load_model()
    feature_cols = meta["feature_cols"]

    sheet_df = get_sheet_as_df()

    if sheet_df.empty:
        print("Sheet is empty.")
        return

    now_utc = datetime.now(pytz.utc)
    snap = compute_snapshot(now_utc, [SEASON])

    if not snap:
        print("No completed games with enough rolling stats yet. No picks generated.")
        return

    schedule_df = _load_sched([SEASON])
    candidates, counts = eligible_candidates(sheet_df, schedule_df, now_utc)

    if counts["missing_results"]:
        print_missing_results(counts["missing_results"])
        return

    decision_week = counts["decision_week"]
    print("Decision week: " + str(decision_week))

    if counts["stale_line"]:
        print(
            "Skipped " + str(counts["stale_line"]) + " game(s) whose opening line is older than "
            + str(PICK_MAX_LINE_AGE_MINUTES) + " minutes (decision moment passed)."
        )
    if counts["not_decidable"]:
        print("Skipped " + str(counts["not_decidable"]) + " game(s) outside the decision week.")
    if counts["no_previous_game"]:
        print("Skipped " + str(counts["no_previous_game"]) + " game(s) where a team has no previous game this season.")

    if not candidates:
        print("No decidable OPEN_CAPTURED games with fresh opening lines.")
        return

    scored = []
    manual_review = []
    skipped_past = counts["past"]
    skipped_existing_pick = counts["existing_pick"]
    skipped_missing_snapshot = 0

    for cand in candidates:
        idx = cand["idx"]
        row = cand["row"]
        home = cand["home"]
        away = cand["away"]
        home_rest = cand["home_rest"]
        away_rest = cand["away_rest"]

        if home not in snap or away not in snap:
            skipped_missing_snapshot += 1
            continue

        try:
            home_spread = float(row.get("opening_line"))
            home_odds = float(row.get("opening_odds"))
        except Exception:
            continue

        away_odds = -110.0
        used_fallback = True

        if "opening_away_odds" in sheet_df.columns:
            away_odds_value = row.get("opening_away_odds")

            try:
                parsed_away_odds = float(away_odds_value)

                if not pd.isna(parsed_away_odds):
                    away_odds = parsed_away_odds
                    used_fallback = False
            except Exception:
                pass

        division = 1 if DIVISION_MAP.get(home) == DIVISION_MAP.get(away) else 0

        vals = {
            "diff_pf": snap[home]["pf"] - snap[away]["pf"],
            "diff_pa": snap[home]["pa"] - snap[away]["pa"],
            "diff_margin": snap[home]["margin"] - snap[away]["margin"],
            "diff_rest": float(home_rest - away_rest),
            "division": float(division),
        }

        X = np.array([[vals.get(c, 0.0) for c in feature_cols]], dtype=float)

        mu = float(model.predict(X)[0])
        sigma = float(meta.get("sigma", 1.0))

        p_home, p_away = cover_probs(mu, sigma, home_spread)

        fair_home, fair_away = remove_vig(
            american_to_implied_prob(home_odds),
            american_to_implied_prob(away_odds),
        )

        # Edge is measured against the break-even probability of the price actually
        # taken (vig included), not the no-vig fair probability, so edge > 0 means +EV.
        edge_home = p_home - american_to_implied_prob(home_odds)
        edge_away = p_away - american_to_implied_prob(away_odds)

        if edge_home >= edge_away:
            side = "HOME"
            pick_team = home
            pick_line = home_spread
            pick_odds = home_odds
            model_p = p_home
            fair_p = fair_home
            edge = edge_home
        else:
            side = "AWAY"
            pick_team = away
            pick_line = -home_spread
            pick_odds = away_odds
            model_p = p_away
            fair_p = fair_away
            edge = edge_away

        if not np.isfinite(edge):
            continue

        if edge < MIN_EDGE_THRESHOLD:
            continue

        pick_record = {
            "idx": idx,
            "pick_team": pick_team,
            "side": side,
            "pick_line": pick_line,
            "pick_odds": pick_odds,
            "model_p": model_p,
            "fair_p": fair_p,
            "edge": edge,
            "matchup": away + " @ " + home,
            "used_fallback": used_fallback,
            "mu_home_margin": mu,
            "sigma": sigma,
        }

        live_spread = get_cached_live_home_spread(home, away)
        curr_line = float(row.get("current_line")) if pd.notna(row.get("current_line")) else (live_spread if live_spread is not None else home_spread)

        alert = audit_candidate_pick(
            pick_team=pick_team,
            pick_side=side,
            open_line=home_spread,
            current_line=curr_line,
            mu_home_margin=mu,
            edge=edge,
            max_auto_edge=MAX_AUTO_EDGE,
        )

        pick_record["integrity_alert"] = alert

        if alert.recommendation == "SUPPRESS":
            print("  SUPPRESSED pick: " + away + " @ " + home + " | " + "; ".join(alert.flags))
            continue
        elif alert.recommendation == "MANUAL_REVIEW":
            manual_review.append(pick_record)
        else:
            scored.append(pick_record)

    scored.sort(key=lambda x: x["edge"], reverse=True)
    manual_review.sort(key=lambda x: x["edge"], reverse=True)

    already = picks_recorded_for_week(sheet_df, decision_week)
    picks = apply_weekly_cap(scored, sheet_df, decision_week, MAX_PICKS)
    print("Week " + str(decision_week) + ": " + str(already) + " pick(s) already recorded, weekly cap " + str(MAX_PICKS) + ".")

    if skipped_past:
        print("Skipped " + str(skipped_past) + " past game(s).")

    if skipped_existing_pick:
        print("Skipped " + str(skipped_existing_pick) + " game(s) that already had a pick.")

    if skipped_missing_snapshot:
        print("Skipped " + str(skipped_missing_snapshot) + " game(s) missing rolling team snapshots.")

    if manual_review:
        print("")
        print("Manual review only. These edges are above MAX_AUTO_EDGE=" + str(MAX_AUTO_EDGE) + ":")

        for p in manual_review:
            print(
                "  REVIEW: "
                + p["matchup"]
                + " | "
                + p["pick_team"]
                + " "
                + p["side"]
                + " "
                + str(p["pick_line"])
                + " @"
                + str(p["pick_odds"])
                + " | model "
                + str(round(p["model_p"] * 100, 1))
                + "%"
                + " vs fair "
                + str(round(p["fair_p"] * 100, 1))
                + "%"
                + " | edge "
                + str(round(p["edge"] * 100, 1))
                + "%"
                + " | predicted home margin "
                + str(round(p["mu_home_margin"], 2))
            )

        print("")

    if not picks:
        print("No auto-picks cleared the edge filter.")
        return

    ws = get_master_sheet()
    headers = ws.row_values(1)
    col = {h: i + 1 for i, h in enumerate(headers)}

    if "pick_made_at" not in col:
        raise SystemExit("Master_Bets has no pick_made_at column. Run python -m src.ensure_away_odds_columns first.")

    batch_update_cells(ws, build_pick_updates(picks, col, now_utc))

    print("Auto-recommended " + str(len(picks)) + " picks:")

    for p in picks:
        print(
            "  "
            + p["matchup"]
            + " | "
            + p["pick_team"]
            + " "
            + p["side"]
            + " "
            + str(p["pick_line"])
            + " @"
            + str(p["pick_odds"])
            + " | model "
            + str(round(p["model_p"] * 100, 1))
            + "%"
            + " vs fair "
            + str(round(p["fair_p"] * 100, 1))
            + "%"
            + " | edge "
            + str(round(p["edge"] * 100, 1))
            + "%"
        )


if __name__ == "__main__":
    main()
