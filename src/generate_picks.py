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
from src.injury_guard import audit_candidate_pick
from src.key_numbers import is_integer_spread
from src.model_utils import DIVISION_MAP, compute_snapshot, cover_probs
from src.odds_utils import american_to_implied_prob, remove_vig
from src.sheet_utils import batch_update_cells, get_master_sheet, get_sheet_as_df

MAX_PICKS = int(os.getenv("MAX_PICKS", "5"))


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

    status = (
        sheet_df.get("status", pd.Series("", index=sheet_df.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )

    opening_line_exists = (
        sheet_df.get("opening_line", pd.Series(np.nan, index=sheet_df.index))
        .notna()
    )

    candidates = sheet_df[(status == "OPEN_CAPTURED") & opening_line_exists]

    if candidates.empty:
        print("No OPEN_CAPTURED games with opening lines found.")
        return

    scored = []
    manual_review = []
    skipped_past = 0
    skipped_existing_pick = 0
    skipped_missing_snapshot = 0

    for idx, row in candidates.iterrows():
        if has_existing_pick(row.get("pick_team")):
            skipped_existing_pick += 1
            continue

        kickoff = parse_kickoff(row.get("kickoff_time"))

        if pd.isna(kickoff):
            continue

        if kickoff <= now_utc:
            skipped_past += 1
            continue

        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()

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

        home_rest = int((kickoff - snap[home]["last_date"]).days)
        away_rest = int((kickoff - snap[away]["last_date"]).days)

        if home_rest < 0 or away_rest < 0:
            continue

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

        edge_home = p_home - fair_home
        edge_away = p_away - fair_away

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

        alert = audit_candidate_pick(
            pick_team=pick_team,
            pick_side=side,
            open_line=home_spread,
            current_line=home_spread,
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

    picks = scored[:MAX_PICKS]

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

    batch_update_cells(ws, updates)

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
