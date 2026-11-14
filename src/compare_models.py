"""
Multi-Model Performance Scorecard & Slate Comparator.

Compares:
1. Model v1 Baseline (Master_Bets live paper bets)
2. Model v2 Baseline (V2_Shadow: Elo + Rest differential)
3. Model v3 EPA-Enhanced (V3_Shadow: Play-by-Play EPA + Discrete Key Numbers + Platt Calibration)
4. Model v3 High-Conviction Tier (Edge >= 3.5%)
5. Pre-Bet Safeguard Audits (Injury & RLM Guard suppressed danger bets)

Outputs:
- Terminal scorecard
- GitHub Actions Step Summary ($GITHUB_STEP_SUMMARY)
- Google Sheets 'Model_Scorecard' tab
"""

import math
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None

from src.config import FLAT_BET_SIZE, LOCAL_TZ, MASTER_SHEET_NAME, MIN_EDGE_THRESHOLD, SEASON
from src.key_numbers import calculate_spread_probabilities, market_implied_mean
from src.odds_utils import american_to_implied_prob, remove_vig
from src.staking import cap_slate_stakes, compute_push_aware_stake
from src.v2_sheet_utils import get_or_create_worksheet

SCORECARD_TAB = "Model_Scorecard"
# Margin sigma used to value line movement in CLV (market close as the mean margin)
CLV_SIGMA = 13.1
# Shadow rows logged before this are legacy (R2): excluded from any promotion sample.
SHADOW_LEGACY_CUTOFF_UTC = "2026-11-03T00:00:00Z"
IN_WEEK = "in_week"
LEGACY = "legacy"
REGIMES = (IN_WEEK, LEGACY)


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        s = str(value).strip().replace("%", "")
        if not s or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(value, default=None):
    f = safe_float(value)
    return int(round(f)) if f is not None else default


def profit_units(odds, stake=FLAT_BET_SIZE):
    try:
        odds = float(odds)
        if odds > 0:
            return stake * odds / 100.0
        elif odds < 0:
            return stake * 100.0 / (-odds)
        return 0.0
    except Exception:
        return 0.0


def fetch_tab_df(tab_name):
    """Safely pulls any worksheet tab into a DataFrame."""
    try:
        ws = get_or_create_worksheet(tab_name)
        all_vals = ws.get_all_values()
        if not all_vals or len(all_vals) < 2:
            return pd.DataFrame()
        headers = [h.strip() for h in all_vals[0]]
        return pd.DataFrame(all_vals[1:], columns=headers)
    except (Exception, SystemExit) as exc:
        print(f"Warning: Failed to load tab '{tab_name}': {exc}")
        return pd.DataFrame()


def load_final_scores(season=SEASON):
    """Loads all finalized game scores for the season via nflverse."""
    if nfl is None:
        return {}
    try:
        sched = nfl.import_schedules([season])
        sched = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()
        scores = {}
        for _, g in sched.iterrows():
            gid = str(g.get("game_id", "")).strip()
            if gid:
                scores[gid] = {
                    "home_score": int(g["home_score"]),
                    "away_score": int(g["away_score"]),
                    "home_team": str(g.get("home_team", "")).strip().upper(),
                    "away_team": str(g.get("away_team", "")).strip().upper(),
                }
        return scores
    except Exception as exc:
        print(f"Warning: Failed to fetch nflverse schedules: {exc}")
        return {}


def clv_win_probability(pick_line, closing_pick_line, sigma=CLV_SIGMA):
    """
    Win probability gained (or lost) from line movement, in probability points.
    Treats the closing line as the market's margin and compares the key-number-aware
    no-push cover probability at the line taken vs at the closing line, so a move off
    3 or 7 counts for more than a move between dead numbers. Price changes are ignored;
    clv_fair_probability is the price-aware version.
    """
    close_mu = -float(closing_pick_line)
    p_taken = calculate_spread_probabilities(close_mu, sigma, float(pick_line))["cond_home_win"]
    p_close = calculate_spread_probabilities(close_mu, sigma, float(closing_pick_line))["cond_home_win"]
    return float(p_taken - p_close)


def _devig_home(home_odds, away_odds):
    """De-vigged home probability from a two-way price pair, or None if either is missing."""
    h, a = safe_float(home_odds), safe_float(away_odds)
    if h is None or a is None or h == 0 or a == 0:
        return None
    return remove_vig(american_to_implied_prob(h), american_to_implied_prob(a))[0]


def clv_fair_probability(pick_side, pick_line, open_home_odds, open_away_odds,
                         close_home_line, close_home_odds, close_away_odds,
                         total_line=None, sigma=CLV_SIGMA):
    """
    Price-aware CLV: fair probability of the ticket at the close minus its fair
    probability when taken, in probability points.

    The closing home line and its de-vigged price fix the market's mean margin m*;
    the ticket (pick side at pick_line) is then valued at m*. Returns None when a
    price or the closing line is missing, so callers fall back to clv_win_probability.
    """
    side = str(pick_side).strip().upper()
    q_open_home = _devig_home(open_home_odds, open_away_odds)
    q_close_home = _devig_home(close_home_odds, close_away_odds)
    close_line = safe_float(close_home_line)
    line = safe_float(pick_line)
    if q_open_home is None or q_close_home is None or close_line is None or line is None or side not in {"HOME", "AWAY"}:
        return None
    total = safe_float(total_line)
    mean = market_implied_mean(close_line, q_close_home, sigma, total)
    if side == "HOME":
        q_open = q_open_home
        p_ticket = calculate_spread_probabilities(mean, sigma, line, total)["cond_home_win"]
    else:
        q_open = 1.0 - q_open_home
        p_ticket = calculate_spread_probabilities(mean, sigma, -line, total)["cond_away_win"]
    return float(p_ticket - q_open)


def parse_logged_at(value):
    """UTC timestamp from either logged_at_utc format (ISO in V2/V3, '... UTC' in V4), else None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if s.upper().endswith(" UTC"):
        s = s[:-4].strip() + "+00:00"
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


_LEGACY_CUTOFF = parse_logged_at(SHADOW_LEGACY_CUTOFF_UTC)


def master_regime_map(master_df):
    """{game_id: Master_Bets.decision_regime} (blank when unset)."""
    out = {}
    if master_df is None or master_df.empty or "game_id" not in master_df.columns:
        return out
    for _, row in master_df.iterrows():
        gid = str(row.get("game_id", "")).strip()
        if gid:
            val = row.get("decision_regime", "")
            out[gid] = "" if val is None or (isinstance(val, float) and math.isnan(val)) else str(val).strip()
    return out


def v1_row_regime(row):
    """A v1 bet is in_week only if Master_Bets says so; blank and lookahead are legacy."""
    val = row.get("decision_regime", "")
    val = "" if val is None or (isinstance(val, float) and math.isnan(val)) else str(val).strip()
    return IN_WEEK if val == IN_WEEK else LEGACY


def shadow_row_regime(row, master_regimes):
    """
    in_week only when the row was logged at/after SHADOW_LEGACY_CUTOFF_UTC and its game was
    decided in-week on Master_Bets. A missing timestamp or Master_Bets row counts as legacy.
    """
    ts = parse_logged_at(row.get("logged_at_utc"))
    if ts is None or ts < _LEGACY_CUTOFF:
        return LEGACY
    gid = str(row.get("game_id", "")).strip()
    return IN_WEEK if master_regimes.get(gid) == IN_WEEK else LEGACY


def filter_regime(df, regime, master_regimes, kind="shadow"):
    """Rows of df in `regime`; kind='v1' uses the row's own decision_regime."""
    if df is None or df.empty or regime is None:
        return df
    if kind == "v1":
        mask = df.apply(lambda r: v1_row_regime(r) == regime, axis=1)
    else:
        mask = df.apply(lambda r: shadow_row_regime(r, master_regimes) == regime, axis=1)
    return df[mask.astype(bool)] if len(mask) else df


def guard_counts(decision, edge):
    """A SUPPRESS_DANGER row is a suppressed bet only if it cleared the edge threshold."""
    return decision == "SUPPRESS_DANGER" and edge is not None and edge >= MIN_EDGE_THRESHOLD


def closing_reference_line(row):
    """Home closing spread used for CLV: the multi-book consensus close when captured, else the single-book close."""
    consensus = safe_float(row.get("closing_consensus_line"))
    return consensus if consensus is not None else safe_float(row.get("closing_line"))


def bet_week(row, game_id):
    """Week from the sheet row, falling back to the game_id (e.g. 2026_10_DAL_IND -> 10)."""
    week = safe_int(row.get("week"))
    if week is None:
        parts = str(game_id or "").split("_")
        week = safe_int(parts[1]) if len(parts) > 1 else None
    return week


def weekly_clv_prob(clv_evaluated):
    """Average CLV win probability (in %) per week, over bets with a recorded close."""
    by_week = {}
    for b in clv_evaluated:
        if b.get("week") is None or b.get("clv_prob") is None:
            continue
        by_week.setdefault(int(b["week"]), []).append(b["clv_prob"])
    return {w: sum(p) / len(p) * 100.0 for w, p in sorted(by_week.items())}


def grade_single_bet(side, line, odds, home_score, away_score, closing_line=None, game_id=None, stake_units=None, week=None,
                     open_home_odds=None, open_away_odds=None, close_single_line=None,
                     close_home_odds=None, close_away_odds=None, total_line=None):
    """
    Grades a single spread bet and calculates realized CLV.

    closing_line (consensus when captured) drives the line-only numbers. The price-aware
    clv_fair_prob uses the single-book close (close_single_line with its prices), since the
    consensus line has no price; it falls back to clv_prob when a price is missing.
    """
    side = str(side).strip().upper()
    line = float(line)
    odds = float(odds) if odds is not None else -110.0

    if side == "HOME":
        margin = home_score - away_score
        closing_pick_line = closing_line if (closing_line is not None and not pd.isna(closing_line)) else None
    else:
        margin = away_score - home_score
        closing_pick_line = -closing_line if (closing_line is not None and not pd.isna(closing_line)) else None

    cover = margin + line
    if cover > 0:
        result = "WIN"
        units = profit_units(odds, FLAT_BET_SIZE)
    elif cover < 0:
        result = "LOSS"
        units = -FLAT_BET_SIZE
    else:
        result = "PUSH"
        units = 0.0

    k_stake = float(stake_units) if (stake_units is not None and float(stake_units) > 0) else FLAT_BET_SIZE
    if result == "WIN":
        k_pnl = profit_units(odds, k_stake)
    elif result == "LOSS":
        k_pnl = -k_stake
    else:
        k_pnl = 0.0

    clv_pts = None
    clv_beat = None
    clv_result = None
    clv_prob = None
    if closing_pick_line is not None:
        clv_pts = line - float(closing_pick_line)
        clv_beat = clv_pts > 0
        clv_result = "BEAT" if clv_pts > 1e-9 else ("LOSE" if clv_pts < -1e-9 else "TIE")
        clv_prob = clv_win_probability(line, float(closing_pick_line))

    clv_fair = clv_fair_probability(
        side, line, open_home_odds, open_away_odds,
        close_single_line, close_home_odds, close_away_odds, total_line=total_line,
    )
    clv_fair_priced = clv_fair is not None
    if clv_fair is None:
        clv_fair = clv_prob

    return {
        "game_id": game_id,
        "week": week,
        "result": result,
        "units": units,
        "kelly_stake": round(k_stake, 2),
        "kelly_units": round(k_pnl, 4),
        "clv_points": clv_pts,
        "clv_beat": clv_beat,
        "clv_result": clv_result,
        "clv_prob": clv_prob,
        "clv_fair_prob": clv_fair,
        "clv_fair_priced": clv_fair_priced,
        "margin": margin,
        "cover": cover,
    }


def compute_metrics(graded_bets, model_name, tier_name="All Picks"):
    """Aggregates win rate, units, ROI, and CLV stats."""
    n_bets = len(graded_bets)
    if n_bets == 0:
        return {
            "model": model_name,
            "tier": tier_name,
            "bets": 0,
            "record": "0-0-0",
            "win_pct": 0.0,
            "units": 0.0,
            "roi_pct": 0.0,
            "kelly_units": 0.0,
            "kelly_roi_pct": 0.0,
            "clv_beat_pct": 0.0,
            "clv_tie_pct": 0.0,
            "clv_lose_pct": 0.0,
            "avg_clv": 0.0,
            "avg_clv_prob_pct": 0.0,
            "avg_clv_fair_prob_pct": 0.0,
            "clv_fair_se_pct": 0.0,
            "clv_fair_priced": 0,
            "weekly_clv_prob_pct": {},
            "clv_weeks": 0,
            "clv_positive_weeks": 0,
            "clv_positive_week_pct": 0.0,
        }

    wins = sum(1 for b in graded_bets if b["result"] == "WIN")
    losses = sum(1 for b in graded_bets if b["result"] == "LOSS")
    pushes = sum(1 for b in graded_bets if b["result"] == "PUSH")

    decided = wins + losses
    win_pct = (wins / decided * 100.0) if decided > 0 else 0.0

    total_units = sum(b["units"] for b in graded_bets)
    roi_pct = (total_units / (n_bets * FLAT_BET_SIZE) * 100.0) if n_bets > 0 else 0.0

    kelly_units = round(sum(b.get("kelly_units", b["units"]) for b in graded_bets), 2)
    total_k_staked = sum(b.get("kelly_stake", FLAT_BET_SIZE) for b in graded_bets)
    kelly_roi_pct = round(kelly_units / total_k_staked * 100.0, 2) if total_k_staked > 0 else 0.0

    # Beat, tie and lose are shares of all bets with a recorded close (they sum to 100)
    clv_evaluated = [b for b in graded_bets if b["clv_points"] is not None]
    fair = [b["clv_fair_prob"] for b in graded_bets if b.get("clv_fair_prob") is not None]
    clv_fair_priced = sum(1 for b in graded_bets if b.get("clv_fair_priced"))
    avg_clv_fair_prob_pct = (sum(fair) / len(fair) * 100.0) if fair else 0.0
    if len(fair) > 1:
        mean_f = sum(fair) / len(fair)
        sd = math.sqrt(sum((f - mean_f) ** 2 for f in fair) / (len(fair) - 1))
        clv_fair_se_pct = sd / math.sqrt(len(fair)) * 100.0
    else:
        clv_fair_se_pct = 0.0
    if clv_evaluated:
        n_close = len(clv_evaluated)
        clv_beat_pct = sum(1 for b in clv_evaluated if b.get("clv_result") == "BEAT") / n_close * 100.0
        clv_lose_pct = sum(1 for b in clv_evaluated if b.get("clv_result") == "LOSE") / n_close * 100.0
        clv_tie_pct = sum(1 for b in clv_evaluated if b.get("clv_result") == "TIE") / n_close * 100.0
        avg_clv = sum(b["clv_points"] for b in clv_evaluated) / n_close
        probs = [b["clv_prob"] for b in clv_evaluated if b.get("clv_prob") is not None]
        avg_clv_prob_pct = (sum(probs) / len(probs) * 100.0) if probs else 0.0
        weekly = weekly_clv_prob(clv_evaluated)
    else:
        weekly = {}
        clv_beat_pct = 0.0
        clv_tie_pct = 0.0
        clv_lose_pct = 0.0
        avg_clv = 0.0
        avg_clv_prob_pct = 0.0

    return {
        "model": model_name,
        "tier": tier_name,
        "bets": n_bets,
        "record": f"{wins}W - {losses}L - {pushes}P",
        "win_pct": round(win_pct, 1),
        "units": round(total_units, 2),
        "roi_pct": round(roi_pct, 2),
        "kelly_units": kelly_units,
        "kelly_roi_pct": kelly_roi_pct,
        "clv_beat_pct": round(clv_beat_pct, 1),
        "clv_tie_pct": round(clv_tie_pct, 1),
        "clv_lose_pct": round(clv_lose_pct, 1),
        "avg_clv": round(avg_clv, 2),
        "avg_clv_prob_pct": round(avg_clv_prob_pct, 2),
        # Promotion rule conditions 2 and 4 (price-aware; line-only where no closing price)
        "avg_clv_fair_prob_pct": round(avg_clv_fair_prob_pct, 2),
        "clv_fair_se_pct": round(clv_fair_se_pct, 2),
        "clv_fair_priced": clv_fair_priced,
        # Promotion rule condition 3: share of weeks with a positive Avg CLV Win Prob
        "weekly_clv_prob_pct": {w: round(v, 2) for w, v in weekly.items()},
        "clv_weeks": len(weekly),
        "clv_positive_weeks": sum(1 for v in weekly.values() if v > 0),
        "clv_positive_week_pct": round(sum(1 for v in weekly.values() if v > 0) / len(weekly) * 100.0, 1) if weekly else 0.0,
    }


def first_priced_decision(df, line_col):
    """
    Keeps each game's first logged evaluation that carries a line. Rows logged before the
    opening line existed have a blank line; keeping them as the "first" decision would
    discard the real Tuesday evaluation.
    """
    if df is None or df.empty or "game_id" not in df.columns:
        return df
    if line_col in df.columns:
        df = df[df[line_col].map(lambda v: safe_float(v) is not None)]
    return df.drop_duplicates(subset=["game_id"], keep="first")


def v3_kelly_stake_map(v3_df, max_slate_exposure=10.0):
    """
    Push-aware quarter-Kelly stakes for v3's active picks, capped per week.

    V3_Shadow does not log a stake column, so this recomputes stakes the same way the
    slate matrix does (and the way log_v4_shadow stores v4_stake_units). Pass the graded
    rows (first priced decision per game) so stakes match the bets being graded.
    """
    from collections import defaultdict

    if v3_df is None or v3_df.empty:
        return {}
    by_week = defaultdict(list)
    for _, row in v3_df.iterrows():
        dec = str(row.get("v3_decision", "")).strip().upper()
        if dec not in {"SHADOW_PICK", "HIGH_CONVICTION"}:
            continue
        gid = str(row.get("game_id", "")).strip()
        odds = safe_float(row.get("v3_best_odds"), -110.0)
        prob = safe_float(row.get("v3_model_probability"), 0.52)
        p_push = safe_float(row.get("p_push"), 0.05)
        units = compute_push_aware_stake(prob * (1.0 - p_push), p_push, odds)["recommended_units"]
        by_week[safe_int(row.get("week"), 0)].append({"game_id": gid, "recommended_units": units})

    stakes = {}
    for cands in by_week.values():
        for b in cap_slate_stakes(cands, max_slate_exposure=max_slate_exposure):
            stakes[b["game_id"]] = float(b["recommended_units"])
    return stakes


def evaluate_all_models(master_df, v2_df, v3_df, scores, v4_df=None, regime=None):
    """Extracts, grades, and aggregates performance across all models (v1, v2, v3, v4).

    Shadow tabs are graded on each game's FIRST logged evaluation (the Tuesday decision,
    matching v1). Keeping later re-evaluations let line moves toward the model turn picks
    into PASS and drop them, biasing CLV and making the head-to-head unequal.

    regime ('in_week' or 'legacy') keeps only rows of that regime before picking each
    game's first row, so a legacy row never shadows an in-week one. None grades everything.
    """
    if master_df is None:
        master_df = pd.DataFrame()
    if v2_df is None:
        v2_df = pd.DataFrame()
    if v3_df is None:
        v3_df = pd.DataFrame()
    full_master_df = master_df
    show_v4 = v4_df is not None and not v4_df.empty
    if regime is not None:
        regimes = master_regime_map(master_df)
        v2_df = filter_regime(v2_df, regime, regimes)
        v3_df = filter_regime(v3_df, regime, regimes)
        v4_df = filter_regime(v4_df, regime, regimes)
        master_df = filter_regime(master_df, regime, regimes, kind="v1")
    if v2_df is not None and not v2_df.empty and "game_id" in v2_df.columns:
        v2_df = v2_df.drop_duplicates(subset=["game_id"], keep="first")
    v3_df = first_priced_decision(v3_df, "v3_best_line")
    v4_df = first_priced_decision(v4_df, "v4_best_line")

    # Closes come from every Master_Bets row: a shadow bet's game may be legacy for v1
    closing_map = {}
    master_by_gid = {}
    if not full_master_df.empty:
        for _, row in full_master_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            if gid:
                master_by_gid[gid] = row
                c_line = closing_reference_line(row)
                if c_line is not None:
                    closing_map[gid] = c_line

    def close_prices(gid):
        m = master_by_gid.get(gid)
        if m is None:
            return {}
        return {
            "close_single_line": safe_float(m.get("closing_line")),
            "close_home_odds": safe_float(m.get("closing_odds")),
            "close_away_odds": safe_float(m.get("closing_away_odds")),
            "total_line": safe_float(m.get("total_line")),
        }

    # 1. Evaluate v1 (Master_Bets)
    v1_bets = []
    if not master_df.empty:
        for _, row in master_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            p_team = str(row.get("pick_team", "")).strip().upper()
            if not p_team or p_team == "NAN":
                continue
            p_side = str(row.get("pick_side", "")).strip().upper()
            p_line = safe_float(row.get("pick_line"))
            p_odds = safe_float(row.get("pick_odds"), -110.0)

            if p_line is None:
                continue

            if gid in scores:
                sc = scores[gid]
                g = grade_single_bet(
                    side=p_side,
                    line=p_line,
                    odds=p_odds,
                    home_score=sc["home_score"],
                    away_score=sc["away_score"],
                    closing_line=closing_map.get(gid),
                    game_id=gid,
                    week=bet_week(row, gid),
                    open_home_odds=safe_float(row.get("opening_odds")),
                    open_away_odds=safe_float(row.get("opening_away_odds")),
                    **close_prices(gid),
                )
                v1_bets.append(g)

    # 2. Evaluate v2 (V2_Shadow)
    v2_bets = []
    if not v2_df.empty:
        for _, row in v2_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            dec = str(row.get("v2_decision", "")).strip().upper()
            if dec != "SHADOW_PICK":
                continue
            side = str(row.get("v2_best_side", "")).strip().upper()
            line = safe_float(row.get("v2_best_line"))
            odds = safe_float(row.get("v2_best_odds"), -110.0)

            if line is None or not side:
                continue

            if gid in scores:
                sc = scores[gid]
                g = grade_single_bet(
                    side=side,
                    line=line,
                    odds=odds,
                    home_score=sc["home_score"],
                    away_score=sc["away_score"],
                    closing_line=closing_map.get(gid),
                    game_id=gid,
                    week=bet_week(row, gid),
                    open_home_odds=safe_float(row.get("home_odds")),
                    open_away_odds=safe_float(row.get("away_odds")),
                    **close_prices(gid),
                )
                v2_bets.append(g)

    # 3. Evaluate v3 (V3_Shadow)
    v3_bets = []
    v3_high_bets = []
    suppressed_bets = []

    if not v3_df.empty:
        v3_stakes = v3_kelly_stake_map(v3_df)
        for _, row in v3_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            dec = str(row.get("v3_decision", "")).strip().upper()
            side = str(row.get("v3_best_side", "")).strip().upper()
            line = safe_float(row.get("v3_best_line"))
            odds = safe_float(row.get("v3_best_odds"), -110.0)
            edge = safe_float(row.get("v3_edge"), 0.0)

            if line is None or not side:
                continue

            if gid in scores:
                sc = scores[gid]
                g = grade_single_bet(
                    side=side,
                    line=line,
                    odds=odds,
                    home_score=sc["home_score"],
                    away_score=sc["away_score"],
                    closing_line=closing_map.get(gid),
                    game_id=gid,
                    week=bet_week(row, gid),
                    stake_units=v3_stakes.get(gid),
                    open_home_odds=safe_float(row.get("home_odds")),
                    open_away_odds=safe_float(row.get("away_odds")),
                    **close_prices(gid),
                )

                if dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
                    v3_bets.append(g)
                    if dec == "HIGH_CONVICTION" or edge >= 0.035:
                        v3_high_bets.append(g)
                elif guard_counts(dec, edge):
                    suppressed_bets.append(g)

    # 4. Evaluate v4 (V4_Shadow)
    v4_bets = []
    v4_high_bets = []
    if v4_df is not None and not v4_df.empty:
        for _, row in v4_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            dec = str(row.get("v4_decision", "")).strip().upper()
            side = str(row.get("v4_best_side", "")).strip().upper()
            line = safe_float(row.get("v4_best_line"))
            odds = safe_float(row.get("v4_best_odds"), -110.0)
            edge = safe_float(row.get("v4_edge"), 0.0)

            if line is None or not side:
                continue

            if gid in scores:
                sc = scores[gid]
                g = grade_single_bet(
                    side=side,
                    line=line,
                    odds=odds,
                    home_score=sc["home_score"],
                    away_score=sc["away_score"],
                    closing_line=closing_map.get(gid),
                    game_id=gid,
                    week=bet_week(row, gid),
                    stake_units=safe_float(row.get("v4_stake_units")),
                    open_home_odds=safe_float(row.get("home_odds")),
                    open_away_odds=safe_float(row.get("away_odds")),
                    **close_prices(gid),
                )

                if dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
                    v4_bets.append(g)
                    if dec == "HIGH_CONVICTION" or edge >= 0.035:
                        v4_high_bets.append(g)
                elif guard_counts(dec, edge) and gid not in [b.get("game_id") for b in suppressed_bets]:
                    suppressed_bets.append(g)

    summary_rows = [
        compute_metrics(v1_bets, "Model v1 (Master_Bets Baseline)", "All Picks"),
        compute_metrics(v2_bets, "Model v2 (Elo + Rest Baseline)", "All Picks"),
        compute_metrics(v3_bets, "Model v3 (EPA + Discrete Keys)", "All Picks"),
        compute_metrics(v3_high_bets, "Model v3 High-Conviction (>=3.5%)", "High-Conviction"),
    ]

    if show_v4:
        summary_rows.append(compute_metrics(v4_bets, "Model v4 (EPA+QB+Luck+Totals)", "All Picks"))
        summary_rows.append(compute_metrics(v4_high_bets, "Model v4 High-Conviction (>=3.5%)", "High-Conviction"))

    # Guard audit (what would have happened to suppressed danger bets?)
    guard_metrics = compute_metrics(suppressed_bets, "Guard Suppressed Danger Bets", "Hypothetical Placed")

    counts_dict = {
        "v1_count": len(v1_bets),
        "v2_count": len(v2_bets),
        "v3_count": len(v3_bets),
        "v3_high_count": len(v3_high_bets),
        "v4_count": len(v4_bets),
        "v4_high_count": len(v4_high_bets),
        "suppressed_count": len(suppressed_bets),
    }

    return summary_rows, guard_metrics, counts_dict


def evaluate_regimes(master_df, v2_df, v3_df, scores, v4_df=None):
    """{'in_week': (summary, guard, counts), 'legacy': (...)}; only in_week feeds promotion."""
    return {
        r: evaluate_all_models(master_df, v2_df, v3_df, scores, v4_df=v4_df, regime=r)
        for r in REGIMES
    }


def build_head_to_head_slate(master_df, v2_df, v3_df, v4_df=None):
    """Builds a side-by-side comparison matrix for the active upcoming week across v1, v2, v3, v4."""
    all_weeks = []
    dfs_to_check = [master_df, v2_df, v3_df]
    if v4_df is not None and not v4_df.empty:
        dfs_to_check.append(v4_df)
    for df in dfs_to_check:
        if not df.empty and "week" in df.columns:
            w_vals = pd.to_numeric(df["week"], errors="coerce").dropna()
            all_weeks.extend(w_vals.tolist())

    now_utc = datetime.now(pytz.utc)

    # Group records by game_id
    m_dict = {str(r.get("game_id", "")).strip(): r for _, r in master_df.iterrows()} if not master_df.empty else {}
    v2_dict = {str(r.get("game_id", "")).strip(): r for _, r in v2_df.iterrows()} if not v2_df.empty else {}
    v3_dict = {str(r.get("game_id", "")).strip(): r for _, r in v3_df.iterrows()} if not v3_df.empty else {}
    v4_dict = {str(r.get("game_id", "")).strip(): r for _, r in v4_df.iterrows()} if (v4_df is not None and not v4_df.empty) else {}

    all_gids = set(m_dict.keys()) | set(v2_dict.keys()) | set(v3_dict.keys()) | set(v4_dict.keys())

    # Pre-compute and cap portfolio stakes across the active slate (per-week)
    from collections import defaultdict
    v3_by_week = defaultdict(list)
    v4_by_week = defaultdict(list)

    for gid in sorted(all_gids):
        r4 = v4_dict.get(gid, {})
        r3 = v3_dict.get(gid, {})
        r2 = v2_dict.get(gid, {})
        rm = m_dict.get(gid, {})

        w = safe_int(r4.get("week") or r3.get("week") or rm.get("week") or r2.get("week"), 8)

        v3_dec = str(r3.get("v3_decision", "")).strip().upper()
        if v3_dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
            v3_odds = safe_float(r3.get("v3_best_odds"), -110.0)
            v3_prob = safe_float(r3.get("v3_model_probability"), 0.52)
            p_push = safe_float(r3.get("p_push"), 0.05)
            uncond_win_prob = v3_prob * (1.0 - p_push)
            s_info = compute_push_aware_stake(uncond_win_prob, p_push, v3_odds)
            v3_by_week[w].append({"game_id": gid, "recommended_units": s_info["recommended_units"]})

        r4 = v4_dict.get(gid, {})
        v4_dec = str(r4.get("v4_decision", "")).strip().upper()
        if v4_dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
            v4_odds = safe_float(r4.get("v4_best_odds"), -110.0)
            v4_prob = safe_float(r4.get("v4_model_probability"), 0.52)
            p_push_4 = safe_float(r4.get("p_push"), 0.05)
            uncond_win_prob_4 = v4_prob * (1.0 - p_push_4)
            s_info_4 = compute_push_aware_stake(uncond_win_prob_4, p_push_4, v4_odds)
            v4_by_week[w].append({"game_id": gid, "recommended_units": s_info_4["recommended_units"]})

    v3_capped_map = {}
    for w, cands in v3_by_week.items():
        for b in cap_slate_stakes(cands, max_slate_exposure=10.0):
            v3_capped_map[b["game_id"]] = b["recommended_units"]

    v4_capped_map = {}
    for w, cands in v4_by_week.items():
        for b in cap_slate_stakes(cands, max_slate_exposure=10.0):
            v4_capped_map[b["game_id"]] = b["recommended_units"]

    slate_rows = []
    for gid in sorted(all_gids):
        r4 = v4_dict.get(gid, {})
        r3 = v3_dict.get(gid, {})
        r2 = v2_dict.get(gid, {})
        rm = m_dict.get(gid, {})

        ko_raw = r4.get("kickoff_utc") or r3.get("kickoff_utc") or rm.get("kickoff_time") or r2.get("kickoff_utc")
        ko = pd.to_datetime(ko_raw, utc=True, errors="coerce")

        week = safe_int(r4.get("week") or r3.get("week") or rm.get("week") or r2.get("week"), 8)
        matchup = str(r4.get("matchup") or r3.get("matchup") or rm.get("matchup") or f"{rm.get('away_team','')} @ {rm.get('home_team','')}").strip()
        if not matchup or matchup == "@":
            continue

        h_line = safe_float(r4.get("home_line") or r3.get("home_line") or rm.get("opening_line") or r2.get("home_line"))
        h_odds = safe_float(r4.get("home_odds") or r3.get("home_odds") or rm.get("opening_odds") or r2.get("home_odds"))

        # v1 Pick
        v1_team = str(rm.get("pick_team", "")).strip().upper()
        v1_line = safe_float(rm.get("pick_line"))
        v1_pick_str = f"{v1_team} {v1_line:+.1f}" if (v1_team and v1_team != "NAN" and v1_line is not None) else "PASS"

        # v2 Pick
        v2_dec = str(r2.get("v2_decision", "")).strip().upper()
        v2_team = str(r2.get("v2_best_team", "")).strip().upper()
        v2_line = safe_float(r2.get("v2_best_line"))
        v2_pick_str = f"{v2_team} {v2_line:+.1f}" if (v2_dec == "SHADOW_PICK" and v2_team and v2_line is not None) else ("SUPPRESS" if v2_dec == "SUPPRESS_DANGER" else "PASS")

        # v3 Pick & Edge
        v3_dec = str(r3.get("v3_decision", "")).strip().upper()
        v3_team = str(r3.get("v3_best_team", "")).strip().upper()
        v3_line = safe_float(r3.get("v3_best_line"))
        v3_edge = safe_float(r3.get("v3_edge"), 0.0)

        if v3_dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
            v3_odds = safe_float(r3.get("v3_best_odds"), -110.0)
            v3_prob = safe_float(r3.get("v3_model_probability"), 0.52)
            p_push = safe_float(r3.get("p_push"), 0.05)
            uncond_win_prob = v3_prob * (1.0 - p_push)
            stake_units = v3_capped_map.get(gid, compute_push_aware_stake(uncond_win_prob, p_push, v3_odds)["recommended_units"])
            v3_pick_str = f"{v3_team} {v3_line:+.1f} (+{v3_edge*100:.1f}%) [{stake_units:.1f}u]"
            if v3_dec == "HIGH_CONVICTION" or v3_edge >= 0.035:
                v3_pick_str += " ★"
        elif v3_dec == "SUPPRESS_DANGER":
            v3_pick_str = "SUPPRESS_DANGER"
        else:
            v3_pick_str = "PASS"

        # v4 Pick & Edge
        v4_dec = str(r4.get("v4_decision", "")).strip().upper()
        v4_team = str(r4.get("v4_best_team", "")).strip().upper()
        v4_line = safe_float(r4.get("v4_best_line"))
        v4_edge = safe_float(r4.get("v4_edge"), 0.0)

        if v4_dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
            v4_odds = safe_float(r4.get("v4_best_odds"), -110.0)
            v4_prob = safe_float(r4.get("v4_model_probability"), 0.52)
            p_push_4 = safe_float(r4.get("p_push"), 0.05)
            uncond_win_prob_4 = v4_prob * (1.0 - p_push_4)
            stake_units_4 = v4_capped_map.get(gid, compute_push_aware_stake(uncond_win_prob_4, p_push_4, v4_odds)["recommended_units"])
            v4_pick_str = f"{v4_team} {v4_line:+.1f} (+{v4_edge*100:.1f}%) [{stake_units_4:.1f}u]"
            if v4_dec == "HIGH_CONVICTION" or v4_edge >= 0.035:
                v4_pick_str += " ★"
        elif v4_dec == "SUPPRESS_DANGER":
            v4_pick_str = "SUPPRESS_DANGER"
        else:
            v4_pick_str = "PASS"

        # Consensus evaluation
        picks = []
        if v1_pick_str not in {"PASS", ""}:
            picks.append(("v1", v1_team))
        if v2_pick_str not in {"PASS", "SUPPRESS", ""}:
            picks.append(("v2", v2_team))
        if v3_dec in {"SHADOW_PICK", "HIGH_CONVICTION"} and v3_team:
            picks.append(("v3", v3_team))
        if v4_dec in {"SHADOW_PICK", "HIGH_CONVICTION"} and v4_team:
            picks.append(("v4", v4_team))

        if not picks:
            consensus = "ALL PASS"
        elif len(picks) == 1:
            consensus = f"SOLO ({picks[0][0].upper()})"
        else:
            teams = {p[1] for p in picks}
            if len(teams) == 1:
                consensus = f"UNANIMOUS ({teams.pop()})"
            else:
                consensus = "DISAGREEMENT"

        slate_rows.append({
            "week": week,
            "matchup": matchup,
            "kickoff_utc": str(ko) if pd.notna(ko) else "",
            "market_home_line": f"{h_line:+.1f}" if h_line is not None else "-",
            "v1_pick": v1_pick_str,
            "v2_pick": v2_pick_str,
            "v3_pick": v3_pick_str,
            "v4_pick": v4_pick_str,
            "consensus": consensus,
            "v3_decision": v3_dec,
            "v4_decision": v4_dec,
            "is_future": bool(pd.notna(ko) and ko > now_utc),
        })

    return pd.DataFrame(slate_rows)


def weekly_clv_table(summary_rows):
    """Header + one row per model: positive weeks, then Avg CLV Win Prob for each week."""
    weeks = sorted({w for r in summary_rows for w in r.get("weekly_clv_prob_pct", {})})
    header = ["Model", "Tier", "Weeks +"] + [f"Wk {w}" for w in weeks]
    rows = []
    for r in summary_rows:
        weekly = r.get("weekly_clv_prob_pct", {})
        n = r.get("clv_weeks", 0)
        pos = f"{r.get('clv_positive_weeks', 0)}/{n} ({r.get('clv_positive_week_pct', 0.0):.0f}%)" if n else "-"
        rows.append([r["model"], r["tier"], pos] + [f"{weekly[w]:+.2f}%" if w in weekly else "-" for w in weeks])
    return header, rows


def _print_summary_table(summary_rows, guard_metrics):
    print(f"{'Model / Architecture':<35} | {'Bets':<5} | {'Record':<14} | {'Win %':<6} | {'Units':<8} | {'ROI %':<7} | {'CLV Beat':<8} | {'CLV Tie':<7} | {'CLV Lose':<8} | {'Fair CLV':<8} | {'Kelly U':<8} | {'Kelly ROI':<9}")
    print("-" * 160)

    for row in summary_rows:
        has = row['bets'] > 0
        u_str = f"{row['units']:+.2f}u" if has else "0.00u"
        roi_str = f"{row['roi_pct']:+.1f}%" if has else "0.0%"
        clv_str = f"{row['clv_beat_pct']:.1f}%" if has else "0.0%"
        tie_str = f"{row.get('clv_tie_pct', 0.0):.1f}%" if has else "0.0%"
        lose_str = f"{row.get('clv_lose_pct', 0.0):.1f}%" if has else "0.0%"
        fair_str = f"{row.get('avg_clv_fair_prob_pct', 0.0):+.2f}%" if has else "0.00%"
        k_u_str = f"{row['kelly_units']:+.2f}u" if has else "0.00u"
        k_roi_str = f"{row['kelly_roi_pct']:+.1f}%" if has else "0.0%"
        print(f"{row['model']:<35} | {row['bets']:<5} | {row['record']:<14} | {row['win_pct']:>5.1f}% | {u_str:>8} | {roi_str:>7} | {clv_str:>8} | {tie_str:>7} | {lose_str:>8} | {fair_str:>8} | {k_u_str:>8} | {k_roi_str:>9}")

    print("-" * 94)
    g_bets = guard_metrics['bets']
    if g_bets > 0:
        g_units = f"{guard_metrics['units']:+.2f}u"
        saved = -guard_metrics['units']
        saved_str = f"{saved:+.2f}u" if saved > 0 else "0.00u"
        print(f"{'Guard Suppressed Bets (Saved)':<35} | {g_bets:<5} | {guard_metrics['record']:<14} | Would Have PnL: {g_units} (Estimated Saved: {saved_str})")
    else:
        print(f"{'Guard Suppressed Bets (Saved)':<35} | 0     | 0W - 0L - 0P   | No completed games with suppressed danger yet.")


def print_terminal_dashboard(summary_rows, guard_metrics, slate_df, legacy=None):
    """Renders a clean CLI dashboard. legacy = (summary_rows, guard_metrics) for pre-R1 rows."""
    print("\n" + "=" * 94)
    print("        NFL MULTI-MODEL COMPARISON SCORECARD (v1 vs v2 vs v3 vs v4)       ")
    print("=" * 94)
    print("IN-WEEK DECISIONS (promotion sample)")
    _print_summary_table(summary_rows, guard_metrics)
    print("=" * 94)
    if legacy is not None:
        print("\nLEGACY (excluded from promotion): look-ahead v1 picks and shadow rows logged before "
              + SHADOW_LEGACY_CUTOFF_UTC)
        _print_summary_table(*legacy)
        print("=" * 94)

    w_header, w_rows = weekly_clv_table(summary_rows)
    print("\nWEEKLY AVG CLV WIN PROB (promotion rule: positive in >= 60% of weeks bet)")
    print(" | ".join([f"{w_header[0]:<35}", f"{w_header[2]:<12}"] + [f"{h:>7}" for h in w_header[3:]]))
    for wr in w_rows:
        print(" | ".join([f"{wr[0]:<35}", f"{wr[2]:<12}"] + [f"{v:>7}" for v in wr[3:]]))

    # Active week slate
    if not slate_df.empty:
        future_slate = slate_df[slate_df["is_future"]].copy()
        if future_slate.empty:
            future_slate = slate_df.tail(16).copy()

        print("\n" + "=" * 115)
        print("                        ACTIVE SLATE HEAD-TO-HEAD PICKS MATRIX                             ")
        print("=" * 115)
        print(f"{'Matchup':<12} | {'Spread':<7} | {'v1 Baseline':<13} | {'v2 Elo/Rest':<13} | {'v3 EPA-Keys':<22} | {'v4 Full-Matrix':<22} | {'Consensus':<13}")
        print("-" * 115)
        for _, r in future_slate.iterrows():
            print(f"{r['matchup']:<12} | {r['market_home_line']:<7} | {r['v1_pick']:<13} | {r['v2_pick']:<13} | {r['v3_pick']:<22} | {r.get('v4_pick','-'):<22} | {r['consensus']:<13}")
        print("=" * 115 + "\n")


def publish_github_summary(summary_rows, guard_metrics, slate_df, legacy=None):
    """Appends Markdown dashboard to GitHub Actions Step Summary if running in CI."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    md_lines = []
    md_lines.append("## 🏈 NFL Multi-Model Performance Scorecard\n")
    md_lines.append(f"*Last updated: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")

    def md_table(rows_in, guard):
        md_lines.append("| Model / Architecture | Bets | Record | Win % | Units PnL | ROI % | CLV Beat / Tie / Lose % | Avg CLV Fair Prob | Kelly Units | Kelly ROI % |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in rows_in:
            u_str = f"**{r['units']:+.2f}u**" if r['bets'] > 0 else "0.00u"
            roi_str = f"**{r['roi_pct']:+.1f}%**" if r['bets'] > 0 else "0.0%"
            btl = f"{r['clv_beat_pct']:.0f} / {r.get('clv_tie_pct', 0.0):.0f} / {r.get('clv_lose_pct', 0.0):.0f}"
            fair = f"{r.get('avg_clv_fair_prob_pct', 0.0):+.2f}% ± {r.get('clv_fair_se_pct', 0.0):.2f}"
            md_lines.append(f"| {r['model']} | {r['bets']} | {r['record']} | {r['win_pct']:.1f}% | {u_str} | {roi_str} | {btl} | {fair} | {r['kelly_units']:+.2f}u | {r['kelly_roi_pct']:+.1f}% |")
        if guard['bets'] > 0:
            saved = -guard['units']
            saved_str = f"+{saved:.2f}u" if saved > 0 else "0.00u"
            md_lines.append(f"| 🛡️ *Injury/RLM Suppressed (Saved)* | {guard['bets']} | {guard['record']} | - | Would PnL: {guard['units']:+.2f}u | Saved: **{saved_str}** | - | - | - | - |")

    md_lines.append("### In-week decisions (promotion sample)\n")
    md_table(summary_rows, guard_metrics)
    if legacy is not None:
        md_lines.append("\n### Legacy (excluded from promotion)\n")
        md_table(*legacy)

    md_lines.append("\n### 📋 Active Slate Head-to-Head Comparison\n")
    md_lines.append("| Matchup | Market Spread | v1 Baseline | v2 Elo/Rest | v3 EPA-Enhanced | v4 Full Matrix | Consensus |")
    md_lines.append("| :--- | :---: | :---: | :---: | :--- | :--- | :---: |")

    if not slate_df.empty:
        future_slate = slate_df[slate_df["is_future"]].copy()
        if future_slate.empty:
            future_slate = slate_df.tail(16).copy()

        for _, r in future_slate.iterrows():
            v4_p = r.get("v4_pick", "-")
            md_lines.append(f"| **{r['matchup']}** | {r['market_home_line']} | `{r['v1_pick']}` | `{r['v2_pick']}` | `{r['v3_pick']}` | `{v4_p}` | **{r['consensus']}** |")

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print("Wrote scorecard summary to $GITHUB_STEP_SUMMARY.")
    except Exception as exc:
        print(f"Warning: Could not write to $GITHUB_STEP_SUMMARY: {exc}")


# Columns are appended, never reordered. CLV Beat/Tie/Lose % are shares of bets with a close.
SCORECARD_HEADER = (
    "Model", "Tier", "Bets", "Record", "Win %", "Net Units", "ROI %", "CLV Beat %",
    "CLV Tie %", "Avg CLV Pts", "Avg CLV Win Prob", "Kelly Units", "Kelly ROI %",
    "CLV Lose %", "Avg CLV Fair Prob", "Avg CLV Fair Prob SE",
)


def _sheet_summary_block(summary_rows, guard_metrics):
    rows = [list(SCORECARD_HEADER)]
    for r in summary_rows:
        rows.append([
            r["model"],
            r["tier"],
            r["bets"],
            r["record"],
            f"{r['win_pct']:.1f}%",
            f"{r['units']:+.2f}u",
            f"{r['roi_pct']:+.1f}%",
            f"{r['clv_beat_pct']:.1f}%",
            f"{r.get('clv_tie_pct', 0.0):.1f}%",
            f"{r['avg_clv']:+.2f}",
            f"{r.get('avg_clv_prob_pct', 0.0):+.2f}%",
            f"{r['kelly_units']:+.2f}u",
            f"{r['kelly_roi_pct']:+.1f}%",
            f"{r.get('clv_lose_pct', 0.0):.1f}%",
            f"{r.get('avg_clv_fair_prob_pct', 0.0):+.2f}%",
            f"{r.get('clv_fair_se_pct', 0.0):.2f}%",
        ])

    g_bets = guard_metrics["bets"]
    if g_bets > 0:
        saved = -guard_metrics["units"]
        rows.append([
            "Injury / RLM Guard Suppressed",
            "Danger Bets Filtered",
            g_bets,
            guard_metrics["record"],
            "-",
            f"Would PnL: {guard_metrics['units']:+.2f}u",
            f"Saved: {saved:+.2f}u",
        ] + ["-"] * (len(SCORECARD_HEADER) - 7))
    return rows


def write_google_sheet_scorecard(summary_rows, guard_metrics, slate_df, legacy=None):
    """Writes or overwrites the Model_Scorecard tab in Google Sheets.

    summary_rows/guard_metrics are the in-week block; legacy = (summary_rows, guard_metrics)
    is written below it as a separate block excluded from promotion.
    """
    now_utc_str = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_et_str = datetime.now(pytz.timezone(LOCAL_TZ)).strftime("%Y-%m-%d %I:%M %p %Z")

    rows = []
    rows.append(["NFL MULTI-MODEL PERFORMANCE SCORECARD", "", "", "", "", "", "", "", ""])
    rows.append([f"Updated: {now_utc_str} ({now_et_str})", "", "", "", "", "", "", "", ""])
    rows.append([""])

    # Section 1: Summary Table (in-week decisions: the promotion sample)
    rows.append(["MODEL PERFORMANCE SUMMARY: IN-WEEK DECISIONS (promotion sample)", "", "", "", "", "", "", "", ""])
    rows.extend(_sheet_summary_block(summary_rows, guard_metrics))

    rows.append([""])
    rows.append([""])

    if legacy is not None:
        rows.append([f"LEGACY (excluded from promotion): look-ahead v1 picks, shadow rows logged before {SHADOW_LEGACY_CUTOFF_UTC}"])
        rows.extend(_sheet_summary_block(*legacy))
        rows.append([""])
        rows.append([""])

    # Section 2: Weekly CLV (promotion rule condition 3)
    w_header, w_rows = weekly_clv_table(summary_rows)
    rows.append(["WEEKLY AVG CLV WIN PROB (promotion rule: positive in >= 60% of weeks bet)"])
    rows.append(w_header)
    rows.extend(w_rows)

    rows.append([""])
    rows.append([""])

    # Section 3: Active Slate
    rows.append(["ACTIVE SLATE HEAD-TO-HEAD MATRIX", "", "", "", "", "", "", "", ""])
    rows.append(["Week", "Matchup", "Kickoff (UTC)", "Market Spread", "v1 Baseline", "v2 Elo/Rest", "v3 EPA-Enhanced", "v4 Full Matrix", "Consensus"])

    if not slate_df.empty:
        future_slate = slate_df[slate_df["is_future"]].copy()
        if future_slate.empty:
            future_slate = slate_df.tail(16).copy()

        for _, r in future_slate.iterrows():
            rows.append([
                r["week"],
                r["matchup"],
                r["kickoff_utc"],
                r["market_home_line"],
                r["v1_pick"],
                r["v2_pick"],
                r["v3_pick"],
                r.get("v4_pick", "-"),
                r["consensus"],
            ])

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            ws = get_or_create_worksheet(SCORECARD_TAB)
            ws.clear()
            ws.update(values=rows, range_name="A1")
            print(f"Successfully updated '{SCORECARD_TAB}' tab in Google Sheets.")
            break
        except (Exception, SystemExit) as exc:
            if attempt < max_retries and any(err in str(exc) for err in ("429", "Quota", "500", "502", "503", "504")):
                time.sleep(5 * attempt)
                continue
            print(f"Warning: Could not write to Google Sheets '{SCORECARD_TAB}': {exc}")
            break


def build_mock_data():
    """Generates realistic demonstration data representing Week 8 and prior settled bets."""
    # Historical settled games
    settled_scores = {
        "2026_01_BAL_KC": {"home_score": 27, "away_score": 20, "home_team": "KC", "away_team": "BAL"},
        "2026_01_GB_PHI": {"home_score": 34, "away_score": 29, "home_team": "PHI", "away_team": "GB"},
        "2026_02_CIN_KC": {"home_score": 26, "away_score": 25, "home_team": "KC", "away_team": "CIN"},
        "2026_02_CHI_HOU": {"home_score": 19, "away_score": 13, "home_team": "HOU", "away_team": "CHI"},
    }

    # Master_Bets (v1)
    master_rows = [
        {"game_id": "2026_01_BAL_KC", "week": 1, "pick_team": "KC", "pick_side": "HOME", "pick_line": -3.0, "pick_odds": -110.0, "closing_line": -3.0},
        {"game_id": "2026_01_GB_PHI", "week": 1, "pick_team": "PHI", "pick_side": "HOME", "pick_line": -2.5, "pick_odds": -105.0, "closing_line": -3.0},
        {"game_id": "2026_02_CIN_KC", "week": 2, "pick_team": "CIN", "pick_side": "AWAY", "pick_line": 5.5, "pick_odds": -110.0, "closing_line": 6.0},
        {"game_id": "2026_02_CHI_HOU", "week": 2, "pick_team": "HOU", "pick_side": "HOME", "pick_line": -6.5, "pick_odds": -110.0, "closing_line": -6.0},
        # Active Week 8
        {"game_id": "2026_08_CAR_GB", "week": 8, "matchup": "CAR @ GB", "home_team": "GB", "away_team": "CAR", "opening_line": -3.0, "opening_odds": -105.0, "pick_team": "GB", "pick_side": "HOME", "pick_line": -3.0, "pick_odds": -105.0, "kickoff_time": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LV_NYJ", "week": 8, "matchup": "LV @ NYJ", "home_team": "NYJ", "away_team": "LV", "opening_line": 1.5, "opening_odds": -110.0, "pick_team": "NAN", "kickoff_time": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_MIN_DET", "week": 8, "matchup": "MIN @ DET", "home_team": "DET", "away_team": "MIN", "opening_line": -2.5, "opening_odds": -110.0, "pick_team": "DET", "pick_side": "HOME", "pick_line": -2.5, "pick_odds": -110.0, "kickoff_time": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LAC_LA", "week": 8, "matchup": "LAC @ LA", "home_team": "LA", "away_team": "LAC", "opening_line": -7.0, "opening_odds": -105.0, "pick_team": "NAN", "kickoff_time": "2026-11-01 21:05:00+00:00"},
        {"game_id": "2026_08_TEN_CIN", "week": 8, "matchup": "TEN @ CIN", "home_team": "CIN", "away_team": "TEN", "opening_line": -10.0, "opening_odds": -110.0, "pick_team": "CIN", "pick_side": "HOME", "pick_line": -10.0, "pick_odds": -110.0, "kickoff_time": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_NE_MIA", "week": 8, "matchup": "NE @ MIA", "home_team": "MIA", "away_team": "NE", "opening_line": 8.5, "opening_odds": -110.0, "pick_team": "NE", "pick_side": "AWAY", "pick_line": -8.5, "pick_odds": -110.0, "kickoff_time": "2026-11-01 18:00:00+00:00"},
    ]

    # V2_Shadow
    v2_rows = [
        {"game_id": "2026_01_BAL_KC", "week": 1, "v2_decision": "SHADOW_PICK", "v2_best_team": "KC", "v2_best_side": "HOME", "v2_best_line": -3.0, "v2_best_odds": -110.0, "v2_edge": 0.024},
        {"game_id": "2026_01_GB_PHI", "week": 1, "v2_decision": "PASS", "v2_edge": 0.008},
        {"game_id": "2026_02_CIN_KC", "week": 2, "v2_decision": "SHADOW_PICK", "v2_best_team": "CIN", "v2_best_side": "AWAY", "v2_best_line": 5.5, "v2_best_odds": -110.0, "v2_edge": 0.021},
        {"game_id": "2026_02_CHI_HOU", "week": 2, "v2_decision": "PASS", "v2_edge": 0.012},
        # Active Week 8
        {"game_id": "2026_08_CAR_GB", "week": 8, "matchup": "CAR @ GB", "home_team": "GB", "away_team": "CAR", "home_line": -3.0, "v2_decision": "SHADOW_PICK", "v2_best_team": "GB", "v2_best_side": "HOME", "v2_best_line": -3.0, "v2_best_odds": -105.0, "v2_edge": 0.022, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LV_NYJ", "week": 8, "matchup": "LV @ NYJ", "home_team": "NYJ", "away_team": "LV", "home_line": 1.5, "v2_decision": "PASS", "v2_edge": 0.011, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_MIN_DET", "week": 8, "matchup": "MIN @ DET", "home_team": "DET", "away_team": "MIN", "home_line": -2.5, "v2_decision": "PASS", "v2_edge": 0.015, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LAC_LA", "week": 8, "matchup": "LAC @ LA", "home_team": "LA", "away_team": "LAC", "home_line": -7.0, "v2_decision": "SHADOW_PICK", "v2_best_team": "LA", "v2_best_side": "HOME", "v2_best_line": -7.0, "v2_best_odds": -105.0, "v2_edge": 0.021, "kickoff_utc": "2026-11-01 21:05:00+00:00"},
        {"game_id": "2026_08_TEN_CIN", "week": 8, "matchup": "TEN @ CIN", "home_team": "CIN", "away_team": "TEN", "home_line": -10.0, "v2_decision": "SHADOW_PICK", "v2_best_team": "CIN", "v2_best_side": "HOME", "v2_best_line": -10.0, "v2_best_odds": -110.0, "v2_edge": 0.023, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_NE_MIA", "week": 8, "matchup": "NE @ MIA", "home_team": "MIA", "away_team": "NE", "home_line": 8.5, "v2_decision": "PASS", "v2_edge": 0.009, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
    ]

    # V3_Shadow
    v3_rows = [
        {"game_id": "2026_01_BAL_KC", "week": 1, "v3_decision": "SHADOW_PICK", "v3_best_team": "KC", "v3_best_side": "HOME", "v3_best_line": -3.0, "v3_best_odds": -110.0, "v3_edge": 0.028},
        {"game_id": "2026_01_GB_PHI", "week": 1, "v3_decision": "SHADOW_PICK", "v3_best_team": "PHI", "v3_best_side": "HOME", "v3_best_line": -2.5, "v3_best_odds": -105.0, "v3_edge": 0.038},
        {"game_id": "2026_02_CIN_KC", "week": 2, "v3_decision": "SHADOW_PICK", "v3_best_team": "CIN", "v3_best_side": "AWAY", "v3_best_line": 5.5, "v3_best_odds": -110.0, "v3_edge": 0.036},
        {"game_id": "2026_02_CHI_HOU", "week": 2, "v3_decision": "PASS", "v3_edge": 0.014},
        # Active Week 8
        {"game_id": "2026_08_CAR_GB", "week": 8, "matchup": "CAR @ GB", "home_team": "GB", "away_team": "CAR", "home_line": -3.0, "v3_decision": "SHADOW_PICK", "v3_best_team": "GB", "v3_best_side": "HOME", "v3_best_line": -3.0, "v3_best_odds": -105.0, "v3_edge": 0.0212, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LV_NYJ", "week": 8, "matchup": "LV @ NYJ", "home_team": "NYJ", "away_team": "LV", "home_line": 1.5, "v3_decision": "SHADOW_PICK", "v3_best_team": "LV", "v3_best_side": "AWAY", "v3_best_line": -1.5, "v3_best_odds": 105.0, "v3_edge": 0.0350, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_MIN_DET", "week": 8, "matchup": "MIN @ DET", "home_team": "DET", "away_team": "MIN", "home_line": -2.5, "v3_decision": "SHADOW_PICK", "v3_best_team": "MIN", "v3_best_side": "AWAY", "v3_best_line": 2.5, "v3_best_odds": 100.0, "v3_edge": 0.0201, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LAC_LA", "week": 8, "matchup": "LAC @ LA", "home_team": "LA", "away_team": "LAC", "home_line": -7.0, "v3_decision": "SHADOW_PICK", "v3_best_team": "LA", "v3_best_side": "HOME", "v3_best_line": -7.0, "v3_best_odds": -105.0, "v3_edge": 0.0238, "kickoff_utc": "2026-11-01 21:05:00+00:00"},
        {"game_id": "2026_08_TEN_CIN", "week": 8, "matchup": "TEN @ CIN", "home_team": "CIN", "away_team": "TEN", "home_line": -10.0, "v3_decision": "SUPPRESS_DANGER", "v3_best_team": "CIN", "v3_best_side": "HOME", "v3_best_line": -10.0, "v3_best_odds": -110.0, "v3_edge": 0.0221, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_NE_MIA", "week": 8, "matchup": "NE @ MIA", "home_team": "MIA", "away_team": "NE", "home_line": 8.5, "v3_decision": "SUPPRESS_DANGER", "v3_best_team": "NE", "v3_best_side": "AWAY", "v3_best_line": -8.5, "v3_best_odds": -110.0, "v3_edge": 0.0153, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
    ]

    # V4_Shadow
    v4_rows = [
        {"game_id": "2026_01_BAL_KC", "week": 1, "v4_stake_units": 0.5, "v4_decision": "SHADOW_PICK", "v4_best_team": "KC", "v4_best_side": "HOME", "v4_best_line": -3.0, "v4_best_odds": -110.0, "v4_edge": 0.032},
        {"game_id": "2026_01_GB_PHI", "week": 1, "v4_stake_units": 1.25, "v4_decision": "HIGH_CONVICTION", "v4_best_team": "PHI", "v4_best_side": "HOME", "v4_best_line": -2.5, "v4_best_odds": -105.0, "v4_edge": 0.041},
        {"game_id": "2026_02_CIN_KC", "week": 2, "v4_stake_units": 0.8, "v4_decision": "HIGH_CONVICTION", "v4_best_team": "CIN", "v4_best_side": "AWAY", "v4_best_line": 5.5, "v4_best_odds": -110.0, "v4_edge": 0.039},
        {"game_id": "2026_02_CHI_HOU", "week": 2, "v4_decision": "PASS", "v4_edge": 0.012},
        # Active Week 8
        {"game_id": "2026_08_CAR_GB", "week": 8, "matchup": "CAR @ GB", "home_team": "GB", "away_team": "CAR", "home_line": -3.0, "v4_decision": "SHADOW_PICK", "v4_best_team": "GB", "v4_best_side": "HOME", "v4_best_line": -3.0, "v4_best_odds": -105.0, "v4_edge": 0.0245, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LV_NYJ", "week": 8, "matchup": "LV @ NYJ", "home_team": "NYJ", "away_team": "LV", "home_line": 1.5, "v4_decision": "HIGH_CONVICTION", "v4_best_team": "LV", "v4_best_side": "AWAY", "v4_best_line": -1.5, "v4_best_odds": 105.0, "v4_edge": 0.0382, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_MIN_DET", "week": 8, "matchup": "MIN @ DET", "home_team": "DET", "away_team": "MIN", "home_line": -2.5, "v4_decision": "PASS", "v4_edge": 0.0142, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_LAC_LA", "week": 8, "matchup": "LAC @ LA", "home_team": "LA", "away_team": "LAC", "home_line": -7.0, "v4_decision": "SHADOW_PICK", "v4_best_team": "LA", "v4_best_side": "HOME", "v4_best_line": -7.0, "v4_best_odds": -105.0, "v4_edge": 0.0260, "kickoff_utc": "2026-11-01 21:05:00+00:00"},
        {"game_id": "2026_08_TEN_CIN", "week": 8, "matchup": "TEN @ CIN", "home_team": "CIN", "away_team": "TEN", "home_line": -10.0, "v4_decision": "SUPPRESS_DANGER", "v4_best_team": "CIN", "v4_best_side": "HOME", "v4_best_line": -10.0, "v4_best_odds": -110.0, "v4_edge": 0.0210, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
        {"game_id": "2026_08_NE_MIA", "week": 8, "matchup": "NE @ MIA", "home_team": "MIA", "away_team": "NE", "home_line": 8.5, "v4_decision": "PASS", "v4_edge": 0.008, "kickoff_utc": "2026-11-01 18:00:00+00:00"},
    ]

    return pd.DataFrame(master_rows), pd.DataFrame(v2_rows), pd.DataFrame(v3_rows), pd.DataFrame(v4_rows), settled_scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Model Comparison Engine")
    parser.add_argument("--test", action="store_true", help="Run with demonstration mock data")
    args = parser.parse_args()

    print("=" * 70)
    print("Running Multi-Model Comparison Engine (v1 vs v2 vs v3 vs v4)...")
    print("=" * 70)

    has_creds = bool(os.getenv("GSHEET_CREDENTIALS") and os.getenv("SPREADSHEET_ID"))

    if args.test or not has_creds:
        if not has_creds:
            print("Notice: GSHEET_CREDENTIALS / SPREADSHEET_ID not detected.")
            print("Running demonstration test with sample Week 8 and historical data...\n")
        else:
            print("Running in test mode with demonstration sample data...\n")
        master_df, v2_df, v3_df, v4_df, scores = build_mock_data()
    else:
        # 1. Pull Live Data from Google Sheets
        master_df = fetch_tab_df(MASTER_SHEET_NAME)
        v2_df = fetch_tab_df("V2_Shadow")
        v3_df = fetch_tab_df("V3_Shadow")
        v4_df = fetch_tab_df("V4_Shadow")

        # Fallback to local cache if Google Sheet tabs are empty
        if v4_df.empty and os.path.exists("data/v4_shadow_picks.csv"):
            v4_df = pd.read_csv("data/v4_shadow_picks.csv")

        if master_df.empty and v2_df.empty and v3_df.empty and v4_df.empty:
            print("No data available in Google Sheet tabs or local shadow caches.")
            return

        # 2. Fetch ground truth scores from nflverse
        scores = load_final_scores(SEASON)
        print(f"Loaded finalized scores for {len(scores)} games in season {SEASON}.")

    # 3. Grade & Aggregate Performance, in-week and legacy kept apart
    blocks = evaluate_regimes(master_df, v2_df, v3_df, scores, v4_df=v4_df)
    summary_rows, guard_metrics, counts = blocks[IN_WEEK]
    legacy = blocks[LEGACY][:2]

    # 4. Build Active Slate Comparison
    slate_df = build_head_to_head_slate(master_df, v2_df, v3_df, v4_df=v4_df)

    # 5. Output to Console, GitHub Actions, and Google Sheets
    print_terminal_dashboard(summary_rows, guard_metrics, slate_df, legacy=legacy)
    publish_github_summary(summary_rows, guard_metrics, slate_df, legacy=legacy)

    if has_creds and not args.test:
        write_google_sheet_scorecard(summary_rows, guard_metrics, slate_df, legacy=legacy)

    print("Multi-Model comparison complete.")


if __name__ == "__main__":
    main()
