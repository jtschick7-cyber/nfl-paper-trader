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
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None

from src.config import FLAT_BET_SIZE, LOCAL_TZ, MASTER_SHEET_NAME, SEASON
from src.staking import compute_push_aware_stake
from src.v2_sheet_utils import get_or_create_worksheet

SCORECARD_TAB = "Model_Scorecard"


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


def grade_single_bet(side, line, odds, home_score, away_score, closing_line=None):
    """Grades a single spread bet and calculates realized CLV."""
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

    clv_pts = None
    clv_beat = None
    if closing_pick_line is not None:
        clv_pts = line - float(closing_pick_line)
        clv_beat = clv_pts > 0

    return {
        "result": result,
        "units": units,
        "clv_points": clv_pts,
        "clv_beat": clv_beat,
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
            "clv_beat_pct": 0.0,
            "avg_clv": 0.0,
        }

    wins = sum(1 for b in graded_bets if b["result"] == "WIN")
    losses = sum(1 for b in graded_bets if b["result"] == "LOSS")
    pushes = sum(1 for b in graded_bets if b["result"] == "PUSH")

    decided = wins + losses
    win_pct = (wins / decided * 100.0) if decided > 0 else 0.0

    total_units = sum(b["units"] for b in graded_bets)
    roi_pct = (total_units / (n_bets * FLAT_BET_SIZE) * 100.0) if n_bets > 0 else 0.0

    clv_evaluated = [b for b in graded_bets if b["clv_points"] is not None]
    if clv_evaluated:
        clv_beats = sum(1 for b in clv_evaluated if b["clv_beat"])
        clv_beat_pct = clv_beats / len(clv_evaluated) * 100.0
        avg_clv = sum(b["clv_points"] for b in clv_evaluated) / len(clv_evaluated)
    else:
        clv_beat_pct = 0.0
        avg_clv = 0.0

    return {
        "model": model_name,
        "tier": tier_name,
        "bets": n_bets,
        "record": f"{wins}W - {losses}L - {pushes}P",
        "win_pct": round(win_pct, 1),
        "units": round(total_units, 2),
        "roi_pct": round(roi_pct, 2),
        "clv_beat_pct": round(clv_beat_pct, 1),
        "avg_clv": round(avg_clv, 2),
    }


def evaluate_all_models(master_df, v2_df, v3_df, scores):
    """Extracts, grades, and aggregates performance across all 3 models."""
    closing_map = {}
    master_by_gid = {}
    if not master_df.empty:
        for _, row in master_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            if gid:
                master_by_gid[gid] = row
                c_line = safe_float(row.get("closing_line"))
                if c_line is not None:
                    closing_map[gid] = c_line

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
                )
                v2_bets.append(g)

    # 3. Evaluate v3 (V3_Shadow)
    v3_bets = []
    v3_high_bets = []
    suppressed_bets = []

    if not v3_df.empty:
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
                )

                if dec == "SHADOW_PICK":
                    v3_bets.append(g)
                    if edge >= 0.035:
                        v3_high_bets.append(g)
                elif dec == "SUPPRESS_DANGER":
                    suppressed_bets.append(g)

    summary_rows = [
        compute_metrics(v1_bets, "Model v1 (Master_Bets Baseline)", "All Picks"),
        compute_metrics(v2_bets, "Model v2 (Elo + Rest Baseline)", "All Picks"),
        compute_metrics(v3_bets, "Model v3 (EPA + Discrete Keys)", "All Picks"),
        compute_metrics(v3_high_bets, "Model v3 High-Conviction (>=3.5%)", "High-Conviction"),
    ]

    # Guard audit (what would have happened to suppressed danger bets?)
    guard_metrics = compute_metrics(suppressed_bets, "Guard Suppressed Danger Bets", "Hypothetical Placed")

    return summary_rows, guard_metrics, {
        "v1_count": len(v1_bets),
        "v2_count": len(v2_bets),
        "v3_count": len(v3_bets),
        "v3_high_count": len(v3_high_bets),
        "suppressed_count": len(suppressed_bets),
    }


def build_head_to_head_slate(master_df, v2_df, v3_df):
    """Builds a side-by-side comparison matrix for the active upcoming week."""
    # Find current active week
    all_weeks = []
    for df in [master_df, v2_df, v3_df]:
        if not df.empty and "week" in df.columns:
            w_vals = pd.to_numeric(df["week"], errors="coerce").dropna()
            all_weeks.extend(w_vals.tolist())

    now_utc = datetime.now(pytz.utc)

    # Group records by game_id
    m_dict = {str(r.get("game_id", "")).strip(): r for _, r in master_df.iterrows()} if not master_df.empty else {}
    v2_dict = {str(r.get("game_id", "")).strip(): r for _, r in v2_df.iterrows()} if not v2_df.empty else {}
    v3_dict = {str(r.get("game_id", "")).strip(): r for _, r in v3_df.iterrows()} if not v3_df.empty else {}

    all_gids = set(m_dict.keys()) | set(v2_dict.keys()) | set(v3_dict.keys())

    slate_rows = []
    for gid in sorted(all_gids):
        r3 = v3_dict.get(gid, {})
        r2 = v2_dict.get(gid, {})
        rm = m_dict.get(gid, {})

        ko_raw = r3.get("kickoff_utc") or rm.get("kickoff_time") or r2.get("kickoff_utc")
        ko = pd.to_datetime(ko_raw, utc=True, errors="coerce")

        # Focus on upcoming games or active week
        week = safe_int(r3.get("week") or rm.get("week") or r2.get("week"), 8)
        matchup = str(r3.get("matchup") or rm.get("matchup") or f"{rm.get('away_team','')} @ {rm.get('home_team','')}").strip()
        if not matchup or matchup == "@":
            continue

        h_line = safe_float(r3.get("home_line") or rm.get("opening_line") or r2.get("home_line"))
        h_odds = safe_float(r3.get("home_odds") or rm.get("opening_odds") or r2.get("home_odds"))

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

        if v3_dec == "SHADOW_PICK":
            v3_odds = safe_float(r3.get("v3_best_odds"), -110.0)
            v3_prob = safe_float(r3.get("v3_model_probability"), 0.52)
            p_push = safe_float(r3.get("p_push"), 0.05)
            stake_info = compute_push_aware_stake(v3_prob, p_push, v3_odds)
            stake_units = stake_info["recommended_units"]
            v3_pick_str = f"{v3_team} {v3_line:+.1f} (+{v3_edge*100:.1f}%) [{stake_units:.1f}u]"
            if v3_edge >= 0.035:
                v3_pick_str += " ★"
        elif v3_dec == "SUPPRESS_DANGER":
            v3_pick_str = "SUPPRESS_DANGER"
        else:
            v3_pick_str = "PASS"

        # Consensus evaluation
        picks = []
        if v1_pick_str not in {"PASS", ""}:
            picks.append(("v1", v1_team))
        if v2_pick_str not in {"PASS", "SUPPRESS", ""}:
            picks.append(("v2", v2_team))
        if v3_dec == "SHADOW_PICK" and v3_team:
            picks.append(("v3", v3_team))

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
            "consensus": consensus,
            "v3_decision": v3_dec,
            "is_future": bool(pd.notna(ko) and ko > now_utc),
        })

    return pd.DataFrame(slate_rows)


def print_terminal_dashboard(summary_rows, guard_metrics, slate_df):
    """Renders a clean CLI dashboard."""
    print("\n" + "=" * 90)
    print("           NFL MULTI-MODEL COMPARISON SCORECARD (v1 vs v2 vs v3)           ")
    print("=" * 90)

    print(f"{'Model / Architecture':<34} | {'Bets':<5} | {'Record':<14} | {'Win %':<6} | {'Units':<8} | {'ROI %':<7} | {'CLV Beat':<8}")
    print("-" * 90)

    for row in summary_rows:
        u_str = f"{row['units']:+.2f}u" if row['bets'] > 0 else "0.00u"
        roi_str = f"{row['roi_pct']:+.1f}%" if row['bets'] > 0 else "0.0%"
        clv_str = f"{row['clv_beat_pct']:.1f}%" if row['bets'] > 0 else "0.0%"
        print(f"{row['model']:<34} | {row['bets']:<5} | {row['record']:<14} | {row['win_pct']:>5.1f}% | {u_str:>8} | {roi_str:>7} | {clv_str:>8}")

    print("-" * 90)
    g_bets = guard_metrics['bets']
    if g_bets > 0:
        g_units = f"{guard_metrics['units']:+.2f}u"
        saved = -guard_metrics['units']
        saved_str = f"{saved:+.2f}u" if saved > 0 else "0.00u"
        print(f"{'Guard Suppressed Bets (Saved)':<34} | {g_bets:<5} | {guard_metrics['record']:<14} | Would Have PnL: {g_units} (Estimated Saved: {saved_str})")
    else:
        print(f"{'Guard Suppressed Bets (Saved)':<34} | 0     | 0W - 0L - 0P   | No completed games with suppressed danger yet.")
    print("=" * 90)

    # Active week slate
    if not slate_df.empty:
        future_slate = slate_df[slate_df["is_future"]].copy()
        if future_slate.empty:
            future_slate = slate_df.tail(16).copy()

        print("\n" + "=" * 90)
        print("                 ACTIVE SLATE HEAD-TO-HEAD PICKS MATRIX                   ")
        print("=" * 90)
        print(f"{'Matchup':<12} | {'Spread':<7} | {'v1 Baseline':<14} | {'v2 Elo/Rest':<14} | {'v3 EPA-Enhanced':<24} | {'Consensus':<14}")
        print("-" * 90)
        for _, r in future_slate.iterrows():
            print(f"{r['matchup']:<12} | {r['market_home_line']:<7} | {r['v1_pick']:<14} | {r['v2_pick']:<14} | {r['v3_pick']:<24} | {r['consensus']:<14}")
        print("=" * 90 + "\n")


def publish_github_summary(summary_rows, guard_metrics, slate_df):
    """Appends Markdown dashboard to GitHub Actions Step Summary if running in CI."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    md_lines = []
    md_lines.append("## 🏈 NFL Multi-Model Performance Scorecard\n")
    md_lines.append(f"*Last updated: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")

    md_lines.append("| Model / Architecture | Bets | Record | Win % | Units PnL | ROI % | CLV Beat % |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in summary_rows:
        u_str = f"**{r['units']:+.2f}u**" if r['bets'] > 0 else "0.00u"
        roi_str = f"**{r['roi_pct']:+.1f}%**" if r['bets'] > 0 else "0.0%"
        md_lines.append(f"| {r['model']} | {r['bets']} | {r['record']} | {r['win_pct']:.1f}% | {u_str} | {roi_str} | {r['clv_beat_pct']:.1f}% |")

    if guard_metrics['bets'] > 0:
        saved = -guard_metrics['units']
        saved_str = f"+{saved:.2f}u" if saved > 0 else "0.00u"
        md_lines.append(f"| 🛡️ *Injury/RLM Suppressed (Saved)* | {guard_metrics['bets']} | {guard_metrics['record']} | - | Would PnL: {guard_metrics['units']:+.2f}u | Saved: **{saved_str}** | - |")

    md_lines.append("\n### 📋 Active Slate Head-to-Head Comparison\n")
    md_lines.append("| Matchup | Market Spread | v1 Baseline | v2 Elo/Rest | v3 EPA-Enhanced | Consensus |")
    md_lines.append("| :--- | :---: | :---: | :---: | :--- | :---: |")

    if not slate_df.empty:
        future_slate = slate_df[slate_df["is_future"]].copy()
        if future_slate.empty:
            future_slate = slate_df.tail(16).copy()

        for _, r in future_slate.iterrows():
            md_lines.append(f"| **{r['matchup']}** | {r['market_home_line']} | `{r['v1_pick']}` | `{r['v2_pick']}` | `{r['v3_pick']}` | **{r['consensus']}** |")

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print("Wrote scorecard summary to $GITHUB_STEP_SUMMARY.")
    except Exception as exc:
        print(f"Warning: Could not write to $GITHUB_STEP_SUMMARY: {exc}")


def write_google_sheet_scorecard(summary_rows, guard_metrics, slate_df):
    """Writes or overwrites the Model_Scorecard tab in Google Sheets."""
    now_utc_str = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    now_et_str = datetime.now(pytz.timezone(LOCAL_TZ)).strftime("%Y-%m-%d %I:%M %p %Z")

    rows = []
    rows.append(["NFL MULTI-MODEL PERFORMANCE SCORECARD", "", "", "", "", "", ""])
    rows.append([f"Updated: {now_utc_str} ({now_et_str})", "", "", "", "", "", ""])
    rows.append([""])

    # Section 1: Summary Table
    rows.append(["MODEL PERFORMANCE SUMMARY", "", "", "", "", "", ""])
    rows.append(["Model", "Tier", "Bets", "Record", "Win %", "Net Units", "ROI %", "CLV Beat %", "Avg CLV Pts"])

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
            f"{r['avg_clv']:+.2f}",
        ])

    # Guard Audit row
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
            "-",
            "-",
        ])

    rows.append([""])
    rows.append([""])

    # Section 2: Active Slate
    rows.append(["ACTIVE SLATE HEAD-TO-HEAD MATRIX", "", "", "", "", "", ""])
    rows.append(["Week", "Matchup", "Kickoff (UTC)", "Market Spread", "v1 Baseline", "v2 Elo/Rest", "v3 EPA-Enhanced", "Consensus"])

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
                r["consensus"],
            ])

    try:
        ws = get_or_create_worksheet(SCORECARD_TAB)
        ws.clear()
        ws.update(values=rows, range_name="A1")
        print(f"Successfully updated '{SCORECARD_TAB}' tab in Google Sheets.")
    except (Exception, SystemExit) as exc:
        print(f"Warning: Could not write to Google Sheets '{SCORECARD_TAB}': {exc}")


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

    return pd.DataFrame(master_rows), pd.DataFrame(v2_rows), pd.DataFrame(v3_rows), settled_scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Model Comparison Engine")
    parser.add_argument("--test", action="store_true", help="Run with demonstration mock data")
    args = parser.parse_args()

    print("=" * 70)
    print("Running Multi-Model Comparison Engine (v1 vs v2 vs v3)...")
    print("=" * 70)

    has_creds = bool(os.getenv("GSHEET_CREDENTIALS") and os.getenv("SPREADSHEET_ID"))

    if args.test or not has_creds:
        if not has_creds:
            print("Notice: GSHEET_CREDENTIALS / SPREADSHEET_ID not detected.")
            print("Running demonstration test with sample Week 8 and historical data...\n")
        else:
            print("Running in test mode with demonstration sample data...\n")
        master_df, v2_df, v3_df, scores = build_mock_data()
    else:
        # 1. Pull Live Data from Google Sheets
        master_df = fetch_tab_df(MASTER_SHEET_NAME)
        v2_df = fetch_tab_df("V2_Shadow")
        v3_df = fetch_tab_df("V3_Shadow")

        if master_df.empty and v2_df.empty and v3_df.empty:
            print("No data available in Google Sheet tabs.")
            return

        # 2. Fetch ground truth scores from nflverse
        scores = load_final_scores(SEASON)
        print(f"Loaded finalized scores for {len(scores)} games in season {SEASON}.")

    # 3. Grade & Aggregate Performance
    summary_rows, guard_metrics, counts = evaluate_all_models(master_df, v2_df, v3_df, scores)

    # 4. Build Active Slate Comparison
    slate_df = build_head_to_head_slate(master_df, v2_df, v3_df)

    # 5. Output to Console, GitHub Actions, and Google Sheets
    print_terminal_dashboard(summary_rows, guard_metrics, slate_df)
    publish_github_summary(summary_rows, guard_metrics, slate_df)

    if has_creds and not args.test:
        write_google_sheet_scorecard(summary_rows, guard_metrics, slate_df)

    print("Multi-Model comparison complete.")


if __name__ == "__main__":
    main()
