"""Top-down player_receptions scanner (P4, notes/implementation-plan/P4-prop-scanner.md).

    .venv/bin/python -u -m src.props.topdown_props                  # reads origin/odds-data
    .venv/bin/python -u -m src.props.topdown_props --source DIR     # a checkout of the branch

The rules and constants below were frozen in the spec on 2026-11-24, before any evaluation
data (only P1's probe file existed, and it is excluded). Do not tune them;
tests/test_topdown_props.py asserts each one.

- Sightings: every `open`-phase snapshot. Each (event, player) with a two-sided ANCHOR_BOOK
  quote gets lam_open (Poisson, prop_pricing.implied_lambda on the de-vigged price). Every
  US_TARGETS / OFFSHORE offer on that player, at any line and one-sided or not, is priced
  against it; EV >= FLAG_EV flags. The anchor books are never priced or flagged.
- The ALT_ANCHOR_BOOK table repeats everything with BetOnline as the anchor (`anchor` column).
- Close: the event's latest `close`-phase snapshot captured before kickoff. lam_close comes
  from the same anchor book at the close; close_ev is the offer priced against it (primary
  CLV). close_ev_median prices against the median lam of every two-sided classed book at the
  close. No anchor quote at the close, or no close snapshot: close_ev is NaN, `close_status`
  says which.
- ev_nb / close_ev_nb redo the implied lam and the price under NB(r = NB_SENSITIVITY_R).
- `probe`-phase files are ignored. Books in no class are ignored and listed once.

Builder's choice the spec leaves open: when a book quotes several two-sided lines for one
player, its lam comes from the line whose de-vigged q_over is nearest 0.5 (ties: lower line).

Writes data/topdown_props_offers.csv. Read-only otherwise. No Odds API call, no Sheet write.
"""

import argparse
import os
import statistics

import numpy as np
import pandas as pd

from src.odds_utils import TEAM_ABBR_MAP
from src.props.load_prop_archive import DEFAULT_REF, list_prop_snapshots, read_prop_snapshot
from src.props.prop_pricing import NB_SENSITIVITY_R, implied_lambda, price_prop_offer, prop_quotes
from src.topdown_replay import cluster_se
from src.total_pricing import devig_two_way

# ---- frozen (spec 2026-11-24) ---------------------------------------------------------------
ANCHOR_BOOK = "pinnacle"                    # the only sharp prop book returned
ALT_ANCHOR_BOOK = "betonlineag"             # sensitivity anchor only, never flagged
US_TARGETS = {"draftkings", "fanduel", "betmgm", "betrivers", "williamhill_us",
              "espnbet", "fanatics", "ballybet", "hardrockbet"}
OFFSHORE = {"bovada", "betus", "mybookieag"}  # reported, never in the gate
FLAG_EV = 0.02
GATE_MIN_OFFERS = 150
GATE_MIN_GAMES = 30
GATE_MIN_Z = 2.0
# ---------------------------------------------------------------------------------------------

OUT_PATH = "data/topdown_props_offers.csv"
ANCHORS = (ANCHOR_BOOK, ALT_ANCHOR_BOOK)
SIDES = ("OVER", "UNDER")


def book_class(book):
    if book == ANCHOR_BOOK:
        return "anchor"
    if book == ALT_ANCHOR_BOOK:
        return "alt_anchor"
    if book in US_TARGETS:
        return "us"
    if book in OFFSHORE:
        return "offshore"
    return None


def _main_line_lambdas(quotes):
    """{(book, player): (line, lam_poisson, lam_nb)} from each book's two-sided quotes."""
    best = {}
    for q in quotes:
        if q["over_odds"] is None or q["under_odds"] is None:
            continue
        p, _ = devig_two_way(q["over_odds"], q["under_odds"])
        key = (q["book"], q["player"])
        rank = (abs(p - 0.5), q["line"])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, q["line"], p)
    return {k: (line, implied_lambda(line, p), implied_lambda(line, p, r=NB_SENSITIVITY_R))
            for k, (_, line, p) in best.items()}


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _close_for(snaps, kickoff):
    """The latest close-phase snapshot captured strictly before kickoff, or None."""
    k = _utc(kickoff)
    before = [s for s in snaps if _utc(s["captured_at"]) < k]
    return max(before, key=lambda s: _utc(s["captured_at"])) if before else None


def evaluate_open(snap, close, ignored=None):
    """Rows for one open-phase snapshot, priced against both anchors, with the close attached."""
    resp = snap["response"]
    kickoff = resp.get("commence_time") or snap.get("kickoff")
    captured = _utc(snap["captured_at"])
    quotes = []
    for q in prop_quotes(resp):
        if book_class(q["book"]) is None:
            if ignored is not None:
                ignored.add(q["book"])
            continue
        quotes.append(q)
    lam_open = _main_line_lambdas(quotes)
    lam_close, close_median = {}, {}
    if close is not None:
        cq = [q for q in prop_quotes(close["response"]) if book_class(q["book"]) is not None]
        if ignored is not None:
            ignored.update(q["book"] for q in prop_quotes(close["response"]) if book_class(q["book"]) is None)
        lam_close = _main_line_lambdas(cq)
        by_player = {}
        for (_, player), (_, lam, _) in lam_close.items():
            by_player.setdefault(player, []).append(lam)
        close_median = {p: statistics.median(v) for p, v in by_player.items()}

    home, away = resp.get("home_team"), resp.get("away_team")
    rows = []
    for q in quotes:
        cls = book_class(q["book"])
        if cls not in ("us", "offshore"):
            continue
        one_sided = q["over_odds"] is None or q["under_odds"] is None
        upd = q.get("last_update")
        age = (captured - _utc(upd)).total_seconds() / 60.0 if upd else np.nan
        for anchor in ANCHORS:
            a = lam_open.get((anchor, q["player"]))
            if a is None:
                continue
            a_line, lam, lam_nb = a
            c = lam_close.get((anchor, q["player"]))
            status = "no_close_snapshot" if close is None else ("ok" if c else "anchor_missing_at_close")
            med = close_median.get(q["player"]) if c else None
            for side, odds in (("OVER", q["over_odds"]), ("UNDER", q["under_odds"])):
                if odds is None:
                    continue
                ev = price_prop_offer(lam, side, q["line"], odds)[2]
                rows.append({
                    "event_id": resp.get("id"), "kickoff": kickoff,
                    "home_team": TEAM_ABBR_MAP.get(home, home), "away_team": TEAM_ABBR_MAP.get(away, away),
                    "anchor": anchor, "player": q["player"], "player_name": q["player_name"],
                    "book": q["book"], "book_class": cls, "side": side,
                    "line": q["line"], "odds": odds, "one_sided": one_sided,
                    "anchor_line": a_line, "same_line": q["line"] == a_line,
                    "lam_open": lam, "ev": ev,
                    "ev_nb": price_prop_offer(lam_nb, side, q["line"], odds, r=NB_SENSITIVITY_R)[2],
                    "flagged": ev >= FLAG_EV,
                    "lam_close": c[1] if c else np.nan,
                    "close_ev": price_prop_offer(c[1], side, q["line"], odds)[2] if c else np.nan,
                    "close_ev_nb": price_prop_offer(c[2], side, q["line"], odds, r=NB_SENSITIVITY_R)[2] if c else np.nan,
                    "close_ev_median": price_prop_offer(med, side, q["line"], odds)[2] if med is not None else np.nan,
                    "close_status": status,
                    "hours_to_kickoff": (_utc(kickoff) - captured).total_seconds() / 3600.0,
                    "quote_age_min": age,
                    "snapshot_ts": snap["captured_at"],
                    "close_ts": close["captured_at"] if close is not None else None,
                })
    return rows


COLUMNS = ["event_id", "kickoff", "home_team", "away_team", "anchor", "player", "player_name", "book",
           "book_class", "side", "line", "odds", "one_sided", "anchor_line", "same_line", "lam_open", "ev",
           "ev_nb", "flagged", "lam_close", "close_ev", "close_ev_nb", "close_ev_median", "close_status",
           "hours_to_kickoff", "quote_age_min", "snapshot_ts", "close_ts"]


def replay(snapshots, ignored=None):
    """DataFrame of every priced offer in every open-phase snapshot (probe files ignored)."""
    closes = {}
    for s in snapshots:
        if s.get("phase") == "close":
            closes.setdefault(s["response"].get("id"), []).append(s)
    rows = []
    for s in snapshots:
        if s.get("phase") != "open":
            continue
        resp = s["response"]
        kickoff = resp.get("commence_time") or s.get("kickoff")
        rows += evaluate_open(s, _close_for(closes.get(resp.get("id"), []), kickoff), ignored)
    return pd.DataFrame(rows, columns=COLUMNS)


# ---- report and gate --------------------------------------------------------------------------

def gate_group(df):
    """The pre-registered group: Pinnacle-anchored, flagged, US-target offers with a close_ev."""
    if df.empty:
        return df
    return df[(df["anchor"] == ANCHOR_BOOK) & df["flagged"].astype(bool) & (df["book_class"] == "us")
              & df["close_ev"].notna()]


def drift_base(df):
    """All Pinnacle-anchored US-target offers with a close_ev, flagged or not."""
    if df.empty:
        return df
    return df[(df["anchor"] == ANCHOR_BOOK) & (df["book_class"] == "us") & df["close_ev"].notna()]


def _top2_games(g):
    return list(g.groupby("event_id")["close_ev"].sum().sort_values(ascending=False).index[:2])


def _side_means(g):
    if g.empty:
        return {}
    return {s: float(x["close_ev"].mean()) for s, x in g.groupby("side")}


def gate_summary(g, all_us):
    """offers, games, mean_close_ev, close_ev_se (game-clustered), mean_close_ev_ex_top2, and
    per-side means of the gate group (side_close_ev) and of all US offers (side_all_close_ev)."""
    n = len(g)
    if n == 0:
        return {"offers": 0, "games": 0, "mean_close_ev": float("nan"), "close_ev_se": float("nan"),
                "mean_close_ev_ex_top2": float("nan"), "side_close_ev": {},
                "side_all_close_ev": _side_means(all_us)}
    rest = g[~g["event_id"].isin(_top2_games(g))]
    return {"offers": n, "games": int(g["event_id"].nunique()), "mean_close_ev": float(g["close_ev"].mean()),
            "close_ev_se": cluster_se(g["close_ev"], g["event_id"]),
            "mean_close_ev_ex_top2": float(rest["close_ev"].mean()) if len(rest) else float("nan"),
            "side_close_ev": _side_means(g), "side_all_close_ev": _side_means(all_us)}


def drift_ok(s):
    """Every side in the gate group beats that side's all-offer mean close_ev."""
    sides = s["side_close_ev"]
    return bool(sides) and all(
        side in s["side_all_close_ev"] and m > s["side_all_close_ev"][side] for side, m in sides.items())


def gate(s, season_settled=False):
    """'pass' / 'fail' per the frozen test; 'inconclusive' once week 18 has settled with
    neither; 'pending' before that (weekly re-runs are descriptive only)."""
    size = s["offers"] >= GATE_MIN_OFFERS and s["games"] >= GATE_MIN_GAMES
    m, se, ex = s["mean_close_ev"], s["close_ev_se"], s["mean_close_ev_ex_top2"]
    if size and m > 0 and se > 0 and m >= GATE_MIN_Z * se and ex > 0 and drift_ok(s):
        return "pass"
    if size and m <= 0:
        return "fail"
    return "inconclusive" if season_settled else "pending"


def group_stats(g, name):
    n = len(g)
    res = g["result"] if n and "result" in g else pd.Series(dtype=object)
    c = g["close_ev"].dropna() if n else pd.Series(dtype=float)
    return {
        "group": name, "offers": n, "games": int(g["event_id"].nunique()) if n else 0,
        "players": int(g["player"].nunique()) if n and "player" in g else 0,
        "mean_ev": float(g["ev"].mean()) if n else float("nan"),
        "close_ev": float(c.mean()) if len(c) else float("nan"),
        "se_game": cluster_se(g["close_ev"], g["event_id"]) if len(c) else float("nan"),
        "close_ev_nb": float(g["close_ev_nb"].mean()) if len(c) else float("nan"),
        "close_ev_median": float(g["close_ev_median"].mean()) if len(c) else float("nan"),
        "beat_close_pct": 100 * float((c > 0).mean()) if len(c) else float("nan"),
        "W-L-P-V": f"{(res == 'WIN').sum()}-{(res == 'LOSS').sum()}-{(res == 'PUSH').sum()}-{(res == 'VOID').sum()}",
        "units": float(g.loc[res.isin(["WIN", "LOSS", "PUSH", "VOID"]).values, "units"].sum()) if len(res) else 0.0,
    }


def _table(rows):
    if not rows:
        return "(none)"
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}")


def report(df, unresolved=(), season_settled=False):
    lines = []
    prim = df[df["anchor"] == ANCHOR_BOOK] if not df.empty else df
    g = gate_group(df)
    base = drift_base(df)
    rest = g[~g["event_id"].isin(_top2_games(g))] if len(g) else g
    lines += ["1-2. Gate group: Pinnacle anchor, flagged (EV >= 2%), US targets, with a close_ev",
              _table([group_stats(g, "gate group"), group_stats(rest, "gate group, top 2 games removed")])]
    gs = gate_summary(g, base)
    z = gs["mean_close_ev"] / gs["close_ev_se"] if gs["close_ev_se"] and gs["close_ev_se"] > 0 else float("nan")
    lines.append(f"gate: {gate(gs, season_settled)}  ({gs['offers']} offers on {gs['games']} games; need "
                 f"{GATE_MIN_OFFERS} on {GATE_MIN_GAMES}; z = {z:.2f}; drift rule "
                 f"{'met' if drift_ok(gs) else 'not met'})")

    f = prim[prim["flagged"].astype(bool) & (prim["book_class"] == "us")] if len(prim) else prim
    lines += ["", "3a. Flagged US offers by book", _table([group_stats(x, b) for b, x in f.groupby("book")]) if len(f) else "(none)",
              "", "3b. Flagged US offers, Over vs Under",
              _table([group_stats(x, s) for s, x in f.groupby("side")]) if len(f) else "(none)",
              "", "3c. Flagged US offers, one-sided vs two-sided",
              _table([group_stats(x, "one-sided" if k else "two-sided") for k, x in f.groupby("one_sided")]) if len(f) else "(none)",
              "", "3d. Flagged US offers, same line as Pinnacle vs different line",
              _table([group_stats(x, "same line" if k else "different line") for k, x in f.groupby("same_line")]) if len(f) else "(none)",
              "", "(close_ev_nb and close_ev_median sit beside close_ev in every table.)"]
    alt = df[df["anchor"] == ALT_ANCHOR_BOOK] if not df.empty else df
    ga = alt[alt["flagged"].astype(bool) & (alt["book_class"] == "us")] if len(alt) else alt
    lines += ["", f"3e. Alternate anchor ({ALT_ANCHOR_BOOK}): flagged US offers, all US offers",
              _table([group_stats(ga, "flagged US"), group_stats(alt[alt["book_class"] == "us"] if len(alt) else alt, "all US")])]
    off = prim[prim["book_class"] == "offshore"] if len(prim) else prim
    lines += ["", "3f. Offshore offers (Pinnacle anchor; never in the gate)",
              _table([group_stats(off[off["flagged"].astype(bool)] if len(off) else off, "flagged offshore"),
                      group_stats(off, "all offshore")])]
    allus = prim[prim["book_class"] == "us"] if len(prim) else prim
    lines += ["", "4. Drift check: all US-target offers by side (Pinnacle anchor), flagged or not",
              _table([group_stats(x, f"all US {s}") for s, x in allus.groupby("side")]) if len(allus) else "(none)"]

    lines += ["", "5. Coverage"]
    if df.empty:
        lines.append("no priced offers")
    else:
        lines.append(f"Pinnacle-anchored offers: {len(prim)} on {prim['event_id'].nunique()} games, "
                     f"{prim['player'].nunique()} players, books: {', '.join(sorted(prim['book'].unique()))}")
        lines.append(f"BetOnline-anchored offers: {len(alt)}")
        cs = prim["close_status"].value_counts().to_dict()
        lines.append(f"close status (Pinnacle rows): {cs}")
        if "result" in df:
            gr = prim["grade_status"].value_counts().to_dict() if "grade_status" in prim else {}
            lines.append(f"grading status (Pinnacle rows): {gr}")
    if unresolved:
        lines.append("ungraded players (never guessed):")
        lines += [f"  {u['game']}: {u['player_name']} ({u['reason']})" for u in unresolved]
    return "\n".join(lines)


def load_snapshots(source=DEFAULT_REF):
    """All archived prop snapshots (the replay ignores probe files)."""
    return [read_prop_snapshot(p, source) for _, p in list_prop_snapshots(source)]


def main():
    from src.props.player_stats import attach_results, load_player_data

    ap = argparse.ArgumentParser(description="Top-down player_receptions scanner over the prop archive")
    ap.add_argument("--source", default=DEFAULT_REF, help="git ref or a checkout directory of odds-data")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--no-refresh", action="store_true", help="grade from the cached nflverse files")
    ap.add_argument("--season-settled", action="store_true", help="week 18 has settled: report inconclusive, not pending")
    args = ap.parse_args()

    snaps = load_snapshots(args.source)
    phases = pd.Series([s.get("phase") for s in snaps]).value_counts().to_dict() if snaps else {}
    print(f"{len(snaps)} prop snapshots by phase: {phases} (probe files ignored)")
    ignored = set()
    df = replay(snaps, ignored)
    if ignored:
        print("books outside every class, ignored: " + ", ".join(sorted(ignored)))
    unresolved = []
    if len(df):
        stats, snapc = load_player_data(args.season, refresh=not args.no_refresh)
        df, unresolved = attach_results(df, stats, snapc)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} priced offers on {df['event_id'].nunique() if len(df) else 0} games; wrote {OUT_PATH}\n")
    pd.set_option("display.width", 240)
    print(report(df, unresolved, args.season_settled))


if __name__ == "__main__":
    main()
