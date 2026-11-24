"""Sharp-vs-soft top-down replay v2, spreads and totals (P2, notes/implementation-plan/P2-topdown-v2.md).

    .venv/bin/python -u -m src.topdown_v2                  # reads origin/odds-data
    .venv/bin/python -u -m src.topdown_v2 --source DIR     # a checkout of the branch

The rules and constants below were frozen in the spec on 2026-11-24, before any evaluation
data. Do not tune them; tests/test_topdown_v2.py asserts each one.

Per snapshot and event, for each market (spread, total):
- every book's two-sided quote gives an implied mean (home margin for spreads via
  market_consensus.implied_mean, total points via total_pricing.total_implied_mean);
- the anchor is the weighted median of the sharp-group votes (one vote per group, the first
  book present in the group); no group present means the market is skipped;
- only US_TARGETS and OFFSHORE books are priced against the anchor; EV >= FLAG_EV flags.

Each first sighting of (event, market, book, side, line, price) before that market's close
(its last snapshot strictly before kickoff with an anchor) is priced again against the
closing anchor (close_ev, the primary CLV) and against the plain median of every book's
implied mean at the close (close_ev_allbooks). Spread rows keep the home-perspective line in
`line`, as Phase 3 does. Only games that kicked off by the archive's latest snapshot count.

Writes data/topdown_v2_offers.csv. Read-only otherwise. Imports from, never edits,
src/market_consensus.py and src/topdown_replay.py (the pre-registered Phase 3 test).
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

from src.load_odds_archive import DEFAULT_REF, iter_archive
from src.market_consensus import book_quotes, crosses_key, implied_mean, price_offer, weighted_median
from src.odds_utils import TEAM_ABBR_MAP
from src.topdown_replay import cluster_se
from src.total_pricing import TOTAL_SIGMA, devig_two_way, price_total_offer, total_implied_mean, total_quotes

# ---- frozen (spec 2026-11-24) ---------------------------------------------------------------
SHARP_GROUPS = [            # one vote per group; first book present in the group is used
    (("pinnacle",), 3.0),
    (("circasports",), 2.0),
    (("betonlineag", "lowvig"), 2.0),
]
US_TARGETS = {"fanduel", "draftkings", "betmgm", "betrivers", "williamhill_us",
              "espnbet", "fanatics", "ballybet", "hardrockbet"}
OFFSHORE = {"betus", "mybookieag", "bovada"}   # reported, never in the gate
FLAG_EV = 0.015
PERSIST_EV = 0.005
PERSIST_HOURS = 6
IN_WEEK_HOURS = 144
SPREAD_SIGMA = 12.8          # = market_consensus.DEFAULT_SIGMA
EVAL_START = "2026-11-25T00:00:00Z"
GATE_MIN_OFFERS = 80
GATE_MIN_GAMES = 35
GATE_MIN_Z = 2.0
# ---------------------------------------------------------------------------------------------

SHARP_BOOKS = {b for books, _ in SHARP_GROUPS for b in books}
SIDES = {"spread": ("HOME", "AWAY"), "total": ("OVER", "UNDER")}
OUT_PATH = "data/topdown_v2_offers.csv"
GAMES_PATH = "data/nflverse_cache/games.csv"


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _utc_or_none(x):
    try:
        return _utc(x) if x else None
    except (TypeError, ValueError):
        return None


def book_class(book):
    return "us" if book in US_TARGETS else "offshore" if book in OFFSHORE else None


def sample_of(kickoff, snapshot_ts):
    """'evaluation' only when both the kickoff and the snapshot are at or after EVAL_START."""
    start = _utc(EVAL_START)
    return "evaluation" if _utc(kickoff) >= start and _utc(snapshot_ts) >= start else "design"


def sharp_anchor(means):
    """(anchor_mean, anchor_groups, voting_books) from {book: implied mean}."""
    votes, weights, books = [], [], []
    for group, w in SHARP_GROUPS:
        b = next((b for b in group if means.get(b) is not None), None)
        if b is not None:
            votes.append(means[b])
            weights.append(w)
            books.append(b)
    if not votes:
        return None, 0, []
    return weighted_median(votes, weights), len(votes), books


def _offer_ev(market, mu, side, line, odds, median_total):
    if market == "spread":
        return price_offer(mu, side, line, odds, SPREAD_SIGMA, median_total)[2]
    return price_total_offer(mu, side, line, odds, TOTAL_SIGMA)[2]


def market_states(event):
    """{market: state} for one event: per-book quotes and implied means, the sharp anchor,
    the anchor line and last_update, and the all-books median mean."""
    sq = book_quotes(event)
    totals = [q["total_line"] for q in sq if q["total_line"] is not None]
    median_total = float(np.median(totals)) if totals else None
    states = {
        "spread": {
            "quotes": {q["book"]: {"line": q["home_line"], "odds": {"HOME": q["home_odds"], "AWAY": q["away_odds"]},
                                   "last_update": q["last_update"]} for q in sq},
            "means": {q["book"]: implied_mean(q, SPREAD_SIGMA, median_total) for q in sq},
        },
        "total": {
            "quotes": {}, "means": {},
        },
    }
    for q in total_quotes(event):
        states["total"]["quotes"][q["book"]] = {"line": q["line"], "last_update": q["last_update"],
                                                "odds": {"OVER": q["over_odds"], "UNDER": q["under_odds"]}}
        states["total"]["means"][q["book"]] = total_implied_mean(
            q["line"], devig_two_way(q["over_odds"], q["under_odds"])[0], TOTAL_SIGMA)
    for st in states.values():
        st["median_total"] = median_total
        st["anchor_mean"], st["anchor_groups"], books = sharp_anchor(st["means"])
        st["anchor_books"] = books
        st["anchor_line"] = float(np.median([st["quotes"][b]["line"] for b in books])) if books else None
        ups = [u for u in (_utc_or_none(st["quotes"][b]["last_update"]) for b in books) if u is not None]
        st["anchor_last_update"] = max(ups) if ups else None
        st["allbooks_mean"] = float(np.median(list(st["means"].values()))) if st["means"] else None
    return states


def evaluate_event(event):
    """(rows, states, ignored_books): one row per US/offshore (market, book, side) priced
    against the sharp anchor; sharp-group books are never priced."""
    states = market_states(event)
    ignored = {b.get("key") for b in event.get("bookmakers", []) or []
               if b.get("key") not in SHARP_BOOKS and book_class(b.get("key")) is None}
    home = TEAM_ABBR_MAP.get(event.get("home_team"), event.get("home_team"))
    away = TEAM_ABBR_MAP.get(event.get("away_team"), event.get("away_team"))
    rows = []
    for market, st in states.items():
        mu = st["anchor_mean"]
        if mu is None:
            continue
        for book, q in st["quotes"].items():
            cls = book_class(book)
            if cls is None:
                continue
            for side in SIDES[market]:
                odds = q["odds"][side]
                if odds is None or odds == 0:
                    continue
                ev = _offer_ev(market, mu, side, q["line"], odds, st["median_total"])
                rows.append({
                    "event_id": event.get("id"), "commence_time": event.get("commence_time"),
                    "home_team": home, "away_team": away, "market": market, "book": book, "book_class": cls,
                    "side": side, "line": q["line"], "odds": odds, "anchor_mean": mu,
                    "anchor_groups": st["anchor_groups"], "anchor_line": st["anchor_line"],
                    "ev": ev, "flagged": ev >= FLAG_EV,
                    "crosses_key": crosses_key(side, q["line"], st["anchor_line"]) if market == "spread" else None,
                    "last_update": q["last_update"], "anchor_last_update": st["anchor_last_update"],
                })
    return rows, states, {b for b in ignored if b}


def replay(snapshots, ignored=None):
    """One row per first sighting of each (event, market, book, side, line, price) before
    that market's close. `ignored`, if a set, collects books outside every class."""
    snaps = sorted(((_utc(t), evs) for t, evs in snapshots), key=lambda x: x[0])
    if not snaps:
        return pd.DataFrame()
    archive_end = snaps[-1][0]
    evaluated = []   # per snapshot: {event_id: (event, rows, states)}
    for t, events in snaps:
        cur = {}
        for e in events:
            rows, states, ign = evaluate_event(e)
            if ignored is not None:
                ignored |= ign
            cur[e.get("id")] = (e, rows, states)
        evaluated.append(cur)

    close = {}   # (event_id, market) -> closing state, from the last pre-kickoff snapshot with an anchor
    for (t, _), cur in zip(snaps, evaluated):
        for eid, (e, _, states) in cur.items():
            k = _utc(e["commence_time"])
            if t >= k or k > archive_end:
                continue
            for market, st in states.items():
                if st["anchor_mean"] is not None:
                    close[(eid, market)] = {"ts": t, **st}

    index = [{(eid, r["market"], r["book"], r["side"]): r for eid, (_, rows, _) in cur.items() for r in rows}
             for cur in evaluated]
    out, seen = [], set()
    for i, ((t, _), cur) in enumerate(zip(snaps, evaluated)):
        nxt_ok = i + 1 < len(snaps) and (snaps[i + 1][0] - t) <= pd.Timedelta(hours=PERSIST_HOURS)
        for eid, (e, rows, _) in cur.items():
            k = _utc(e["commence_time"])
            for r in rows:
                c = close.get((eid, r["market"]))
                if c is None or t >= c["ts"]:
                    continue
                key = (eid, r["market"], r["book"], r["side"], r["line"], r["odds"])
                if key in seen:
                    continue
                seen.add(key)
                if nxt_ok:
                    n = index[i + 1].get((eid, r["market"], r["book"], r["side"]))
                    persisted = bool(n is not None and n["line"] == r["line"] and n["odds"] == r["odds"]
                                     and n["ev"] >= PERSIST_EV)
                else:
                    persisted = np.nan
                lu, alu = _utc_or_none(r["last_update"]), r["anchor_last_update"]
                hours = (k - t).total_seconds() / 3600
                row = dict(r)
                row.update({
                    "close_ts": c["ts"], "close_anchor_mean": c["anchor_mean"],
                    "close_ev": _offer_ev(r["market"], c["anchor_mean"], r["side"], r["line"], r["odds"],
                                          c["median_total"]),
                    "close_ev_allbooks": _offer_ev(r["market"], c["allbooks_mean"], r["side"], r["line"], r["odds"],
                                                   c["median_total"]),
                    "snapshot_ts": t, "hours_to_kickoff": hours, "in_week": hours <= IN_WEEK_HOURS,
                    "quote_age_min": (t - lu).total_seconds() / 60 if lu is not None else np.nan,
                    "anchor_fresher": bool(alu > lu) if (lu is not None and alu is not None) else np.nan,
                    "persisted_next_snap": persisted,
                    "sample": sample_of(k, t),
                })
                out.append(row)
    return pd.DataFrame(out)


def grade(market, side, line, odds, home_score, away_score):
    """(result, units) of a 1u ticket at the offered line and price."""
    if market == "spread":
        diff = (home_score - away_score) + line
        diff = diff if side == "HOME" else -diff
    else:
        diff = (home_score + away_score) - line
        diff = diff if side == "OVER" else -diff
    if diff > 0:
        return "WIN", (odds / 100.0 if odds > 0 else 100.0 / -odds)
    if diff < 0:
        return "LOSS", -1.0
    return "PUSH", 0.0


def attach_results(df, games_path=GAMES_PATH):
    """Adds home_score, away_score, result and units where games.csv has the final score."""
    if df.empty or not os.path.exists(games_path):
        return df.assign(home_score=np.nan, away_score=np.nan, result=None, units=np.nan)
    g = pd.read_csv(games_path, low_memory=False).dropna(subset=["home_score", "away_score"])
    g["date"] = pd.to_datetime(g["gameday"]).dt.date
    scores = {(r.home_team, r.away_team, r.date): (r.home_score, r.away_score) for r in g.itertuples()}
    out = df.copy()
    hs, as_, res, units = [], [], [], []
    for r in out.itertuples():
        d = _utc(r.commence_time).tz_convert("America/New_York").date()
        s = scores.get((r.home_team, r.away_team, d))
        if s is None:
            hs.append(np.nan), as_.append(np.nan), res.append(None), units.append(np.nan)
            continue
        x, u = grade(r.market, r.side, r.line, r.odds, s[0], s[1])
        hs.append(s[0]), as_.append(s[1]), res.append(x), units.append(u)
    out["home_score"], out["away_score"], out["result"], out["units"] = hs, as_, res, units
    return out


# ---- report and gate --------------------------------------------------------------------------

def gate_group(df, sample):
    """The pre-registered group: flagged, US targets, in-week, spreads and totals combined."""
    if df.empty:
        return df
    return df[(df["sample"] == sample) & df["flagged"].astype(bool) & (df["book_class"] == "us")
              & df["in_week"].astype(bool)]


def _top2_games(g):
    return list(g.groupby("event_id")["close_ev"].sum().sort_values(ascending=False).index[:2])


def gate_summary(g):
    """offers, games, mean_close_ev, close_ev_se (game-clustered), mean_close_ev_ex_top2."""
    n = len(g)
    if n == 0:
        return {"offers": 0, "games": 0, "mean_close_ev": float("nan"), "close_ev_se": float("nan"),
                "mean_close_ev_ex_top2": float("nan")}
    rest = g[~g["event_id"].isin(_top2_games(g))]
    return {"offers": n, "games": int(g["event_id"].nunique()), "mean_close_ev": float(g["close_ev"].mean()),
            "close_ev_se": cluster_se(g["close_ev"], g["event_id"]),
            "mean_close_ev_ex_top2": float(rest["close_ev"].mean()) if len(rest) else float("nan")}


def gate(s, season_settled=False):
    """'pass' / 'fail' per the frozen test; 'inconclusive' once week 18 has settled with
    neither; 'pending' before that (weekly re-runs are descriptive only)."""
    size = s["offers"] >= GATE_MIN_OFFERS and s["games"] >= GATE_MIN_GAMES
    m, se, ex = s["mean_close_ev"], s["close_ev_se"], s["mean_close_ev_ex_top2"]
    if size and m > 0 and se > 0 and m >= GATE_MIN_Z * se and ex > 0:
        return "pass"
    if size and m <= 0:
        return "fail"
    return "inconclusive" if season_settled else "pending"


def group_stats(g, name):
    n = len(g)
    graded = g[g["result"].isin(["WIN", "LOSS", "PUSH"])] if n and "result" in g else g.iloc[0:0]
    return {
        "group": name, "offers": n, "games": int(g["event_id"].nunique()) if n else 0,
        "mean_ev": float(g["ev"].mean()) if n else float("nan"),
        "close_ev": float(g["close_ev"].mean()) if n else float("nan"),
        "se_game": cluster_se(g["close_ev"], g["event_id"]) if n else float("nan"),
        "close_ev_allbooks": float(g["close_ev_allbooks"].mean()) if n else float("nan"),
        "beat_close_pct": 100 * float((g["close_ev"] > 0).mean()) if n else float("nan"),
        "W-L-P": f"{(graded['result'] == 'WIN').sum()}-{(graded['result'] == 'LOSS').sum()}-"
                 f"{(graded['result'] == 'PUSH').sum()}",
        "units": float(graded["units"].sum()) if len(graded) else 0.0,
    }


def _table(rows):
    if not rows:
        return "(none)"
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}")


def report(df, sample, season_settled=False):
    """The printed report for one sample ('evaluation' or 'design')."""
    head = ("EVALUATION SAMPLE (evidence)" if sample == "evaluation"
            else "DESIGN SAMPLE, NOT EVIDENCE")
    lines = [f"===== {head} ====="]
    s = df[df["sample"] == sample] if not df.empty else df
    if s.empty:
        lines.append("no rows")
        if sample == "evaluation":
            lines.append(f"gate: {gate(gate_summary(s), season_settled)} (0 offers on 0 games)")
        return "\n".join(lines)
    lines.append(f"{len(s)} first sightings on {s['event_id'].nunique()} games, "
                 f"{s['snapshot_ts'].nunique()} snapshots")
    g = gate_group(df, sample)
    rest = g[~g["event_id"].isin(_top2_games(g))] if len(g) else g
    lines += ["", "1-2. Gate group: flagged, US targets, in-week, spreads + totals",
              _table([group_stats(g, "gate group"), group_stats(rest, "gate group, top 2 games removed")])]
    gs = gate_summary(g)
    z = gs["mean_close_ev"] / gs["close_ev_se"] if gs["close_ev_se"] and gs["close_ev_se"] > 0 else float("nan")
    verdict = gate(gs, season_settled) if sample == "evaluation" else "n/a (design sample)"
    lines.append(f"gate: {verdict}  ({gs['offers']} offers on {gs['games']} games; need "
                 f"{GATE_MIN_OFFERS} on {GATE_MIN_GAMES}; z = {z:.2f})")

    f = s[s["flagged"].astype(bool)]
    rows = []
    for (m, wk, cls), x in f.groupby(["market", "in_week", "book_class"]):
        rows.append(group_stats(x, f"{m} / {'in-week' if wk else '>144h'} / {cls}"))
    lines += ["", "3a. Flagged offers by market x horizon x book class", _table(rows)]
    lines += ["", "3b. Flagged offers by book",
              _table([group_stats(x, b) for b, x in f.groupby("book")])]
    p = g["persisted_next_snap"] if len(g) else pd.Series(dtype=object)
    lines += ["", "3c. Gate group persistence (next snapshot within 6 h, EV >= 0.5%): "
              f"true {int((p == True).sum())}, false {int((p == False).sum())}, "  # noqa: E712
              f"no snapshot within 6 h {int(p.isna().sum())}"]
    fr = [group_stats(x, f"anchor_fresher = {v}") for v, x in g.groupby(g["anchor_fresher"].astype(str))] if len(g) else []
    lines += ["", "3d. Gate group by anchor_fresher", _table(fr)]
    lines += ["", "(close_ev_allbooks sits beside close_ev in every table: the all-books median close.)"]
    base = s[(s["book_class"] == "us") & s["in_week"].astype(bool)]
    lines += ["", "Context: all US in-week first sightings, flagged or not",
              _table([group_stats(base, "all US in-week offers")])]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Sharp-vs-soft top-down replay v2 over the odds-data archive")
    ap.add_argument("--source", default=DEFAULT_REF, help="git ref or a checkout directory of odds-data")
    ap.add_argument("--games", default=GAMES_PATH, help="nflverse games.csv for grading")
    ap.add_argument("--season-settled", action="store_true", help="week 18 has settled: report inconclusive, not pending")
    args = ap.parse_args()

    snaps = list(iter_archive(args.source))
    if not snaps:
        print("no snapshots")
        return
    start = _utc(EVAL_START)
    n_eval = sum(1 for t, _ in snaps if _utc(t) >= start)
    print(f"{len(snaps)} snapshots, {snaps[0][0]:%Y-%m-%d %H:%M} to {snaps[-1][0]:%Y-%m-%d %H:%M} UTC "
          f"({len(snaps) - n_eval} before EVAL_START {EVAL_START}, {n_eval} at or after)")
    ignored = set()
    df = attach_results(replay(snaps, ignored), args.games)
    if ignored:
        print("books outside every class, ignored: " + ", ".join(sorted(ignored)))
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} first sightings on {df['event_id'].nunique() if len(df) else 0} games; wrote {OUT_PATH}\n")
    pd.set_option("display.width", 220)
    print(report(df, "evaluation", args.season_settled))
    print()
    print(report(df, "design"))


if __name__ == "__main__":
    main()
