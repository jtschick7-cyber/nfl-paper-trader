"""Replay of the top-down consensus pricer over the odds-data archive (audit Phase 3).

    .venv/bin/python -m src.topdown_replay                 # reads origin/odds-data
    .venv/bin/python -m src.topdown_replay --source DIR    # a checkout of the branch

Only events that kicked off by the archive's latest snapshot count (their close is final).
For every snapshot taken before an event's last pre-kickoff snapshot (the "close"):
- each book's offer is priced against the leave-one-out consensus (src/market_consensus.py);
  offers with EV >= --stale-ev are flagged, deduplicated on (event, book, side, line, price)
  keeping the first sighting;
- every offer is also priced against the closing consensus mean (all books): that EV is the
  price-aware closing line value (CLV). The gate: flagged offers should beat the close
  (mean closing EV > 0), and beat the primary book's offer on the same side and snapshot.
- graded against the final score where data/nflverse_cache/games.csv has it.

Writes data/topdown_replay_offers.csv. Read-only otherwise: no Sheet, no model files.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

from src.config import PRIMARY_BOOK
from src.load_odds_archive import DEFAULT_REF, iter_archive
from src.market_consensus import (
    DEFAULT_SIGMA, STALE_EV, book_quotes, consensus_mean, evaluate_event, implied_mean, price_offer,
)

OUT_PATH = "data/topdown_replay_offers.csv"


def _kickoff(event):
    return pd.Timestamp(event["commence_time"]).tz_convert("UTC")


def closing_state(snapshots, sigma=DEFAULT_SIGMA):
    """{event_id: dict(close_ts, close_mean, total_line, quotes)} from each event's last
    snapshot strictly before kickoff. Events that kick off after the archive's latest
    snapshot are left out: a later snapshot could still change their close."""
    close = {}
    if not snapshots:
        return close
    archive_end = max(pd.Timestamp(ts) for ts, _ in snapshots)
    for ts, events in snapshots:
        t = pd.Timestamp(ts)
        for e in events:
            k = _kickoff(e)
            if t >= k or k > archive_end:
                continue
            quotes = book_quotes(e)
            if not quotes:
                continue
            totals = [q["total_line"] for q in quotes if q["total_line"] is not None]
            total = float(np.median(totals)) if totals else None
            m = consensus_mean({q["book"]: implied_mean(q, sigma, total) for q in quotes})
            if m is None:
                continue
            close[e["id"]] = {"close_ts": t, "close_mean": m, "total_line": total,
                              "quotes": {q["book"]: q for q in quotes}}
    return close


def replay(snapshots, sigma=DEFAULT_SIGMA, stale_ev=STALE_EV):
    """One row per first sighting of each (event, book, side, line, price) before the close."""
    close = closing_state(snapshots, sigma)
    rows, seen = [], set()
    for ts, events in snapshots:
        t = pd.Timestamp(ts)
        for e in events:
            c = close.get(e.get("id"))
            if c is None or t >= c["close_ts"]:
                continue
            for r in evaluate_event(e, sigma, stale_ev):
                key = (r["event_id"], r["book"], r["side"], r["home_line"], r["odds"])
                if key in seen:
                    continue
                seen.add(key)
                _, _, close_ev = price_offer(c["close_mean"], r["side"], r["home_line"], r["odds"], sigma, c["total_line"])
                cq = c["quotes"].get(r["book"])
                r.update({
                    "snapshot_ts": t, "close_ts": c["close_ts"], "hours_to_kickoff": (_kickoff(e) - t).total_seconds() / 3600,
                    "close_mean": c["close_mean"], "close_ev": close_ev,
                    # the same book at the close: did it move against this side (it corrected)?
                    "book_close_home_line": cq["home_line"] if cq else None,
                    "book_close_odds": (cq["home_odds"] if r["side"] == "HOME" else cq["away_odds"]) if cq else None,
                })
                rows.append(r)
    return pd.DataFrame(rows)


def attach_results(df, games_path="data/nflverse_cache/games.csv"):
    """Adds home_margin and the 1u result / units at the offered line and price."""
    if df.empty or not os.path.exists(games_path):
        return df.assign(home_margin=np.nan, result=None, units=np.nan)
    g = pd.read_csv(games_path, low_memory=False)
    g = g.dropna(subset=["home_score", "away_score"])
    g["date"] = pd.to_datetime(g["gameday"]).dt.date
    g["home_margin"] = g["home_score"] - g["away_score"]
    key = g.set_index(["home_team", "away_team", "date"])["home_margin"].to_dict()
    out = df.copy()
    margins = []
    for _, r in out.iterrows():
        k = pd.Timestamp(r["commence_time"]).tz_convert("America/New_York").date()
        margins.append(key.get((r["home_team"], r["away_team"], k), np.nan))
    out["home_margin"] = margins
    res, units = [], []
    for _, r in out.iterrows():
        if pd.isna(r["home_margin"]):
            res.append(None)
            units.append(np.nan)
            continue
        cover = r["home_margin"] + r["home_line"]
        cover = cover if r["side"] == "HOME" else -cover
        o = r["odds"]
        if cover > 0:
            res.append("WIN")
            units.append(o / 100.0 if o > 0 else 100.0 / -o)
        elif cover < 0:
            res.append("LOSS")
            units.append(-1.0)
        else:
            res.append("PUSH")
            units.append(0.0)
    out["result"], out["units"] = res, units
    return out


def _mean_se(x):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"), len(x)


def cluster_se(values, clusters):
    """SE of the mean of `values` with offers on the same event treated as correlated
    (cluster-robust, G/(G-1) small-sample factor). NaN with fewer than 2 events."""
    d = pd.DataFrame({"x": pd.Series(values, dtype=float).values, "c": list(clusters)}).dropna()
    g = d["c"].nunique()
    if g < 2:
        return float("nan")
    resid = d.groupby("c")["x"].apply(lambda s: float((s - d["x"].mean()).sum()))
    return float(math.sqrt((resid ** 2).sum() * g / (g - 1)) / len(d))


def summarize(df, primary=PRIMARY_BOOK):
    """Table of groups: n, mean EV at sighting, mean closing EV ± SE, share beating the
    close, distinct events, and graded W-L-P / units."""
    groups = {
        "all offers": df,
        f"{primary} offers": df[df["book"] == primary],
        "flagged (EV >= threshold)": df[df["stale"]],
        "flagged, key-number cross": df[df["stale"] & df["key_cross"].notna()],
        "flagged, >= 48h before kickoff": df[df["stale"] & (df["hours_to_kickoff"] >= 48)],
        "flagged, < 48h before kickoff": df[df["stale"] & (df["hours_to_kickoff"] < 48)],
    }
    rows = []
    for name, g in groups.items():
        m, se, n = _mean_se(g["close_ev"])
        graded = g[g["result"].isin(["WIN", "LOSS", "PUSH"])] if "result" in g else g.iloc[0:0]
        rows.append({
            "group": name, "offers": n, "events": g["event_id"].nunique() if n else 0,
            "mean_ev_at_sighting": float(g["ev"].mean()) if n else float("nan"),
            "mean_close_ev": m, "close_ev_se": se,
            "close_ev_se_by_event": cluster_se(g["close_ev"], g["event_id"]) if n else float("nan"),
            "beat_close_pct": 100 * float((g["close_ev"] > 0).mean()) if n else float("nan"),
            "W": int((graded["result"] == "WIN").sum()), "L": int((graded["result"] == "LOSS").sum()),
            "P": int((graded["result"] == "PUSH").sum()), "units": float(graded["units"].sum()) if len(graded) else 0.0,
        })
    return pd.DataFrame(rows)


def paired_vs_primary(df, primary=PRIMARY_BOOK):
    """Flagged offers against the primary book's offer on the same event, side and
    snapshot: mean closing-EV gain ± SE (the line-shopping value of the detector)."""
    p = df[df["book"] == primary][["event_id", "side", "snapshot_ts", "close_ev"]]
    f = df[df["stale"] & (df["book"] != primary)]
    j = f.merge(p, on=["event_id", "side", "snapshot_ts"], suffixes=("", "_primary"))
    return _mean_se(j["close_ev"] - j["close_ev_primary"])


def corrected_share(df):
    """Share of flagged offers whose book had moved against that side by the close."""
    f = df[df["stale"] & df["book_close_home_line"].notna()]
    if f.empty:
        return float("nan"), 0
    worse = []
    for _, r in f.iterrows():
        dl = r["book_close_home_line"] - r["home_line"]
        line_worse = dl < 0 if r["side"] == "HOME" else dl > 0
        same_line = abs(dl) < 1e-9
        price_worse = same_line and _dec(r["book_close_odds"]) < _dec(r["odds"]) - 1e-9
        worse.append(line_worse or price_worse)
    return 100 * float(np.mean(worse)), len(f)


def _dec(o):
    o = float(o)
    return 1 + (o / 100.0 if o > 0 else 100.0 / -o)


def main():
    ap = argparse.ArgumentParser(description="Top-down consensus replay over the odds-data archive")
    ap.add_argument("--source", default=DEFAULT_REF, help="git ref or a checkout directory of odds-data")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--stale-ev", type=float, default=STALE_EV)
    args = ap.parse_args()

    snaps = list(iter_archive(args.source))
    print(f"{len(snaps)} snapshots, {snaps[0][0]:%Y-%m-%d %H:%M} to {snaps[-1][0]:%Y-%m-%d %H:%M} UTC")
    df = attach_results(replay(snaps, args.sigma, args.stale_ev))
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} distinct pre-close offers on {df['event_id'].nunique()} events; wrote {OUT_PATH}\n")
    pd.set_option("display.width", 200)
    print(summarize(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    m, se, n = paired_vs_primary(df)
    print(f"\nFlagged vs {PRIMARY_BOOK} (same event, side, snapshot): closing-EV gain {m:+.4f} ± {se:.4f} on {n} pairs")
    pct, n = corrected_share(df)
    print(f"Flagged offers whose book moved against that side by the close: {pct:.1f}% of {n}")
    flagged = df[df["stale"]]
    if len(flagged):
        print("\nFlagged offers by book: " + ", ".join(f"{b} {c}" for b, c in flagged["book"].value_counts().items()))


if __name__ == "__main__":
    main()
