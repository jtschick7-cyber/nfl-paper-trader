"""Audit Phase 3: consensus pricer, archive reader and replay."""

import gzip
import json
import os

import pandas as pd
import pytest

from src.load_odds_archive import iter_archive, list_snapshots, snapshot_time
from src.market_consensus import (
    DEFAULT_SIGMA, book_quotes, consensus_mean, crosses_key, evaluate_event, implied_mean, price_offer,
    weighted_median,
)
from src.topdown_replay import cluster_se, corrected_share, replay, summarize


def _book(key, home_line, home_odds=-110, away_odds=-110, total=44.5):
    return {"key": key, "title": key, "markets": [
        {"key": "spreads", "outcomes": [
            {"name": "Kansas City Chiefs", "price": home_odds, "point": home_line},
            {"name": "Denver Broncos", "price": away_odds, "point": -home_line}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": total},
                                       {"name": "Under", "price": -110, "point": total}]}]}


def _event(books, eid="e1", kickoff="2026-11-15T18:00:00Z"):
    return {"id": eid, "commence_time": kickoff, "home_team": "Kansas City Chiefs",
            "away_team": "Denver Broncos", "bookmakers": books}


def test_book_quotes_drops_one_sided_and_zero_prices():
    bad = _book("b2", -3.0)
    bad["markets"][0]["outcomes"] = bad["markets"][0]["outcomes"][:1]
    zero = _book("b3", -3.0, home_odds=0)
    q = book_quotes(_event([_book("b1", -3.0), bad, zero]))
    assert [x["book"] for x in q] == ["b1"] and q[0]["total_line"] == 44.5


def test_implied_mean_puts_books_on_one_scale():
    # -3 at -110/-110 and -3 at -130/+110 differ; -2.5 at -110 sits below -3 at -110
    m3 = implied_mean(book_quotes(_event([_book("a", -3.0)]))[0], total_line=44.5)
    m3j = implied_mean(book_quotes(_event([_book("a", -3.0, -130, 110)]))[0], total_line=44.5)
    m25 = implied_mean(book_quotes(_event([_book("a", -2.5)]))[0], total_line=44.5)
    # at -3 / -110 both ways the push mass at 3 puts the mean below 3
    assert 2.5 < m3 < 3.0
    assert m3j > m3 > m25


def test_consensus_drops_outliers_excludes_self_and_weights_sharp_books():
    means = {"lowvig": 3.0, "betonlineag": 3.0, "draftkings": 3.4, "fanduel": 3.4, "stale": 6.0}
    assert consensus_mean(means) == 3.0  # stale dropped, sharp books weigh double
    assert consensus_mean(means, exclude="lowvig") == pytest.approx(3.2)  # 2 + 1 + 1 weights, tie averaged
    assert consensus_mean({"a": 1.0, "b": 2.0}) is None
    assert weighted_median([1, 2, 3], [1, 1, 5]) == 3


def test_stale_book_is_flagged_and_consensus_books_are_not():
    books = [_book("lowvig", -3.0), _book("betonlineag", -3.0), _book("draftkings", -3.0),
             _book("fanduel", -3.0), _book("betrivers", -1.5)]
    rows = pd.DataFrame(evaluate_event(_event(books)))
    flagged = rows[rows["stale"]]
    assert list(zip(flagged["book"], flagged["side"])) == [("betrivers", "HOME")]
    assert flagged.iloc[0]["key_cross"] == 3  # -1.5 wins a 3-point home win that -3 only pushes
    assert rows[(rows["book"] == "draftkings")]["ev"].max() < 0


def test_crosses_key():
    # HOME at -2.5 vs consensus -3.5 wins margin 3 that -3.5 loses
    assert crosses_key("HOME", -2.5, -3.5) == 3
    assert crosses_key("AWAY", -3.5, -2.5) == 3   # AWAY +3.5 vs +2.5 wins a 3-point loss
    assert crosses_key("HOME", -3.5, -2.5) is None  # worse line
    assert crosses_key("HOME", -1.0, -1.5) is None


def test_price_offer_matches_symmetric_pick():
    p_win, p_push, ev = price_offer(0.0, "HOME", 0.0, 100, DEFAULT_SIGMA)
    assert p_win == pytest.approx((1 - p_push) / 2, abs=1e-9) and ev == pytest.approx(0.0, abs=1e-9)


def _write_snap(root, stamp, events):
    d = root / "snapshots" / stamp[:4] / stamp[4:6]
    os.makedirs(d, exist_ok=True)
    with gzip.open(d / f"odds_{stamp}.json.gz", "wb") as f:
        f.write(json.dumps(events).encode())


def test_archive_reader_and_replay(tmp_path):
    consensus = [_book("lowvig", -3.0), _book("betonlineag", -3.0), _book("draftkings", -3.0), _book("fanduel", -3.0)]
    _write_snap(tmp_path, "20261110T150000Z", [_event(consensus + [_book("betrivers", -1.5)])])
    # By the close the stale book has corrected and the market moved to -3.5
    close = [_book(b, -3.5) for b in ("lowvig", "betonlineag", "draftkings", "fanduel", "betrivers")]
    _write_snap(tmp_path, "20261115T170000Z", [_event(close)])
    _write_snap(tmp_path, "20261115T190000Z", [_event(close)])  # after kickoff: ignored
    assert snapshot_time("odds_20261110T150000Z.json.gz").hour == 15
    assert len(list_snapshots(str(tmp_path))) == 3
    snaps = list(iter_archive(str(tmp_path)))
    df = replay(snaps)
    assert df["snapshot_ts"].nunique() == 1  # only the pre-close snapshot is replayed
    f = df[df["stale"]]
    assert list(f["book"]) == ["betrivers"] and f.iloc[0]["close_ev"] > f.iloc[0]["ev"] > 0
    pct, n = corrected_share(df)
    assert (pct, n) == (100.0, 1)
    s = summarize(df.assign(result=None, units=float("nan"))).set_index("group")
    assert s.loc["flagged (EV >= threshold)", "offers"] == 1


def test_cluster_se_counts_events_not_offers():
    x = [1.0, 1.0, 0.0, 0.0]
    assert cluster_se(x, ["a", "a", "b", "b"]) > cluster_se(x, ["a", "b", "c", "d"])
    assert pd.isna(cluster_se([1.0, 2.0], ["a", "a"]))
