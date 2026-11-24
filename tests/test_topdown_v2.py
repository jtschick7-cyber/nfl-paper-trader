"""P2: sharp-vs-soft top-down replay v2 (src/topdown_v2.py)."""

import math

import pandas as pd
import pytest

from src import topdown_v2 as v2
from src.market_consensus import book_quotes, implied_mean

KICK = "2026-11-26T18:00:00Z"
HOME, AWAY = "Detroit Lions", "New England Patriots"


def _book(key, home_line=None, home_odds=-110, away_odds=-110, total=44.5, over_odds=-110, under_odds=-110):
    markets = []
    if home_line is not None:
        markets.append({"key": "spreads", "last_update": "2026-11-25T12:00:00Z", "outcomes": [
            {"name": HOME, "price": home_odds, "point": home_line},
            {"name": AWAY, "price": away_odds, "point": -home_line}]})
    if total is not None:
        markets.append({"key": "totals", "last_update": "2026-11-25T12:00:00Z", "outcomes": [
            {"name": "Over", "price": over_odds, "point": total},
            {"name": "Under", "price": under_odds, "point": total}]})
    return {"key": key, "title": key, "markets": markets}


def _event(books, eid="e1", kickoff=KICK):
    return {"id": eid, "commence_time": kickoff, "home_team": HOME, "away_team": AWAY, "bookmakers": books}


def _ts(s):
    return pd.Timestamp(s)


def _replay(snaps):
    """Snapshots plus a post-kickoff one so the event counts as kicked off."""
    return v2.replay(snaps + [(_ts("2026-11-27T12:00:00Z"), [])])


def test_frozen_constants_match_the_spec():
    assert v2.SHARP_GROUPS == [(("pinnacle",), 3.0), (("circasports",), 2.0), (("betonlineag", "lowvig"), 2.0)]
    assert v2.US_TARGETS == {"fanduel", "draftkings", "betmgm", "betrivers", "williamhill_us",
                             "espnbet", "fanatics", "ballybet", "hardrockbet"}
    assert v2.OFFSHORE == {"betus", "mybookieag", "bovada"}
    assert v2.FLAG_EV == 0.015
    assert v2.PERSIST_EV == 0.005
    assert v2.PERSIST_HOURS == 6
    assert v2.IN_WEEK_HOURS == 144
    assert v2.SPREAD_SIGMA == 12.8
    assert v2.TOTAL_SIGMA == 13.2
    assert v2.EVAL_START == "2026-11-25T00:00:00Z"
    assert v2.GATE_MIN_OFFERS == 80 and v2.GATE_MIN_GAMES == 35 and v2.GATE_MIN_Z == 2.0


def test_sharp_book_is_never_flagged():
    # lowvig hangs DET -1 against betonlineag's -4: +EV against the anchor, but sharp-group
    books = [_book("betonlineag", -4.0), _book("lowvig", -1.0), _book("pinnacle", -4.0),
             _book("circasports", -4.0), _book("fanduel", -4.0)]
    rows, meta, _ = v2.evaluate_event(_event(books))
    assert {r["book"] for r in rows} == {"fanduel"}
    assert meta["spread"]["anchor_groups"] == 3


def test_anchor_is_betonline_then_lowvig():
    bol, lv = _book("betonlineag", -3.0, -120, 100), _book("lowvig", -2.5)
    ev = _event([bol, lv, _book("draftkings", -3.0)])
    _, meta, _ = v2.evaluate_event(ev)
    qs = {q["book"]: q for q in book_quotes(ev)}
    assert meta["spread"]["anchor_groups"] == 1
    assert meta["spread"]["anchor_mean"] == pytest.approx(implied_mean(qs["betonlineag"], 12.8, 44.5))
    ev2 = _event([lv, _book("draftkings", -3.0)])
    _, meta2, _ = v2.evaluate_event(ev2)
    assert meta2["spread"]["anchor_mean"] == pytest.approx(implied_mean(book_quotes(ev2)[0], 12.8, 44.5))


def test_no_anchor_means_no_rows():
    rows, meta, _ = v2.evaluate_event(_event([_book("draftkings", -3.0), _book("fanduel", -1.0)]))
    assert rows == [] and meta["spread"]["anchor_groups"] == 0


def test_ne_at_det_pattern_flags_dk_only():
    # stale DK DET -1.5; LowVig NE +2.5 (sharp: priced never); BetOnline anchors at DET -2.5 -125
    books = [_book("betonlineag", -2.5, -125, 105), _book("lowvig", -2.5, -112, -108),
             _book("draftkings", -1.5), _book("fanduel", -2.5, -125, 105), _book("betus", -2.5, -125, 105)]
    rows, _, _ = v2.evaluate_event(_event(books))
    flagged = [(r["book"], r["market"], r["side"]) for r in rows if r["flagged"]]
    assert flagged == [("draftkings", "spread", "HOME")]
    sides = {}
    for r in rows:
        if r["flagged"]:
            sides.setdefault(r["market"], set()).add(r["side"])
    assert all(len(s) == 1 for s in sides.values())
    by = {(r["book"], r["side"]): r for r in rows if r["market"] == "spread"}
    assert by[("betus", "HOME")]["book_class"] == "offshore" and by[("draftkings", "HOME")]["book_class"] == "us"


def test_stale_total_is_flagged_on_the_right_side():
    books = [_book("betonlineag", -3.0, total=44.5), _book("draftkings", -3.0, total=47.5)]
    rows, meta, _ = v2.evaluate_event(_event(books))
    t = {r["side"]: r for r in rows if r["market"] == "total" and r["book"] == "draftkings"}
    assert t["UNDER"]["flagged"] and not t["OVER"]["flagged"]
    assert meta["total"]["anchor_mean"] == pytest.approx(44.5, abs=1e-6)


def test_unknown_books_are_ignored_and_reported():
    rows, _, ignored = v2.evaluate_event(_event([_book("betonlineag", -3.0), _book("someshop", -1.0)]))
    assert rows == [] and ignored == {"someshop"}


def _snaps_dk(next_dk_line, gap_hours):
    t0 = _ts("2026-11-25T12:00:00Z")
    t1 = t0 + pd.Timedelta(hours=gap_hours)
    base = [_book("betonlineag", -3.0, total=None)]
    return [(t0, [_event(base + [_book("draftkings", -1.5, total=None)])]),
            (t1, [_event(base + [_book("draftkings", next_dk_line, total=None)])])]


def test_persisted_next_snap_true_false_nan():
    def dk_home(df):
        r = df[(df["book"] == "draftkings") & (df["side"] == "HOME")].iloc[0]
        assert r["flagged"]
        return r["persisted_next_snap"]

    assert dk_home(_replay(_snaps_dk(-1.5, 3))) == True  # noqa: E712 (object column)
    assert dk_home(_replay(_snaps_dk(-3.0, 3))) == False  # noqa: E712
    assert math.isnan(dk_home(_replay(_snaps_dk(-1.5, 7))))


def test_replay_close_columns_and_first_sightings():
    df = _replay(_snaps_dk(-3.0, 3))
    # only the t0 snapshot is before the close (t1); one row per book side
    assert len(df) == 2 and set(df["snapshot_ts"]) == {_ts("2026-11-25T12:00:00Z")}
    r = df[df["side"] == "HOME"].iloc[0]
    assert r["close_ev"] == pytest.approx(r["ev"])  # anchor unchanged at the close
    assert r["close_anchor_mean"] == pytest.approx(r["anchor_mean"])
    assert r["close_ev_allbooks"] == pytest.approx(r["close_ev"])  # both books at DET -3 at the close
    assert r["sample"] == "evaluation" and r["in_week"]
    assert r["quote_age_min"] == pytest.approx(0.0)
    assert r["anchor_fresher"] == False  # noqa: E712


def test_replay_skips_games_not_kicked_off():
    assert v2.replay(_snaps_dk(-1.5, 3)).empty


def test_in_week_boundary_and_sample_rule():
    k = _ts(KICK)
    t0 = k - pd.Timedelta(hours=144)
    base = [_book("betonlineag", -3.0, total=None), _book("draftkings", -1.5, total=None)]
    df = _replay([(t0, [_event(base)]), (k - pd.Timedelta(hours=1), [_event(base)])])
    assert df["hours_to_kickoff"].iloc[0] == pytest.approx(144.0) and df["in_week"].all()
    start = _ts(v2.EVAL_START)
    assert v2.sample_of(start, start) == "evaluation"
    assert v2.sample_of(start + pd.Timedelta(days=1), start - pd.Timedelta(seconds=1)) == "design"
    assert v2.sample_of(start - pd.Timedelta(seconds=1), start - pd.Timedelta(days=1)) == "design"


def test_grade_totals_and_spreads():
    assert v2.grade("total", "OVER", 44.0, -110, 24, 20) == ("PUSH", 0.0)
    assert v2.grade("total", "UNDER", 44.0, -110, 24, 20) == ("PUSH", 0.0)
    res, units = v2.grade("total", "OVER", 43.5, -110, 24, 20)
    assert res == "WIN" and units == pytest.approx(100 / 110)
    assert v2.grade("total", "UNDER", 43.5, 120, 24, 20) == ("LOSS", -1.0)
    assert v2.grade("spread", "HOME", -3.0, -110, 24, 21) == ("PUSH", 0.0)
    assert v2.grade("spread", "AWAY", -3.0, 150, 24, 22) == ("WIN", 1.5)


def _summary(offers=80, games=35, mean=0.02, se=0.005, ex_top2=0.01):
    return {"offers": offers, "games": games, "mean_close_ev": mean, "close_ev_se": se,
            "mean_close_ev_ex_top2": ex_top2}


def test_gate_pass_fail_inconclusive_and_top2_rule():
    assert v2.gate(_summary()) == "pass"
    assert v2.gate(_summary(mean=0.0099, se=0.005)) != "pass"         # z below 2
    assert v2.gate(_summary(se=float("nan"))) != "pass"
    assert v2.gate(_summary(ex_top2=0.0)) == "pending"                # top-2 rule
    assert v2.gate(_summary(ex_top2=0.0), season_settled=True) == "inconclusive"
    assert v2.gate(_summary(mean=0.0, ex_top2=-0.01)) == "fail"
    assert v2.gate(_summary(mean=-0.01, games=34)) == "pending"        # size not reached
    assert v2.gate(_summary(offers=79, games=60)) == "pending"
    assert v2.gate(_summary(offers=79, mean=-0.02), season_settled=True) == "inconclusive"


def test_gate_summary_clusters_by_game_and_drops_top_two_games():
    rows = [{"event_id": f"g{g}", "close_ev": v}
            for g in range(10) for v in ((0.5, 0.5) if g < 2 else (-0.01, 0.0))]
    s = v2.gate_summary(pd.DataFrame(rows))
    assert s["offers"] == 20 and s["games"] == 10
    assert s["mean_close_ev"] == pytest.approx((2.0 - 0.08) / 20)
    assert s["mean_close_ev_ex_top2"] == pytest.approx(-0.005)
    assert s["close_ev_se"] > 0
    empty = v2.gate_summary(pd.DataFrame(columns=["event_id", "close_ev"]))
    assert empty["offers"] == 0 and v2.gate(empty) == "pending"
