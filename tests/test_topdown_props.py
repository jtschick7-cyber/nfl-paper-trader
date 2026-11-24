"""P4: top-down player_receptions scanner (src/props/topdown_props.py) and grading
(src/props/player_stats.py)."""

import math

import pandas as pd
import pytest

from src.props import player_stats as ps
from src.props import prop_pricing as pp
from src.props import topdown_props as tp
from src.total_pricing import devig_two_way

KICK = "2026-11-29T18:00:00Z"
OPEN_AT = "2026-11-27T21:05:00Z"
CLOSE_AT = "2026-11-29T17:35:00Z"
HOME, AWAY = "Detroit Lions", "Chicago Bears"


def _book(key, players, last_update="2026-11-27T21:00:00Z"):
    """players: [(name, line, over_odds or None, under_odds or None)]"""
    outs = []
    for name, line, o, u in players:
        if o is not None:
            outs.append({"name": "Over", "description": name, "price": o, "point": line})
        if u is not None:
            outs.append({"name": "Under", "description": name, "price": u, "point": line})
    return {"key": key, "title": key,
            "markets": [{"key": "player_receptions", "last_update": last_update, "outcomes": outs}]}


def _snap(phase, books, captured_at, eid="e1", kickoff=KICK):
    return {"schema": 1, "phase": phase, "market": "player_receptions", "captured_at": captured_at,
            "kickoff": kickoff,
            "response": {"id": eid, "commence_time": kickoff, "home_team": HOME, "away_team": AWAY,
                         "bookmakers": books}}


def _lam(line, o, u, r=None):
    return pp.implied_lambda(line, devig_two_way(o, u)[0], r=r)


# ---- 9. frozen constants ----------------------------------------------------------------------

def test_frozen_constants_equal_spec():
    assert tp.ANCHOR_BOOK == "pinnacle"
    assert tp.ALT_ANCHOR_BOOK == "betonlineag"
    assert tp.US_TARGETS == {"draftkings", "fanduel", "betmgm", "betrivers", "williamhill_us",
                             "espnbet", "fanatics", "ballybet", "hardrockbet"}
    assert tp.OFFSHORE == {"bovada", "betus", "mybookieag"}
    assert tp.FLAG_EV == 0.02
    assert tp.NB_SENSITIVITY_R == 8.0 == pp.NB_SENSITIVITY_R
    assert (pp.LAMBDA_LO, pp.LAMBDA_HI, pp.LAMBDA_TOL) == (0.05, 20.0, 1e-6)
    assert tp.GATE_MIN_OFFERS == 150
    assert tp.GATE_MIN_GAMES == 30
    assert tp.GATE_MIN_Z == 2.0
    assert tp.OUT_PATH == "data/topdown_props_offers.csv"


def test_book_classes():
    assert tp.book_class("pinnacle") == "anchor"
    assert tp.book_class("betonlineag") == "alt_anchor"
    assert tp.book_class("fanduel") == "us"
    assert tp.book_class("bovada") == "offshore"
    assert tp.book_class("someotherbook") is None


# ---- replay: pricing, flags, anchors ----------------------------------------------------------

def _basic_snaps(close_books=None):
    open_books = [
        _book("pinnacle", [("Amon-Ra St. Brown", 6.5, -120, -105), ("Jameson Williams", 3.5, -110, -110)]),
        _book("betonlineag", [("Amon-Ra St. Brown", 6.5, -115, -110), ("Sam LaPorta", 4.5, -110, -110)]),
        _book("draftkings", [("Amon-Ra St. Brown", 6.5, 120, -150), ("Sam LaPorta", 4.5, 105, -135)]),
        _book("betrivers", [("Amon-Ra St. Brown", 5.5, -125, None)], last_update="2026-11-27T20:35:00Z"),
        _book("bovada", [("Jameson Williams", 3.5, -105, -125)]),
        _book("weirdbook", [("Jameson Williams", 3.5, 200, 200)]),
    ]
    if close_books is None:
        close_books = [
            _book("pinnacle", [("Amon-Ra St. Brown", 6.5, -140, 110), ("Jameson Williams", 3.5, -110, -110)]),
            _book("betonlineag", [("Amon-Ra St. Brown", 6.5, -135, 105), ("Sam LaPorta", 4.5, -105, -115)]),
            _book("draftkings", [("Amon-Ra St. Brown", 6.5, -135, 105)]),
        ]
    return [_snap("open", open_books, OPEN_AT), _snap("close", close_books, CLOSE_AT)]


def test_replay_prices_targets_against_pinnacle():
    ignored = set()
    df = tp.replay(_basic_snaps(), ignored)
    assert ignored == {"weirdbook"}
    prim = df[df["anchor"] == "pinnacle"]
    lam = _lam(6.5, -120, -105)
    r = prim[(prim["book"] == "draftkings") & (prim["player"] == "amon-ra st brown") & (prim["side"] == "OVER")].iloc[0]
    assert r["lam_open"] == pytest.approx(lam, abs=1e-6)
    assert r["ev"] == pytest.approx(pp.price_prop_offer(lam, "OVER", 6.5, 120)[2], abs=1e-9)
    assert r["ev_nb"] == pytest.approx(
        pp.price_prop_offer(_lam(6.5, -120, -105, r=8.0), "OVER", 6.5, 120, r=8.0)[2], abs=1e-9)
    assert bool(r["flagged"]) == (r["ev"] >= 0.02)
    assert r["book_class"] == "us" and r["line"] == 6.5 and r["odds"] == 120 and not r["one_sided"]
    assert r["hours_to_kickoff"] == pytest.approx((pd.Timestamp(KICK) - pd.Timestamp(OPEN_AT)).total_seconds() / 3600)
    assert r["quote_age_min"] == pytest.approx(5.0)
    assert r["kickoff"] == KICK and r["event_id"] == "e1"
    # offshore priced too, never US
    assert set(prim[prim["book"] == "bovada"]["book_class"]) == {"offshore"}


def test_one_sided_offer_priced_and_flaggable_never_anchor():
    snaps = [_snap("open", [
        _book("pinnacle", [("Jahmyr Gibbs", 3.5, -110, -110)]),
        _book("betrivers", [("Jahmyr Gibbs", 3.5, 150, None), ("David Montgomery", 1.5, -110, None)]),
        _book("fanduel", [("David Montgomery", 1.5, -110, -110)]),
    ], OPEN_AT)]
    df = tp.replay(snaps)
    br = df[(df["book"] == "betrivers") & (df["player"] == "jahmyr gibbs")]
    assert len(br) == 1 and br.iloc[0]["side"] == "OVER" and bool(br.iloc[0]["one_sided"])
    assert br.iloc[0]["flagged"]  # +150 on a fair coin flip
    # Montgomery has only a one-sided BetRivers quote plus FanDuel; no Pinnacle -> nothing anchors
    assert df[df["player"] == "david montgomery"].empty


def test_one_sided_anchor_quote_is_not_an_anchor():
    snaps = [_snap("open", [
        _book("pinnacle", [("Jahmyr Gibbs", 3.5, -110, None)]),
        _book("fanduel", [("Jahmyr Gibbs", 3.5, 150, -200)]),
    ], OPEN_AT)]
    assert tp.replay(snaps).empty


def test_anchors_never_flagged_and_alt_table():
    df = tp.replay(_basic_snaps())
    assert not df["book"].isin({"pinnacle", "betonlineag"}).any()
    prim, alt = df[df["anchor"] == "pinnacle"], df[df["anchor"] == "betonlineag"]
    # LaPorta has no Pinnacle quote: absent from the primary table, present in the alternate
    assert prim[prim["player"] == "sam laporta"].empty
    la = alt[alt["player"] == "sam laporta"]
    assert set(la["book"]) == {"draftkings"} and len(la) == 2
    assert la.iloc[0]["lam_open"] == pytest.approx(_lam(4.5, -110, -110), abs=1e-6)
    # Williams has no BetOnline quote: primary only
    assert alt[alt["player"] == "jameson williams"].empty
    assert not prim[prim["player"] == "jameson williams"].empty


def test_cross_line_offer_priced_from_anchor_lambda():
    df = tp.replay(_basic_snaps())
    r = df[(df["anchor"] == "pinnacle") & (df["book"] == "betrivers")].iloc[0]
    lam = _lam(6.5, -120, -105)
    assert r["line"] == 5.5 and not r["same_line"]
    assert r["ev"] == pytest.approx(pp.price_prop_offer(lam, "OVER", 5.5, -125)[2], abs=1e-9)


def test_anchor_main_line_when_several_lines():
    # the two-sided anchor line whose de-vigged q_over is nearest 0.5 is used
    snaps = [_snap("open", [
        _book("pinnacle", [("Jahmyr Gibbs", 2.5, -250, 190), ("Jahmyr Gibbs", 3.5, -105, -115)]),
        _book("fanduel", [("Jahmyr Gibbs", 3.5, -110, -110)]),
    ], OPEN_AT)]
    df = tp.replay(snaps)
    assert df.iloc[0]["lam_open"] == pytest.approx(_lam(3.5, -105, -115), abs=1e-6)


# ---- 6. close matching ------------------------------------------------------------------------

def test_close_ev_uses_close_phase_pinnacle():
    df = tp.replay(_basic_snaps())
    prim = df[df["anchor"] == "pinnacle"]
    r = prim[(prim["book"] == "draftkings") & (prim["side"] == "OVER") & (prim["player"] == "amon-ra st brown")].iloc[0]
    lam_c = _lam(6.5, -140, 110)
    assert r["lam_close"] == pytest.approx(lam_c, abs=1e-6)
    assert r["close_ev"] == pytest.approx(pp.price_prop_offer(lam_c, "OVER", 6.5, 120)[2], abs=1e-9)
    assert r["close_ev_nb"] == pytest.approx(
        pp.price_prop_offer(_lam(6.5, -140, 110, r=8.0), "OVER", 6.5, 120, r=8.0)[2], abs=1e-9)
    # median over every two-sided book at the close: pinnacle, betonlineag, draftkings
    med = sorted([lam_c, _lam(6.5, -135, 105), _lam(6.5, -135, 105)])[1]
    assert r["close_ev_median"] == pytest.approx(pp.price_prop_offer(med, "OVER", 6.5, 120)[2], abs=1e-9)
    assert r["close_status"] == "ok"
    # alt table closes against BetOnline at the close
    a = df[(df["anchor"] == "betonlineag") & (df["player"] == "sam laporta") & (df["side"] == "OVER")].iloc[0]
    assert a["lam_close"] == pytest.approx(_lam(4.5, -105, -115), abs=1e-6)


def test_no_close_when_pinnacle_missing_at_close():
    close = [_book("pinnacle", [("Amon-Ra St. Brown", 6.5, -140, 110)])]
    df = tp.replay(_basic_snaps(close_books=close))
    w = df[(df["anchor"] == "pinnacle") & (df["player"] == "jameson williams")]
    assert len(w) == 2
    assert w["close_ev"].isna().all() and set(w["close_status"]) == {"anchor_missing_at_close"}


def test_no_close_snapshot_and_probe_ignored():
    snaps = _basic_snaps()[:1]
    probe = _basic_snaps()[1]
    probe["phase"] = "probe"
    df = tp.replay(snaps + [probe])
    assert df["close_ev"].isna().all() and set(df["close_status"]) == {"no_close_snapshot"}
    # a probe file never yields sightings either
    only_probe = _basic_snaps()[0]
    only_probe["phase"] = "probe"
    assert tp.replay([only_probe]).empty


def test_latest_close_before_kickoff_is_used():
    snaps = _basic_snaps()
    early = _snap("close", [_book("pinnacle", [("Amon-Ra St. Brown", 6.5, 200, -260)])], "2026-11-29T17:20:00Z")
    late = _snap("close", [_book("pinnacle", [("Amon-Ra St. Brown", 6.5, 300, -400)])], "2026-11-29T18:10:00Z")
    df = tp.replay(snaps + [early, late])
    r = df[(df["anchor"] == "pinnacle") & (df["player"] == "amon-ra st brown")].iloc[0]
    assert r["lam_close"] == pytest.approx(_lam(6.5, -140, 110), abs=1e-6)


# ---- 7. grading -------------------------------------------------------------------------------

def _stats(rows):
    return pd.DataFrame([{"player_id": pid, "player_display_name": n, "team": t, "game_id": g,
                          "season": 2026, "week": 13, "receptions": rec} for pid, n, t, g, rec in rows])


def _snaps(rows):
    return pd.DataFrame([{"pfr_player_id": pid, "player": n, "team": t, "game_id": g, "offense_snaps": s}
                         for pid, n, t, g, s in rows])


GID = "2026_13_CHI_DET"


def _offers(players):
    return pd.DataFrame([{"event_id": "e1", "kickoff": KICK, "home_team": "DET", "away_team": "CHI",
                          "player": pp.normalize_name(n), "player_name": n, "side": side, "line": line,
                          "odds": odds} for n, side, line, odds in players])


def test_grade_offer_win_loss_push():
    assert ps.grade_offer("OVER", 4.5, 120, 5) == ("WIN", pytest.approx(1.2))
    assert ps.grade_offer("UNDER", 4.5, -150, 5) == ("LOSS", -1.0)
    assert ps.grade_offer("OVER", 5, -110, 5) == ("PUSH", 0.0)
    assert ps.grade_offer("UNDER", 5.5, -125, 3) == ("WIN", pytest.approx(0.8))


def test_attach_results_win_void_unmatched_and_ambiguous():
    stats = _stats([("p1", "Amon-Ra St. Brown", "DET", GID, 7), ("p2", "Jameson Williams", "DET", GID, 2),
                    ("p3", "Michael Pittman Jr.", "CHI", GID, 4),
                    ("p4", "Josh Smith", "DET", GID, 3), ("p5", "Josh Smith Jr.", "DET", GID, 1)])
    snaps = _snaps([("s1", "Amon-Ra St. Brown", "DET", GID, 60), ("s2", "Jameson Williams", "DET", GID, 45),
                    ("s3", "Michael Pittman", "CHI", GID, 50), ("s6", "Sam LaPorta", "DET", GID, 0),
                    ("s4", "Josh Smith", "DET", GID, 10), ("s5", "Josh Smith Jr.", "DET", GID, 12)])
    offers = _offers([("Amon-Ra St. Brown", "OVER", 6.5, 110), ("Jameson Williams", "OVER", 3.5, -110),
                      ("Michael Pittman", "UNDER", 4, -110), ("Sam LaPorta", "OVER", 4.5, -110),
                      ("Tim Patrick", "OVER", 1.5, -110), ("Josh Smith", "OVER", 1.5, -110)])
    out, unresolved = ps.attach_results(offers, stats, snaps)
    res = dict(zip(out["player"], zip(out["result"], out["units"])))
    assert res["amon-ra st brown"] == ("WIN", pytest.approx(1.1))
    assert res["jameson williams"] == ("LOSS", -1.0)
    assert res["michael pittman"] == ("PUSH", 0.0)
    assert res["sam laporta"] == ("VOID", 0.0)             # zero offensive snaps
    assert out.set_index("player").loc["amon-ra st brown", "receptions"] == 7
    for name in ("tim patrick", "josh smith"):                # unmatched / ambiguous: never guessed
        r, u = res[name]
        assert r is None and (u is None or math.isnan(u))
    reasons = {u["player"]: u["reason"] for u in unresolved}
    assert reasons == {"tim patrick": "unmatched", "josh smith": "ambiguous"}


def test_void_when_no_snap_row_and_pending_when_game_absent():
    stats = _stats([("p1", "Amon-Ra St. Brown", "DET", GID, 0)])
    offers = _offers([("Amon-Ra St. Brown", "OVER", 6.5, 110)])
    out, unresolved = ps.attach_results(offers, stats, _snaps([("s9", "Jared Goff", "DET", GID, 60)]))
    assert out.iloc[0]["result"] == "VOID" and out.iloc[0]["units"] == 0.0 and not unresolved
    out, unresolved = ps.attach_results(offers, stats.iloc[0:0], _snaps([]).reindex(
        columns=["pfr_player_id", "player", "team", "game_id", "offense_snaps"]))
    assert out.iloc[0]["result"] is None and out.iloc[0]["grade_status"] == "pending" and not unresolved


def test_void_excluded_from_w_l():
    g = pd.DataFrame({"event_id": ["a", "a", "b"], "ev": [0.03] * 3, "close_ev": [0.01, -0.01, 0.02],
                      "close_ev_nb": [0.0] * 3, "close_ev_median": [0.0] * 3,
                      "result": ["WIN", "VOID", "LOSS"], "units": [0.9, 0.0, -1.0]})
    s = tp.group_stats(g, "x")
    assert s["W-L-P-V"] == "1-1-0-1"
    assert s["units"] == pytest.approx(-0.1)


# ---- 8. gate ----------------------------------------------------------------------------------

def _summary(**kw):
    s = {"offers": 200, "games": 35, "mean_close_ev": 0.02, "close_ev_se": 0.005,
         "mean_close_ev_ex_top2": 0.015, "side_close_ev": {"OVER": 0.02, "UNDER": 0.03},
         "side_all_close_ev": {"OVER": -0.01, "UNDER": -0.02}}
    s.update(kw)
    return s


def test_gate_pass_fail_inconclusive():
    assert tp.gate(_summary()) == "pass"
    assert tp.gate(_summary(mean_close_ev=0.009)) == "pending"           # z 1.8 < 2.0
    assert tp.gate(_summary(mean_close_ev=0.009), season_settled=True) == "inconclusive"
    assert tp.gate(_summary(mean_close_ev_ex_top2=-0.001), season_settled=True) == "inconclusive"
    assert tp.gate(_summary(mean_close_ev=0.0)) == "fail"
    assert tp.gate(_summary(mean_close_ev=-0.01)) == "fail"
    # size not reached: never pass, never fail
    assert tp.gate(_summary(offers=149)) == "pending"
    assert tp.gate(_summary(games=29, mean_close_ev=-0.05), season_settled=True) == "inconclusive"


def test_gate_drift_rule():
    # flagged Overs no better than all Overs: the Over edge is drift, so no pass
    s = _summary(side_close_ev={"OVER": 0.02, "UNDER": 0.03}, side_all_close_ev={"OVER": 0.02, "UNDER": -0.02})
    assert tp.gate(s, season_settled=True) == "inconclusive"
    s = _summary(side_close_ev={"OVER": 0.02}, side_all_close_ev={"OVER": 0.01, "UNDER": 0.05})
    assert tp.gate(s) == "pass"  # only sides present in the gate group are checked


def test_gate_summary_from_rows():
    rows = []
    for i in range(40):
        for k in range(4):
            rows.append({"event_id": f"g{i}", "side": "OVER" if k % 2 else "UNDER",
                         "close_ev": 0.01 + 0.001 * k + (0.2 if i < 2 else 0.0)})
    g = pd.DataFrame(rows)
    all_us = pd.concat([g, g.assign(close_ev=-0.05)])
    s = tp.gate_summary(g, all_us)
    assert s["offers"] == 160 and s["games"] == 40
    assert s["mean_close_ev_ex_top2"] == pytest.approx(0.0115, abs=1e-9)
    assert s["side_close_ev"]["OVER"] == pytest.approx(g[g["side"] == "OVER"]["close_ev"].mean())
    assert s["side_all_close_ev"]["UNDER"] == pytest.approx(all_us[all_us["side"] == "UNDER"]["close_ev"].mean())
    assert tp.gate(s) == "pass"


def test_gate_group_selection():
    df = pd.DataFrame({
        "anchor": ["pinnacle", "pinnacle", "pinnacle", "betonlineag", "pinnacle"],
        "book_class": ["us", "us", "offshore", "us", "us"],
        "flagged": [True, False, True, True, True],
        "close_ev": [0.01, 0.01, 0.01, 0.01, float("nan")]})
    assert list(tp.gate_group(df).index) == [0]
