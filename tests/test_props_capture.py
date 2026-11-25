import gzip
import json
import os
import random
import re
from datetime import datetime, timedelta, timezone

import pytest

from src.props import capture_props as cp


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Last 2026 regular-season kickoff (week 18, nflverse placeholder 13:00 ET).
LAST_REG_2026 = utc(2027, 1, 10, 18, 0)
CUTOFFS = {2026: LAST_REG_2026}


def event(eid, kickoff):
    return {"id": eid, "sport_key": "americanfootball_nfl", "commence_time": iso(kickoff),
            "home_team": "Home", "away_team": "Away"}


class FakeResponse:
    def __init__(self, payload, headers=None, status=200):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise cp.requests.HTTPError(f"{self.status_code} error")


class FakeGet:
    """Stands in for requests.get: records each call, answers the event-odds endpoint."""

    def __init__(self, remaining=400, last=1, include_quota=True):
        self.calls = []
        self.remaining = remaining
        self.last = last
        self.include_quota = include_quota

    def __call__(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        headers = {}
        if self.include_quota:
            self.remaining -= self.last
            headers = {"x-requests-remaining": str(self.remaining),
                       "x-requests-used": str(500 - self.remaining),
                       "x-requests-last": str(self.last)}
        eid = url.rstrip("/").split("/")[-2]
        return FakeResponse({"id": eid, "bookmakers": []}, headers)


# 1. Open targets -------------------------------------------------------------------

@pytest.mark.parametrize("kickoff,expected", [
    (utc(2026, 11, 29, 18, 0), utc(2026, 11, 27, 21, 0)),   # Sun 1pm ET
    (utc(2026, 11, 29, 14, 30), utc(2026, 11, 27, 21, 0)),  # Sun London
    (utc(2026, 11, 30, 1, 20), utc(2026, 11, 27, 21, 0)),   # SNF
    (utc(2026, 12, 1, 1, 15), utc(2026, 11, 27, 21, 0)),    # MNF
    (utc(2026, 12, 19, 18, 0), utc(2026, 12, 18, 21, 0)),   # Sat afternoon
    (utc(2026, 12, 19, 21, 30), utc(2026, 12, 18, 21, 0)),  # Sat late
    (utc(2026, 12, 4, 1, 15), utc(2026, 12, 3, 21, 15)),    # TNF: k - 4h
    (utc(2026, 11, 26, 17, 30), utc(2026, 11, 26, 13, 30)), # Thanksgiving: k - 4h
    (utc(2026, 12, 25, 18, 0), utc(2026, 12, 25, 14, 0)),   # Christmas Fri: k - 4h
])
def test_open_target_table(kickoff, expected):
    assert cp.open_target(kickoff) == expected


def test_close_target_is_kickoff_minus_25():
    k = utc(2026, 11, 29, 18, 0)
    assert cp.close_target(k) == k - timedelta(minutes=25)


# 2. Due rules at the edges -----------------------------------------------------------

K = utc(2026, 11, 29, 18, 0)


def _due(now, ledger=frozenset()):
    return {(d["phase"], d["event_id"]) for d in cp.due_items([event("ev1", K)], set(ledger), now, cutoffs=CUTOFFS)}


def test_open_due_until_two_hours_before_kickoff():
    assert ("open", "ev1") in _due(K - timedelta(minutes=121))
    assert ("open", "ev1") in _due(K - timedelta(minutes=120))
    assert ("open", "ev1") not in _due(K - timedelta(minutes=119))


def test_open_not_due_before_its_target():
    target = cp.open_target(K)
    assert ("open", "ev1") not in _due(target - timedelta(seconds=1))
    assert ("open", "ev1") in _due(target)


def test_close_window_edges():
    assert ("close", "ev1") not in _due(K - timedelta(minutes=41))
    assert ("close", "ev1") in _due(K - timedelta(minutes=40))
    assert ("close", "ev1") in _due(K - timedelta(minutes=5))
    assert ("close", "ev1") not in _due(K - timedelta(minutes=4))


def test_kicked_off_or_far_events_are_never_due():
    assert _due(K + timedelta(minutes=1)) == set()


# 3. Next wake ------------------------------------------------------------------------

def test_next_wake_is_earliest_future_target_within_deadline():
    events = [event("sun", K), event("snf", utc(2026, 11, 30, 1, 20))]
    now = utc(2026, 11, 27, 17, 30)
    deadline = now + timedelta(minutes=320)
    assert cp.next_wake(events, set(), now, deadline, cutoffs=CUTOFFS) == utc(2026, 11, 27, 21, 0)


def test_next_wake_skips_targets_already_in_ledger():
    events = [event("sun", K)]
    now = utc(2026, 11, 29, 15, 30)
    deadline = now + timedelta(minutes=320)
    assert cp.next_wake(events, {("open", "sun")}, now, deadline, cutoffs=CUTOFFS) == K - timedelta(minutes=25)
    assert cp.next_wake(events, {("open", "sun"), ("close", "sun")}, now, deadline, cutoffs=CUTOFFS) is None


def test_next_wake_none_when_no_target_fits_before_deadline():
    events = [event("sun", K)]
    now = utc(2026, 11, 27, 13, 30)
    assert cp.next_wake(events, set(), now, now + timedelta(minutes=320), cutoffs=CUTOFFS) is None


# 4. Ledger ---------------------------------------------------------------------------

def _write_gz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def test_ledger_from_archive_and_local_files(tmp_path):
    archive = tmp_path / "odds-data"
    _write_gz(archive / "props" / "2026" / "11" / "props_open_evA_20261127T210041Z.json.gz", {})
    _write_gz(archive / "snapshots" / "2026" / "11" / "odds_20261127T210000Z.json.gz", [])
    local = tmp_path / "props_cache"
    local.mkdir()
    (local / "props_open_evB_20261127T210102Z.json").write_text("{}")
    (local / "props_probe_evC_20261127T100000Z.json").write_text("{}")

    ledger = cp.load_ledger(str(archive), str(local))
    assert ledger == {("open", "evA"), ("open", "evB"), ("probe", "evC")}

    events = [event("evA", K), event("evB", K), event("evC", K)]
    at_open = {(d["phase"], d["event_id"]) for d in cp.due_items(events, ledger, utc(2026, 11, 27, 21, 1), cutoffs=CUTOFFS)}
    assert at_open == {("open", "evC")}  # a probe file does not count as the open
    at_close = {(d["phase"], d["event_id"]) for d in cp.due_items(events, ledger, K - timedelta(minutes=25), cutoffs=CUTOFFS)}
    assert at_close == {("close", "evA"), ("close", "evB"), ("close", "evC")}


# 5. Credit guards --------------------------------------------------------------------

def _close_items(n, base=K):
    items = [{"phase": "close", "event_id": f"ev{i:02d}", "kickoff": base + timedelta(minutes=i),
              "target_at": base + timedelta(minutes=i - 25)} for i in range(n)]
    random.Random(7).shuffle(items)
    return items


def test_per_run_cap_takes_earliest_kickoffs_first(tmp_path, monkeypatch):
    fake = FakeGet(remaining=400)
    monkeypatch.setattr(cp.requests, "get", fake)
    budget = cp.CreditBudget(cap=20, floor=80)
    budget.observe({"x-requests-remaining": "400"})

    written, skipped = cp.capture_items(_close_items(25), budget, "KEY", str(tmp_path), now=K - timedelta(minutes=30))

    sent = [c["url"].split("/")[-2] for c in fake.calls]
    assert sent == [f"ev{i:02d}" for i in range(20)]
    assert len(written) == 20
    assert sorted(s["event_id"] for s in skipped) == [f"ev{i:02d}" for i in range(20, 25)]
    assert budget.spent == 20


def test_quota_floor_blocks_a_call(tmp_path, monkeypatch):
    fake = FakeGet(remaining=80)
    monkeypatch.setattr(cp.requests, "get", fake)
    budget = cp.CreditBudget(cap=20, floor=80)
    budget.observe({"x-requests-remaining": "80"})

    written, skipped = cp.capture_items(_close_items(3), budget, "KEY", str(tmp_path), now=K - timedelta(minutes=30))
    assert fake.calls == [] and written == [] and len(skipped) == 3


def test_unknown_quota_allows_one_call_then_floor_applies(tmp_path, monkeypatch):
    fake = FakeGet(remaining=81)  # after the first call: 80 left, so a second would go below 80
    monkeypatch.setattr(cp.requests, "get", fake)
    budget = cp.CreditBudget(cap=20, floor=80)

    written, skipped = cp.capture_items(_close_items(3), budget, "KEY", str(tmp_path), now=K - timedelta(minutes=30))
    assert len(fake.calls) == 1 and len(written) == 1 and len(skipped) == 2
    assert budget.remaining == 80


def test_unknown_quota_stays_at_one_call_without_headers(tmp_path, monkeypatch):
    fake = FakeGet(include_quota=False)
    monkeypatch.setattr(cp.requests, "get", fake)
    budget = cp.CreditBudget(cap=20, floor=80)
    cp.capture_items(_close_items(3), budget, "KEY", str(tmp_path), now=K - timedelta(minutes=30))
    assert len(fake.calls) == 1


def test_unknown_quota_then_ample_quota_continues(tmp_path, monkeypatch):
    fake = FakeGet(remaining=300)
    monkeypatch.setattr(cp.requests, "get", fake)
    budget = cp.CreditBudget(cap=20, floor=80)
    cp.capture_items(_close_items(3), budget, "KEY", str(tmp_path), now=K - timedelta(minutes=30))
    assert len(fake.calls) == 3


def test_actual_request_cost_counts_against_the_cap(tmp_path, monkeypatch):
    fake = FakeGet(remaining=400, last=3)
    monkeypatch.setattr(cp.requests, "get", fake)
    budget = cp.CreditBudget(cap=20, floor=80)
    budget.observe({"x-requests-remaining": "400"})
    cp.capture_items(_close_items(10), budget, "KEY", str(tmp_path), now=K - timedelta(minutes=30))
    # 1st call is assumed to cost 1, then each is known to cost 3: 3 + 3*5 = 18, a 7th would be 21.
    assert len(fake.calls) == 6
    assert budget.spent == 18


# 6. Request --------------------------------------------------------------------------

def test_event_odds_request_params(tmp_path, monkeypatch):
    fake = FakeGet()
    monkeypatch.setattr(cp.requests, "get", fake)
    cp.capture_items(_close_items(1), cp.CreditBudget(cap=20, floor=80), "KEY", str(tmp_path),
                     now=K - timedelta(minutes=30))
    call = fake.calls[0]
    assert call["url"] == "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/ev00/odds"
    params = call["params"]
    assert "regions" not in params
    books = params["bookmakers"].split(",")
    assert 1 <= len(books) <= 10 and all(b and b == b.strip() for b in books)
    assert "pinnacle" in books
    assert params["markets"] == "player_receptions"
    assert params["oddsFormat"] == "american" and params["dateFormat"] == "iso"
    assert params["apiKey"] == "KEY"


def test_default_bookmaker_list_is_ten_or_fewer():
    assert len(cp.BOOKMAKERS) <= 10
    assert len(set(cp.BOOKMAKERS)) == len(cp.BOOKMAKERS)
    assert "lowvig" not in cp.BOOKMAKERS


# 7. Snapshot file --------------------------------------------------------------------

def test_snapshot_file_name_and_schema_with_empty_response(tmp_path, monkeypatch):
    fake = FakeGet(remaining=360)
    monkeypatch.setattr(cp.requests, "get", fake)
    now = utc(2026, 11, 27, 21, 0, 41)
    item = {"phase": "open", "event_id": "abc123", "kickoff": K, "target_at": utc(2026, 11, 27, 21, 0)}
    written, _ = cp.capture_items([item], cp.CreditBudget(cap=20, floor=80), "KEY", str(tmp_path), now=now)

    assert len(written) == 1
    name = os.path.basename(written[0])
    assert name == "props_open_abc123_20261127T210041Z.json"
    assert re.fullmatch(r"props_(open|close|probe)_[A-Za-z0-9-]+_\d{8}T\d{6}Z\.json", name)
    with open(written[0]) as f:
        doc = json.load(f)
    assert doc["schema"] == 1
    assert doc["phase"] == "open"
    assert doc["market"] == "player_receptions"
    assert doc["captured_at"] == "2026-11-27T21:00:41Z"
    assert doc["target_at"] == "2026-11-27T21:00:00Z"
    assert doc["kickoff"] == "2026-11-29T18:00:00Z"
    assert doc["minutes_to_kickoff"] == pytest.approx(2699.3, abs=0.05)  # Fri 21:00:41 to Sun 18:00
    assert doc["bookmakers_requested"] == cp.BOOKMAKERS
    assert doc["quota"] == {"used": 141, "remaining": 359, "last": 1}
    assert doc["response"] == {"id": "abc123", "bookmakers": []}  # written although no book posted yet


def test_failed_call_writes_nothing_and_is_reported(tmp_path, monkeypatch):
    def failing(url, params=None, timeout=None):
        return FakeResponse({"message": "bad"}, {"x-requests-remaining": "300"}, status=500)
    monkeypatch.setattr(cp.requests, "get", failing)
    budget = cp.CreditBudget(cap=20, floor=80)
    written, skipped = cp.capture_items(_close_items(1), budget, "SECRETKEY", str(tmp_path),
                                        now=K - timedelta(minutes=30))
    assert written == [] and skipped[0]["reason"].startswith("error")
    assert "SECRETKEY" not in skipped[0]["reason"]
    assert os.listdir(tmp_path) == []


# Loop --------------------------------------------------------------------------------

class Clock:
    def __init__(self, start):
        self.t = start
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += timedelta(seconds=seconds)


def _events_get(events, odds_fake, events_status=200):
    def get(url, params=None, timeout=None):
        if url.endswith("/events"):
            return FakeResponse(events, {"x-requests-remaining": "400"}, status=events_status)
        return odds_fake(url, params, timeout)
    return get


def test_loop_captures_open_then_sleeps_to_next_target_then_exits(tmp_path, monkeypatch):
    events = [event("sun", K)]
    odds = FakeGet(remaining=400)
    monkeypatch.setattr(cp.requests, "get", _events_get(events, odds))
    clock = Clock(utc(2026, 11, 29, 14, 0))  # open due (target Fri 21:00); close at 17:35 fits
    pushes = []
    archive = tmp_path / "archive"
    archive.mkdir()
    out = tmp_path / "cache"

    code = cp.run_loop(api_key="KEY", out_dir=str(out), now_fn=clock.now, sleep_fn=clock.sleep,
                       ledger_fn=lambda: cp.load_ledger(str(archive), str(out)),
                       push_fn=lambda: pushes.append(1), cutoffs_fn=lambda: CUTOFFS)

    assert code == 0
    phases = sorted(n.split("_")[1] for n in os.listdir(out))
    assert phases == ["close", "open"]
    assert len(odds.calls) == 2
    assert clock.t >= K - timedelta(minutes=25)
    assert len(pushes) == 2


def test_loop_exits_red_when_events_call_fails(tmp_path, monkeypatch):
    odds = FakeGet()
    monkeypatch.setattr(cp.requests, "get", _events_get([], odds, events_status=500))
    clock = Clock(utc(2026, 11, 29, 14, 0))
    code = cp.run_loop(api_key="KEY", out_dir=str(tmp_path), now_fn=clock.now, sleep_fn=clock.sleep,
                       ledger_fn=lambda: set(), push_fn=lambda: None,
                       cutoffs_fn=lambda: CUTOFFS)
    assert code == 1 and odds.calls == []


def test_probe_captures_earliest_event_once(tmp_path, monkeypatch):
    events = [event("later", K + timedelta(days=1)), event("first", K)]
    odds = FakeGet()
    monkeypatch.setattr(cp.requests, "get", _events_get(events, odds))
    clock = Clock(utc(2026, 11, 25, 12, 0))
    code = cp.run_loop(probe=True, api_key="KEY", out_dir=str(tmp_path), now_fn=clock.now,
                       sleep_fn=clock.sleep, ledger_fn=lambda: set(), push_fn=lambda: None,
                       cutoffs_fn=lambda: CUTOFFS)
    assert code == 0
    assert [c["url"].split("/")[-2] for c in odds.calls] == ["first"]
    (name,) = os.listdir(tmp_path)
    assert name.startswith("props_probe_first_")
    with open(tmp_path / name) as f:
        assert json.load(f)["phase"] == "probe"


# 8. Regular season only (owner decision 5) ------------------------------------------

WILD_CARD = utc(2027, 1, 16, 21, 30)   # first playoff Saturday after the 2026 week 18


def _due_at(events, now, cutoffs=CUTOFFS):
    return {(d["phase"], d["event_id"]) for d in cp.due_items(events, set(), now, cutoffs=cutoffs)}


def test_playoff_event_is_never_due_or_targeted():
    events = [event("wc", WILD_CARD)]
    assert _due_at(events, WILD_CARD - timedelta(minutes=25)) == set()
    assert _due_at(events, WILD_CARD - timedelta(hours=3)) == set()
    assert cp.upcoming(events, WILD_CARD - timedelta(days=2), cutoffs=CUTOFFS) == []
    now = WILD_CARD - timedelta(hours=5)
    assert cp.next_wake(events, set(), now, now + timedelta(hours=6), cutoffs=CUTOFFS) is None


def test_last_regular_season_game_is_still_captured():
    events = [event("w18", LAST_REG_2026), event("wc", WILD_CARD)]
    assert _due_at(events, LAST_REG_2026 - timedelta(minutes=25)) == {("close", "w18")}
    assert _due_at(events, LAST_REG_2026 - timedelta(hours=3)) == {("open", "w18")}
    # a week-18 game flexed to SNF after nflverse's 13:00 placeholder still counts
    snf = LAST_REG_2026 + timedelta(hours=7, minutes=20)
    assert _due_at([event("snf", snf)], snf - timedelta(minutes=25)) == {("close", "snf")}


def test_2027_event_allowed_once_its_schedule_is_present():
    k = utc(2027, 9, 12, 17, 0)
    events = [event("w1", k)]
    now = k - timedelta(minutes=25)
    assert _due_at(events, now) == set()                       # 2027 schedule not yet in games.csv
    with_2027 = {**CUTOFFS, 2027: utc(2028, 1, 9, 21, 25)}
    assert _due_at(events, now, with_2027) == {("close", "w1")}


def test_loop_spends_nothing_on_playoff_events(tmp_path, monkeypatch, capsys):
    odds = FakeGet(remaining=400)
    monkeypatch.setattr(cp.requests, "get", _events_get([event("wc", WILD_CARD)], odds))
    clock = Clock(WILD_CARD - timedelta(minutes=30))
    code = cp.run_loop(api_key="KEY", out_dir=str(tmp_path), now_fn=clock.now, sleep_fn=clock.sleep,
                       ledger_fn=lambda: set(), push_fn=lambda: None, cutoffs_fn=lambda: CUTOFFS)
    assert code == 0 and odds.calls == [] and os.listdir(tmp_path) == []
    assert "after the 2026 regular season" in capsys.readouterr().out


def test_missing_schedule_fails_closed(tmp_path, monkeypatch, capsys):
    from src.regular_season import ScheduleUnavailable

    def no_schedule():
        raise ScheduleUnavailable("data/nflverse_cache/games.csv not found")

    calls = []
    monkeypatch.setattr(cp.requests, "get", lambda *a, **k: calls.append(a))
    clock = Clock(K - timedelta(minutes=30))
    code = cp.run_loop(api_key="KEY", out_dir=str(tmp_path), now_fn=clock.now, sleep_fn=clock.sleep,
                       ledger_fn=lambda: set(), push_fn=lambda: None, cutoffs_fn=no_schedule)
    assert code == 1 and calls == []
    out = capsys.readouterr().out
    assert "games.csv not found" in out and "0 credits" in out


def test_season_missing_from_schedule_fails_closed(tmp_path, monkeypatch, capsys):
    odds = FakeGet(remaining=400)
    k = utc(2027, 9, 12, 17, 0)
    monkeypatch.setattr(cp.requests, "get", _events_get([event("w1", k)], odds))
    clock = Clock(k - timedelta(minutes=30))
    code = cp.run_loop(api_key="KEY", out_dir=str(tmp_path), now_fn=clock.now, sleep_fn=clock.sleep,
                       ledger_fn=lambda: set(), push_fn=lambda: None, cutoffs_fn=lambda: CUTOFFS)
    assert code == 1 and odds.calls == []
    assert "no regular-season schedule for 2027" in capsys.readouterr().out


def test_probe_skips_playoff_events(tmp_path, monkeypatch):
    odds = FakeGet()
    monkeypatch.setattr(cp.requests, "get", _events_get([event("wc", WILD_CARD)], odds))
    clock = Clock(WILD_CARD - timedelta(days=2))
    code = cp.run_loop(probe=True, api_key="KEY", out_dir=str(tmp_path), now_fn=clock.now,
                       sleep_fn=clock.sleep, ledger_fn=lambda: set(), push_fn=lambda: None,
                       cutoffs_fn=lambda: CUTOFFS)
    assert code == 1 and odds.calls == []
