import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from src import midweek_snapshot as ms
from src import odds_utils
from src.archive_snapshots import archive_path, archive_snapshots
from src.load_odds_archive import iter_archive, snapshot_time
from src.regular_season import ScheduleUnavailable, Window


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


WED = utc(2026, 12, 2, 17, 31)   # Wednesday
THU = utc(2026, 12, 3, 17, 31)
SAT = utc(2026, 12, 5, 17, 31)
MON, TUE, FRI, SUN = (utc(2026, 11, 30, 17, 31), utc(2026, 12, 1, 17, 31),
                      utc(2026, 12, 4, 17, 31), utc(2026, 12, 6, 17, 31))


class FakeResponse:
    def __init__(self, payload, headers=None, status=200):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ms.requests.HTTPError(f"{self.status_code} error")


LAST_REG_2026 = utc(2027, 1, 10, 18, 0)   # 2026 week 18 (nflverse placeholder 13:00 ET)
FIRST_REG_2026 = utc(2026, 9, 11, 0, 20)
CUTOFFS = {2026: Window(FIRST_REG_2026, LAST_REG_2026)}

UPCOMING = [{"id": "g1", "commence_time": "2026-12-06T18:00:00Z"}]   # inside 168 h of WED, THU and SAT


class FakeGet:
    """Answers /events for free (with `events`) and /odds at `odds_cost` credits; odds statuses are scripted."""

    def __init__(self, remaining=300, odds_cost=2, odds_statuses=None, quota_headers=True,
                 payload=None, events=None):
        self.calls = []
        self.remaining = remaining
        self.odds_cost = odds_cost
        self.odds_statuses = list(odds_statuses or [])
        self.quota_headers = quota_headers
        self.payload = payload if payload is not None else [{"id": "g1", "bookmakers": []}]
        self.events = UPCOMING if events is None else events

    def _headers(self, last):
        if not self.quota_headers:
            return {}
        return {"x-requests-remaining": str(self.remaining), "x-requests-used": str(500 - self.remaining),
                "x-requests-last": str(last)}

    def __call__(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if url.endswith("/events"):
            return FakeResponse(self.events, self._headers(0))
        status = self.odds_statuses.pop(0) if self.odds_statuses else 200
        if isinstance(status, Exception):
            raise status
        self.remaining -= self.odds_cost
        return FakeResponse(self.payload if status < 400 else {"message": "x"}, self._headers(self.odds_cost),
                            status)

    @property
    def odds_calls(self):
        return [c for c in self.calls if c["url"].endswith("/odds/")]


def run(now, tmp_path, monkeypatch, get=None, archive_times=(), ledger_error=None, **kw):
    get = get or FakeGet()
    kw.setdefault("cutoffs_fn", lambda: CUTOFFS)
    monkeypatch.setattr(ms.requests, "get", get)

    def ledger_fn():
        if ledger_error:
            raise ledger_error
        return list(archive_times)

    code = ms.run(api_key="KEY", now=now, out_dir=str(tmp_path / "snaps"), ledger_fn=ledger_fn,
                  sleep_fn=lambda s: None, **kw)
    return code, get


def written(tmp_path):
    d = tmp_path / "snaps"
    return sorted(os.listdir(d)) if d.is_dir() else []


# 1. Request is the existing game-odds request -----------------------------------------

def test_request_params_match_fetch_odds_api(tmp_path, monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"], seen["params"] = url, dict(params)
        return FakeResponse([], {})

    monkeypatch.setattr(odds_utils.requests, "get", fake_get)
    monkeypatch.setattr(odds_utils, "ODDS_API_KEY", "KEY")
    monkeypatch.setattr(odds_utils, "CACHE_DIR", str(tmp_path / "cache"))
    odds_utils.fetch_odds_api(force_refresh=True)

    url, params = ms.odds_request("KEY")
    assert url == seen["url"]
    assert params == seen["params"]
    assert params["regions"] == "us" and params["markets"] == "spreads,totals"
    assert "bookmakers" not in params


def test_estimated_cost_is_markets_times_regions():
    assert ms.estimated_cost() == 2


# 2. Day and time gate -----------------------------------------------------------------

@pytest.mark.parametrize("now", [WED, THU, SAT])
def test_gate_open_on_wed_thu_sat_from_1730(now):
    assert ms.gate_reason(now) is None


@pytest.mark.parametrize("now", [MON, TUE, FRI, SUN])
def test_gate_closed_on_other_days(now):
    assert "weekday" in ms.gate_reason(now)


def test_gate_closed_at_the_1330_wake_and_open_late_in_the_day():
    assert "before" in ms.gate_reason(WED.replace(hour=13, minute=31))
    assert ms.gate_reason(WED.replace(hour=17, minute=25)) is None   # 5-min tolerance
    assert "before" in ms.gate_reason(WED.replace(hour=17, minute=24, second=59))
    assert ms.gate_reason(WED.replace(hour=21, minute=31)) is None
    assert ms.gate_reason(WED.replace(hour=23, minute=59)) is None


def test_closed_gate_makes_no_call(tmp_path, monkeypatch):
    code, get = run(FRI, tmp_path, monkeypatch)
    assert code == 0 and get.calls == [] and written(tmp_path) == []
    code, get = run(WED.replace(hour=13, minute=35), tmp_path, monkeypatch)
    assert code == 0 and get.calls == []


# 3. Ledger: at most one snapshot per day ----------------------------------------------

def test_taken_today_counts_snapshots_since_the_window_start_only():
    start = ms.window_start(WED)
    assert ms.taken_today([start], WED)
    assert ms.taken_today([WED - timedelta(minutes=1)], WED)
    assert not ms.taken_today([start - timedelta(seconds=1)], WED)    # 13:30-wake closing snapshot
    assert not ms.taken_today([WED - timedelta(days=1)], WED)
    assert not ms.taken_today([], WED)


def test_snapshot_already_in_archive_today_makes_no_call(tmp_path, monkeypatch):
    late = WED.replace(hour=21, minute=40)
    code, get = run(late, tmp_path, monkeypatch, archive_times=[WED])
    assert code == 0 and get.odds_calls == [] and written(tmp_path) == []


def test_local_snapshot_today_counts_as_taken(tmp_path, monkeypatch):
    d = tmp_path / "snaps"
    d.mkdir()
    (d / "odds_20261202T173100Z.json").write_text("[]")
    code, get = run(WED.replace(hour=21, minute=40), tmp_path, monkeypatch)
    assert code == 0 and get.odds_calls == []


def test_unreadable_ledger_makes_no_call_and_fails(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, ledger_error=RuntimeError("fetch failed"))
    assert code == 1 and get.odds_calls == [] and written(tmp_path) == []


def test_ledger_reads_the_archive_listing(tmp_path):
    root = tmp_path / "odds-data" / "snapshots" / "2026" / "12"
    root.mkdir(parents=True)
    (root / "odds_20261202T173100Z.json.gz").write_bytes(b"")
    assert ms.load_ledger(str(tmp_path / "odds-data")) == [utc(2026, 12, 2, 17, 31)]


# 4. Capture and the file format the readers parse -------------------------------------

def test_capture_writes_one_snapshot_in_archive_format(tmp_path, monkeypatch):
    payload = [{"id": "g1", "commence_time": "2026-12-06T18:00:00Z", "bookmakers": [{"key": "draftkings"}]}]
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(payload=payload))
    assert code == 0
    assert len(get.odds_calls) == 1
    names = written(tmp_path)
    assert names == ["odds_20261202T173100Z.json"]
    with open(tmp_path / "snaps" / names[0]) as f:
        assert json.load(f) == payload

    dest = tmp_path / "odds-data"
    paths = archive_snapshots(str(tmp_path / "snaps"), str(dest))
    assert paths == [archive_path(str(dest), names[0])]
    assert paths[0].endswith("snapshots/2026/12/odds_20261202T173100Z.json.gz")
    assert list(iter_archive(str(dest))) == [(utc(2026, 12, 2, 17, 31), payload)]
    assert snapshot_time(paths[0]) == utc(2026, 12, 2, 17, 31)


def test_rerun_after_capture_is_idempotent(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch)
    assert code == 0 and len(get.odds_calls) == 1
    code, get = run(WED + timedelta(hours=4), tmp_path, monkeypatch)
    assert code == 0 and get.odds_calls == [] and len(written(tmp_path)) == 1


def test_non_list_response_is_not_written_and_fails(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(payload={"message": "odd"}))
    assert code == 1 and written(tmp_path) == []
    assert len(get.odds_calls) == 1   # a malformed 200 is not retried (it was billed)


# 5. Credit guards -------------------------------------------------------------------

def test_quota_floor_blocks_the_call(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(remaining=81))
    assert code == 1 and get.odds_calls == [] and written(tmp_path) == []


def test_quota_exactly_at_floor_plus_cost_allows_the_call(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(remaining=82))
    assert code == 0 and len(get.odds_calls) == 1


def test_default_floor_matches_props_capture():
    from src.props import capture_props
    assert ms.QUOTA_FLOOR == capture_props.QUOTA_FLOOR == 80
    assert ms.MAX_CALLS_PER_RUN == 2


def test_server_error_retries_once_then_stops_at_the_cap(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(odds_statuses=[502, 503, 200]))
    assert code == 1 and len(get.odds_calls) == 2 and written(tmp_path) == []


def test_server_error_then_success(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch,
                    get=FakeGet(odds_statuses=[ms.requests.ConnectionError("reset"), 200]))
    assert code == 0 and len(get.odds_calls) == 2 and len(written(tmp_path)) == 1


def test_client_error_is_not_retried(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(odds_statuses=[401]))
    assert code == 1 and len(get.odds_calls) == 1


def test_retry_rechecks_the_floor(tmp_path, monkeypatch):
    # 83 left: the first call (502) is billed 2 -> 81 left, and 81 - 2 < 80 blocks the retry.
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(remaining=83, odds_statuses=[502, 200]))
    assert code == 1 and len(get.odds_calls) == 1


def test_unknown_quota_allows_one_call_only(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(quota_headers=False, odds_statuses=[502, 200]))
    assert code == 1 and len(get.odds_calls) == 1
    code, get = run(THU, tmp_path, monkeypatch, get=FakeGet(quota_headers=False))
    assert code == 0 and len(get.odds_calls) == 1


def test_events_call_failure_makes_no_paid_call(tmp_path, monkeypatch):
    class Boom(FakeGet):
        def __call__(self, url, params=None, timeout=None):
            if url.endswith("/events"):
                self.calls.append({"url": url, "params": {}})
                raise ms.requests.ConnectionError("down")
            return super().__call__(url, params, timeout)

    code, get = run(WED, tmp_path, monkeypatch, get=Boom())
    assert code == 1 and get.odds_calls == []


def test_api_key_is_redacted_from_errors(tmp_path, monkeypatch, capsys):
    run(WED, tmp_path, monkeypatch,
        get=FakeGet(odds_statuses=[ms.requests.ConnectionError("https://x?apiKey=KEY"),
                                   ms.requests.ConnectionError("apiKey=KEY")]))
    assert "KEY" not in capsys.readouterr().out.replace("apiKey", "")


# 6. No game due: no paid call (off-season, Pro Bowl week) -----------------------------

def _ev(t):
    return {"id": "g", "commence_time": t.strftime("%Y-%m-%dT%H:%M:%SZ")}


def test_has_upcoming_event_window():
    now = WED
    assert ms.HORIZON_HOURS == 168
    assert ms.has_upcoming_event([_ev(now + timedelta(hours=167))], now, cutoffs=CUTOFFS)
    assert ms.has_upcoming_event([_ev(now + timedelta(hours=168))], now, cutoffs=CUTOFFS)
    assert not ms.has_upcoming_event([_ev(now + timedelta(hours=168, seconds=1))], now, cutoffs=CUTOFFS)
    assert not ms.has_upcoming_event([_ev(now)], now, cutoffs=CUTOFFS)                       # already kicked off
    assert not ms.has_upcoming_event([_ev(now - timedelta(hours=1))], now, cutoffs=CUTOFFS)
    assert not ms.has_upcoming_event([], now, cutoffs=CUTOFFS)
    assert not ms.has_upcoming_event(None, now, cutoffs=CUTOFFS)


def test_has_upcoming_event_skips_malformed_entries():
    now = WED
    good = _ev(now + timedelta(days=2))
    junk = [{"id": "x"}, {"commence_time": "not a time"}, {"commence_time": None}, "str", 7]
    assert not ms.has_upcoming_event(junk, now, cutoffs=CUTOFFS)
    assert ms.has_upcoming_event(junk + [good], now, cutoffs=CUTOFFS)


def test_empty_events_makes_no_paid_call(tmp_path, monkeypatch, capsys):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(events=[]))
    assert code == 0 and get.odds_calls == [] and written(tmp_path) == []
    assert "0 credits spent" in capsys.readouterr().out


@pytest.mark.parametrize("delta", [timedelta(days=10), -timedelta(hours=1)])
def test_only_far_or_past_events_make_no_paid_call(tmp_path, monkeypatch, delta):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(events=[_ev(WED + delta)]))
    assert code == 0 and get.odds_calls == [] and written(tmp_path) == []


@pytest.mark.parametrize("hours", [144, 167])
def test_event_inside_the_horizon_takes_the_snapshot(tmp_path, monkeypatch, hours):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(events=[_ev(WED + timedelta(hours=hours))]))
    assert code == 0 and len(get.odds_calls) == 1 and len(written(tmp_path)) == 1


def test_non_list_events_fails_without_a_paid_call(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, get=FakeGet(events={"message": "odd"}))
    assert code == 1 and get.odds_calls == [] and written(tmp_path) == []


# 7. Readers see only snapshots/ ------------------------------------------------------

def test_spread_readers_import_the_same_lister():
    from src import topdown_replay, topdown_v2
    assert topdown_v2.iter_archive is iter_archive
    assert topdown_replay.iter_archive is iter_archive


# 8. Regular season only (owner decision 5) ------------------------------------------

WED_W18 = utc(2027, 1, 6, 17, 31)    # Wednesday before the 2026 week 18
WED_WC = utc(2027, 1, 13, 17, 31)    # Wednesday before wild-card weekend
WILD_CARD = utc(2027, 1, 16, 21, 30)
WED_W18_2027 = utc(2028, 1, 5, 17, 31)
W18_2027 = utc(2028, 1, 9, 18, 0)
WITH_2027 = {**CUTOFFS, 2027: Window(utc(2027, 9, 10, 0, 20), W18_2027)}


def no_schedule():
    raise ScheduleUnavailable("data/nflverse_cache/games.csv not found")


def test_test_dates_are_gate_days():
    for d in (WED_W18, WED_WC, WED_W18_2027, utc(2027, 8, 25, 17, 31)):
        assert ms.gate_reason(d) is None, d


def test_has_upcoming_event_counts_regular_season_games_only():
    assert ms.has_upcoming_event([_ev(LAST_REG_2026)], WED_W18, cutoffs=CUTOFFS)
    assert not ms.has_upcoming_event([_ev(WILD_CARD)], WED_WC, cutoffs=CUTOFFS)
    assert not ms.has_upcoming_event([_ev(W18_2027)], WED_W18_2027, cutoffs=CUTOFFS)
    assert ms.has_upcoming_event([_ev(W18_2027)], WED_W18_2027, cutoffs=WITH_2027)


def test_playoff_week_makes_no_paid_call(tmp_path, monkeypatch, capsys):
    code, get = run(WED_WC, tmp_path, monkeypatch, get=FakeGet(events=[_ev(WILD_CARD)]))
    assert code == 0 and get.odds_calls == [] and written(tmp_path) == []
    out = capsys.readouterr().out
    assert "regular-season" in out and "0 credits spent" in out


def test_week_18_snapshot_is_still_taken(tmp_path, monkeypatch):
    code, get = run(WED_W18, tmp_path, monkeypatch, get=FakeGet(events=[_ev(LAST_REG_2026)]))
    assert code == 0 and len(get.odds_calls) == 1 and len(written(tmp_path)) == 1


def test_week_18_inside_the_grace_window_counts(tmp_path, monkeypatch):
    snf = LAST_REG_2026 + timedelta(hours=7, minutes=20)
    code, get = run(WED_W18, tmp_path, monkeypatch, get=FakeGet(events=[_ev(snf)]))
    assert code == 0 and len(get.odds_calls) == 1


def test_december_snapshot_is_taken_without_a_schedule(tmp_path, monkeypatch):
    code, get = run(WED, tmp_path, monkeypatch, cutoffs_fn=no_schedule)
    assert code == 0 and len(get.odds_calls) == 1 and len(written(tmp_path)) == 1


def test_january_playoff_fails_closed_without_a_schedule(tmp_path, monkeypatch, capsys):
    code, get = run(WED_WC, tmp_path, monkeypatch, get=FakeGet(events=[_ev(WILD_CARD)]),
                    cutoffs_fn=no_schedule)
    assert code == 1 and get.odds_calls == [] and written(tmp_path) == []
    out = capsys.readouterr().out
    assert "games.csv not found" in out and "0 credits spent" in out


def test_season_missing_from_schedule_fails_closed(tmp_path, monkeypatch, capsys):
    code, get = run(WED_W18_2027, tmp_path, monkeypatch, get=FakeGet(events=[_ev(W18_2027)]))
    assert code == 1 and get.odds_calls == []
    assert "no regular-season schedule for 2027" in capsys.readouterr().out


def test_2027_snapshot_allowed_once_its_schedule_is_present(tmp_path, monkeypatch):
    code, get = run(WED_W18_2027, tmp_path, monkeypatch, get=FakeGet(events=[_ev(W18_2027)]),
                    cutoffs_fn=lambda: WITH_2027)
    assert code == 0 and len(get.odds_calls) == 1


def test_august_preseason_makes_no_paid_call(tmp_path, monkeypatch, capsys):
    code, get = run(utc(2027, 8, 25, 17, 31), tmp_path, monkeypatch,
                    get=FakeGet(events=[_ev(utc(2027, 8, 27, 0, 0))]), cutoffs_fn=lambda: WITH_2027)
    assert code == 0 and get.odds_calls == []
    assert "before the 2027 regular season" in capsys.readouterr().out
