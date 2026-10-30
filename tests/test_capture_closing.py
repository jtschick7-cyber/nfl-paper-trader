"""
Unit tests for hardened closing line capture, pre-kickoff cache catch-up,
RLM idempotency, settled CLV recovery, and verified TNF backfill.
"""

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from src.backfill_closing import VERIFIED_CLOSING_BACKFILLS, backfill_closing_records
from src.capture_closing import (
    build_closing_row_updates,
    capture_closing,
    row_needs_cache_catchup,
    row_needs_closing,
)
from src.compare_models import evaluate_all_models


def _utc(iso_str: str):
    return pd.to_datetime(iso_str, utc=True)


def test_row_needs_closing_two_tier_window():
    kickoff = "2026-10-30T00:15:00+00:00"
    ko_ts = _utc(kickoff)

    base_row = {
        "game_id": "2026_08_CAR_GB",
        "status": "OPEN_CAPTURED",
        "closing_line": float("nan"),
        "closing_captured_at": "",
        "kickoff_time": kickoff,
    }

    # Too early (> 240m pre-kickoff, e.g. 5h before kickoff)
    assert not row_needs_closing(base_row, ko_ts - timedelta(hours=5))

    # Inside widened safety-net window (e.g. 138m pre-kickoff — exact TNF 21:56 UTC run!)
    assert row_needs_closing(base_row, ko_ts - timedelta(minutes=138))

    # Inside prime closing window (e.g. 15m pre-kickoff)
    assert row_needs_closing(base_row, ko_ts - timedelta(minutes=15))

    # Too close / past kickoff (< 2m pre-kickoff -> handled by cache catch-up instead)
    assert not row_needs_closing(base_row, ko_ts - timedelta(minutes=1))
    assert not row_needs_closing(base_row, ko_ts + timedelta(minutes=10))

    # Early safety-net capture at T-138m:
    early_captured_row = dict(
        base_row,
        status="CLOSE_CAPTURED",
        closing_line=-3.5,
        closing_captured_at=(ko_ts - timedelta(minutes=138)).isoformat(),
    )
    # Still outside prime window (T-90m > 60m): should NOT re-fetch
    assert not row_needs_closing(early_captured_row, ko_ts - timedelta(minutes=90))
    # Inside prime window (T-35m <= 60m): SHOULD upgrade once!
    assert row_needs_closing(early_captured_row, ko_ts - timedelta(minutes=35))

    # Once upgraded inside prime window (T-35m), subsequent runs (T-10m) are idempotent
    prime_captured_row = dict(
        early_captured_row,
        closing_line=-3.0,
        closing_captured_at=(ko_ts - timedelta(minutes=35)).isoformat(),
    )
    assert not row_needs_closing(prime_captured_row, ko_ts - timedelta(minutes=10))


def test_row_needs_cache_catchup_includes_settled_and_result_updated():
    kickoff = "2026-10-30T00:15:00+00:00"
    after_ko = _utc("2026-10-30T12:00:00+00:00")

    for st in ("PENDING", "OPEN_CAPTURED", "CLOSE_CAPTURED", "RESULT_UPDATED", "SETTLED"):
        row = {
            "game_id": "2026_08_CAR_GB",
            "status": st,
            "closing_line": float("nan"),
            "kickoff_time": kickoff,
        }
        assert row_needs_cache_catchup(row, after_ko), f"Expected catchup eligible for status={st}"

    # Already has closing_line -> False
    populated = {
        "game_id": "2026_08_CAR_GB",
        "status": "RESULT_UPDATED",
        "closing_line": -3.0,
        "kickoff_time": kickoff,
    }
    assert not row_needs_cache_catchup(populated, after_ko)

    # ERROR status -> False
    err_row = dict(populated, status="ERROR", closing_line=float("nan"))
    assert not row_needs_cache_catchup(err_row, after_ko)


def test_build_closing_row_updates_preserves_postgame_status_and_heals_clv():
    col = {
        "closing_line": 1,
        "closing_odds": 2,
        "closing_away_odds": 3,
        "closing_captured_at": 4,
        "status": 5,
        "clv_points": 6,
        "clv_flag": 7,
        "notes": 8,
    }

    # 1. RESULT_UPDATED no-pick game: status NOT overwritten to CLOSE_CAPTURED
    no_pick_row = {
        "game_id": "2026_08_CAR_GB",
        "home_team": "GB",
        "away_team": "CAR",
        "status": "RESULT_UPDATED",
        "pick_team": "",
        "notes": "",
    }
    u1 = build_closing_row_updates(
        row=no_pick_row,
        row_num=10,
        col=col,
        home_spread=-3.0,
        home_odds=-105.0,
        away_odds=-115.0,
        captured_at_iso="2026-10-29T22:28:15+00:00",
    )
    by_col1 = {item["col"]: item["value"] for item in u1}
    assert col["status"] not in by_col1
    assert by_col1[col["closing_line"]] == -3.0
    assert by_col1[col["closing_odds"]] == -105.0
    assert by_col1[col["closing_away_odds"]] == -115.0
    assert by_col1[col["closing_captured_at"]] == "2026-10-29T22:28:15+00:00"

    # 2. SETTLED game with AWAY pick +3.5 vs closing GB -3.0 (CAR +3.0 -> +0.5 CLV)
    #    and opening GB -1.5 -> GB -3.0 (1.5 pts RLM against CAR AWAY!)
    settled_row = {
        "game_id": "2026_08_CAR_GB",
        "home_team": "GB",
        "away_team": "CAR",
        "opening_line": -1.5,
        "pick_team": "CAR",
        "pick_side": "AWAY",
        "pick_line": 3.5,
        "status": "SETTLED",
        "notes": "flat 1.0u; MISSING_CLOSING_LINE",
    }
    u2 = build_closing_row_updates(
        row=settled_row,
        row_num=11,
        col=col,
        home_spread=-3.0,
        home_odds=-105.0,
        away_odds=-115.0,
        captured_at_iso="2026-10-29T22:28:15+00:00",
    )
    by_col2 = {item["col"]: item["value"] for item in u2}
    assert col["status"] not in by_col2
    assert pytest.approx(by_col2[col["clv_points"]]) == 0.5
    assert by_col2[col["clv_flag"]] is True
    assert "MISSING_CLOSING_LINE" not in by_col2[col["notes"]]
    assert "flat 1.0u" in by_col2[col["notes"]]
    assert by_col2[col["notes"]].count("RLM_ALERT:") == 1

    # Re-running on a row that already has RLM_ALERT does not duplicate it
    settled_row_with_rlm = dict(settled_row, notes=by_col2[col["notes"]])
    u3 = build_closing_row_updates(
        row=settled_row_with_rlm,
        row_num=11,
        col=col,
        home_spread=-3.0,
        home_odds=-105.0,
        away_odds=-115.0,
        captured_at_iso="2026-10-29T22:28:15+00:00",
    )
    by_col3 = {item["col"]: item["value"] for item in u3}
    assert col["notes"] not in by_col3


def test_capture_closing_recovers_cache_before_live_fetch(tmp_path, monkeypatch):
    """
    Simulates a post-kickoff run where:
    - 2026_08_CAR_GB is already RESULT_UPDATED and missing closing_line, with pre-game odds in cache.
    - 2026_08_MIN_DET is 45m before kickoff (live target).
    Verifies cache recovery runs BEFORE fetch_odds_api overwrites odds_latest.json,
    and emits settled_backfilled=true to GITHUB_OUTPUT.
    """
    now_fixed = datetime(2026, 10, 30, 1, 40, 56, tzinfo=pytz.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_fixed if tz else now_fixed.replace(tzinfo=None)

    monkeypatch.setattr("src.capture_closing.datetime", FixedDateTime)
    monkeypatch.setattr("src.capture_closing.ATTEMPTS_FILE", str(tmp_path / "attempts.json"))

    gh_out = tmp_path / "gh_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))

    # Seed pre-kickoff cache file in tmp_path (simulating 22:28:15Z capture)
    latest_cache = tmp_path / "odds_latest.json"
    latest_cache.write_text(
        json.dumps([{
            "id": "tnf",
            "commence_time": "2026-10-30T00:15:00Z",
            "home_team": "Green Bay Packers",
            "away_team": "Carolina Panthers",
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-10-29T22:28:15Z",
                "markets": [{
                    "key": "spreads",
                    "last_update": "2026-10-29T22:28:15Z",
                    "outcomes": [
                        {"name": "Green Bay Packers", "point": -3.0, "price": -105.0},
                        {"name": "Carolina Panthers", "point": 3.0, "price": -115.0},
                    ],
                }],
            }],
        }]),
        encoding="utf-8",
    )

    sheet_df = pd.DataFrame([
        {
            "game_id": "2026_08_CAR_GB",
            "home_team": "GB",
            "away_team": "CAR",
            "kickoff_time": "2026-10-30T00:15:00+00:00",
            "opening_line": -3.5,
            "closing_line": float("nan"),
            "closing_odds": float("nan"),
            "closing_away_odds": float("nan"),
            "closing_captured_at": "",
            "pick_team": "",
            "pick_side": "",
            "status": "RESULT_UPDATED",
            "notes": "",
        },
        {
            "game_id": "2026_08_MIN_DET",
            "home_team": "DET",
            "away_team": "MIN",
            "kickoff_time": (now_fixed + timedelta(minutes=45)).isoformat(),
            "opening_line": -2.5,
            "closing_line": float("nan"),
            "closing_odds": float("nan"),
            "closing_away_odds": float("nan"),
            "closing_captured_at": "",
            "pick_team": "DET",
            "pick_side": "HOME",
            "status": "OPEN_CAPTURED",
            "notes": "",
        },
    ])

    def fake_fetch_odds_api(**kwargs):
        # Overwrite odds_latest.json with live in-play TNF spread + upcoming DET spread!
        live_payload = [
            {
                "id": "tnf",
                "commence_time": "2026-10-30T00:15:00Z",
                "home_team": "Green Bay Packers",
                "away_team": "Carolina Panthers",
                "bookmakers": [{
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2026-10-30T01:40:00Z",
                    "markets": [{
                        "key": "spreads",
                        "last_update": "2026-10-30T01:40:00Z",
                        "outcomes": [
                            {"name": "Green Bay Packers", "point": -14.5, "price": -110.0},
                            {"name": "Carolina Panthers", "point": 14.5, "price": -110.0},
                        ],
                    }],
                }],
            },
            {
                "id": "det",
                "commence_time": (now_fixed + timedelta(minutes=45)).isoformat(),
                "home_team": "Detroit Lions",
                "away_team": "Minnesota Vikings",
                "bookmakers": [{
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": now_fixed.isoformat(),
                    "markets": [{
                        "key": "spreads",
                        "last_update": now_fixed.isoformat(),
                        "outcomes": [
                            {"name": "Detroit Lions", "point": -3.0, "price": -110.0},
                            {"name": "Minnesota Vikings", "point": 3.0, "price": -110.0},
                        ],
                    }],
                }],
            },
        ]
        latest_cache.write_text(json.dumps(live_payload), encoding="utf-8")
        return live_payload

    headers = list(sheet_df.columns)

    class DummyWS:
        def row_values(self, _):
            return headers

    captured_updates = []

    monkeypatch.setattr("src.capture_closing.get_sheet_as_df", lambda: sheet_df)
    monkeypatch.setattr("src.capture_closing.get_master_sheet", lambda: DummyWS())
    monkeypatch.setattr("src.capture_closing.fetch_odds_api", fake_fetch_odds_api)
    monkeypatch.setattr("src.capture_closing.batch_update_cells", lambda ws, u: captured_updates.extend(u))

    capture_closing(cache_dir=str(tmp_path))

    col_idx = {h: i + 1 for i, h in enumerate(headers)}
    row2 = {u["col"]: u["value"] for u in captured_updates if u["row"] == 2}
    row3 = {u["col"]: u["value"] for u in captured_updates if u["row"] == 3}

    # Row 2 (2026_08_CAR_GB): recovered -3.0 (NOT -14.5 in-play!) and kept RESULT_UPDATED status
    assert row2[col_idx["closing_line"]] == -3.0
    assert row2[col_idx["closing_odds"]] == -105.0
    assert row2[col_idx["closing_away_odds"]] == -115.0
    assert row2[col_idx["closing_captured_at"]] == "2026-10-29T22:28:15+00:00"
    assert col_idx["status"] not in row2

    # Row 3 (2026_08_MIN_DET): captured live -3.0 and transitioned to CLOSE_CAPTURED
    assert row3[col_idx["closing_line"]] == -3.0
    assert row3[col_idx["status"]] == "CLOSE_CAPTURED"

    assert "settled_backfilled=true" in gh_out.read_text(encoding="utf-8")


def test_backfill_tnf_and_shadow_clv_evaluation(monkeypatch):
    """
    Verifies VERIFIED_CLOSING_BACKFILLS populates 2026_08_CAR_GB idempotently
    and unlocks CLV grading for shadow models v2, v3, and v4 in compare_models.
    """
    sheet_df = pd.DataFrame([{
        "game_id": "2026_08_CAR_GB",
        "week": 8,
        "home_team": "GB",
        "away_team": "CAR",
        "kickoff_time": "2026-10-30T00:15:00+00:00",
        "opening_line": -3.5,
        "opening_odds": -115.0,
        "opening_away_odds": -105.0,
        "closing_line": float("nan"),
        "closing_odds": float("nan"),
        "closing_away_odds": float("nan"),
        "closing_captured_at": "",
        "pick_team": "",
        "pick_side": "",
        "pick_line": float("nan"),
        "pick_odds": float("nan"),
        "status": "RESULT_UPDATED",
        "notes": "",
    }])
    headers = list(sheet_df.columns)

    class DummyWS:
        col_count = len(headers)

        def row_values(self, _):
            return headers

    applied = []

    def apply_updates(ws, updates):
        applied.extend(updates)
        for u in updates:
            r_idx = u["row"] - 2
            c_name = headers[u["col"] - 1]
            sheet_df.at[r_idx, c_name] = u["value"]

    monkeypatch.setattr("src.backfill_closing.get_sheet_as_df", lambda: sheet_df)
    monkeypatch.setattr("src.backfill_closing.get_master_sheet", lambda: DummyWS())
    monkeypatch.setattr("src.backfill_closing.batch_update_cells", apply_updates)

    n1 = backfill_closing_records(VERIFIED_CLOSING_BACKFILLS)
    assert n1 == 1
    assert sheet_df.loc[0, "closing_line"] == -3.0
    assert sheet_df.loc[0, "closing_odds"] == -105.0
    assert sheet_df.loc[0, "closing_away_odds"] == -115.0
    assert sheet_df.loc[0, "closing_captured_at"] == "2026-10-29T22:28:15+00:00"
    assert sheet_df.loc[0, "status"] == "RESULT_UPDATED"

    # Second run is strictly idempotent
    applied.clear()
    n2 = backfill_closing_records(VERIFIED_CLOSING_BACKFILLS)
    assert n2 == 0
    assert applied == []

    # Verify compare_models now grades CLV for shadow models on 2026_08_CAR_GB
    scores = {"2026_08_CAR_GB": {"home_score": 24, "away_score": 17, "home_team": "GB", "away_team": "CAR"}}
    v2_df = pd.DataFrame([{
        "game_id": "2026_08_CAR_GB",
        "week": 8,
        "v2_decision": "SHADOW_PICK",
        "v2_best_team": "CAR",
        "v2_best_side": "AWAY",
        "v2_best_line": 3.5,
        "v2_best_odds": -105.0,
        "v2_edge": 0.025,
    }])
    v3_df = pd.DataFrame([{
        "game_id": "2026_08_CAR_GB",
        "week": 8,
        "v3_decision": "SHADOW_PICK",
        "v3_best_team": "CAR",
        "v3_best_side": "AWAY",
        "v3_best_line": 3.5,
        "v3_best_odds": -105.0,
        "v3_edge": 0.030,
    }])
    v4_df = pd.DataFrame([{
        "game_id": "2026_08_CAR_GB",
        "week": 8,
        "v4_decision": "HIGH_CONVICTION",
        "v4_best_team": "CAR",
        "v4_best_side": "AWAY",
        "v4_best_line": 3.5,
        "v4_best_odds": -105.0,
        "v4_edge": 0.040,
        "v4_stake_units": 1.0,
    }])

    summary_rows, _, _ = evaluate_all_models(sheet_df, v2_df, v3_df, scores, v4_df=v4_df)
    by_model = {r["model"]: r for r in summary_rows}

    # CAR +3.5 vs closing CAR +3.0 (GB -3.0) = +0.5 CLV (100% CLV Beat)
    assert by_model["Model v2 (Elo + Rest Baseline)"]["clv_beat_pct"] == 100.0
    assert by_model["Model v2 (Elo + Rest Baseline)"]["avg_clv"] == 0.5
    assert by_model["Model v3 (EPA + Discrete Keys)"]["clv_beat_pct"] == 100.0
    assert by_model["Model v3 (EPA + Discrete Keys)"]["avg_clv"] == 0.5
    assert by_model["Model v4 (EPA+QB+Luck+Totals)"]["clv_beat_pct"] == 100.0
    assert by_model["Model v4 (EPA+QB+Luck+Totals)"]["avg_clv"] == 0.5
