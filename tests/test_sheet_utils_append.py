"""P1-7: append_rows is retried only when the request was rejected, never after a 5xx."""

import pytest

import src.sheet_utils as su


class _WS:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def append_rows(self, rows, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_quota_rejection_is_retried(monkeypatch):
    monkeypatch.setattr(su.time, "sleep", lambda s: None)
    ws = _WS([Exception("APIError: [429]: Quota exceeded")])
    assert su.append_rows_safely(ws, [[1]]) == "ok" and ws.calls == 2


def test_server_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(su.time, "sleep", lambda s: None)
    ws = _WS([Exception("APIError: [503]: backend error")])
    with pytest.raises(Exception):
        su.append_rows_safely(ws, [[1]])
    assert ws.calls == 1
