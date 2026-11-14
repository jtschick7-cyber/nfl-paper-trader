import pytest
from src.sheet_utils import ensure_columns, verify_row_game_ids


class FakeWS:
    def __init__(self, headers, col_count, gids=None):
        self.headers = list(headers)
        self.col_count = col_count
        self.gids = gids or []
        self.added = 0
        self.writes = []

    def row_values(self, _):
        return list(self.headers)

    def col_values(self, _):
        return list(self.gids)

    def add_cols(self, n):
        self.added += n
        self.col_count += n

    def batch_update(self, chunk, value_input_option=None):
        self.writes.extend(chunk)


def test_ensure_columns_appends_only_missing_headers_and_grows_grid():
    ws = FakeWS(["game_id", "closing_line"], col_count=2)
    cols = ensure_columns(ws, ["closing_line", "closing_consensus_line"])
    assert cols["closing_consensus_line"] == 3
    assert ws.added == 1
    assert ws.writes == [{"range": "C1", "values": [["closing_consensus_line"]]}]

    ws = FakeWS(["game_id", "closing_consensus_line"], col_count=10)
    ensure_columns(ws, ["closing_consensus_line"])
    assert ws.writes == [] and ws.added == 0


def test_verify_row_game_ids_drops_rows_that_moved():
    ws = FakeWS([], 0, gids=["game_id", "A", "B"])
    assert verify_row_game_ids(ws, {2: "A", 3: "X", 9: "Z"}, 1) == {2}
    assert verify_row_game_ids(ws, {}, 1) == set()


def test_retry_api_call_retries_connection_reset(monkeypatch):
    import requests
    from src import sheet_utils

    monkeypatch.setattr(sheet_utils.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("Connection reset by peer")
        return "ok"

    assert sheet_utils.retry_api_call(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_api_call_gives_up_after_max_attempts(monkeypatch):
    import pytest
    import requests
    from src import sheet_utils

    monkeypatch.setattr(sheet_utils.time, "sleep", lambda _s: None)

    def always():
        raise requests.exceptions.ConnectionError("reset")

    with pytest.raises(requests.exceptions.ConnectionError):
        sheet_utils.retry_api_call(always, max_attempts=3)


class HeaderWS:
    """Minimal worksheet: a header row, a grid width, and the A1 writes made to it."""

    def __init__(self, headers, col_count=26):
        self.headers = list(headers)
        self.col_count = col_count
        self.writes = []

    def row_values(self, _):
        out = list(self.headers)
        while out and out[-1] == "":
            out.pop()
        return out

    def add_cols(self, n):
        self.col_count += n

    def update(self, values, range_name):
        from src.sheet_utils import col_to_letter

        self.writes.append((range_name, values))
        start = 0
        while range_name[start].isalpha():
            start += 1
        letters = range_name[:start]
        col = 0
        for ch in letters:
            col = col * 26 + (ord(ch) - 64)
        assert col_to_letter(col) == letters
        row = values[0]
        assert col - 1 + len(row) <= self.col_count, "write past the grid"
        while len(self.headers) < col - 1 + len(row):
            self.headers.append("")
        for i, v in enumerate(row):
            self.headers[col - 1 + i] = v


def test_ensure_tab_headers_writes_empty_sheet():
    from src.sheet_utils import ensure_tab_headers

    ws = HeaderWS([])
    ensure_tab_headers(ws, ["a", "b", "c"])
    assert ws.headers == ["a", "b", "c"]


def test_ensure_tab_headers_identical_row_is_untouched():
    from src.sheet_utils import ensure_tab_headers

    ws = HeaderWS(["a", "b", "c"])
    ensure_tab_headers(ws, ["a", "b", "c"])
    assert ws.writes == []


def test_ensure_tab_headers_prefix_appends_and_grows_grid():
    from src.sheet_utils import ensure_tab_headers

    ws = HeaderWS(["a", "b"], col_count=2)
    ensure_tab_headers(ws, ["a", "b", "artifact_id", "decision_regime"])
    assert ws.headers == ["a", "b", "artifact_id", "decision_regime"]
    assert ws.col_count >= 4
    # existing cells are never rewritten
    assert all(rng != "A1" for rng, _ in ws.writes)
    assert "" not in ws.headers and len(set(ws.headers)) == len(ws.headers)


def test_ensure_tab_headers_conflict_raises_with_both_rows():
    from src.sheet_utils import ensure_tab_headers

    ws = HeaderWS(["a", "x"])
    with pytest.raises(ValueError) as exc:
        ensure_tab_headers(ws, ["a", "b", "c"])
    assert "x" in str(exc.value) and "b" in str(exc.value)
    assert ws.writes == []
    # a sheet with more columns than the code expects is also a conflict
    with pytest.raises(ValueError):
        ensure_tab_headers(HeaderWS(["a", "b", "c", "d"]), ["a", "b", "c"])


def test_artifact_id_uses_hash_and_training_span(tmp_path):
    import hashlib

    from src.artifacts import artifact_id

    p = tmp_path / "m.pkl"
    p.write_bytes(b"model-bytes")
    h = hashlib.sha256(b"model-bytes").hexdigest()[:8]
    assert artifact_id(str(p), {"trained_seasons": [2014, 2012, 2025]}, "v3") == f"v3_2012-2025_{h}"
    assert artifact_id(str(p), {"trained_on_seasons": [1999, 2024]}, "v2") == f"v2_1999-2024_{h}"
    assert artifact_id(str(p), {}, "v4") == f"v4_{h}"
