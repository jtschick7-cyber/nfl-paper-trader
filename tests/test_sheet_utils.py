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
