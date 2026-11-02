import gzip
import json

from src.archive_snapshots import archive_path, archive_snapshots


def test_archive_snapshots_gzips_once_into_year_month_tree(tmp_path):
    src = tmp_path / "snapshots"
    src.mkdir()
    payload = [{"id": "g1", "home_team": "Kansas City Chiefs"}]
    (src / "odds_20261101T170000Z.json").write_text(json.dumps(payload))
    (src / "odds_latest.json").write_text("[]")  # not a timestamped snapshot
    dest = tmp_path / "archive"

    written = archive_snapshots(str(src), str(dest))

    target = dest / "snapshots" / "2026" / "11" / "odds_20261101T170000Z.json.gz"
    assert written == [str(target)]
    with gzip.open(target, "rt") as f:
        assert json.load(f) == payload

    # Re-running is a no-op, and identical bytes keep git from seeing a change
    before = target.read_bytes()
    assert archive_snapshots(str(src), str(dest)) == []
    assert target.read_bytes() == before


def test_archive_path_rejects_untimestamped_names(tmp_path):
    assert archive_path(str(tmp_path), "odds_latest.json") is None
    assert archive_path(str(tmp_path), "odds_raw_2026-11-01.json") is None
