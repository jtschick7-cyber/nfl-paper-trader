import gzip
import json
import os

from src.archive_snapshots import archive_snapshots
from src.load_odds_archive import list_snapshots
from src.props.archive_props import archive_path, archive_props
from src.props.load_prop_archive import list_prop_snapshots, parse_prop_name, read_prop_snapshot


def test_archive_path_maps_into_props_year_month(tmp_path):
    dest = str(tmp_path)
    assert archive_path(dest, "data/props_cache/props_close_abc123_20261129T173512Z.json") == \
        os.path.join(dest, "props", "2026", "11", "props_close_abc123_20261129T173512Z.json.gz")
    assert archive_path(dest, "odds_20261129T173512Z.json") is None
    assert archive_path(dest, "props_latest.json") is None
    assert archive_path(dest, "props_open_abc_2026.json") is None
    assert archive_path(dest, "props_weird_abc_20261129T173512Z.json") is None
    assert archive_path(dest, "props_open_abc_20261129T173512Z.json.gz") is None


def test_archive_props_gzips_once_and_bytes_are_stable(tmp_path):
    src = tmp_path / "props_cache"
    src.mkdir()
    doc = {"schema": 1, "phase": "open", "response": {"bookmakers": []}}
    (src / "props_open_abc123_20261127T210041Z.json").write_text(json.dumps(doc))
    (src / "odds_20261127T210041Z.json").write_text("[]")
    (src / "props_bad.json").write_text("{}")
    dest = tmp_path / "odds-data"

    written = archive_props(str(src), str(dest))
    target = dest / "props" / "2026" / "11" / "props_open_abc123_20261127T210041Z.json.gz"
    assert written == [str(target)]
    with gzip.open(target, "rt") as f:
        assert json.load(f) == doc

    before = target.read_bytes()
    assert archive_props(str(src), str(dest)) == []
    assert target.read_bytes() == before
    # A fresh archive of the same file gives identical bytes (mtime=0).
    dest2 = tmp_path / "again"
    archive_props(str(src), str(dest2))
    assert (dest2 / "props" / "2026" / "11" / target.name).read_bytes() == before


def test_parse_prop_name():
    p = parse_prop_name("props/2026/11/props_close_abc123_20261129T173512Z.json.gz")
    assert p["phase"] == "close" and p["event_id"] == "abc123"
    assert p["captured_at"].isoformat() == "2026-11-29T17:35:12+00:00"
    assert parse_prop_name("odds_20261129T173512Z.json.gz") is None


def test_spread_and_prop_readers_do_not_see_each_other(tmp_path):
    root = tmp_path / "odds-data"
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    (snaps / "odds_20261127T210000Z.json").write_text(json.dumps([{"id": "g1"}]))
    archive_snapshots(str(snaps), str(root))
    props = tmp_path / "props_cache"
    props.mkdir()
    (props / "props_open_g1_20261127T210041Z.json").write_text(json.dumps({"phase": "open"}))
    (props / "props_close_g1_20261129T173512Z.json").write_text(json.dumps({"phase": "close"}))
    archive_props(str(props), str(root))

    spread = list_snapshots(str(root))
    assert [os.path.basename(p) for _, p in spread] == ["odds_20261127T210000Z.json.gz"]

    prop = list_prop_snapshots(str(root))
    assert [os.path.basename(p) for _, p in prop] == [
        "props_open_g1_20261127T210041Z.json.gz", "props_close_g1_20261129T173512Z.json.gz"]
    assert read_prop_snapshot(prop[1][1], str(root)) == {"phase": "close"}
