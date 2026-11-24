"""Copies player-prop snapshots into the `odds-data` archive tree as gzip files (task P1).

Same rules as src/archive_snapshots.py: existing files are kept and gzip mtime=0 keeps the
bytes deterministic. scripts/push_prop_snapshots.sh runs this against a checkout of the branch.

Layout: <dest>/props/YYYY/MM/props_<phase>_<event_id>_<YYYYMMDDTHHMMSSZ>.json.gz
"""

import glob
import gzip
import os
import shutil
import sys

from src.props import PROPS_CACHE_DIR
from src.props.load_prop_archive import parse_prop_name


def archive_path(dest_root: str, snapshot_path: str):
    """Archive location for a raw prop snapshot, or None if its name is not a valid one."""
    name = os.path.basename(snapshot_path)
    parsed = parse_prop_name(name) if name.endswith(".json") else None
    if parsed is None:
        return None
    t = parsed["captured_at"]
    return os.path.join(dest_root, "props", f"{t.year:04d}", f"{t.month:02d}", name + ".gz")


def archive_props(src_dir: str, dest_root: str):
    """Gzips each prop snapshot not yet archived; returns the list of archive paths written."""
    written = []
    for src in sorted(glob.glob(os.path.join(src_dir, "props_*.json"))):
        dest = archive_path(dest_root, src)
        if dest is None or os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(src, "rb") as f_in, gzip.GzipFile(dest, "wb", mtime=0) as f_out:
            shutil.copyfileobj(f_in, f_out)
        written.append(dest)
    return written


def main(argv):
    src_dir = argv[1] if len(argv) > 2 else PROPS_CACHE_DIR
    dest_root = argv[-1]
    written = archive_props(src_dir, dest_root)
    print(f"Archived {len(written)} new prop snapshot(s) into {dest_root}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.props.archive_props [PROPS_DIR] DEST_ROOT")
    main(sys.argv)
