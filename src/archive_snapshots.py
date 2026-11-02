"""
Copies raw Odds API snapshots into a durable archive tree as gzip files.

The Actions cache that holds data/odds_cache/snapshots/ is evicted after 7 idle days
and pruned to 72h by _save_timestamped_snapshot, so scripts/push_odds_snapshots.sh
runs this against a checkout of the `odds-data` branch and commits the result there.

Layout: <dest>/snapshots/YYYY/MM/odds_<YYYYMMDDTHHMMSSZ>.json.gz (existing files are kept).
"""

import glob
import gzip
import os
import shutil
import sys

from src.odds_utils import SNAPSHOTS_DIR


def archive_path(dest_root: str, snapshot_path: str):
    """Archive location for a snapshot file, or None if its name has no timestamp slug."""
    name = os.path.basename(snapshot_path)
    if not (name.startswith("odds_") and name.endswith("Z.json")) or len(name) < len("odds_YYYYMM"):
        return None
    year, month = name[5:9], name[9:11]
    if not (year.isdigit() and month.isdigit()):
        return None
    return os.path.join(dest_root, "snapshots", year, month, name + ".gz")


def archive_snapshots(src_dir: str, dest_root: str):
    """Gzips each snapshot not yet archived; returns the list of archive paths written."""
    written = []
    for src in sorted(glob.glob(os.path.join(src_dir, "odds_*.json"))):
        dest = archive_path(dest_root, src)
        if dest is None or os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # mtime=0 keeps the gzip bytes deterministic, so re-runs never produce a diff.
        with open(src, "rb") as f_in, gzip.GzipFile(dest, "wb", mtime=0) as f_out:
            shutil.copyfileobj(f_in, f_out)
        written.append(dest)
    return written


def main(argv):
    src_dir = argv[1] if len(argv) > 2 else SNAPSHOTS_DIR
    dest_root = argv[-1]
    written = archive_snapshots(src_dir, dest_root)
    print(f"Archived {len(written)} new odds snapshot(s) into {dest_root}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.archive_snapshots [SNAPSHOT_DIR] DEST_ROOT")
    main(sys.argv)
