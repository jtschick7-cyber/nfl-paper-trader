"""Reader for the gzipped Odds API snapshots on the `odds-data` branch (audit Phase 3).

Layout (src/archive_snapshots.py): snapshots/YYYY/MM/odds_<YYYYMMDDTHHMMSSZ>.json.gz, each the
raw /odds response (a list of events with every book's spreads and totals).

The source is either a local checkout of the branch (a directory holding `snapshots/`) or a
git ref such as `origin/odds-data`, read with `git show` so nothing is checked out.
Read-only; no network.
"""

import gzip
import json
import os
import re
import subprocess
from datetime import datetime, timezone

DEFAULT_REF = "origin/odds-data"
_NAME = re.compile(r"odds_(\d{8}T\d{6}Z)\.json\.gz$")


def snapshot_time(name):
    """UTC datetime from a snapshot file name, or None."""
    m = _NAME.search(os.path.basename(name))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _is_dir_source(source):
    return os.path.isdir(source)


def list_snapshots(source=DEFAULT_REF, repo="."):
    """[(utc datetime, path)] sorted by time. `path` is a file path for a directory source,
    else the path inside the git ref."""
    if _is_dir_source(source):
        paths = []
        for root, _, files in os.walk(os.path.join(source, "snapshots")):
            paths += [os.path.join(root, f) for f in files]
    else:
        out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", source, "snapshots/"],
                             check=True, capture_output=True, text=True).stdout
        paths = out.split()
    rows = [(snapshot_time(p), p) for p in paths]
    return sorted((t, p) for t, p in rows if t is not None)


def read_snapshot(path, source=DEFAULT_REF, repo="."):
    """The event list of one snapshot ([] if it is not a list)."""
    if _is_dir_source(source):
        with gzip.open(path, "rb") as f:
            raw = f.read()
    else:
        raw = gzip.decompress(subprocess.run(["git", "-C", repo, "show", f"{source}:{path}"],
                                             check=True, capture_output=True).stdout)
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def iter_archive(source=DEFAULT_REF, repo="."):
    """Yields (utc datetime, events) for every snapshot in time order."""
    for t, p in list_snapshots(source, repo):
        yield t, read_snapshot(p, source, repo)
