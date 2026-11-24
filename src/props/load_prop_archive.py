"""Reader for the gzipped player-prop snapshots on the `odds-data` branch (task P1).

Layout (src/props/archive_props.py): props/YYYY/MM/props_<phase>_<event_id>_<YYYYMMDDTHHMMSSZ>.json.gz,
each a JSON object with capture metadata and the raw /events/{id}/odds response.

Mirrors src/load_odds_archive.py but reads only props/. The source is a directory holding
props/ or a git ref such as `origin/odds-data`, read with `git show`. Read-only; no network.
"""

import gzip
import json
import os
import re
import subprocess
from datetime import datetime, timezone

DEFAULT_REF = "origin/odds-data"
PHASES = ("open", "close", "probe")
_NAME = re.compile(r"(?:^|/)props_(open|close|probe)_([A-Za-z0-9-]+)_(\d{8}T\d{6}Z)\.json(?:\.gz)?$")


def parse_prop_name(name):
    """{"phase", "event_id", "captured_at"} from a prop snapshot file name (raw or .gz), or None."""
    m = _NAME.search(name.replace(os.sep, "/"))
    if not m:
        return None
    t = datetime.strptime(m.group(3), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return {"phase": m.group(1), "event_id": m.group(2), "captured_at": t}


def _is_dir_source(source):
    return os.path.isdir(source)


def list_prop_snapshots(source=DEFAULT_REF, repo="."):
    """[(utc datetime, path)] of archived prop snapshots, sorted by capture time. `path` is a file
    path for a directory source, else the path inside the git ref."""
    if _is_dir_source(source):
        paths = []
        for root, _, files in os.walk(os.path.join(source, "props")):
            paths += [os.path.join(root, f) for f in files]
    else:
        out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", source, "props/"],
                             check=True, capture_output=True, text=True).stdout
        paths = out.split()
    rows = []
    for p in paths:
        parsed = parse_prop_name(p) if p.endswith(".json.gz") else None
        if parsed:
            rows.append((parsed["captured_at"], p))
    return sorted(rows)


def read_prop_snapshot(path, source=DEFAULT_REF, repo="."):
    """The JSON object of one prop snapshot."""
    if _is_dir_source(source):
        with gzip.open(path, "rb") as f:
            raw = f.read()
    else:
        raw = gzip.decompress(subprocess.run(["git", "-C", repo, "show", f"{source}:{path}"],
                                             check=True, capture_output=True).stdout)
    return json.loads(raw)
