"""Identifiers for model artifacts, stamped on every shadow row (R2)."""

import hashlib


def artifact_hash(path, length=8):
    """First `length` hex characters of the file's sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def artifact_id(path, artifact, prefix):
    """e.g. 'v3_2012-2025_1a2b3c4d'; the span is left out when the artifact records no seasons."""
    seasons = None
    if isinstance(artifact, dict):
        seasons = artifact.get("trained_seasons") or artifact.get("trained_on_seasons")
    digest = artifact_hash(path)
    if seasons:
        years = [int(s) for s in seasons]
        return f"{prefix}_{min(years)}-{max(years)}_{digest}"
    return f"{prefix}_{digest}"
