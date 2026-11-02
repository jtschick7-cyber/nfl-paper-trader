#!/usr/bin/env bash
# Commits new raw Odds API snapshots (gzipped) to the `odds-data` branch so they outlive
# the Actions cache. The branch is an orphan holding only snapshots/, so main never grows.
# Needs `permissions: contents: write` on the calling job; uses the checkout's GITHUB_TOKEN.
set -euo pipefail

BRANCH="${ODDS_DATA_BRANCH:-odds-data}"
SNAP_DIR="${SNAP_DIR:-data/odds_cache/snapshots}"
REPO_ROOT="$(pwd)"
WORK="$(mktemp -d)"
trap 'git -C "$REPO_ROOT" worktree remove --force "$WORK" >/dev/null 2>&1 || true' EXIT

if [ ! -d "$SNAP_DIR" ] || [ -z "$(ls -A "$SNAP_DIR" 2>/dev/null)" ]; then
  echo "No odds snapshots to archive."
  exit 0
fi

checkout_branch() {
  git -C "$REPO_ROOT" worktree remove --force "$WORK" >/dev/null 2>&1 || true
  rm -rf "$WORK"
  if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git fetch --quiet --depth=1 origin "$BRANCH"
    git worktree add --quiet --detach "$WORK" FETCH_HEAD
  else
    git worktree add --quiet --detach "$WORK"
    git -C "$WORK" checkout --quiet --orphan "$BRANCH"
    git -C "$WORK" rm -rf --quiet . >/dev/null 2>&1 || true
    git -C "$WORK" clean -fdxq
  fi
}

# Retry: a parallel run (Tuesday capture vs closing loop) may push first; snapshot names
# are unique timestamps, so re-applying onto the new tip is always conflict-free.
for attempt in 1 2 3; do
  checkout_branch
  python -m src.archive_snapshots "$SNAP_DIR" "$WORK"
  git -C "$WORK" add -A snapshots
  if git -C "$WORK" diff --cached --quiet; then
    echo "odds-data branch already has every snapshot."
    exit 0
  fi
  git -C "$WORK" -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet -m "Archive odds snapshots ($(date -u +%Y-%m-%dT%H:%MZ))"
  if git -C "$WORK" push --quiet origin "HEAD:refs/heads/$BRANCH"; then
    echo "Pushed snapshots to $BRANCH."
    exit 0
  fi
  echo "Push to $BRANCH rejected (attempt $attempt); retrying on the new tip."
  sleep $((attempt * 5))
done
echo "Failed to push snapshots to $BRANCH." >&2
exit 1
