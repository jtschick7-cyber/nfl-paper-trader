#!/usr/bin/env bash
# Commits new player-prop snapshots (gzipped) to the `odds-data` branch under props/ (task P1).
# Copy of scripts/push_odds_snapshots.sh for data/props_cache. The closing loop pushes
# snapshots/ to the same branch at the same times, hence the same rebase-and-retry.
# Needs `permissions: contents: write` on the calling job; uses the checkout's GITHUB_TOKEN.
set -euo pipefail

BRANCH="${ODDS_DATA_BRANCH:-odds-data}"
SNAP_DIR="${SNAP_DIR:-data/props_cache}"
REPO_ROOT="$(pwd)"
WORK="$(mktemp -d)"
trap 'git -C "$REPO_ROOT" worktree remove --force "$WORK" >/dev/null 2>&1 || true' EXIT

if [ ! -d "$SNAP_DIR" ] || [ -z "$(ls -A "$SNAP_DIR" 2>/dev/null)" ]; then
  echo "No prop snapshots to archive."
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

# Retry: a parallel run (closing loop vs props capture) may push first; snapshot names
# are unique timestamps, so re-applying onto the new tip is always conflict-free.
for attempt in 1 2 3; do
  checkout_branch
  python -m src.props.archive_props "$SNAP_DIR" "$WORK"
  git -C "$WORK" add -A props
  if git -C "$WORK" diff --cached --quiet; then
    echo "odds-data branch already has every prop snapshot."
    exit 0
  fi
  git -C "$WORK" -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet -m "Archive prop snapshots ($(date -u +%Y-%m-%dT%H:%MZ))"
  if git -C "$WORK" push --quiet origin "HEAD:refs/heads/$BRANCH"; then
    echo "Pushed prop snapshots to $BRANCH."
    exit 0
  fi
  echo "Push to $BRANCH rejected (attempt $attempt); retrying on the new tip."
  sleep $((attempt * 5))
done
echo "Failed to push prop snapshots to $BRANCH." >&2
exit 1
