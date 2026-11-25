# Mid-week game-odds snapshots

Approved by the owner on 2026-11-24, answering the owner question in `notes/implementation-plan/P2-topdown-v2.md`. Built on `claude/midweek-snapshots`.

## What it does

- `midweek_snapshots.yml` runs `python -m src.midweek_snapshot`, then `scripts/push_odds_snapshots.sh` with `SNAP_DIR=data/midweek_cache/snapshots`.
- One full-slate `/odds` call on Wed, Thu and Sat from 17:30 UTC. The request is the existing game-odds request (`regions=us`, `markets=spreads,totals`, `oddsFormat=american`, `dateFormat=iso`, no `bookmakers` filter). A test compares it with `fetch_odds_api`'s params. Cost about 2 credits, so about 6 a week.
- The file is `odds_<YYYYMMDDTHHMMSSZ>.json` with the raw response, archived as `snapshots/YYYY/MM/odds_<slug>.json.gz` on `odds-data`. `src.topdown_v2` reads it unchanged.
- No existing workflow, request or reader was edited. No Sheet access; only `ODDS_API_KEY`.

## Scheduling and self-gating

- The Worker's `30 13,17,21 * * MON,WED,THU,FRI,SAT` entry dispatches it third, after `closing_line_monitor.yml` and `props_capture.yml`. Still 4 Cloudflare crons. Backup GitHub cron `47 17 * * 3,4,6`.
- The job calls nothing unless it is Wed, Thu or Sat (UTC) and at or after 17:25 UTC (the 17:30 wake with 5 min of tolerance, or a later wake if that one failed). The 13:30 wake and Mon/Fri wakes exit at once.
- At most one snapshot a day: before any paid call it fetches `origin/odds-data` and skips if any snapshot since 17:25 UTC today is archived (or in its local dir). A closing-loop snapshot in that window counts too, since it covers the same slate. If the ledger cannot be read, nothing is called (exit 1).
- No game due, no call (added 2026-11-25): the free `/events` list must hold a game kicking off within 168 h, so the off-season and the week before the Super Bowl spend 0 credits. A non-list `/events` body exits 1 without a paid call.
- Known gap: if the paid call succeeds but all three push attempts fail, the snapshot is lost and a later wake that day may pay again (at most 3 wakes a day).

## Credit guards

- At most 2 paid calls per run; the second only retries a network error, 429 or 5xx. A 4xx or a non-list 200 is not retried.
- Quota floor 80 (same as `src/props/capture_props.py` `QUOTA_FLOOR`): the free `/events` call reports the quota, and a call is made only if `remaining − 2 ≥ 80`. While the quota is unknown, at most one call. Env overrides `MIDWEEK_MAX_CALLS`, `MIDWEEK_QUOTA_FLOOR`.
- Weekly burn before this: about 36 credits (spread pipeline) + about 30 (props). With this, about 72 a week, about 310 of the 500 a month.

## Phase 3 replay sees them too

`src.topdown_replay` (Phase 3) also reads every file under `snapshots/`, so its weekly re-run will include these snapshots from 12-02 on. No Phase 3 rule was changed; more snapshots only give it more sightings and a denser "next snapshot" for persistence.
