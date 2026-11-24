# P1. Prospective `player_receptions` prop archive

**Source:** owner's roadmap brief of 2026-11-24 (Step 3), with the changes agreed in review the same day. **Effort:** M. **Branch:** `claude/p1-props-capture`. **Depends on:** nothing.

Capture only. No pricing, grading, Sheet tab or decision. Nothing in the spread pipeline changes.

## Deadline

Each week without capture loses that week's prop open/close pairs for good.

- Merged **and** the Worker redeployed by **Fri 2026-11-27 17:30 UTC**: week 12's Friday open, Sunday and Monday closes are captured.
- Otherwise by **Thu 2026-12-03 13:30 UTC**, for week 13 (TNF open is due about 21:15 UTC that day).

## Goal

For every NFL game from now on, archive two multi-book `player_receptions` snapshots on `odds-data` under `props/`:

- **open**: after Friday's injury reports (Thursday afternoon for TNF);
- **close**: about 25 minutes before kickoff, after inactives (announced at T-90).

Later tasks (prop pricer, top-down prop scanner, bottom-up receptions model) read only this archive.

## Decisions

Agreed 2026-11-24. Do not reopen without asking.

1. **Separate workflow.** `props_capture.yml`, concurrency group `props-capture`. `closing_loop.py`, `closing_line_monitor.yml`, `tuesday_opening.yml`, `src/config.py` and `src/odds_utils.py` are not edited.
2. **No new Cloudflare crons.** The Worker has 4 crons; the free plan allows 5 per Worker (`external_trigger/README.md`). `SCHEDULE` already maps a cron to a list of workflows, so `props_capture.yml` is added to the two existing closing-loop crons. The props job paces itself to its targets, as `closing_loop.py` does. Those crons already cover London, Thanksgiving, Christmas (Fri 12-25) and late-season Saturdays.
3. **Explicit `bookmakers=` list instead of `regions=us`.** The Odds API bills each group of up to 10 bookmakers as one region. A 10-book list costs the same 1 credit per event and lets Pinnacle in. `lowvig` is left out: it is BetOnline's reduced-vig brand on the same feed as `betonlineag`. The game-odds request is unchanged, so the pre-registered Phase 3 replay is unaffected.
4. **Game list from the free `/events` endpoint**, not the Sheet. The props job needs no Sheet secrets and no `decision_window` logic.
5. **One archive file per (event, phase).** The file listing then serves as the ledger that stops a second run from spending credits on a game already captured.

## Design

### Modules (new package `src/props/`)

- `src/props/__init__.py`
- `src/props/capture_props.py`: the self-pacing loop below. CLI: `python -m src.props.capture_props [--probe]`.
- `src/props/archive_props.py`: gzips `data/props_cache/props_*.json` into `<dest>/props/YYYY/MM/<name>.gz`, skipping files already there. Same rules as `src/archive_snapshots.py`, including `mtime=0` for deterministic bytes.
- `src/props/load_prop_archive.py`: `list_prop_snapshots(source="origin/odds-data")` and `read_prop_snapshot(path, source)`, mirroring `src/load_odds_archive.py` but reading only `props/`. Read-only, no network.
- `scripts/push_prop_snapshots.sh`: copy of `scripts/push_odds_snapshots.sh` with `SNAP_DIR=data/props_cache`, `python -m src.props.archive_props` and `git add -A props`. Keep the three-attempt rebase-and-retry: the closing loop pushes to the same branch at the same times. Leave the original script alone.

### Constants (top of `capture_props.py`, env-overridable)

```python
MARKET = "player_receptions"
BOOKMAKERS = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us",
              "betrivers", "betonlineag", "bovada", "betus", "mybookieag"]   # <= 10, see Probe
CLOSE_LEAD_MIN = 25          # target: kickoff - 25 min
CLOSE_WINDOW_MIN = (40, 5)   # close is due from kickoff-40 to kickoff-5
OPEN_LATEST_MIN = 120        # an open is no longer due after kickoff-2h
MAX_CREDITS_PER_RUN = 20     # PROPS_MAX_CREDITS
QUOTA_FLOOR = 80             # PROPS_QUOTA_FLOOR: credits always left for the spread pipeline
MAX_RUNTIME_MINUTES = 320    # same as closing_loop.py; workflow timeout 345
```

### Targets

For each event from `GET /v4/sports/americanfootball_nfl/events` (0 credits) with kickoff `k` in the next 6 days:

- `open_target(k) = min(k - 4h, F)`, where `F` is the first Friday 21:00 UTC at or after `k - 4 days`.
- `close_target(k) = k - 25 min`.

Expected values; each is a test case:

| Game (kickoff UTC) | open target | why |
|---|---|---|
| Sun 18:00 (1pm ET) | Fri 21:00 | after Friday injury reports |
| Sun 14:30 (London) | Fri 21:00 | |
| Mon 01:20 (SNF) | Fri 21:00 | |
| Tue 01:15 (MNF) | Fri 21:00 | |
| Sat 18:00 / 21:30 | Fri 21:00 | |
| Fri 01:15 (TNF) | Thu 21:15 | k − 4h; that Friday 21:00 is after kickoff |
| Thu 17:30 (Thanksgiving) | Thu 13:30 | k − 4h |
| Fri 18:00 (Christmas) | Fri 14:00 | k − 4h |

**Due rules.** Late runs catch up; nothing is ever captured twice.

- An open is due when `now >= open_target`, `now <= k - 120 min`, and `("open", event_id)` is not in the ledger.
- A close is due when `k - 40 min <= now <= k - 5 min`, and `("close", event_id)` is not in the ledger.

**Loop.** Each iteration:

1. Fetch the events list and refresh the ledger: `git fetch --depth=1 origin odds-data`, then `list_prop_snapshots`, plus local files in `data/props_cache/`.
2. Capture every due item.
3. Push, then sleep until the earliest future target within the runtime deadline. If there is none, exit 0.

If the events call fails, log it, retry at the next wake, and exit 1 at the end of the run so it shows red.

### The capture call

`GET /v4/sports/americanfootball_nfl/events/{event_id}/odds` with `bookmakers=<comma list>`, `markets=player_receptions`, `oddsFormat=american`, `dateFormat=iso`. **No `regions` parameter.**

Credit guards, checked before every call:

- **Per-run cap.** Spend at most `MAX_CREDITS_PER_RUN` per run, counting 1 per call, or the actual `x-requests-last` once known. Over the cap, capture the earliest kickoffs first and log the rest as skipped.
- **Quota floor.** Never call when the last seen `x-requests-remaining` minus this batch's cost would drop below `QUOTA_FLOOR`.
- **Unknown quota.** If `x-requests-remaining` has not been seen yet (the free events call may not return it), allow at most one capture call. Read the header from that call, then apply the floor.

### Snapshot file

`data/props_cache/props_<phase>_<event_id>_<YYYYMMDDTHHMMSSZ>.json`, archived as `props/YYYY/MM/<same name>.gz`. Content:

```json
{"schema": 1, "phase": "open|close|probe", "market": "player_receptions",
 "captured_at": "2026-11-27T21:00:41Z", "target_at": "2026-11-27T21:00:00Z",
 "kickoff": "2026-11-29T18:00:00Z", "minutes_to_kickoff": 2579.3,
 "bookmakers_requested": ["pinnacle", "..."],
 "quota": {"used": 140, "remaining": 360, "last": 1},
 "response": {"...": "raw /events/{id}/odds JSON, unmodified"}}
```

Write the file even when no book has posted the market yet (empty `bookmakers`), so the attempt is not paid for twice.

### Probe (first live run)

`--probe` captures the earliest upcoming event once, as phase `probe` (later tasks ignore it), costing 1 credit.

- Run it once by manual `workflow_dispatch` (input `probe: true`) after merge.
- Read the file. Record in the PR which of the 10 books returned `player_receptions` and the `x-requests-last` cost.
- Drop any book that returned nothing and add a replacement (candidates: `espnbet`, `fanatics`, `hardrockbet`, `ballybet`), keeping 10 or fewer. If Pinnacle returns no props, say so; the later scanner then has no sharp prop anchor, which affects how much it can show.

### Workflow `.github/workflows/props_capture.yml`

- **Triggers:** `workflow_dispatch` (boolean input `probe`), plus a sparse GitHub `schedule:` backup for opens only (e.g. `"47 16 * * 4,5"`). Every run is idempotent, so a duplicate does no harm.
- **Permissions and concurrency:** top level `permissions: contents: read`; group `props-capture` with `cancel-in-progress: false`.
- **Job:** `capture` with `permissions: contents: write`, `timeout-minutes: 345` and env `ODDS_API_KEY` only (no Sheet secrets).
- **Steps:**
  - checkout with `fetch-depth: 0` (or an explicit fetch of `odds-data`);
  - Python 3.12 with pip cache; `pip install -r requirements.txt`;
  - `python -m src.props.capture_props`;
  - `bash scripts/push_prop_snapshots.sh` with `if: always()` (it re-pushes anything the in-loop pushes missed).

### Worker

In `external_trigger/src/index.js` `SCHEDULE`, append `"props_capture.yml"` **after** `"closing_line_monitor.yml"` in:

- `"30 13,17,21 * * MON,THU,FRI,SAT"`
- `"30 10,15,20 * * SUN"`

`runCron` stops at the first failed dispatch, and this order means a props failure can never block the closing loop. `wrangler.toml` does not change. After merge the owner redeploys from their machine with `cd external_trigger && npm run deploy`.

## Tests (write first)

`tests/test_props_capture.py`:

1. `open_target` for all eight rows of the table.
2. Due rules at the edges: open at `k - 121m` (due) and `k - 119m` (not due); close at `k - 41m`, `k - 40m`, `k - 5m` and `k - 4m`.
3. Next wake: the earliest future target within the deadline; `None` when none fits.
4. Ledger: an event with an archived or local `open` file is skipped for open but still due for close.
5. Credit guards: with 25 due events and a cap of 20, exactly 20 calls go out, earliest kickoffs first. The quota floor blocks a call. With unknown quota, one call is allowed and the floor then applies.
6. Request built with mocked `requests`: no `regions` key; `bookmakers` has 10 or fewer comma-separated keys; `markets == "player_receptions"`.
7. File name and JSON schema; an empty response is still written.

`tests/test_archive_props.py`:

1. Name-to-path mapping into `props/YYYY/MM/`. `odds_*.json` and malformed names are ignored. Gzip output is byte-identical on a re-run.
2. **Non-interference:** a directory source holding both `snapshots/` and `props/`. `load_odds_archive.list_snapshots` returns only the spread snapshots, and `list_prop_snapshots` returns only props.

`tests/test_workflows.py`: add a test for `props_capture.yml`. It checks the group and cancel setting, top-level `contents: read`, job-level `contents: write`, no `GSHEET`/`SPREADSHEET` secret referenced, and `workflow_dispatch` present. The existing tests (Python 3.12, no cron at minute 0) must pass for the new file.

`external_trigger/test/index.test.js`: both closing crons dispatch `closing_line_monitor.yml` then `props_capture.yml`, and the cron count is unchanged.

Before and after: `.venv/bin/python -m pytest tests/ -q` and `node --test external_trigger/test/*.test.js` are green (1630 and 8 at `aa677fb`).

## Budget

| Pipeline | Credits / week | Credits / month (about 4.3 weeks) |
|---|---|---|
| Existing spreads and totals (measured 432 → 396, 11-17 to 11-24) | ~36 | ~155 |
| Props open (1 per game) | ~14–16 | ~65 |
| Props close (1 per game) | ~14–16 | ~65 |
| **Total** | **~66–68** | **~285 / 500** |

A second prop market would double the props lines. That is out of scope.

## Live check (after the first Friday and Sunday)

- Every Sat/Sun/Mon game has one `open` file, with `captured_at` within 15 min of its target. Every played game has one `close` file with `minutes_to_kickoff` between 5 and 40.
- Credits spent by the props runs equal the number of files (`x-requests-last` = 1 each), and no run spent more than 20.
- `git ls-tree origin/odds-data snapshots/` gains only the closing loop's files. `python -m src.topdown_replay` output for the existing snapshots is unchanged.
- The closing loop's own checks still pass (closes T-10 to T-17 min in `Master_Bets`).

## Out of scope

Name normalisation, Negative Binomial pricing, the top-down prop scanner, grading, any Sheet tab, other prop markets, and adding Pinnacle to the game-odds request (that would change the Phase 3 replay's book pool and needs its own decision).

## Owner answers (2026-11-24)

| Question | Answer |
|---|---|
| Spend 1 credit on the probe run after merge? | Yes |
| Quota floor of 80 credits kept for the spread pipeline? | Yes, 80 |
| Can you redeploy the Worker by Fri 11-27 17:30 UTC? If not, week 12 is only covered by the GitHub backup crons, which are often hours late. | Owner redeploys from their machine once the build PR is merged; steps in `external_trigger/README.md` |
