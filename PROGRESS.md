# Progress

## Current state

- **Branch:** `claude/midweek-snapshots` (PR #33 open, waits for owner merge). PRs #14–#32 are merged: P1 (#26), P2 (#27), the P4 spec (#29), P3 (#30), the PROGRESS refresh and P5 spec (#31), and P4 (#32).
- **Last task:** P4 prop scanner built (`notes/p4-prop-scanner.md`). First run: the archive held only P1's probe (excluded), so the evaluation is empty and the gate is pending (0 of 150 offers, 0 of 30 games).
- **P1 is live.** The Worker was redeployed twice; its crons now include `30 13,17,21 * * MON,WED,THU,FRI,SAT` (closing loop + props capture). Probe run 45488926007 cost 1 credit and found `player_receptions` at Pinnacle, DraftKings, FanDuel, BetOnline and Bovada (15 players), BetMGM (12) and BetRivers (12, Over only); none at `williamhill_us`, `betus`, `mybookieag`.
- **Mid-week snapshots:** `claude/midweek-snapshots` (PR open). After merge the owner redeploys the Worker (`cd external_trigger && npm run deploy`), ideally before Wed 2026-12-02 13:30 UTC.
- **Next task:** P5a (Stage A of `notes/implementation-plan/P5-receptions-model.md`, rules frozen; branch `claude/p5a-receptions-stage-a`). Also the P1 live check after the first Friday and Sunday (P1 spec "Live check").
- **Owner decisions (2026-11-24):** (a) yes: after the Wed 11-25 GB @ LA captures are checked, replace `williamhill_us` with `espnbet` unless it posted props, and `betus`/`mybookieag` with `fanatics`/`hardrockbet`, before Fri 11-27 21:00 UTC; (b) yes: add ~3 mid-week game-odds snapshots a week for P2's gate (~6 credits a week); (c) no: no historical prop odds are bought, so P5's market test (Stage B) uses only the 2026 archive.
- **Weekly re-runs:** `src.topdown_replay` (pre-registered Phase 3 test; 34 of 100 flagged offers, 17 of 40 played games on 11-24), `src.topdown_v2` (frozen P2 gate on the evaluation sample from 2026-11-25; 0 of 80 offers, 0 of 35 games on 11-24) and `src.props.topdown_props` after `git fetch origin odds-data` (frozen P4 gate; 0 of 150 offers, 0 of 30 games on 11-24). Descriptive only, never stop early for a pass. Phase 4 at the off-season. No retrain queued. Owner decided (2026-11-21): v4 keeps logging as a shadow until the season ends, then retires unless the R3 raw-lean test clearly favours it; off-season v3 retrain with `split_neutral_teams`.
- **Calendar:** Sat 11-28 check of the Thanksgiving/Black Friday props captures and the Friday opens; Tue 12-01 regular live check after the 15:17 UTC run.
- **Still to see live:** the first graded in-week rows (week 12 settles 11-26 to 12-01), so the in-week summary and in-week RAW LEAN CLV fill; a SUPPRESS_DANGER row (needs |raw lean| >= 7 pts and an edge above the threshold); `Model v3 [<artifact_id>]` lines (appear only when the in-week rows hold two artifacts, so after a retrain).

## Task queue

| Task | Spec | Status |
|---|---|---|
| R1 decision timing | `notes/implementation-plan/R1-decision-timing.md` | done (PR #14), live check passed 11-24 |
| R2 scorecard hygiene | `notes/implementation-plan/R2-scorecard-hygiene.md` | done (PR #15), live check passed 11-24 |
| R3 raw-lean tracking | `notes/implementation-plan/R3-raw-lean-tracking.md` | done (PR #16), backfilled 11-14, live check passed 11-24 |
| R4 production backtest | `notes/implementation-plan/R4-production-backtest.md` | done (PR #17) |
| R5 price-consistent anchor | `notes/implementation-plan/R5-price-consistent-anchor.md` | done (PR #18), live check passed 11-24 |
| Audit Phase 1 safety fixes | owner's audit, Phase 1 row | done (PR #19), live check passed 11-24 |
| Audit Phase 2 feature study | `notes/phase2-feature-study.md` | done (PR #20) |
| Audit P0-7 v4 QB shock + v4 study | `notes/p0-7-v4-qb-study.md` | done (PR #21) |
| Audit Phase 3 market consensus | `notes/phase3-market-consensus.md` | offline replay done (PR #22), live tab held; weekly re-run |
| P1 prop archive capture | `notes/implementation-plan/P1-props-capture.md` | built and merged 11-24 (PR #26); Worker redeployed, probe done (run 45488926007); live check after the first Friday and Sunday |
| P2 top-down v2 (spreads + totals) | `notes/implementation-plan/P2-topdown-v2.md` | built 11-24 (`notes/p2-topdown-v2.md`); weekly re-run, gate pending |
| P3 totals kill test | `notes/implementation-plan/P3-totals-kill-test.md` | done 11-24: KILL (`notes/p3-totals-kill-test.md`); bottom-up totals closed |
| P4 prop scanner | `notes/implementation-plan/P4-prop-scanner.md` | built 11-24 (`notes/p4-prop-scanner.md`); weekly re-run, gate pending |
| Mid-week game-odds snapshots | owner answer to the P2 owner question (11-24); `notes/midweek-snapshots.md` | built 11-24 (`claude/midweek-snapshots`); awaiting merge and Worker redeploy before Wed 12-02 13:30 UTC |
| P5 receptions model | `notes/implementation-plan/P5-receptions-model.md` | spec written 11-24 (rules frozen); P5a after P4, P5b only after a Stage A pass |

## Log

### 2026-11-24: mid-week game-odds snapshots built

- Owner approved the P2 owner question: one extra game-odds snapshot on Wed/Thu/Sat from 17:30 UTC, about 2 credits each, so the P2 gate (80 offers on 35 games) can fill this season.
- New `src/midweek_snapshot.py` and `midweek_snapshots.yml` (group `midweek-snapshots`, top-level `contents: read`, job `contents: write`, `ODDS_API_KEY` only, backup cron `47 17 * * 3,4,6`). Same request as `fetch_odds_api` (a test compares the params); files land in `odds-data` `snapshots/YYYY/MM/` via `push_odds_snapshots.sh` with `SNAP_DIR`, so `src.topdown_v2` reads them unchanged.
- Self-gating: Wed/Thu/Sat, at or after 17:25 UTC, and no call if `odds-data` already holds any snapshot since 17:25 UTC today; ledger read before any paid call, nothing called if it fails.
- Guards: 2 paid calls per run (second only retries network/429/5xx), quota floor 80 (as P1) read from the free `/events` call, one call while the quota is unknown. Weekly burn about 36 (spreads) + 30 (props) + 6 = 72.
- Worker `SCHEDULE`: `midweek_snapshots.yml` appended to the `30 13,17,21 * * MON,WED,THU,FRI,SAT` entry; still 4 crons. Phase 3 `src.topdown_replay` also reads `snapshots/`, so it sees these too; no Phase 3 rule changed. No existing workflow, reader or request edited.
- Tests: true iff `pytest tests/ -q` reports 1746 passed (31 new) and the Worker tests report 11 passed. No Odds API call was made locally. Owner: after merge, `cd external_trigger && npm run deploy`, ideally before Wed 12-02 13:30 UTC (the GitHub backup cron covers late days meanwhile).

### 2026-11-24: P4 top-down player_receptions scanner built

- New `src/props/prop_pricing.py`: `normalize_name`, `prop_quotes`, Poisson `prop_probabilities` (integer lines push), `implied_lambda` (bisection on [0.05, 20] to 1e-6), `price_prop_offer`. It also has the NB(r = 8) sensitivity. The de-vig is `total_pricing.devig_two_way`, and the EV is `key_numbers.calculate_expected_value`.
- New `src/props/topdown_props.py`: prices every US and offshore offer in each `open` snapshot against Pinnacle's λ*, and repeats this against BetOnline (`anchor` column). It flags at EV ≥ 2% and closes against the latest `close` snapshot before kickoff. It also computes `close_ev_nb` and `close_ev_median`, plus `close_status` and the frozen gate (pass, fail, pending, or inconclusive with `--season-settled`). Probe files are ignored.
- New `src/props/player_stats.py`: grades from the nflverse release assets `stats_player_week_2026.parquet` and `snap_counts_2026.parquet` (cached in `data/nflverse_cache/`). The join is on normalized name, team and game. Zero snaps or no snap row is VOID. Unmatched, ambiguous and snaps-without-stats players are listed and never graded.
- Every spec constant is frozen and asserted in a test. The first 37 tests were committed (`7141f0c`) before the code and before the first run.
- Builder choices where the spec is silent are listed in the note. The main one: a book with several two-sided lines for a player is priced off the line nearest 50/50; the report counts how often that fired.
- Owner-approved follow-up the same day, before any capture existed: a player with snaps but no stats row is graded as 0 catches when the name is seen for that team elsewhere in the season, otherwise listed. The report also lists games still pending 3+ days after kickoff.
- First run, archive as of ~21:50 UTC: 1 prop file, the probe, which is excluded. So 0 priced offers and the gate is pending. A smoke check with the probe relabelled in memory priced 126 offers per anchor across the 5 returning books, BetRivers' one-sided Overs included. No EVs were looked at.
- `total_pricing`, `topdown_replay`, `market_consensus`, `topdown_v2`, the workflows, the Worker, `src/config.py` and `models/` were not edited. No Odds API call and no Sheet write.
- `.gitignore`: added `data/topdown_props_*.csv`.
- Tests: true iff `pytest tests/ -q` reports 1756 passed (41 new, after P3's 1715) and the Worker tests report 10 passed.
- The weekly re-run fills the note's tables. The GB @ LA open and close files should land by ~01:30 UTC 11-26, and the Thanksgiving and Black Friday files after that.

### 2026-11-24: P3 offline totals kill test: KILL

- New `src/totals/features.py` builds team-game PBP stats (dropback EPA, neutral sec/play and PROE, plays, red-zone trips and TDs, points). Its leak-free 8-game rolling windows (min 3) carry across seasons, and fold-time fills and the weather encoding use training seasons only.
- New `src/totals/backtest_totals.py` runs the walk-forward (2014–2025, train 2012..S−1), StandardScaler + Ridge with `alpha` from {1,3,10,30,100} by an inner TimeSeriesSplit, and a nested β against t* = `total_implied_mean`. `src/total_pricing.py` was reused unchanged.
- The code and tests were committed (`8506432`) before any outcome was computed. One run.
- 2020–2025: pooled β̂ +0.068 ± 0.100 (t +0.68) and Δ log loss +0.00011 ± 0.00008 (the model is worse), so the β path is not met.
- Wind ≥ 15, 2020–2025: mean (actual − t*) −1.60 ± 1.23 (t −1.30), Δ log loss +0.00098, so the wind path is not met. Verdict: **KILL**.
- Every fold picked `alpha` 100, the top of the grid. The fold βs fell from +0.29 (2014) to about 0 from 2019 on. 2020–2025 placed no bets at 0.55/0.45.
- Descriptive only, not grounds to reopen: indoor games 2020–2025 finished +1.92 ± 0.59 over t*. Caveats in the note: game-time weather, a single-source close, and an underpowered wind subset (detects ≥ about 2.5 pts).
- `games.csv` has 114 high-wind REG games in 2020–2025, not the spec's 121, which probably included playoffs. The pipeline matches the raw counts.
- Tests: true iff `pytest tests/ -q` reports 1715 passed (21 new) and the Worker tests report 10 passed.
- Older PROGRESS log entries (11-13 to 11-21) were moved unchanged to `notes/archive/`.

### 2026-11-24: P2 top-down v2 replay (spreads and totals)

- New `src/total_pricing.py` (P3 reuses it): discretised-Normal totals with σ 13.2 and a continuity correction, `devig_two_way`, `total_implied_mean` (bisection), `price_total_offer`, `total_quotes`.
- New `src/topdown_v2.py`: sharp-group anchor (BetOnline/LowVig as one vote), only US and offshore books priced, flag at EV ≥ 1.5%, a per-market close, `close_ev` and `close_ev_allbooks`, persistence, freshness, a design/evaluation sample split and grading of spreads and totals. Writes `data/topdown_v2_offers.csv` (gitignored).
- Every spec constant is frozen and asserted in a test. `gate()` returns pass, fail or inconclusive, or `pending` before week 18 settles (`--season-settled`).
- Phase 3 files were not edited.
- First run, archive 6466ba8: 52 snapshots, all before `EVAL_START`, so the evaluation sample is empty and the gate is pending (0 of 80 offers, 0 of 35 games).
- Design sample, not evidence: 6,052 first sightings on 47 games. The gate group is 9 offers on 9 games, close_ev +2.1% ± 1.0% (game-clustered), +0.9% with the top 2 games removed, 5-4.
- `anchor_groups` was 1 on every row: only BetOnline is in the archive. No book fell outside the classes.
- Tests: true iff `pytest tests/ -q` reports 1694 passed (28 new, after P1's 1666) and the Worker tests report 10 passed.
- Owner question still open (spec): add mid-week snapshots so the 80/35 size can be reached this season. Without them, expect the gate to still be pending at week 18.

### 2026-11-24: P1 probe and Wednesday Worker cron

- Probe run 45488926007 (GB @ LA, 1 credit, 395 left): 7 of 10 books returned `player_receptions`, all with both Over and Under except BetRivers (Over only). Pinnacle, DraftKings, FanDuel, BetOnline and Bovada had 15 players; BetMGM had 12. `williamhill_us`, `betus` and `mybookieag` returned nothing; recheck on the first real captures before replacing them.
- Week 12 has a Wednesday game (GB @ LA, 01:00 UTC Thu) and the Worker had no Wednesday cron. Added `WED` to `30 13,17,21 * * MON,THU,FRI,SAT`, still 4 crons. The closing loop and props capture now both run on Wednesdays. The owner has since redeployed the Worker (twice).

### 2026-11-24: P1 prop archive capture built

- New package `src/props/`: `capture_props.py` (self-pacing loop, `--probe`), `archive_props.py` (gzip into `props/YYYY/MM/`, `mtime=0`), `load_prop_archive.py` (reads only `props/`). New `scripts/push_prop_snapshots.sh` (copy of the odds push script, same 3-attempt retry).
- Targets as specified: open = min(k − 4h, first Fri 21:00 UTC at or after k − 4d), due until k − 120 min; close target k − 25, due k − 40 to k − 5. Game list from the free `/events` call; ledger = `origin/odds-data` `props/` listing plus local `data/props_cache/` files (gitignored).
- Request: `/events/{id}/odds`, 10-book `bookmakers=` list incl. Pinnacle, `markets=player_receptions`, no `regions`. Guards: 20 credits/run (earliest kickoffs first), 80-credit floor, one call while the quota is unknown. Env overrides `PROPS_MAX_CREDITS`, `PROPS_QUOTA_FLOOR`, `PROPS_BOOKMAKERS`, `PROPS_MAX_MINUTES`.
- A run exits 1 (red) when the events call or ledger read failed, a capture call errored, or a due capture was skipped by a guard. If the ledger can't be read, that wake captures nothing, so no game is paid for twice. A run whose budget is used up exits early so the next queued run starts with a fresh cap.
- `props_capture.yml`: group `props-capture`, top-level `contents: read`, job `contents: write`, `ODDS_API_KEY` only, timeout 345, backup cron `47 16 * * 4,5`, `probe` input. Worker `SCHEDULE`: `props_capture.yml` appended after `closing_line_monitor.yml` on both closing crons; still 4 crons, `wrangler.toml` unchanged.
- The spread pipeline is unchanged: `closing_loop.py`, `closing_line_monitor.yml`, `tuesday_opening.yml`, `src/config.py`, `src/odds_utils.py` and `push_odds_snapshots.sh` were not edited. A test checks that the spread and prop readers each list only their own files.
- No Odds API call was made locally. Tests: pytest 1630 → 1666, node 8 → 10, all green.
- Owner to do after merge: redeploy the Worker, dispatch the probe, then swap out any book that returned no props (candidates in the spec).

### 2026-11-24: live check of the Tuesday run (week 12) and Phase 3 replay

- Read-only. Tuesday run 45423885996: decision week 12, 16 games captured, 93 future games skipped; v1 picked 5, v2/v3/v4 picked none.
- Phase 1 pass: week-11 Sunday and Monday closes landed T-10 to T-17 min (week 10 was T-45); `Bets w/ Close` present; slate is week 12 only; "Restore play-by-play cache" step ran (first-run miss, saved). Only |raw lean| >= 7 game: v3 NE@LAC (−7.4), flagged SUSPICIOUS_OUTLIER, PASS since edge −1.97% is below threshold (R2 rule).
- R5 pass: new rows carry `v3_2012-2025_132e614a` and `v4_2012-2025_995bf8ae`; no V3/V4 pick; one artifact in-week, so no per-artifact line yet (by design).
- R1 pass: no week 13+ row has an opening line; all 16 week-12 rows have `opening_book`, `opening_quote_at`, `decision_regime = in_week`, and the 5 picks have `pick_made_at`. Requests remaining 432 (11-17) → 396 (11-24).
- R2/R3 pass: the three shadow tabs end with `artifact_id`, `decision_regime`, raw-lean columns; 48 new rows, none with blank `raw_lean_side`; legacy v2 6 / v3 4 bets; guard count 3; RAW LEAN CLV sections present (in-week empty until week 12 settles).
- Phase 3 replay: first run gave 41 flagged offers on 23 games, but 7 were unplayed week-12 games scored against the latest snapshot. Owner approved a fix: `closing_state` now skips games that kick off after the archive's latest snapshot (new test). Re-run: 47 played games, 34 flagged on 17, closing EV +3.1% ± 1.4% (game-clustered), 68% beat the close. Test not reached (needs 100 on 40).

Older entries (2026-11-13 to 11-21): `notes/archive/PROGRESS-2026-11-13-to-11-21.md`.
