# Progress log archive (2026-11-24, part 1)

Moved unchanged from `PROGRESS.md` on 2026-11-25.

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
