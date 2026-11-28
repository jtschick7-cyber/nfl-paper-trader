# Progress

## Current state

- **Branch:** `claude/amazing-johnson-kfhn3v` (prop book swap, PR #37; owner merges before Thu 12-03 ~21:15 UTC, the week-14 TNF open). PRs #14–#36 and #38 are merged on `main`.
- **2026-11-28:** schedule source fixed (PR #38, merged): nfl_data_py's habitatring.com mirror returns 404, so Nightly Results failed at 05:17 UTC; `src/nflverse_schedule.py` now reads the nflverse GitHub copy. DEN @ PIT settles at the next Nightly Results run.
- **Last task:** regular-season filter: props capture and mid-week snapshots always count Sep–Dec kickoffs (no schedule needed) and count Jan–Aug kickoffs only inside the season's REG window from nflverse `games.csv` (±24 h grace), failing closed for Jan–Aug without a schedule (`src/regular_season.py`). Before that, Nov 25 audit hardening: prop grading never sums a playoff rematch into a regular-season game; the mid-week snapshot spends nothing when no game kicks off within 168 h; `props_capture.yml` uses a shallow checkout. Before that, P5a Stage A: **FAIL**, Step 5 closed.
- **P1 is live.** The Worker was redeployed twice; its crons now include `30 13,17,21 * * MON,WED,THU,FRI,SAT` (closing loop + props capture). Probe run 45488926007 cost 1 credit and found `player_receptions` at Pinnacle, DraftKings, FanDuel, BetOnline and Bovada (15 players), BetMGM (12) and BetRivers (12, Over only); none at `williamhill_us`, `betus`, `mybookieag`.
- **Mid-week snapshots:** live. Owner redeployed the Worker on 11-24 (version `916d228c`, 4 crons incl. `30 13,17,21 * * MON,WED,THU,FRI,SAT`). First snapshot `odds_20261125T173058Z` (2 credits); the 21:30 wake and a late backup-cron run (01:11 UTC Thu) called nothing.
- **Owner decision (2026-11-25):** the 2026 P2/P4 gates are regular-season only; final reports run on the `odds-data` tag `eval-2026-final` set at week-18 settle (`notes/implementation-plan/README.md` decision 5). Props capture and mid-week snapshots now stop after the last regular-season kickoff (2026: 2027-01-10 18:00 UTC placeholder, plus 24 h grace) and skip August preseason, once this PR merges; a failed schedule download never blocks Sep–Dec capture. Prop book swap (`betus`→`fanatics`, `mybookieag`→`hardrockbet`, `williamhill_us`→`espnbet`) is in the open PR on `claude/amazing-johnson-kfhn3v`; it must merge before Thu 11-26 13:30 UTC (the first Thanksgiving open).
- **Next task:** Step 5 is closed (P5a FAIL), so there is no P5b. Next is the P1 live check after the first Friday and Sunday (P1 spec "Live check"), and the owner's next roadmap item.
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
| P1 prop archive capture | `notes/implementation-plan/P1-props-capture.md` | built and merged 11-24 (PR #26); Worker redeployed, probe done (run 45488926007); first open/close 11-25 (GB @ LA); book swap 11-26; live check after the first Friday and Sunday |
| P2 top-down v2 (spreads + totals) | `notes/implementation-plan/P2-topdown-v2.md` | built 11-24 (`notes/p2-topdown-v2.md`); weekly re-run, gate pending |
| P3 totals kill test | `notes/implementation-plan/P3-totals-kill-test.md` | done 11-24: KILL (`notes/p3-totals-kill-test.md`); bottom-up totals closed |
| P4 prop scanner | `notes/implementation-plan/P4-prop-scanner.md` | built 11-24 (`notes/p4-prop-scanner.md`); weekly re-run, gate pending |
| Mid-week game-odds snapshots | owner answer to the P2 owner question (11-24); `notes/midweek-snapshots.md` | built and merged 11-24 (PR #33); Worker redeployed 11-24; first snapshot 11-25 17:30 UTC |
| P5 receptions model | `notes/implementation-plan/P5-receptions-model.md` | P5a done 11-24: Stage A **FAIL** on calibration (`notes/p5a-receptions-stage-a.md`); Step 5 closed, no P5b |

## Log

### 2026-11-28: schedule source fix; Sat P1 live check

- Nightly Results run 46111441456 failed: `nfl.import_schedules` reads `http://www.habitatring.com/games.csv`, now HTTP 404. The same 46-column file at `raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv` is current: 168 finals, DEN @ PIT 5–27 included.
- New `src/nflverse_schedule.import_schedules` has the same contract as the old call and tries GitHub first, with the old mirror as fallback. It is used by `update_results`, `compare_models.load_final_scores` (which had silently returned no scores on failure), `audit_state`, `fetch_schedule` and `model_utils._load_sched` (which returned an empty schedule in CI without a cache).
- No workflow, frozen file, model, Sheet or Odds API touched.
- P1 live check, holiday and Friday part: GB @ LA, the 3 Thanksgiving games and DEN @ PIT each have 1 open (captured at target) and 1 close (25–30 min before kickoff). All 11 Sun/Mon games have Friday 21:00 opens. Every file cost 1 credit, the Friday run cost 11, and 338 credits remain. No empty `bookmakers` file.
- In the same check, `williamhill_us`, `betus` and `mybookieag` posted nothing in all 16 files (PR #37 unmerged). NE @ LAC's open has 5 books and no Pinnacle.
- Closing loop: the holiday closes landed at T-10.4 to T-12.2 min (pass band T-10 to T-17). The mid-week job skipped Friday correctly; the 11-27 17:30 snapshot is the closing loop's.
- P4 descriptive run: 0 ungraded players across 613 graded rows (no name mismatches). Gate pending at 7 offers on 4 games; `drift_ok` met.
- Still to check: the Sunday/Monday closes (P1 "after the first Sunday").
- Tests: true iff `pytest tests/ -q` reports 1871 passed (8 new).

### 2026-11-26: first prop open/close, first mid-week snapshot, prop book swap

- GB @ LA (kickoff 11-26 01:00 UTC): `props_open_…_20261125T210000Z` (captured by the 17:30 run after sleeping to its 21:00 target, k − 240 min) and `props_close_…_20261126T003500Z` (21:30 run, k − 25 min). 1 credit each; remaining 390 after the open, 385 after the close.
- Players per book, open / close: Pinnacle 15/15, DraftKings 15/15, FanDuel 15/15, Bovada 15/15, BetMGM 13/13, BetRivers 12/11 (Over only), BetOnline 15 at the open and **absent at the close**. `williamhill_us`, `betus`, `mybookieag`: none in the probe, the open or the close.
- Owner decision (a) applied: `_DEFAULT_BOOKS` in `src/props/capture_props.py` swaps `williamhill_us`→`espnbet`, `betus`→`fanatics`, `mybookieag`→`hardrockbet`. Still 10 books (one region of billing) with Pinnacle; credit cost unchanged. All three new books are in P4's frozen `US_TARGETS`, so no P4 rule changes; P4's offshore class now has only Bovada in the capture list. A new test pins the list and checks every book is in a P4 class.
- `PROPS_BOOKMAKERS` overrides still win. No workflow or Worker change, so no redeploy.
- Mid-week snapshot (first real run): Wed 11-25 17:30 wake wrote `odds_20261125T173058Z` (30 events, 2 credits, 393 → 391). The 21:30 wake found it and skipped; the backup cron's late run at 01:11 UTC Thu was before the window and called nothing.
- Worker: owner redeployed on 11-24 after #31–#34 (version `916d228c`).
- Tests: true iff `pytest tests/ -q` reports 1863 passed, 1 skipped.

### 2026-11-25: regular-season cutoff for props capture and mid-week snapshots

- New `src/regular_season.py`: Sep–Dec kickoffs (UTC) always count and never load the schedule. Jan–Aug kickoffs must fall inside the season's first-to-last `game_type == REG` kickoff in `data/nflverse_cache/games.csv` (ET to UTC; season = year if month >= 3 else year - 1), so playoffs and August preseason spend nothing and 2027's January games count once `games.csv` holds 2027.
- 24 h grace on both ends: nflverse lists week 18 at a 13:00 ET placeholder until flex, so a week-18 SNF still counts; wild card is always >= 5 days later.
- `capture_props.upcoming()` / `due_items()` / `next_wake()` take `cutoffs`; skipped events are logged once per run. `midweek_snapshot.has_upcoming_event()` counts REG games only.
- Fail closed for Jan–Aug only (review fix, owner-approved): no readable `games.csv`, or no rows for the season, means 0 credits on those events and exit 1. A failed nflverse download never blocks Sep–Dec capture. Playoff and preseason skips exit 0.
- `props_capture.yml` and `midweek_snapshots.yml` gain a `python -m src.regular_season --fetch` step (`games.csv` is gitignored); the step never fails.
- No frozen file, model, Sheet or Odds API touched.
- Tests: true iff `pytest tests/ -q` reports 1863 passed (30 new) and the Worker tests report 11 passed.

### 2026-11-25: Nov 25 audit hardening (Fixes 1, 2, 4)

- `src/props/player_stats.py`: drops postseason rows (`season_type` / `game_type` != REG) before grading, and returns `ambiguous_game` (listed, ungraded) when two games match one (season, away, home), so a playoff rematch is never summed in. Descriptive W-L only; the gate uses `close_ev`.
- `src/midweek_snapshot.py`: reads the free `/events` body and skips with 0 credits unless a game kicks off within 168 h; a non-list body exits 1 before any paid call. Test fake now returns an upcoming game by default.
- `props_capture.yml`: shallow checkout, like the other odds-data writers.
- Owner decision 5 (regular-season-only gates) recorded in the plan README and as dated scope clarifications in the P2 and P4 specs; no number or rule changed.
- No frozen file, model, Sheet or Odds API touched. Older 11-24 log entries moved unchanged to `notes/archive/`.
- Tests: true iff `pytest tests/ -q` reports 1833 passed (10 new) and the Worker tests report 11 passed.

### 2026-11-24: P5a receptions model, Stage A walk-forward: FAIL

- New `src/props/data_sources.py` downloads and caches the nflverse `stats_player_week`, `snap_counts`, `injuries` and `players` assets and joins snaps PFR → GSIS. 204 of 241,911 REG snap rows have no match and are dropped.
- New `src/props/features.py` builds the frozen universe and the point-in-time features: role windows (8/4 games across seasons), game script, injury report shares, `qb_out`, team pace and opponent windows. `depth_rank` uses an as-of-kickoff roster, never realised participation.
- New `src/props/train_receptions.py`: StandardScaler + PoissonRegressor, alpha by forward inner season splits, NB r per position group by ML on [1, 200] from out-of-fold means, and a 0..14 PMF with the ≥ 15 tail lumped. New `src/props/backtest_receptions.py` holds the metrics, calibration, gate and report.
- The tests were committed (`9342779`) before the code, and the code (`03bcb5e`) before the single run. Builder choices where the spec is silent are listed in the note.
- Owner decision before the run, from the spec text only: test season 2018 has no inner fold under the frozen alpha rule, so it is **skipped**. The gate seasons are unaffected and no constant changed.
- Universe: 28,404 player-games for 2016–2025; 20,579 are scored (2019–2025), 17,761 of them in the gate seasons.
- Result, pooled 2020–2025: Δ LL_sum −0.0430 ± 0.0050 (−8.67 SE), injury subset −0.0571, Δ < 0 in 6 of 6 seasons. Full log loss Δ −0.0248 and CRPS Δ −0.0272 point the same way.
- Calibration fails. 15 of the 29 bins with n ≥ 500 miss the 2-SE band, mostly with observed over-rates above the model's P(over). Verdict: **FAIL**. Step 5 stops, and `receptions_vs_market.py` was not built.
- Data: `games.csv` came from `nflverse/nfldata` on GitHub, because `nfl_data_py`'s schedule host does not resolve here. PBP 2016–2025 was filled by `src.build_features`, and its rewrite of the tracked `data/team_game_*.csv` was reverted. `.gitignore`: added `data/receptions_stage_a.csv`.
- `models/`, the workflows, the Worker, `src/config.py`, `src/totals/*` and the other listed modules were not edited. No Odds API call and no Sheet write.
- Tests: true iff `pytest tests/ -q` reports 1792 passed (36 new, after P4's 1756) and the Worker tests report 10 passed.

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

Older entries: `notes/archive/PROGRESS-2026-11-24-part1.md` (11-24: P2, P1 probe and capture, week-12 live check) and `notes/archive/PROGRESS-2026-11-13-to-11-21.md`.
