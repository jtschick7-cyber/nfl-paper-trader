# Progress

## Current state

- **Branch:** `main`. PRs #14–#22 (R1–R5, audit Phases 1–3, P0-7) are merged; the 2026-11-24 live check is on `claude/live-check-2026-11-24`.
- **Last task:** live check of the 2026-11-24 Tuesday run (week 12, the first in-week slate). Every check passed; see the log.
- **Next task:** re-run `src.topdown_replay` weekly on the growing archive (pre-registered test in the Phase 3 note; 34 of 100 flagged offers, 17 of 40 played games on 11-24); Phase 4 at the off-season. No retrain queued. Owner decided (2026-11-21): v4 keeps logging as a shadow until the season ends, then retires unless the R3 raw-lean test clearly favours it; off-season v3 retrain with `split_neutral_teams`.
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

## Log

### 2026-11-24: live check of the Tuesday run (week 12) and Phase 3 replay

- Read-only. Tuesday run 45423885996: decision week 12, 16 games captured, 93 future games skipped; v1 picked 5, v2/v3/v4 picked none.
- Phase 1 pass: week-11 Sunday and Monday closes landed T-10 to T-17 min (week 10 was T-45); `Bets w/ Close` present; slate is week 12 only; "Restore play-by-play cache" step ran (first-run miss, saved). Only |raw lean| >= 7 game: v3 NE@LAC (−7.4), flagged SUSPICIOUS_OUTLIER, PASS since edge −1.97% is below threshold (R2 rule).
- R5 pass: new rows carry `v3_2012-2025_132e614a` and `v4_2012-2025_995bf8ae`; no V3/V4 pick; one artifact in-week, so no per-artifact line yet (by design).
- R1 pass: no week 13+ row has an opening line; all 16 week-12 rows have `opening_book`, `opening_quote_at`, `decision_regime = in_week`, and the 5 picks have `pick_made_at`. Requests remaining 432 (11-17) → 396 (11-24).
- R2/R3 pass: the three shadow tabs end with `artifact_id`, `decision_regime`, raw-lean columns; 48 new rows, none with blank `raw_lean_side`; legacy v2 6 / v3 4 bets; guard count 3; RAW LEAN CLV sections present (in-week empty until week 12 settles).
- Phase 3 replay: first run gave 41 flagged offers on 23 games, but 7 were unplayed week-12 games scored against the latest snapshot. Owner approved a fix: `closing_state` now skips games that kick off after the archive's latest snapshot (new test). Re-run: 47 played games, 34 flagged on 17, closing EV +3.1% ± 1.4% (game-clustered), 68% beat the close. Test not reached (needs 100 on 40).

### 2026-11-21: Audit Phase 3 consensus pricer and archive replay

- Offline only: no Sheet tab, workflow or decision change.
- `src/market_consensus.py` (per-book m*, leave-one-out trimmed weighted consensus, EV per offer, stale flag at EV >= 1%, key-number cross), `src/load_odds_archive.py` (reads `odds-data` via git), `src/topdown_replay.py` (price-aware CLV against the closing consensus, game-clustered SE).
- Archive: 40 snapshots, 1–20 Nov 2026, 63 games. Flagged 38 offers on 21 games: closing EV +2.4% ± 1.0%, 68% beat the close, 15-14 graded. Promising, too small.
- Pre-registered pass: >= 100 flagged on >= 40 games, closing EV > 0 by >= 2 SE; only then a live `TopDown_Shadow` tab.
- Tests: 1,629 passed.

### 2026-11-21: Audit P0-7 v4 QB shock and v4 study

- Offline only, no `.pkl` changes; defaults stay legacy, so the live v4 logger is unchanged.
- `qb_version="fixed"`: actual starters in training, a −0.10 backup prior, and a same-shrinkage baseline (0 for an unchanged starter). `fetch_pregame_starter_overrides(version="fixed", week=…)`: decision-week report only, earlier-season backup or `REPLACEMENT_BACKUP`.
- New bug: OAK/SD/STL schedule codes never matched LV/LAC/LA in the feature tables (about 7% of 2012–2019 rows got zero features). `canonical_teams=True` in the v3 and v4 builders.
- `backtest_production --feature-set` now covers v4: `legacy`, `qb`, `split_neutral`, `split_neutral_qb`, `split_neutral_qb_teams`; v3 adds `split_neutral_teams`.
- Result: the QB fix removes the artifact but no gain (log loss 0.69350 → 0.69368; the market prices starter changes). The team-code fix is the best of each model (v3 0.69340). Probe 0 and calibration within 2 SE for all. Recommend retiring v4 at the off-season.
- Tests: 1,621 passed, 8 Node tests passed.

### 2026-11-21: Audit Phase 2 offline feature study

- Owner: offline study only, no `.pkl` changes; stacked on Phase 1.
- `epa_features`: `play_split="dropback"` (no scramble double count) and `version="fixed"` (correct sign plus offence/defence split). Defaults stay legacy, so the live loggers and artifacts are unchanged.
- v3 builder: `matchup_version`, `neutral_sites`, a `home_field` column, a game_id kickoff tiebreak, NaN for missing lines; `_load_sched` keeps `location`.
- `backtest_production --feature-set legacy|sign|sign_scramble|split|split_neutral`.
- Result: no gate improves beyond noise (log loss within 0.0002, RMSE 12.873 → 12.869); recommend folding split_neutral into the next off-season retrain.
- Tests: 1,610 passed.

### 2026-11-21: Audit Phase 1 safety fixes

- No model artifact retrained; v3/v4 `.pkl` unchanged, so the R3 test is not interrupted.
- decision.py: no PICK on an assumed (one-sided) price; a finite edge beats NaN. Loggers: NaN edge is PASS, zero prices rejected, v4 stops if the existing rows cannot be read.
- Outlier guard now gets the raw margin (it was dead on the blended one) and keeps a 0.0 pick'em. Live behaviour change: games with |raw lean| >= 7 pts become SUPPRESS_DANGER.
- build_features: refresh on missing completed game_ids (not max week, no file-age gate), atomic CSV writes, fail instead of dropping the live season, global SSL bypass removed.
- Closing: one more upgrade inside T-20m; odds cache 240 s; opening/closing matched by kickoff within 5 days.
- Scorecard: slate deduped like grading and cut to the decision week, `Bets w/ Close` appended, fallback away price gives line-only CLV, Step Summary gains weekly-CLV and raw-lean tables, Sheet write no longer clears first.
- Workflows: PBP cache, odds cache on the V3 logger, closing backfill in its own `sheet-writer` job.
- Deferred: consensus fair prob and EV line shopping (Phase 3), v2 retire-or-retrain (owner), nfl_data_py migration.
- Tests: 1,604 passed, 8 Node tests passed.

### 2026-11-14: R5 price-consistent anchor

- Owner: keep v4 as a second shadow model, not a separate promotion candidate.
- `src/price_anchor.py`: m* (Brent + cache, matches R2's `market_implied_mean` to 1e-6), `fit_price_anchor` (β vs m*, σ iterated, diagnostic calibrator slope + SE). `shadow_decision(anchor="price")`; no anchor field = old path, bit-identical.
- A vs B settled by the walk-forward backtest: a tie (v3 log loss 0.69350 both, RMSE 12.8715 vs 12.8716; v4 likewise). Kept A.
- v3 new rule: 142 bets, 52.1% ± 4.2, +3.3u; probe 0 picks, max |p − q| 8e-13; every n ≥ 200 bucket within 2 SE. Baseline: 149 bets, 47.5%, −0.4u, 118 probe picks.
- Retrained (2012–2025): v3 β 0.0212 ± 0.0704, diag slope 0.61 ± 0.60; v4 β 0.0266 ± 0.0704, 0.64 ± 0.60. `fit_fold` on all seasons reproduces both exactly. New ids `v3_…_132e614a`, `v4_…_995bf8ae`.
- Scorecard: in-week block adds a `Model v3 [<artifact_id>]` line per artifact when a tab has more than one (it did not split by artifact before). Backfill skips price-anchored artifacts.
- Tests: xfail removed (passes); the R4 standard-price lock kept for the legacy rule and a new lock added (v3 needs 67.5 pts at −110 on −5.5).
- v4 totals-conditioned σ: no calibration difference (log loss 0.69345 with, 0.69346 without).
- `pytest tests/`: 1,524 passed, 0 xfailed.

### 2026-11-14: R4 production backtest

- `src/decision.py`: `shadow_decision` (blend, key numbers, calibrator, break-even edges, `PICK`/`PASS` before the guard). v2/v3/v4 loggers call it; a characterization test copies the old inline code and matches it to 1e-9 (1,211 cases: three models, totals, integer and half-point lines, five calibrator kinds).
- Locks: v3 today cannot bet at −110/−110 up to 60 pts disagreement (R5 expected to change); zero disagreement at −3, +110/−130 picks, as `xfail(strict)` named for R5.
- `src/backtest_production.py`: walk-forward 2014–2025, ridge/β/calibrator per fold on earlier seasons only (ids recorded and tested), zero-disagreement probe, CSVs to `data/backtest_production_*.csv` (gitignored). v4 behind `--model v4`.
- Check: fitting on all of 2012–2025 reproduces the committed v3 artifact exactly (β, SE, calibrator). Needed a stable sort: ties in kickoff time otherwise moved β to 0.0349.
- v3 baseline: 149 bets in 3,150 games, 47.5% ± 4.2, −0.4u; 117 at +105 or longer, 104 also probe picks; no bets 2023–2025 (closing prices top out at +105); raw-lean hit 50.4% ± 0.9.
- β per fold 0.04–0.33; early folds are small (SE 0.13–0.20), 2019+ fall in 0.07–0.13 with SE 0.07–0.10.
- README: command listed; old backtests flagged as not modelling the live rule; "judge on the live scorecard alone" now says production backtest first.
- `pytest tests/`: 1,391 passed, 1 xfailed (1,219 new, nearly all the parametrized characterization test).

### 2026-11-14: R3 raw-lean tracking

- Owner accepted the pre-registered test as written; it is in the README under the promotion rule.
- `src/raw_lean.py`: `raw_lean_cells` (4 decimals), `lean_side`, `unblend_margin`. v2/v3/v4 loggers append `raw_mu_home_margin`, `raw_lean_pts`, `raw_lean_side` after R2's columns, taken before the blend.
- `src/backfill_raw_leans.py` + manual `backfill_raw_leans.yml` (sheet-writer, dry run by default): fills blanks only; β by `artifact_id`, else by sigma; pre-fix v3 skipped; v4 side blank under 0.01 deviation.
- Dump dry run: V3 27 leans, 19 at ≥ 1.5, 15/27 differ from `v3_best_side` (all match the review). V4 42 rows, 2 blank sides, 23/40 differ (review: 24/42, counting the two ambiguous rows).
- `compare_models.evaluate_raw_leans`: first row per game, lean side at the opening line and price, `clv_fair_probability` with line-only fallback counted; all and ≥ 1.5 cuts; always-home (paired diff SE/t) and always-underdog on the same games; in-week and legacy blocks; status line for the registered test. Section goes last on `Model_Scorecard`.
- Spec deviation: v3 unblend precision is 0.5e-4/β ≈ 1.4e-3, not 1e-3; the test asserts that bound.
- Two R2 tests updated for the three new trailing columns (positions, not assertions, changed).
- `pytest tests/`: 172 passed (33 new).

### 2026-11-14: R2 scorecard hygiene

- `sheet_utils.ensure_tab_headers`: writes an empty header row, appends missing cells to a prefix, raises otherwise. Replaces the three loggers' `ensure_headers` (v3/v4 no longer overwrite row 1). Checked against the 11-13 dump: all three live header rows equal the old `HEADERS`.
- Shadow tabs append `artifact_id` (`src/artifacts.py`: span + sha256[:8]) and `decision_regime = in_week`. v3/v4 rows carry the artifact's `model_version` (env var still overrides); v2 unchanged. v4's local CSV is rewritten once with the new header.
- Loggers: `shadow_decision()` returns `PASS` below `MIN_EDGE_THRESHOLD` whatever the guard says. Scorecard counts `SUPPRESS_DANGER` only at or above the threshold.
- `compare_models`: `SHADOW_LEGACY_CUTOFF_UTC`, `parse_logged_at` (both formats), `evaluate_all_models(regime=...)`, `evaluate_regimes`. In-week block first, then "Legacy (excluded from promotion)"; weekly CLV is in-week only.
- Price-aware CLV: `clv_fair_probability` + `key_numbers.market_implied_mean` (matches the spec table). Scorecard appends `CLV Lose %`, `Avg CLV Fair Prob`, `Avg CLV Fair Prob SE`; beat/tie/lose are now shares of all bets with a close.
- README: every promotion condition (2, 3, 4, 6) uses Avg CLV Fair Prob on in-week rows (owner agreed to 3 and 6 on 2026-11-14; weekly table switched too); 40-bet SE note; v3/v4 cannot reach 2% at −110; `edge` scale change on 2026-11-02; `clv_points` not authoritative.
- Offline run on the 11-13 dump (scores from `game_result`): in-week block empty; legacy v1 18, v2 6, v3 4; guard 3 (TEN@CIN, NYJ@KC, ARI@SEA).
- `pytest tests/`: 139 passed (34 new). One existing assertion changed on purpose: beat % in `test_ties_reported_separately_from_beat_rate` is now 25 (share of all bets), not 50.

### 2026-11-14: R1 decision timing

- New `src/decision_window.py` (pure): decision week, decidable, missing-results guard (5 h grace), `previous_game_day` from the scheduled (not completed) previous game.
- `capture_opening`: decidable games only, Odds API cache max 30 min, skips live markets, writes `opening_book`, `opening_quote_at`, `decision_regime = in_week`. Selection split into `plan_opening_capture`.
- `generate_picks`: same guards; skips openers older than `PICK_MAX_LINE_AGE_MINUTES` (90); `MAX_PICKS` = 5 is now a weekly cap; writes `pick_made_at`; v1 rest from `previous_game_day`. Split into `eligible_candidates`, `apply_weekly_cap`, `build_pick_updates`.
- v2/v3/v4 shadow loggers: same guards; rest from scheduled previous game (v2 raw days; v3/v4 `min(days, 21)`, default 7).
- Four columns appended via `EXTRA_COLUMNS`. One-off `src/migrate_lookahead_openers.py` + manual `migrate_lookahead_openers.yml` (sheet-writer group, dry run by default).
- README: decision window, weekly cap, workflow table. `pytest tests/`: 105 passed (30 new).
- Not verifiable locally: the live-run checks in "Current state" and the `v1_pick_timing.py` check on the first `in_week` week.

### 2026-11-13: independent review

- Reviewed the findings note against code, artifacts and the live Sheet. Full write-up in `FABLE_REVIEW.md`.
- Main results: v1's baseline has look-ahead bias (11 of 23 timed picks); every v2/v3 scorecard bet is an artifact; v3/v4 probabilities are driven by the line number; the shadow models cannot reach a 2% edge at standard prices.
- `notes/shadow-model-findings.md` is superseded as a backlog. Its fix 1 survives, amended, as R3. Its fixes 2 and 5 were already done.
- Added `scripts/review_repro/` (three read-only scripts) and `notes/implementation-plan/` (R1–R5 specs).
- `data/sheet_dump/` is gitignored and holds a local Sheet dump from 2026-11-13.
- No source code changed. `pytest tests/`: 75 passed.
- Handoff items from 2026-11-09, now checked in the Sheet: V3/V4 rows do append after existing data (runs of 11-04 and 11-10); week 9 MNF close was captured and settled.
