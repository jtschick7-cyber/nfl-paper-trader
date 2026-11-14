# Progress

## Current state

- **Branch:** `claude/r3-raw-lean-tracking` (PR open). R1 merged as PR #14, R2 as PR #15.
- **Last task:** R3, raw-lean tracking (2026-11-14).
- **Next task:** **R4, production backtest.** Spec: `notes/implementation-plan/R4-production-backtest.md`.
- **After the first live Tuesday run with R1 (still pending):** no week-13 row has an opening line; every new pick has `pick_made_at`, `opening_book`, `opening_quote_at`, `decision_regime = in_week`; `x-requests-remaining` in the capture log looks sane.
- **After the next scorecard refresh (R2 merged):** `V2_Shadow`, `V3_Shadow`, `V4_Shadow` end with `artifact_id`, `decision_regime`; the loggers ran without a header error; `Model_Scorecard` legacy block holds v2's 6 and v3's 4 old bets; guard count is 3.
- **After R3 merges:** run the "Backfill raw leans" workflow as a dry run, review it (expect V2 57, V3 27 with 19 at ≥ 1.5 pts and 93 pre-fix skipped, V4 42 with 2 blank sides, plus any rows logged since 11-13), then once with `apply`. After the next Tuesday run: new shadow rows end with `raw_mu_home_margin`, `raw_lean_pts`, `raw_lean_side`, and `Model_Scorecard` ends with the raw-lean section.

## Task queue

| Task | Spec | Status |
|---|---|---|
| R1 decision timing | `notes/implementation-plan/R1-decision-timing.md` | done (PR #14), live check pending |
| R2 scorecard hygiene | `notes/implementation-plan/R2-scorecard-hygiene.md` | done (PR), live check pending |
| R3 raw-lean tracking | `notes/implementation-plan/R3-raw-lean-tracking.md` | done (PR), backfill and live check pending |
| R4 production backtest | `notes/implementation-plan/R4-production-backtest.md` | not started (independent of R1–R3) |
| R5 price-consistent anchor | `notes/implementation-plan/R5-price-consistent-anchor.md` | not started (needs R4, R2) |

## Log

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
