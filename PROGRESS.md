# Progress

## Current state

- **Branch:** `claude/r1-decision-timing` (PR open). Merge before Tue 2026-11-17 15:17 UTC so week 12 is the first clean `in_week` week.
- **Last task:** R1, fix decision timing (2026-11-14).
- **Next task:** **R2, scorecard hygiene.** Spec: `notes/implementation-plan/R2-scorecard-hygiene.md`. Owner question for R2 is in the plan README table.
- **After the first live Tuesday run with R1:** check the Sheet: no week-13 row has an opening line; every new pick has `pick_made_at`, `opening_book`, `opening_quote_at`, `decision_regime = in_week`; `x-requests-remaining` in the capture log looks sane.
- **If R1 merges after the deadline:** run the `Migrate look-ahead openers` workflow (dry run first, then tick `apply`).

## Task queue

| Task | Spec | Status |
|---|---|---|
| R1 decision timing | `notes/implementation-plan/R1-decision-timing.md` | done (PR), live check pending |
| R2 scorecard hygiene | `notes/implementation-plan/R2-scorecard-hygiene.md` | not started |
| R3 raw-lean tracking | `notes/implementation-plan/R3-raw-lean-tracking.md` | not started |
| R4 production backtest | `notes/implementation-plan/R4-production-backtest.md` | not started (independent of R1–R3) |
| R5 price-consistent anchor | `notes/implementation-plan/R5-price-consistent-anchor.md` | not started (needs R4, R2) |

## Log

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
