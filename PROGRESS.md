# Progress

## Current state

- **Branch:** `main`. The review and plan were added on `claude/review-and-plan`, cut from `99dae0c`; no source code has changed since that commit.
- **Last task:** independent review of `notes/shadow-model-findings.md` (2026-11-13). Output: `FABLE_REVIEW.md`.
- **Next task:** **R1, fix decision timing.** Spec: `notes/implementation-plan/R1-decision-timing.md`. Read `notes/implementation-plan/README.md` first.
- **Deadline:** R1 live before Tue 2026-11-17 15:17 UTC keeps week 12 clean. Later than that, run the migration step in the R1 spec.
- **Open questions for the owner:** the four in the table in `notes/implementation-plan/README.md`. Only the R1 one (weekly `MAX_PICKS` value, default 5) is needed to start.

## Task queue

| Task | Spec | Status |
|---|---|---|
| R1 decision timing | `notes/implementation-plan/R1-decision-timing.md` | not started |
| R2 scorecard hygiene | `notes/implementation-plan/R2-scorecard-hygiene.md` | not started |
| R3 raw-lean tracking | `notes/implementation-plan/R3-raw-lean-tracking.md` | not started |
| R4 production backtest | `notes/implementation-plan/R4-production-backtest.md` | not started (independent of R1–R3) |
| R5 price-consistent anchor | `notes/implementation-plan/R5-price-consistent-anchor.md` | not started (needs R4, R2) |

## Log

### 2026-11-13: independent review

- Reviewed the findings note against code, artifacts and the live Sheet. Full write-up in `FABLE_REVIEW.md`.
- Main results: v1's baseline has look-ahead bias (11 of 23 timed picks); every v2/v3 scorecard bet is an artifact; v3/v4 probabilities are driven by the line number; the shadow models cannot reach a 2% edge at standard prices.
- `notes/shadow-model-findings.md` is superseded as a backlog. Its fix 1 survives, amended, as R3. Its fixes 2 and 5 were already done.
- Added `scripts/review_repro/` (three read-only scripts) and `notes/implementation-plan/` (R1–R5 specs).
- `data/sheet_dump/` is gitignored and holds a local Sheet dump from 2026-11-13.
- No source code changed. `pytest tests/`: 75 passed.
- Handoff items from 2026-11-09, now checked in the Sheet: V3/V4 rows do append after existing data (runs of 11-04 and 11-10); week 9 MNF close was captured and settled.
