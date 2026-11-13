# Implementation plan for the FABLE_REVIEW recommendations (R1–R5)

Written 2026-11-13 at `main` = `99dae0c`. The reasoning and evidence are in [`FABLE_REVIEW.md`](../../FABLE_REVIEW.md); this folder is the build spec. Each task file is self-contained: read `PROGRESS.md`, then only the file for the task you are on.

## Order and dependencies

| Task | Spec | Effort | Depends on | Touches live picks? |
|---|---|---|---|---|
| R1 | [R1-decision-timing.md](R1-decision-timing.md) | S | none | **Yes**, v1 baseline restarts |
| R2 | [R2-scorecard-hygiene.md](R2-scorecard-hygiene.md) | S | R1 (uses `decision_regime`) | No, reporting only |
| R3 | [R3-raw-lean-tracking.md](R3-raw-lean-tracking.md) | M | R2 (header helper, price-aware CLV) | No, evaluation only |
| R4 | [R4-production-backtest.md](R4-production-backtest.md) | M | none (can run in parallel with R2/R3) | No, offline |
| R5 | [R5-price-consistent-anchor.md](R5-price-consistent-anchor.md) | M | R4 | Shadow tabs only |

One task per session, one branch and PR per task (`claude/r1-decision-timing`, and so on).

**R1 has a deadline.** The Tuesday workflow captures next week's look-ahead openers every time it runs. If R1 is live before **Tue 2026-11-17 15:17 UTC**, week 12 is the first clean week with no migration. After that, run the migration step in the R1 spec.

## Decisions already taken

These follow from the review and from the owner saying "implement R1–R5 as recommended". Do not reopen them without asking.

1. The v1 baseline restarts. Picks made under the old timing are kept but labelled `lookahead` and reported separately.
2. `MIN_EDGE_THRESHOLD` stays at 0.02. β is changed only by the walk-forward fit, never by hand.
3. v2/v3 shadow rows logged before 2026-11-03 00:00 UTC are legacy and excluded from any promotion sample.
4. v1's model is not replaced and nothing is promoted this season.

## Questions to put to the owner before the task that needs them

| Task | Question | Default if they have no preference |
|---|---|---|
| R1 | `MAX_PICKS` becomes a weekly cap. Keep it at 5? | 5 |
| R2 | The README promotion rule's CLV metric changes to the price-aware one. OK to edit `README.md:37-46`? | Yes |
| R3 | Pre-registered test (text is in the R3 spec). Accept as written? | Yes |
| R5 | Retire v4, or keep it as a second shadow model? | Keep, but not as a separate promotion candidate |

## Rules for every task

- Tests first for anything quantitative. Never weaken an existing test to make it pass. `pytest tests/` is green before and after (75 tests at the start).
- No retraining of `models/*.pkl` except in R5.
- Sheet schema: only ever **append** columns. Never reorder or rename.
- Every script that mutates the Sheet outside the normal workflows defaults to dry-run and needs `--apply`.
- New `Master_Bets` columns go in `EXTRA_COLUMNS` in `src/ensure_away_odds_columns.py` so the Tuesday run creates them inside the `sheet-writer` group.
- Update `PROGRESS.md` when a task lands: about 10 bullets, and move the "Next task" pointer.

## Checking against the live Sheet

```bash
.venv/bin/python scripts/read_sheet.py Master_Bets
```

The reader is view-only (service account key path is in `.env` as `GSHEET_KEY_FILE`). Tabs: `Master_Bets`, `V2_Shadow`, `V3_Shadow`, `V4_Shadow`, `Model_Scorecard`. Workflows are the only writers; a local session cannot write to the Sheet and should not try. Verify writers with unit tests and, after merge, by reading the Sheet after the next scheduled run.

## Review evidence, reproducible from the repo

| Script | Shows |
|---|---|
| `scripts/review_repro/anchor_and_price_checks.py` | Calibrators in the artifacts; a +110 price triggers a pick with zero model input; shadow models cannot reach 2% at −110; the key-number anchor's cover probabilities and their historical record |
| `scripts/review_repro/v3_beta_rederive.py` | β = 0.0369 ± 0.0710; raw-lean ATS record; calibrator slope 0.361 ± 0.453; line explains 96.5% of calibrator input |
| `scripts/review_repro/v1_pick_timing.py` | Which v1 picks were made after results the line pre-dates; the rest days each pick was computed with |

The third needs a Sheet dump in `data/sheet_dump/` (gitignored); the commands are in its docstring. These scripts are also the regression checks for R1, R4 and R5: each spec says what their output should become.

## Facts worth knowing before you start

- `nfl_data_py` schedule: `spread_line` is positive when the home team is favoured, so `home_line = -spread_line`. `home_spread_odds`, `away_spread_odds` and `total_line` are populated for every 2012–2025 regular-season game but one.
- A Bash hook in the owner's environment rewrites `cat` and `grep`; use the Read tool or a short Python script to view files.
- `gspread` 6.2.1 `get_all_records()` raises if the header row has duplicate values, including two blank cells. Any new shadow-tab column must get its header written first (R2 builds the helper).
- The opening-capture, pick and shadow steps all run in `.github/workflows/tuesday_opening.yml`, five times a week (Tue 15:17, 18:17, 23:17 and Wed/Thu 18:17 UTC), plus an external Cloudflare trigger. GitHub starts crons hours late, so nothing may assume a run happens at its scheduled time.
- Line numbers in the specs are as of `99dae0c`. Search for the quoted code if they have drifted.
