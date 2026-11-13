# R3. Raw-lean tracking

**Review sections:** F1, D, W2, M6. **Effort:** M. **Branch:** `claude/r3-raw-lean`. **Depends on:** R2 (`ensure_tab_headers`, `clv_fair_probability`, regimes).

Evaluation only. Picks and probabilities do not change.

## Why

The shadow models cannot bet under the current design, so there is nothing to grade. The raw lean (which side each model favours before the market blend, and by how much) can be graded on every game. This is `notes/shadow-model-findings.md` fix 1 with the corrections from the review.

Expect little: over 2012–2025 the v3 raw lean covered 49.8% against the closing line (n = 2,974). What this measures instead is whether the lean predicts movement from the in-week open to the close.

## Definitions

- `market_mu` = `-home_line` (the opening line logged on the row).
- `raw_mu_home_margin` = the model's margin before `blend_margin`. For v2 this is its ordinary margin.
- `raw_lean_pts` = `raw_mu_home_margin - market_mu`. Positive means the model leans to the home side.
- `raw_lean_side` = `HOME` if `raw_lean_pts` > 0, `AWAY` if < 0, blank if exactly 0.

The lean must come from these, never from `v*_best_side`: that column disagrees with the raw-lean side in 15 of 27 v3 rows and 24 of 42 v4 rows.

## Changes

### Loggers

Append to the end of `HEADERS` (after R2's two columns) in `log_v2_shadow.py`, `log_v3_shadow.py`, `log_v4_shadow.py`: `raw_mu_home_margin`, `raw_lean_pts`, `raw_lean_side`.

- v3: keep the pre-blend value before `log_v3_shadow.py:376-378` overwrites `mu`.
- v4: same at `log_v4_shadow.py:431-434`. Write these three with 4 decimals; v4 rounds its other columns to 2 and that is what made backfill imprecise.
- v2: `raw_mu_home_margin` equals `mu`.

### Backfill: new `src/backfill_raw_leans.py` (one-off, dry-run by default, `--apply` to write)

Runs inside the `sheet-writer` group. Fills the three columns only where they are blank.

| Tab | Rows | Formula | Note |
|---|---|---|---|
| V2_Shadow | all | `raw = mu_home_margin` | v2's artifact has not changed since 2026-10-19 |
| V3_Shadow | `sigma` = 12.8145 (current artifact) | `raw = market_mu + (mu_home_margin - market_mu) / 0.03693757162735288` | exact; `mu` is logged to 4 decimals |
| V3_Shadow | `sigma` = 13.2147 (pre-fix) | skip | features were hardcoded to zero |
| V4_Shadow | all current-artifact rows | same with β = 0.042324548578395654 | `mu` is logged to 2 decimals: error ±0.12 pts. Leave `raw_lean_side` blank when `abs(mu_home_margin - market_mu) < 0.01` |

Read β from the artifact by `artifact_id` rather than hardcoding once R2's stamp exists; the constants above are for rows that predate it. All backfilled rows are legacy regime, so they inform but do not count toward the test below.

### Grading: `src/compare_models.py`

New "RAW LEAN CLV" section on `Model_Scorecard`, `in_week` regime, first row per game per model, games with a recorded close:

- The ticket is `raw_lean_side` at the row's opening line and that side's opening price.
- Metric: `clv_fair_probability` (R2). Fall back to `clv_win_probability` when closing prices are missing and flag the count.
- Per model, for all leans and for `abs(raw_lean_pts) >= 1.5`: n, mean, standard error, t-statistic, share positive, and mean `raw_lean_pts`.
- **Baselines on the same games:** "always home" and "always underdog" (the side with the positive line; skip pick'em). Report each model's mean minus the always-home mean, with its standard error computed on the paired differences.
- A legacy block underneath with the same columns, labelled as not counting.

Do not sum across models: v2, v3 and v4 share features and their leans are correlated.

### Pre-registered test

Copy this into the README under the promotion rule, dated, before any in-week lean has a close:

> **Raw-lean test, registered 2026-11-13.** Primary: v3, all in-week leans, price-aware CLV. The lean carries information if, with at least 100 graded lean-games, the mean paired difference against the always-home baseline has t ≥ 2. Everything else in the raw-lean section (v2, v4, the ≥ 1.5-point cut, the always-underdog comparison) is descriptive and decides nothing. No change to β, the 2% threshold or the live model is made from this table during the 2026 season; a pass means the model earns a re-fit in the offseason.

About 109 regular-season games remained when this was written. This season can show an effect of roughly a point or more, not half a point.

## Tests (write first)

- New `tests/test_raw_lean.py`:
  - sign and size for home and away leans, and zero;
  - v3/v4: `raw_lean_pts * beta` equals the logged blended deviation;
  - backfill formula returns the raw margin from a blended one to 1e-3 for v3 precision, and leaves the side blank for the v4 ambiguous case;
  - the pre-fix v3 rows are skipped.
- `tests/test_scorecard.py`: lean grading uses `raw_lean_side`, not `v3_best_side`, when they differ; baselines are computed on the same games; legacy rows stay out of the primary block.
- Logger test: the three new headers are appended last and existing column positions are unchanged.

## Done when

- `pytest tests/` green.
- After the next live Tuesday run, new shadow rows carry the three columns, and `Model_Scorecard` has the raw-lean section.
- Backfill dry-run output reviewed, then applied once; V3 weeks 10–11 show 27 backfilled leans, 19 of them at 1.5 points or more (the review's count).
- README carries the registered test.

## Out of scope

- Any use of leans to place bets.
- A second "decision line" later in the week (listed in the review as optional).
