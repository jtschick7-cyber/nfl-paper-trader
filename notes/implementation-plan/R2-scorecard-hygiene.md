# R2. Scorecard hygiene

**Review sections:** S, M7, M8, M9, F1 point 6, defects a5, a8, a10. **Effort:** S. **Branch:** `claude/r2-scorecard-hygiene`. **Depends on:** R1 (`decision_regime`).

Reporting only. No pick or probability changes.

## Problem

1. The scorecard pools rows from different model artifacts under one name, and pools look-ahead picks with in-week picks.
2. "Guard suppressed" counts rows that were never bets.
3. CLV ignores price, although closing prices are stored.
4. "CLV Beat %" hides ties.
5. The shadow tabs have no safe way to gain columns.

## Changes

### 1. Shared header migration: `src/sheet_utils.py`

```python
def ensure_tab_headers(ws, headers) -> None
```

- Empty sheet: write `headers`.
- Existing header row is a prefix of `headers` (same names, same order): grow the grid if needed and write the missing header cells.
- Anything else: raise with both rows in the message. Never rewrite existing header cells.

Replace the three private `ensure_headers` functions (`log_v2_shadow.py:119`, `log_v3_shadow.py:134`, `log_v4_shadow.py:133`) with it. v3 and v4 currently overwrite the header row on mismatch; that behaviour goes.

### 2. Artifact and regime stamps on shadow rows

Append to the end of `HEADERS` in each logger: `artifact_id`, `decision_regime`.

- `artifact_id`: first 8 hex characters of the sha256 of the `.pkl` file, plus the training span when the artifact has one, for example `v3_2012-2025_1a2b3c4d`. Put the helper in `src/sheet_utils.py` or a small `src/artifacts.py`.
- `decision_regime`: `in_week` (after R1 every logged row is).
- Replace the stale `MODEL_VERSION` defaults (`log_v3_shadow.py:45`, `log_v4_shadow.py:57`) with the artifact's own `model_version`. Leave v2's default alone: v2's dedupe key includes it (`log_v2_shadow.py:254`), and changing it would re-log every open game.

### 3. Regime-aware grading: `src/compare_models.py`

- Constant `SHADOW_LEGACY_CUTOFF_UTC = "2026-11-03T00:00:00Z"`. A shadow row is **legacy** if `logged_at_utc` is before it. Parse both formats in use: ISO in V2/V3, `"2026-11-03 21:25:00 UTC"` in V4.
- A v1 bet is **legacy** if its `Master_Bets.decision_regime` is not `in_week`.
- A shadow row for a game whose `Master_Bets.decision_regime` is not `in_week` is also legacy (joins on `game_id`).
- For v3/v4, grade the first priced row per game **within the regime**, so a legacy row does not shadow an in-week one.
- Summary table: one row per model for `in_week`, then a clearly separated "Legacy (excluded from promotion)" block. The weekly CLV table and the promotion-rule columns use `in_week` only.

### 4. Guard statistic

- `log_v3_shadow.py:461-470` and `log_v4_shadow.py:498-507`: check `best_edge >= MIN_EDGE_THRESHOLD` first; below it the decision is `PASS` whatever the guard says. Match `log_v2_shadow.py:355-363`.
- `compare_models.py:446` and `:482`: count a `SUPPRESS_DANGER` row only if its edge is at least `MIN_EDGE_THRESHOLD`, so existing rows are graded correctly too.

### 5. Price-aware CLV

New function in `src/compare_models.py`, alongside `clv_win_probability`:

```python
def clv_fair_probability(pick_side, pick_line, open_home_odds, open_away_odds,
                         close_home_line, close_home_odds, close_away_odds,
                         total_line=None, sigma=CLV_SIGMA) -> float | None
```

Definition: fair probability of the ticket at the close, minus its fair probability when taken.

1. `q_open` = de-vigged probability of the pick side from the opening pair.
2. `q_close_home` = de-vigged home probability from the closing pair at `close_home_line`.
3. `m*` = `market_implied_mean(close_home_line, q_close_home, sigma, total_line)` (below).
4. `p_ticket` = no-push cover probability of the pick side at `pick_line` under mean `m*`.
5. Return `p_ticket - q_open`.

Properties to test: zero when line and prices are unchanged; equals `q_close - q_open` when only the price moved; positive when the line moved toward the pick.

Return `None` when any closing price is missing; callers then fall back to the existing line-only number. Use `closing_line` with `closing_odds`/`closing_away_odds` (single book). The consensus line has no price, so the price-aware number uses the single-book close and the line-only number keeps using consensus. Say so in the README.

New helper in `src/key_numbers.py`:

```python
def market_implied_mean(home_line, fair_home, sigma, total_line=None) -> float
```

Bisection on the mean so that `calculate_spread_probabilities(mean, sigma, home_line, total_line)["cond_home_win"] == fair_home`. The function is monotone in the mean. Bracket `[-home_line - 20, -home_line + 20]`, 60 iterations. Reference values at sigma 12.8145:

| home_line | fair_home | mean |
|---|---|---|
| −3 | 0.5000 | 2.791 |
| −3 | 0.4566 | 1.483 |
| −2.5 | 0.5000 | 1.465 |
| −7.5 | 0.5000 | 8.625 |
| +7.5 | 0.5000 | −8.625 |
| −10 | 0.5000 | 10.578 |

R5 reuses this function; build it here.

### 6. Scorecard columns

- Add "Avg CLV Fair Prob" (price-aware) and its standard error. Keep "Avg CLV Win Prob" for continuity.
- Replace "CLV Beat %" semantics: show beat, tie and lose as shares of all bets with a close (three numbers), not beat over non-ties.
- `Master_Bets.clv_points` (`update_results.py:149-160`, `capture_closing.py:224-231`): leave the stored values, but document in the README that the scorecard number is authoritative. Do not rewrite settled rows.

### 7. README

- "Promoting a shadow model to live" (`README.md:37-46`): condition 2 and 4 use Avg CLV Fair Prob, `in_week` rows only. Add the honest note from the review: at the observed per-bet SD of about 5.5 points, 40 bets cannot separate +0.5% from zero. Ask the owner before changing the thresholds themselves.
- "Market blend": state that under the current design v3 and v4 cannot reach a 2% edge at standard prices.
- Note that `Master_Bets.edge` is on the no-vig scale for rows written before 2026-11-02 and on the break-even scale after.

## Tests (write first)

- `tests/test_sheet_utils.py`: `ensure_tab_headers` for empty, prefix, identical and conflicting header rows; after a prefix migration the header row has no blanks or duplicates.
- `tests/test_key_numbers.py`: `market_implied_mean` reproduces the table above to 0.01 and round-trips to `fair_home` within 1e-6.
- `tests/test_compare_clv.py`: the three properties of `clv_fair_probability`; `None` when closing odds are missing.
- `tests/test_scorecard.py`: a legacy v3 pick and an in-week v3 pass on different games are reported in different blocks; a `SUPPRESS_DANGER` row with edge 0.0153 is not counted; mixed `logged_at_utc` formats parse.
- A logger-level test that `SUPPRESS_DANGER` is only assigned at or above the threshold.

## Done when

- `pytest tests/` green.
- After the next live scorecard refresh: v2's 6 and v3's 4 old bets appear only in the legacy block; the guard count for the old rows is 3, not 6 (TEN@CIN, NYJ@KC, ARI@SEA remain; NE@MIA, JAX@BAL and DAL@IND drop out); the in-week block is empty or holds only post-R1 rows.
- All three shadow tabs have `artifact_id` and `decision_regime` headers and the loggers still run (`get_all_records()` does not raise).

## Out of scope

- Raw-lean columns and the lean scorecard section (R3).
- Changing promotion thresholds.
- Recomputing historical `edge` or `clv_points` cells.
