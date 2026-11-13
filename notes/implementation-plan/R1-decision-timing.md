# R1. Fix decision timing

**Review sections:** M1, M2, M3, defects a1, a2, a3, a6. **Effort:** S. **Branch:** `claude/r1-decision-timing`.

## Problem

1. `src/capture_opening.py:68-100` captures an "opening" line for every future `PENDING` game the Odds API returns, including next week's look-ahead lines. Median lead is 9.4 days.
2. `src/generate_picks.py:248` caps picks per run, not per week, and `:126` prices every pick at `opening_line` however old it is. Later runs add picks at last week's line using this week's results. There is no pick timestamp.
3. Rest days are computed from each team's last completed game as of now, which is the wrong game when a week is still to be played.
4. `fetch_odds_api()` defaults to a 12-hour cache (`src/odds_utils.py:122`), and `opening_captured_at` is stamped with the run time, not the quote time. The book is not recorded.

## Target behaviour

A game is **decidable** in a run only when all three hold:

- **Current week:** its `week` equals the smallest week that still has a game kicking off after now.
- **Horizon:** kickoff is between now and now + 7.5 days (a backstop against bad week data).
- **Results are in:** every current-season game that kicked off more than 5 hours ago has a final score in the schedule data. If not, the run captures nothing and picks nothing, prints which games are missing, and exits 0. A later run retries.

The week rule is deliberately not a pure day count: runs start hours late, and a late Thursday run is less than 7 days from next week's Thursday game.

For decidable games:

- Opening capture, v1 picks and all three shadow evaluations happen in the same run, on an Odds API response fetched no more than 30 minutes ago.
- v1 picks a game only if its opening line was captured within the last 90 minutes. An older line means the decision moment has passed: no pick.
- `MAX_PICKS` is a weekly cap: allowed this run = `MAX_PICKS` minus picks already recorded for that week.

## Changes

### New: `src/decision_window.py` (pure functions, no I/O)

```python
def current_decision_week(games, now_utc) -> int | None
    # games: iterable of (week, kickoff_utc). Smallest week with a kickoff > now.

def is_decidable(week, kickoff_utc, now_utc, decision_week, max_days=7.5) -> bool

def missing_results(schedule_df, now_utc, grace_hours=5.0) -> list[str]
    # game_ids with kickoff < now - grace and no home_score/away_score.

def previous_game_day(schedule_df, team, kickoff_utc)
    # Eastern calendar day of the team's latest SCHEDULED game with kickoff < kickoff_utc
    # (played or not), same season. None if there is none.
```

`schedule_df` is what `src.model_utils._load_sched([SEASON])` returns (all scheduled games, `kickoff_utc` already built).

### `src/capture_opening.py`

- Compute `decision_week` from the sheet rows; skip rows that are not decidable.
- Return early with a message when `missing_results(...)` is non-empty.
- Call `fetch_odds_api(max_age_seconds=1800)`. Skip a game whose `market_last_update` is later than its kickoff (a live market), as `capture_closing.py:454-457` does. Do not skip on an old `market_last_update` alone: if that field means "last changed" rather than "last checked", a quiet line would never be captured.
- Write three more cells per captured game: `opening_book` (`odds_row["bookmaker"]`), `opening_quote_at` (`odds_row["market_last_update"]`), `decision_regime` = `in_week`.
- The existing branch that fills `total_line` on `OPEN_CAPTURED` rows stays, restricted to decidable rows.

Quota: this adds at most a handful of Odds API calls a week. The closing loop is the main consumer of the 500-a-month free tier; check the `x-requests-remaining` value the fetch prints after the first run.

### `src/generate_picks.py`

- Same decidable and results guards.
- Skip a candidate when `now - opening_captured_at` > `PICK_MAX_LINE_AGE_MINUTES` (env, default 90). Count and print the skips.
- Weekly cap as above (replace `picks = scored[:MAX_PICKS]`).
- Write `pick_made_at` (UTC ISO) with each pick.
- Replace the rest calculation at `:146-152` with `previous_game_day`. Keep v1's convention: integer Eastern calendar days, skip the game if either team has no previous game this season.

### Shadow loggers: `src/log_v2_shadow.py`, `src/log_v3_shadow.py`, `src/log_v4_shadow.py`

- Add the decidable and results guards to each row loop. No new columns here (R2 and R3 add them).
- Rest: use `previous_game_day` at `model_v2_core.py:415` (inside `compute_live_features_from_state`; pass the schedule in or compute rest in the caller), `log_v3_shadow.py:224-229`, `log_v4_shadow.py:185-190`. Keep each model's training convention:
  - v2: raw day difference, no cap.
  - v3 and v4: `min(days, 21)`, default 7 when there is no previous game. Note that training carries the previous game across seasons; the 21-day cap makes that equivalent in practice, so same-season lookup with the default is fine. State this in a comment.

### `src/ensure_away_odds_columns.py`

Add to `EXTRA_COLUMNS`: `opening_book`, `opening_quote_at`, `pick_made_at`, `decision_regime`.

`src/sheet_utils.py:get_sheet_as_df` needs no change (the new columns are strings).

### New: `src/migrate_lookahead_openers.py` (one-off, dry-run by default)

Only needed if look-ahead openers for a week later than the current decision week exist when R1 deploys (true for week 12 if R1 lands after Tue 2026-11-17 15:17 UTC).

For rows with status `OPEN_CAPTURED`, kickoff in the future, `week` > current decision week:

- **No pick on the row:** copy `opening_line`, `opening_odds`, `opening_away_odds`, `opening_captured_at` into `lookahead_line`, `lookahead_odds`, `lookahead_away_odds`, `lookahead_captured_at` (create via `ensure_columns`), blank the four opening cells, set status `PENDING`. The next in-week run re-captures them.
- **Pick on the row:** leave the row alone and set `decision_regime` = `lookahead`.

Print the planned changes; write only with `--apply`. It must run inside the `sheet-writer` concurrency group, so run it as a temporary `workflow_dispatch` step or by hand while no workflow is running, not from a laptop without write credentials.

Rows with a blank `decision_regime` are legacy by definition; no backfill is required.

## What happens to weeks 10 and 11

Nothing is rewritten. Their openers are already captured and their picks already made, so under the new rules no further picks are added to them (the 90-minute rule) and they stay legacy. Week 12 is the first `in_week` week if R1 is live by the deadline above.

## Tests (write first)

New `tests/test_decision_window.py`:

- `current_decision_week`: Tuesday after MNF returns next week; Thursday before and after the Thursday kickoff returns this week; a late Thursday run 6.97 days before next Thursday's game still returns this week.
- `is_decidable`: next week's game is false even inside 7.5 days; this week's Monday game is true on Tuesday.
- `missing_results`: a game 6 hours old with no score is reported; one 3 hours old is not.
- `previous_game_day`: uses the scheduled previous game even when it has no score yet; a bye gives 14 days; a Thursday-to-Sunday turnaround gives the right count. Include the two cases from the review: HOU@LAC week 9 must give (7, 14) and BUF@MIN week 9 (8, 8), not (21, 14) and (15, 22).

Add `tests/test_pick_timing.py`. The existing suite tests pure row-level functions and does not mock gspread, and `capture_opening()` and `generate_picks.main()` are single I/O functions today. So first pull the selection logic out into pure functions that take the sheet DataFrame, the parsed spreads, the schedule and `now`, and return the cell updates (the way `row_needs_closing` and `build_closing_row_updates` are split out in `src/capture_closing.py`). Test those:

- `capture_opening` writes nothing for a game in a later week.
- `capture_opening` writes nothing when a result is missing.
- `generate_picks` skips a candidate whose opening is 91 minutes old.
- Weekly cap: with 3 picks already on the week and `MAX_PICKS=5`, at most 2 are added.
- Every pick write includes `pick_made_at`.

## Done when

- `pytest tests/` is green with the new tests.
- `scripts/review_repro/v1_pick_timing.py` on a fresh dump, for the first in-week week, shows every pick's earliest match at or after `opening_captured_at` with `prior_wk_games_already_played` equal to the full prior-week count **and** the line captured after those games. (After R1 it is simpler to assert `pick_made_at - opening_captured_at` < 90 minutes for every `in_week` pick.)
- After the first live Tuesday run: no row in a later week has an opening line; every new pick has `pick_made_at`, `opening_book`, `opening_quote_at`, `decision_regime = in_week`.
- `README.md` workflow table and the "Edge and staking" paragraph describe the decision window and the weekly cap.

## Out of scope

- Capturing look-ahead openers into their own columns as an ongoing series.
- Re-pricing a stale opener at the current quote.
- Any change to the v1 model, thresholds or the closing-line capture.
