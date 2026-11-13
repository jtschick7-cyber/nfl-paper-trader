# Shadow models aren't betting: findings and fix plan (2026-11-13)

## Repo state
- On `main`, in sync with `origin/main`. Head is `b555cc8`.
- Not committed: `scripts/read_sheet.py`, a read-only Sheet reader.
  - It uses the view-only service account `sheet-reader-local`, whose key is at `~/.config/nfl-paper-trader/sheet-reader.json`. `.env` holds `SPREADSHEET_ID` and `GSHEET_KEY_FILE`.
  - Run `.venv/bin/python scripts/read_sheet.py <TabName>`.

## Scorecard as of 2026-11-13
| Model | Bets | Record | CLV Beat % | Avg CLV Win Prob |
|---|---|---|---|---|
| v1 | 18 | 7–11 | 61.5% | −0.52% |
| v2 | 6 | 2–4 | 25.0% | −1.18% |
| v3 | 4 | 3–1 | 0.0% | −2.94% |
| v4 | 0 | — | — | — |

v2, v3 and v4 are set to PASS on every week 10 and week 11 game. At this rate none of them will reach the promotion rule's sample of 40 bets over 6 weeks.

## What's wrong

1. **The edge bar now includes the vig, so the shadow models can't reach it.** Commit `bf3e4aa` (Nov 2) changed the measure: edge is now model probability minus the break-even price at the odds taken (about 52.4% at -110), instead of minus the no-vig fair price. That's correct, but every edge dropped by about 2.4 points. The bet bar is still `MIN_EDGE_THRESHOLD = 0.02` in `src/config.py`, so a bet needs a cover probability of at least 54.4%.
   - Average edge for v2 went from +1.0% (weeks 8–9) to −1.2% (weeks 10–11). For v3 it went from +1.3% to −1.3%. The best edge any game reached in weeks 10–11 was +0.5%.
2. **v3 and v4 barely move off the market line.** Commit `97761e1` blends each prediction as `mu = market + β·(model − market)`. β was fit on 2012–2025 games and came out at about 0.04 ± 0.07 (`models/metadata_v3.json`, `models/metadata_v4.json`), so each prediction is about 96% the market's number. v4's margin averages 0.09 points from the line, so it can essentially never clear the bar. The calibrators for both models also report `no_edge`.
3. **v1 is measured on the easier scale.** v1 still computes edge against the no-vig price (`implied_probability_no_vig` in `Master_Bets`). It claims 5–9% edges on almost every game, but its average CLV is about zero. So it's overconfident, and the four models aren't being compared on the same scale.

Problems 1 and 2 are deliberate, honest changes, not bugs. The trouble is that the models no longer produce any bets, so they can't be evaluated.

## Fix plan

1. **Track raw-lean CLV for v2, v3 and v4.** This is for evaluation only and doesn't change the real picks.
   - For every priced game, log each model's raw lean: the side it favors before the market blend and with no edge threshold. Record its size (model margin minus market margin, in points).
   - Grade these leans against the close with the same CLV win-probability math the scorecard already uses.
   - Add a "Raw Lean CLV" section to `Model_Scorecard` showing leans, CLV Beat %, Avg CLV Win Prob, and the same split for leans of 1.5 points or more.
   - This gives about 13–14 graded leans per model per week instead of zero. If the closing line keeps moving toward a model's lean, its features carry information even though they aren't strong enough to bet on.
   - The files involved are `src/log_v2_shadow.py`, `src/log_v3_shadow.py`, `src/log_v4_shadow.py` and `src/compare_models.py`. Add new columns at the end of each shadow tab so existing rows keep their positions.
2. **Measure v1's edge the same way (after the vig).** Recompute the v1 edge against the break-even price at the odds taken, in the pick generator and in `Master_Bets` from now on. Decide whether v1 keeps betting at 2% under the stricter measure, or log the stricter edge in a new column first and change the threshold later.
3. **Leave the β blend and the 2% threshold alone for real picks.** Lowering the bar just to get bets would undo the honest fix.
4. **Tests:** add unit tests for computing the raw lean and grading its CLV, then run `pytest tests/`.
5. **Commit:** commit `scripts/read_sheet.py` along with this work, on a `claude/` branch with a PR.

## Decision point (around week 14–15)
- If v3 or v4 raw leans show positive CLV over 40 or more games, look at raising β, or bet only the larger leans.
- If every model, v1 included, is still at or below zero on CLV, take that as "no edge found" and rethink the features or the betting timing, for example betting at open versus near close.
