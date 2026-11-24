# P3. Offline NFL totals kill test

**Source:** owner's roadmap brief of 2026-11-24 (Step 2), with the changes agreed in review the same day. **Effort:** S–M (about one session). **Branch:** `claude/p3-totals-kill-test`. **Depends on:** `src/total_pricing.py` as specified in [P2](P2-topdown-v2.md). If P3 is built first, create that module exactly as P2 specifies, with its tests.

Offline only. No network, no Sheet, no workflow, no artifact.

## Question

Does a bottom-up totals model built from public play-by-play and weather carry any information about the final total beyond the closing total and its price, in 2020–2025? If not, no live totals pipeline gets built.

This is a **kill test**: cheap, one shot, rules frozen here. The prior is low:

- the market prices pace, efficiency and wind heavily;
- the spread models' incremental β turned negative after 2019.

Do not peek. Nobody computes the outcome metrics below before the code and tests are committed. The spec-time checks were only the data's shape: totals residual SD and the count of high-wind games.

## Facts (checked 2026-11-24)

- **`games.csv`:**
  - `total_line`, `over_odds` and `under_odds` are present for 95% of 2020–2025 games. These are nflverse closing lines from one source, not a multi-book close.
  - `roof` (`dome`, `closed`, `open`, `outdoors`), `wind` and `temp` are also present. `wind` is known for 87% of 2020–2025 outdoor games.
  - **Weather is recorded at game time, not forecast.** That flatters any weather effect; a live system would only have a forecast.
- **Sample:** 1,155 played outdoor or open-roof regular-season games in 2020–2025. Of these, **121** had wind ≥ 15 mph (26, 30, 14, 14, 16 and 21 by season).
- **Residual SD:** SD of (total − closing `total_line`) over 2012–2025 is 13.24 (n = 3,663). For 121 games that gives an SE of about 1.2 points: the wind subset can only detect a market bias of about 2.5 points or more.
- **PBP:** `data/epa_cache/pbp_2010.parquet` to `pbp_2025.parquet` hold `pass_oe`, `xpass`, `wp`, `qtr`, `game_seconds_remaining`, `drive`, `posteam`, `epa`, `pass`, `rush`, `yardline_100` and `play_type`. `src/epa_features.parse_pbp_to_game_epa(pbp, play_split="dropback")` gives team-game offensive and defensive pass and rush EPA.

## Design

### Files

- `src/totals/__init__.py`
- `src/totals/features.py`: team-game stats from PBP, then leak-free game-level features.
- `src/totals/backtest_totals.py`: walk-forward fit, metrics, and the kill/keep verdict. CLI: `.venv/bin/python -u -m src.totals.backtest_totals`.
- **Output:** `data/totals_backtest_games.csv` (add to `.gitignore` if no pattern covers it) and `notes/p3-totals-kill-test.md` with the tables and the verdict.

### Team-game stats (per team per game, from PBP)

- `off_pass_epa`, `off_rush_epa`, `def_pass_epa`, `def_rush_epa` from `parse_pbp_to_game_epa(..., play_split="dropback")`.
- `neutral_sec_per_play`: mean seconds between consecutive offensive plays of the same `posteam` in the same `drive`, on plays in Q1–Q3 with `wp` in [0.20, 0.80].
- `neutral_pass_oe`: mean `pass_oe` on those neutral plays.
- `off_plays`: offensive plays.
- `rz_trips` and `rz_tds`: drives reaching `yardline_100 <= 20`, and those ending in a TD.
- `points_for` and `points_against` (from `games.csv`), as a baseline.

### Game features (frozen)

All rolling windows use only that team's games with kickoff strictly before the target kickoff.

- **Window:** last 8 games, carried across seasons, with a minimum of 3. With fewer than 3 prior games, the feature is the league mean of the training seasons.
- **Matchup sums** (home plus away, so they map to the total):
  - `pass_env = home_off_pass + away_def_pass + away_off_pass + home_def_pass` (EPA per play);
  - `rush_env`, the same for rush;
  - `pace = home_sec_per_play + away_sec_per_play`;
  - `pass_oe_sum`;
  - `plays_sum`;
  - `rz_td_rate_sum`, each team's rate shrunk to the league rate with 20 pseudo-trips: `(tds + 20·league) / (trips + 20)`;
  - `points_env = home_pf + away_pf + home_pa + away_pa` (the baseline feature).
- **Weather:**
  - `is_indoor` is true when `roof` is `dome` or `closed`;
  - `wind` is 0 when indoor; for an unknown outdoor wind, use the training-season median and set `wind_missing` to 1;
  - `high_wind` is `wind >= 15` (outdoor);
  - `cold` is `temp <= 32` (outdoor).
- **No market inputs** among the features. `total_line` and the prices are used only as the anchor.

### Walk-forward (frozen)

- **Seasons:** test seasons S = 2014…2025, regular season only. Train on 2012…S−1. 2010–2011 PBP only warms up the rolling windows.
- **Model:** StandardScaler plus Ridge on `actual_total`. Choose `alpha` from {1, 3, 10, 30, 100} by a 5-fold `TimeSeriesSplit` inside the training seasons only. One model; no gradient boosting (it is too easy to overfit on 3,600 games).
- **Anchor:** `t* = total_implied_mean(total_line, q_over, 13.2)`, where `q_over` comes from `devig_two_way(over_odds, under_odds)`. Games without a line are kept for training, dropped from evaluation.
- **β per fold, nested as in `backtest_production.fit_fold`:** a no-intercept OLS of `(actual − t*)` on `(raw_mu − t*)`, fitted on walk-forward predictions inside the training seasons. It is clipped to [0, 1] for the blend, and the unclipped value and SE are reported.
- **Blend:** `mu_blend = t* + β_fold·(raw_mu − t*)`, then `p_over = total_probabilities(mu_blend, total_line)`, push-excluded.

### Metrics (all out of sample, by period 2014–2019 and 2020–2025)

1. **Pooled β̂ ± SE:** a no-intercept OLS of `(actual − t*)` on `(raw_mu − t*)` over the test-season predictions. Report it for all games, for outdoor games with wind ≥ 15, and for indoor games.
2. **Over/under log loss**, push games excluded: model `p_over` against the market's `q_over`. Report Δ = model − market, with a paired SE over games. Negative means the model is better.
3. **Wind check without a model:** mean `(actual − t*)` ± SE on outdoor games with wind ≥ 15, and on games with wind < 15. If the market under-adjusts for wind, the first is negative.
4. **Descriptive only:** the record and units at the closing price for games where model `p_over` is 0.55 or more, or 0.45 or less.

### Kill/keep rule (frozen)

Decide on 2020–2025 only.

- **Keep**, meaning a follow-up spec for live tracking with forecast weather (owner decision), only if either:
  - pooled β̂ > 0 with t ≥ 1.5 **and** Δ log loss < 0; or
  - the wind ≥ 15 subset shows mean `(actual − t*)` with |t| ≥ 2.0 **and** Δ log loss < 0 on that subset.
- **Kill** otherwise. Write the tables and the verdict to `notes/p3-totals-kill-test.md` and stop. Do not try other feature sets, windows or models after seeing the result. That would turn a kill test into a search.
- Either way, note that game-time weather overstates what a forecast-based live system could get.

## Tests (write first) `tests/test_totals_features.py`, `tests/test_backtest_totals.py`

1. **Leak-free:** on a synthetic schedule, a game's features change when an earlier game's PBP changes and do not change when a later game's PBP changes.
2. `neutral_sec_per_play` on a synthetic drive. Plays outside Q1–Q3 or outside the `wp` range are excluded.
3. The red-zone shrink formula, including zero trips, which gives the league rate.
4. **Weather encoding:**
   - indoor gives wind 0 and `high_wind` 0;
   - an unknown outdoor wind gives the training median and `wind_missing` 1;
   - the median comes from training seasons only.
5. **Walk-forward:**
   - no test-season row in any fold's training data;
   - the `alpha` choice and β never see the test season;
   - on synthetic data with a known slope, β̂ recovers it.
6. The kill/keep function on synthetic summaries: each of the two keep paths, and kill.

The full suite stays green.

## Deliverables

- The code and tests above, plus `notes/p3-totals-kill-test.md` with:
  - the β, log-loss and wind tables by period;
  - the verdict;
  - one paragraph on the caveats: game-time weather, a single-source closing line, and the wind subset's power.
- `PROGRESS.md`: about 10 bullets. If killed, mark totals bottom-up as closed. Top-down totals in P2 are unaffected.
