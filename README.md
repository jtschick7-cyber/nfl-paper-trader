# 🏈 NFL Paper Trader & Multi-Model Forecasting Pipeline

A free paper-trading pipeline that prices NFL point spreads against live sportsbook odds, records opening and closing lines, measures closing line value (CLV), and compares several models side by side in Google Sheets. Everything runs on GitHub Actions (free for a public repo), the Google Sheets API and the free tier of The Odds API.

---

## 📌 Models

One model places paper bets; the others are logged in shadow tabs and graded on the same games so they can be compared before any of them is promoted.

| Model | Features | Margin & probability | Tab |
| :--- | :--- | :--- | :--- |
| **v1 (live)** | Rolling points scored/allowed, margin, rest | Ridge margin → key-number cover probability (uncalibrated) | `Master_Bets` |
| **v2 (shadow)** | Elo ratings, rest differential, divisional flag | Ridge margin → key-number cover probability → logistic calibration | `V2_Shadow` |
| **v3 (shadow)** | v2 + `nflverse` pass/rush EPA, success rate, early-down EPA | Constrained ridge → **market blend** → key numbers → calibration | `V3_Shadow` |
| **v4 (shadow)** | v3 + point-in-time QB starter shocks and turnover-luck regression | Constrained ridge → **market blend** → key numbers → calibration | `V4_Shadow` |

v3 and v4 are trained on 2012–2025 (`src/train_model_v3.py`, `src/train_model_v4.py`).

### Market blend (`src/market_blend.py`)
The closing spread predicts margins better than any of the box-score models, so v3/v4 treat the market as the prior and move off it only as far as the model's disagreement has predicted ATS results out of sample:

`mu_blend = market_mu + beta * (model_mu - market_mu)`, with `market_mu = -home_line`.

`beta` is fit on walk-forward predictions and clipped to [0, 1]. The current artifacts have beta ≈ 0.04 (v3 0.037, v4 0.042, standard error ≈ 0.07), so v3 and v4 stay within a fraction of a point of the line and PASS on almost every game. That is the expected, honest outcome: the models are not yet showing an edge the market misses. Under the current design v3 and v4 **cannot reach a 2% edge at standard prices** (−110 both sides): a blend this close to the line cannot move the cover probability to the 54.4% that −110 needs for a 2% edge. A pick can still appear on a plus price, with no model input behind it (see `scripts/review_repro/anchor_and_price_checks.py`).

### `no_edge` calibrator (`src/model_v2_core.py`)
The logistic calibrator is fit on out-of-sample cover outcomes. If its slope comes out zero or negative (raw probabilities carry no or inverted information), the model is flagged `no_edge` instead of being given a forced positive slope; every probability is then 0.5 and the model makes no picks.

### Edge and staking
Edge is the calibrated cover probability minus the break-even probability of the price actually taken (vig included), so a positive edge means positive expected value. `Master_Bets.edge` is on that break-even scale only for rows written from 2026-11-02; older rows were written against the no-vig probability and are not recomputed. Picks need `MIN_EDGE_THRESHOLD` (2%); edges above `MAX_AUTO_EDGE` (10%) are sent to manual review rather than auto-bet. Key-number push weights (3, 7, 6, 10, …) are in `src/key_numbers.py`.

v1 bets flat stakes on at most `MAX_PICKS` (5) picks **per week**: each run may add only `MAX_PICKS` minus the picks already recorded for that week, highest edge first.

### Decision window (`src/decision_window.py`)
A game is decided only in the run where it is **decidable**: its week is the smallest week that still has a game kicking off after now, it kicks off within 7.5 days, and every current-season game that kicked off more than 5 hours ago has a final score. If a score is missing, the run captures nothing and picks nothing and a later run retries. Opening capture (on an Odds API response no more than 30 minutes old), v1 picks and the v2–v4 shadow evaluations all happen in that run. v1 picks a game only if its opening line was captured within `PICK_MAX_LINE_AGE_MINUTES` (90); an older line means the decision moment has passed. Each capture records `opening_book`, `opening_quote_at` and `decision_regime = in_week`; each pick records `pick_made_at`. Rest days count from each team's latest scheduled game, played or not. Rows with a blank `decision_regime` (and rows marked `lookahead`) were decided under the old timing, against openers up to ~10 days old, and are reported separately.

### 🛡️ Pre-bet safeguards (`src/injury_guard.py`)
- **Reverse line movement**: lines moving ≥ 1.5 points against the pick side are suppressed and flagged `RLM_ALERT` at close.
- **Model–market divergence**: a model margin more than 7 points from the line is treated as an unmodeled injury or weather shock and not auto-bet.

### Promoting a shadow model to live
A shadow model (v2, v3 or v4) replaces v1 only if all of these hold at a weekly checkpoint (after the Tuesday scorecard):
1. **Sample:** at least **40 graded bets with a recorded close**, across at least **6 different weeks**.
2. **Beats the close:** **Avg CLV Fair Prob ≥ +0.5%** over all those bets.
3. **Holds up week to week:** Avg CLV Win Prob is positive in **at least 60% of the weeks** it bet.
4. **Beats v1:** its Avg CLV Fair Prob is at least **0.5 points higher than v1's** over the same weeks.
5. **Profit is reported but not required.** At 40 bets it's mostly noise.
6. **Timing:** if two models qualify, the one with the higher Avg CLV Win Prob wins. The switch happens at the next Tuesday run, never mid-week, and the date and numbers get recorded here.

Only the scorecard's **in-week** block counts: v1 bets with `decision_regime = in_week`, and shadow rows logged from 2026-11-03 00:00 UTC for games decided in-week. Look-ahead v1 picks and older shadow rows are shown in a separate **Legacy (excluded from promotion)** block.

**Avg CLV Fair Prob** (price-aware) is the fair probability of the ticket at the close minus its fair probability when taken. The closing home line and its de-vigged price at the primary book fix the market's mean margin (`market_implied_mean` in `src/key_numbers.py`); the ticket's no-push cover probability under that mean is compared with the de-vigged probability of the price taken. It uses the single-book close (`closing_line`, `closing_odds`, `closing_away_odds`) because the consensus line has no price; a bet without closing prices falls back to the line-only number. The scorecard shows its standard error next to it.

**Avg CLV Win Prob** (line only) is the average change in cover probability between the line taken and the closing line, ignoring price; it is kept for continuity. For condition 3, the scorecard's **Weekly Avg CLV Win Prob** table shows each model's average by week and its positive weeks (`Weeks +`); only weeks with at least one bet that has a recorded close count.

**How much 40 bets can tell you.** The per-bet CLV has a standard deviation of about 5.5 points, so the standard error of a 40-bet average is about 0.9 points: 40 bets cannot separate a true +0.5% from zero. Read conditions 2 and 4 together with the SE column; the thresholds themselves are the owner's call and have not been changed.

---

## 📊 Google Sheet

1. **`Master_Bets`**: the live ledger. Opening lines, v1 picks, closing lines, scores, CLV and units.
2. **`V2_Shadow`**, **`V3_Shadow`**, **`V4_Shadow`**: shadow evaluations for each model.
3. **`Model_Scorecard`**: side-by-side record, units, ROI, CLV beat/tie/lose shares (each over all bets with a close, so they sum to 100%), average CLV in points, in win probability and price-aware (with its standard error), plus the active-slate head-to-head matrix. An in-week block (the promotion sample) comes first and a legacy block below it. Each model is graded on its **first** logged decision per game within each block. "Guard suppressed" counts only `SUPPRESS_DANGER` rows whose edge cleared `MIN_EDGE_THRESHOLD`; the loggers now record `PASS` below the threshold whatever the guard says.

Shadow tabs end with `artifact_id` (training span plus the first 8 hex characters of the `.pkl` sha256, e.g. `v3_2012-2025_1a2b3c4d`) and `decision_regime`. Loggers add missing header cells with `ensure_tab_headers` (`src/sheet_utils.py`), which appends to an existing header row but never rewrites it and stops if the row is not a prefix of the expected headers.

### Closing line and CLV columns (`Master_Bets`)
| Column | Meaning |
| :--- | :--- |
| `closing_line` | Home closing spread at the primary book (DraftKings, then FanDuel, then the next book in `BOOK_PRIORITY`) |
| `closing_odds` / `closing_away_odds` | Closing price on the home and away side at that book |
| `closing_consensus_line` | Median home closing spread across every book in the Odds API response |
| `closing_captured_at` | Timestamp of the market update that was captured |

The scorecard's CLV uses `closing_consensus_line` when present (one book's number is noisy) and falls back to `closing_line`. CLV is reported in points and as the change in cover probability using key-number push weights, so moving off 3 counts for more than moving between dead numbers. New columns are created by `src/ensure_away_odds_columns.py` during the Tuesday run.

`Master_Bets.clv_points` is written at close and settle time and is not recomputed; where it differs from the scorecard, the scorecard is authoritative.

---

## ⚙️ GitHub Actions workflows

| Workflow | When (UTC) | What |
| :--- | :--- | :--- |
| `tuesday_opening.yml` | Tue 15:17, 18:17, 23:17; Wed/Thu 18:17 | Schedule, then for the current decision week only (see **Decision window**): opening lines, v1 picks, v2–v4 shadow logs; scorecard. Later runs in the week only act if an earlier one stopped for missing results |
| `closing_line_monitor.yml` | Gameday cron (Wed–Mon) | **Self-pacing closing loop** (below), then scorecard if settled bets got a close |
| `nightly_results.yml` | Daily 05:17 | Catch up missed closes from cached snapshots, settle results, scorecard |
| `v2_shadow_logger.yml`, `v3_shadow_logger.yml` | Manual | Re-run the shadow loggers |
| `migrate_lookahead_openers.yml` | Manual, one-off | R1 migration: move look-ahead openers for a later week out of the opening columns (dry run unless `apply` is ticked) |
| `ci_tests.yml` | Push / PR to `main` | `pytest tests/` |

### On-time starts (`external_trigger/`)
GitHub's `schedule:` triggers start hours late and skip firings, so a Cloudflare Worker calls `workflow_dispatch` for the nightly, Tuesday and closing workflows at fixed UTC times. The GitHub crons stay as a backup; duplicate starts are safe. Setup and schedule: [`external_trigger/README.md`](external_trigger/README.md).

### Self-pacing closing loop (`src/closing_loop.py`)
GitHub drops most dense cron triggers on busy days, so the closing monitor does not rely on a fixed cadence. Whichever cron run fires first stays alive (up to `CLOSING_LOOP_MAX_MINUTES`, default 320) and sleeps until fixed leads before each kickoff (`CLOSING_WAKE_LEADS`, default 45 and 10 minutes), capturing then. `src/capture_closing.py` captures any missing close 240–2 minutes before kickoff and upgrades an early capture once inside the last hour, after inactives. Games missed entirely are recovered by the nightly run from pre-kickoff snapshots in the odds cache.

### Concurrency
Sheet writes address rows by position, so the short writers (Tuesday, nightly, v2/v3 loggers and the closing workflow's scorecard job) share the `sheet-writer` concurrency group and never overlap. The hours-long closing loop keeps its own group: GitHub keeps only one pending run per group and cancels it when a newer one queues, so a loop inside `sheet-writer` would let the next closing cron cancel a Tuesday or nightly run waiting behind it. Instead, before writing, the loop re-reads the `game_id` column and drops any update whose row no longer holds the game it read.

### Durable odds snapshots
Every Odds API fetch is saved as a timestamped snapshot in `data/odds_cache/snapshots/`, which lives only in the Actions cache (pruned to 72 hours, evicted after 7 idle days). The Tuesday and closing workflows run `scripts/push_odds_snapshots.sh`, which gzips new snapshots into the orphan **`odds-data`** branch under `snapshots/YYYY/MM/`. `main` never carries them. To read them:

```bash
git fetch origin odds-data && git worktree add ../odds-data origin/odds-data
```

---

## 🛠️ Local development

Python 3.12 (dependencies are pinned in `requirements.txt`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

Useful commands:

```bash
python -m src.build_features          # refresh EPA feature caches
python -m src.train_model_v4          # retrain v4 (constrained ridge + market blend + calibrator)
python -m src.backtest_epa --test-seasons 2024 2025   # walk-forward backtest
python -m src.compare_models          # rebuild the scorecard
```

Backtests are walk-forward with no lookahead. Small high-edge subsets from past backtests were too small to mean anything and are no longer quoted here; judge the models on the live scorecard's CLV, which needs far fewer bets than win rate to separate skill from noise.

---

## 🔒 Cost and secrets
- **Free**: The Odds API free tier (500 calls/month) with a 12-hour cache for openers and a short cache for closes; GitHub Actions minutes are free for public repos; Google Sheets API is free.
- **No secrets in code**: `GSHEET_CREDENTIALS`, `SPREADSHEET_ID` and `ODDS_API_KEY` come from GitHub Actions secrets.
- `nfl_data_py` is archived upstream in favour of `nflreadpy`; it is pinned at 0.3.2 (the last release that allows pandas 2) until a migration.
