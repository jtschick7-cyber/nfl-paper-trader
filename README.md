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

`beta` is fit on walk-forward predictions and clipped to [0, 1]. The current artifacts have beta ≈ 0.04 (v3 0.037, v4 0.042, standard error ≈ 0.07), so v3 and v4 stay within a fraction of a point of the line and PASS on almost every game. That is the expected, honest outcome: the models are not yet showing an edge the market misses.

### `no_edge` calibrator (`src/model_v2_core.py`)
The logistic calibrator is fit on out-of-sample cover outcomes. If its slope comes out zero or negative (raw probabilities carry no or inverted information), the model is flagged `no_edge` instead of being given a forced positive slope; every probability is then 0.5 and the model makes no picks.

### Edge and staking
Edge is the calibrated cover probability minus the break-even probability of the price actually taken (vig included), so a positive edge means positive expected value. Picks need `MIN_EDGE_THRESHOLD` (2%); edges above `MAX_AUTO_EDGE` (10%) are sent to manual review rather than auto-bet. Key-number push weights (3, 7, 6, 10, …) are in `src/key_numbers.py`.

### 🛡️ Pre-bet safeguards (`src/injury_guard.py`)
- **Reverse line movement**: lines moving ≥ 1.5 points against the pick side are suppressed and flagged `RLM_ALERT` at close.
- **Model–market divergence**: a model margin more than 7 points from the line is treated as an unmodeled injury or weather shock and not auto-bet.

---

## 📊 Google Sheet

1. **`Master_Bets`**: the live ledger. Opening lines, v1 picks, closing lines, scores, CLV and units.
2. **`V2_Shadow`**, **`V3_Shadow`**, **`V4_Shadow`**: shadow evaluations for each model.
3. **`Model_Scorecard`**: side-by-side record, units, ROI, CLV beat/tie rates, average CLV in points and in win probability, plus the active-slate head-to-head matrix. Each model is graded on its **first** logged decision per game.

### Closing line and CLV columns (`Master_Bets`)
| Column | Meaning |
| :--- | :--- |
| `closing_line` | Home closing spread at the primary book (DraftKings, then FanDuel, then the next book in `BOOK_PRIORITY`) |
| `closing_odds` / `closing_away_odds` | Closing price on the home and away side at that book |
| `closing_consensus_line` | Median home closing spread across every book in the Odds API response |
| `closing_captured_at` | Timestamp of the market update that was captured |

The scorecard's CLV uses `closing_consensus_line` when present (one book's number is noisy) and falls back to `closing_line`. CLV is reported in points and as the change in cover probability using key-number push weights, so moving off 3 counts for more than moving between dead numbers. New columns are created by `src/ensure_away_odds_columns.py` during the Tuesday run.

---

## ⚙️ GitHub Actions workflows

| Workflow | When (UTC) | What |
| :--- | :--- | :--- |
| `tuesday_opening.yml` | Tue 15:00, 18:00, 23:00; Wed/Thu 18:00 | Schedule, opening lines, v1 picks, v2–v4 shadow logs, scorecard |
| `closing_line_monitor.yml` | Gameday cron (Wed–Mon) | **Self-pacing closing loop** (below), then scorecard if settled bets got a close |
| `nightly_results.yml` | Daily 05:00 | Catch up missed closes from cached snapshots, settle results, scorecard |
| `v2_shadow_logger.yml`, `v3_shadow_logger.yml` | Manual | Re-run the shadow loggers |
| `ci_tests.yml` | Push / PR to `main` | `pytest tests/` |

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
