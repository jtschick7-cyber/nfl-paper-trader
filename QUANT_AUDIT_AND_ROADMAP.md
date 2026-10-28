# NFL Paper-Trader: Quantitative, Data Engineering & Systems Architecture Audit

**Date:** October 28, 2026  
**Repository:** `nfl-paper-trader`  
**Scope:** End-to-end review of operational health, quantitative/mathematical integrity, CI/CD & pipeline technical debt, and Model v4 strategic roadmap.

---

## Executive Summary

An end-to-end audit of `nfl-paper-trader` uncovered **six critical (P0) bugs** currently affecting live shadow evaluations, probability calibration, and scorecard grading, alongside several high-leverage architectural and quantitative improvements:

1. **Hardcoded Zero Context Features in Model v3 Live Inference (`src/log_v3_shadow.py`)**: `elo_diff`, `diff_margin`, and `diff_rest`—which account for **80.8% of Model v3's standardized Ridge weights**—are hardcoded to `0.0` at inference time.
2. **Frozen EPA Feature Cache in CI (`src/log_v3_shadow.py` & `v3_shadow_logger.yml`)**: `data/team_game_epa.csv` is committed to Git (frozen at 2026 Week 7), and `log_v3_shadow.py` only runs `build_features()` if the file does not exist on disk.
3. **Inverted Platt Calibrator Slope (`-0.3194`) in `models/nfl_spread_model_v3.pkl` (`src/train_model_v3.py`, `src/backtest_epa.py`)**: Fitting `LogisticRegression(C=1.0)` in-sample on raw probabilities without log-odds transformation, side symmetry, or a monotonicity guard produced a **negative slope (`-0.3194`)**, causing higher raw cover probabilities to map to *lower* calibrated probabilities.
4. **Dead `diff_early_down_epa` Feature (`src/epa_features.py`)**: `compute_rolling_epa_snapshot` and `get_matchup_epa_features` omit `off_early_down_epa` and `def_early_down_epa`, causing `diff_early_down_epa` to evaluate to `0.0` across training, backtesting, and production.
5. **High-Conviction v3 Picks Excluded from Scorecard (`src/compare_models.py` vs. `src/log_v3_shadow.py`)**: `log_v3_shadow.py` labels picks with `edge >= 3.5%` as `"HIGH_CONVICTION"`, but `compare_models.py` filters strictly on `dec == "SHADOW_PICK"`, dropping all high-conviction plays from `v3_bets`, `v3_high_bets`, and the Active Slate Matrix.
6. **Conditional-vs-Unconditional Probability Mismatch in Kelly Staking (`src/compare_models.py` & `src/staking.py`)**: `v3_model_probability` is a conditional non-push cover probability $P(\text{Win} \mid \text{No Push})$, but `compare_models.py` passes it directly into `compute_push_aware_stake` alongside `p_push` without scaling by $(1 - p_{\text{push}})$, subtracting `p_push` solely from `loss_prob` and inflating stake sizes on integer spreads.

---

## 1. Codebase & Operational Health Audit

### 1.1 Critical Production Bugs

#### Bug 1: Hardcoded `0.0` Context Features in `src/log_v3_shadow.py` (Lines 243–246)
In `src/log_v3_shadow.py`:
```python
diff_rest = 0.0   # Default rest differential
diff_margin = 0.0
elo_diff = 0.0    # Model default
```
Per `models/metadata_v3.json`, the standardized Ridge coefficients are:
- `elo_diff`: `+3.8576`
- `diff_margin`: `+1.1406`
- `diff_rest`: `+0.7227`
- `diff_success_rate`: `+0.7940`
- `diff_pass_epa`: `+0.5598`
- `diff_rush_epa`: `+0.0071`
- `diff_early_down_epa`: `0.0000`
- `division`: `-0.0708`

Hardcoding `elo_diff`, `diff_margin`, and `diff_rest` to `0.0` zeroes out **80.8% of the model's linear signal** and introduces standardization bias via `scaler.transform()` wherever the training feature mean is non-zero.

**Remediation:** Compute live point-in-time Elo, rolling 4-game scoring margin, and rest days from historical schedules (1999–present or 2023–present) prior to `kickoff` in `src/log_v3_shadow.py`, matching the training pipeline.

---

#### Bug 2: Static EPA CSV Cache Never Refreshes in CI (`src/log_v3_shadow.py` Lines 137–140)
```python
if not os.path.exists(EPA_DATA_PATH):
    print(f"EPA data missing at {EPA_DATA_PATH}. Building on the fly...")
    from src.build_features import build_features
    build_features()
```
Because `data/team_game_epa.csv` is tracked in Git (ending at Week 7, `2026_07_TB_CAR`) and `.github/workflows/v3_shadow_logger.yml` does not run `python -m src.build_features`, `os.path.exists(EPA_DATA_PATH)` is always `True`.

**Remediation:** Always refresh current-season play-by-play data in CI (or check if the maximum completed week in `nfl.import_schedules([SEASON])` exceeds the maximum week in `data/team_game_epa.csv` for `SEASON`, and trigger `build_features()` when stale).

---

#### Bug 3: `diff_early_down_epa` Omitted from Rolling Snapshot & Matchup Builder (`src/epa_features.py` Lines 174–248)
`parse_pbp_to_game_epa` (lines 98–133) computes `off_early_down_epa` and `def_early_down_epa`, but:
1. `compute_rolling_epa_snapshot` (lines 174–177) defines:
   ```python
   epa_metrics = [
       "off_pass_epa", "off_rush_epa", "off_total_epa", "off_success_rate",
       "def_pass_epa", "def_rush_epa", "def_total_epa", "def_success_rate"
   ]
   ```
   omitting `"off_early_down_epa"` and `"def_early_down_epa"`.
2. `get_matchup_epa_features` (lines 197–248) never calculates or returns `"diff_early_down_epa"`.

**Remediation:** Add `"off_early_down_epa"` and `"def_early_down_epa"` to `epa_metrics` in `compute_rolling_epa_snapshot`, and compute `diff_early_down_epa` in `get_matchup_epa_features`:
```python
home_early_matchup = h["roll_off_early_down_epa"] - a["roll_def_early_down_epa"]
away_early_matchup = a["roll_off_early_down_epa"] - h["roll_def_early_down_epa"]
diff_early_down_epa = home_early_matchup - away_early_matchup
```

---

#### Bug 4: `HIGH_CONVICTION` Picks Dropped by `src/compare_models.py` (Lines 288–291 & 367)
- `src/log_v3_shadow.py` (lines 315–318) sets `decision = "HIGH_CONVICTION"` when `best_edge >= 0.035` and `"SHADOW_PICK"` when `MIN_EDGE_THRESHOLD <= best_edge < 0.035`.
- `src/compare_models.py` (lines 288 and 367) only checks `if dec == "SHADOW_PICK":`.

**Remediation:** Update `src/compare_models.py` lines 288, 367, and 387 to treat `{"SHADOW_PICK", "HIGH_CONVICTION"}` as active v3 picks:
```python
if dec in {"SHADOW_PICK", "HIGH_CONVICTION"}:
    v3_bets.append(g)
    if dec == "HIGH_CONVICTION" or edge >= 0.035:
        v3_high_bets.append(g)
```

---

#### Bug 5: Injury Guard Reverse Line Movement (RLM) Check is Inactive (`src/injury_guard.py`)
In `src/generate_picks.py` (lines 215–223), `src/generate_picks_v2.py` (lines 249–257), `src/log_v2_shadow.py` (lines 339–347), `src/log_v3_shadow.py` (lines 300–308), and `src/backtest_epa.py` (lines 327–335), `audit_candidate_pick` is invoked with `open_line=home_line, current_line=home_line`. Because both lines are identical, `check_line_movement` never fires.

**Remediation:**
1. During Tuesday shadow logging (which runs 30–35 minutes after opening capture, or on the 18:00 / 23:00 UTC runs), pass `open_line=row["opening_line"]` and `current_line` from the latest cached/live Odds API spread if available.
2. In `capture_closing.py`, when capturing closing lines 30–90 minutes prior to kickoff, run `check_line_movement(open_line=row["opening_line"], current_line=odds_row["home_spread"], pick_side=row["pick_side"])` and flag any pick where the closing line moved $\ge 1.5$ points against the opening pick in `notes` (or a `closing_rlm_alert` column).

---

### 1.2 Time Zones, Daylight Saving Time (DST), and Schedule Flexing

1. **Flexed NFL Games Keep Stale Kickoff Times (`src/fetch_schedule.py` Lines 34–48)**:
   `sync_schedule()` skips any `game_id` already present in `Master_Bets` (`if gsis_id in existing_ids: continue`). Because all 272 games are loaded in Week 1, mid-to-late-season flexed games retain their initial placeholder kickoff time, causing `capture_closing.py` to trigger at the wrong time.
   - **Remediation:** In `sync_schedule()`, for existing rows with `status in {"PENDING", "OPEN_CAPTURED"}`, compare `kickoff_iso(game)` against the sheet's `kickoff_time` and batch-update any changed kickoff timestamps.
2. **DST Ambiguity (`src/fetch_schedule.py` Line 16)**:
   Replace `pytz.timezone(LOCAL_TZ).localize(datetime.combine(gameday, t))` with `pytz.timezone(LOCAL_TZ).localize(datetime.combine(gameday, t), is_dst=False)` (or `zoneinfo.ZoneInfo(LOCAL_TZ)`).
3. **Inconsistent Rest-Day Calculation (`src/train_model_v3.py` Lines 119–120 vs. `src/model_v2_core.py` Line 215)**:
   `(kickoff - h_last).days` in `train_model_v3.py` and `backtest_epa.py` truncates fractional 24-hour periods in UTC, whereas `model_v2_core.py` computes calendar days in `America/New_York` (`local_day = kickoff_utc.dt.tz_convert("America/New_York").dt.normalize()`). Standardize all modules on Eastern calendar days.

---

### 1.3 The Odds API Quota Protection (500 Calls/Month)

1. **Unmatched / Off-the-Board Game Loop in `capture_closing.py` (Lines 85–106, 166–175)**:
   If one game in the 30–90 minute window is off the board (e.g., due to injury uncertainty), `closing_line` remains `NaN` in `Master_Bets`. Because `max_age_seconds=300` equals the 5-minute cron interval and GitHub Actions caches are immutable per key, `capture_closing.py` will call The Odds API on **all 12 cron runs** during that 60-minute window.
   - **Remediation:** Increase `max_age_seconds` in `capture_closing.py` to `900` (15 minutes) or `1200` (20 minutes), and record a `closing_attempt_at` timestamp or cap retries at 2 per game.
2. **UTC Midnight Cache Filename Rollover (`src/odds_utils.py` Lines 101–102)**:
   `today = datetime.now().strftime("%Y-%m-%d")` rolls over at `00:00 UTC` (8:00 PM EDT / 7:00 PM EST)—right before TNF, SNF, and MNF kickoffs (`00:15 UTC` / `01:15 UTC`), invalidating any cache file written minutes earlier.
   - **Remediation:** Use a single rolling cache file path (`data/odds_cache/odds_latest.json`) for age checks (`time.time() - os.path.getmtime(cache_file) < max_age_seconds`), while optionally archiving dated copies.

---

### 1.4 Google Sheets (`gspread`) Batch Efficiency & Resilience

1. **Non-Atomic `ws.clear()` in `src/compare_models.py` (Lines 570–573)**:
   `ws.clear()` followed by `ws.update()` without retry logic leaves `Model_Scorecard` empty if `ws.update()` encounters a transient 429/5xx error. Wrap `ws.clear()` + `ws.update()` in the exponential/linear backoff helper from `src/sheet_utils.py`.
2. **Redundant Sheet Reads & Auth**:
   Modify `get_sheet_as_df(ws=None)` in `src/sheet_utils.py` to accept an existing `ws` object (or return `(df, ws, headers)`), eliminating duplicate `get_master_sheet()` and `ws.row_values(1)` calls across `capture_opening.py`, `capture_closing.py`, `update_results.py`, and `generate_picks.py`.
3. **Unify `src/sheet_utils.py` and `src/v2_sheet_utils.py`**:
   Consolidate `get_or_create_worksheet` into `src/sheet_utils.py` with consistent `SCOPES` and retry wrappers on `get_all_values()`, `get_all_records()`, `update()`, and `append_rows()`.

---

## 2. Quantitative & Mathematical Integrity

### 2.1 Discrete Key Numbers Engine (`src/key_numbers.py`)

1. **Mean Shift Distortion ($\mathbb{E}[M] \neq \mu$)**:
   In `discrete_margin_probabilities(mu, sigma=13.4, max_margin=50)` (lines 87–122), multiplying a non-zero-mean Gaussian `raw_mass` ($\mathcal{N}(\mu, \sigma^2)$) by a symmetric zero-centered weight vector `weights` ($w(|m|)$) shifts the expected value away from $\mu$. Furthermore, `gauss_freq` on line 111 is evaluated at `0` rather than averaging across the empirical spread distribution.
   - **Remediation:** After applying `adjusted_mass = raw_mass * weights` and normalizing, apply an exponential tilting / small mean-correction shift $\delta$ to the Gaussian location parameter $\mu' = \mu + \delta$ (or one-step Newton adjustment) so that $\sum_m m \cdot P(M = m) = \mu$ holds to within $10^{-4}$ points.
2. **Tie Probability (`m = 0`) Factor-of-2 Bug (Line 113)**:
   For $m > 0$, `MARGIN_FREQUENCIES[abs_m]` sums the empirical frequencies of $+m$ and $-m$ (`2.0 * gauss_freq`). For $m = 0$, there is only a single integer bin (`0`), not two, and clipping `ratio` at `0.7` leaves $P(M = 0) \approx 2.1\%$ instead of the true $0.3\%$ regular-season tie rate.
   - **Remediation:** Handle `abs_m == 0` explicitly without `2.0 *` or the `0.7` lower clip:
     ```python
     if abs_m == 0:
         weights[idx] = emp_freq / gauss_freq if gauss_freq > 0 else 1.0
     else:
         ratio = emp_freq / (2.0 * gauss_freq)
         weights[idx] = np.clip(ratio, 0.7, 1.6)
     ```

---

### 2.2 Platt Calibration & Walk-Forward Backtester (`src/train_model_v3.py`, `src/backtest_epa.py`)

1. **In-Sample Overfitting & Negative Calibrator Slope (`-0.3194`)**:
   In `models/metadata_v3.json`:
   ```json
   "calibrator_params": [0.17370070321525555, -0.3193983666588068]
   ```
   `src/train_model_v3.py` (lines 228–252) and `src/backtest_epa.py` (lines 207–221) fit `LogisticRegression(C=1.0)` in-sample on raw `cond_home_win` probabilities. Because regularized Ridge margin predictions have lower variance than market spreads, `cond_home_win` correlates with being the underdog; combined with heavy L2 regularization on a low-variance 1D input, the learned slope is **negative (`-0.3194`)**.
   - **Remediation:** Replace in-sample `LogisticRegression()` in `src/train_model_v3.py` and `src/backtest_epa.py` with the out-of-sample walk-forward log-odds calibration architecture from `src/model_v2_core.py` (`fit_calibrator` & `calibrate_probability`, lines 433–521):
     - Collect out-of-sample walk-forward predictions across prior seasons.
     - Include both `HOME` (`p_raw`, `home_win`, `weight=0.5`) and `AWAY` (`1 - p_raw`, `away_win`, `weight=0.5`) to enforce zero-intercept symmetry.
     - Regress on log-odds $\text{logit}(p) = \ln(p / (1 - p))$ with `C=1e6`.
     - Enforce monotonicity: if `coef <= 0.0`, fall back to `"identity"` (or a conservative shrinkage slope `coef = 0.25`).
2. **FiveThirtyEight Elo Autocorrelation Denominator Bug (`src/train_model_v3.py` Line 72 & `src/backtest_epa.py` Line 44)**:
   ```python
   mov_mult = math.log(max(mov, 1) + 1.0) * (2.2 / ((home_elo - away_elo) * 0.001 + 2.2))
   ```
   When a road favorite (`away_elo > home_elo`) wins by a large margin, `(home_elo - away_elo)` is negative, shrinking the denominator below `2.2` and *amplifying* the favorite's Elo gain instead of dampening it.
   - **Remediation:**
     ```python
     elo_diff_winner = (h_adj - away_elo) if actual_margin >= 0 else (away_elo - h_adj)
     mov_mult = math.log(max(mov, 1) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))
     ```
3. **Elo Warm-Up & Cross-Season Rolling Bleed**:
   - In `src/backtest_epa.py` (line 172), load historical schedules starting from at least `2020` (or `1999`) so Elo ratings and training folds are properly warmed up before the first test season.
   - In `compute_rolling_epa_snapshot` (`src/epa_features.py` line 179), apply off-season regression toward `0.0` (e.g., `0.50` weight on prior-season games when a team has played fewer than `window` games in the current season).

---

### 2.3 Ported Modules (`src/staking.py`, `src/qb_features.py`, `src/turnover_luck.py`)

1. **`src/staking.py` — Exact Analytical Push-Aware Kelly Formula & Probability Scaling**:
   - **Analytical Formula:** Maximizing $G(f) = w \ln(1 + fb) + l \ln(1 - f)$ yields:
     $$f^* = \max\!\left(0,\; \frac{w b - l}{b (w + l)}\right)$$
     Replace `scipy.optimize.minimize_scalar` in `full_kelly_fraction` (lines 66–88) with this exact closed-form expression.
   - **Conditional Probability Bug in `src/compare_models.py` (Lines 369–371):**
     Because `v3_model_probability` is conditional on no push ($P(\text{Win} \mid \text{No Push})$), convert it to unconditional win probability `w = v3_prob * (1.0 - p_push)` before calling `compute_push_aware_stake(w, p_push, v3_odds)`.
2. **`src/qb_features.py` — Empirical Bayes Shrinkage Constant & Weighted Aggregations**:
   - Increase `SHRINKAGE_K` (line 32) from `4.0` to `60.0` dropbacks to reflect the true noise-to-signal variance ratio ($\sigma^2_{\text{play}} / \sigma^2_{\text{QB}}$) of NFL dropback EPA.
   - In `compute_rolling_qb_ratings` (lines 201–208), compute `avg_success`, `avg_sack`, and `avg_int` weighted by `dropbacks` rather than taking the unweighted mean across games.
3. **`src/turnover_luck.py` — Full-Team (Offense + Defense) Fumble Recovery Tracking**:
   - Update `parse_pbp_turnovers` (lines 58–79) and `compute_rolling_turnover_luck` (lines 135–152) to include opponent fumbles forced/seen on defense (`def_fumbles_seen`, `def_fumbles_recovered`) so defensive fumble recovery luck is captured alongside offensive fumble recovery luck.
   - Use the Bayesian-shrunk recovery rate discrepancy `(shrunk_rate - 0.50) * total_fumbles` to compute `net_fumble_luck` so small-sample windows are automatically regularized.

---

## 3. Pipeline Inefficiencies & Technical Debt

1. **Consolidate Staggered Tuesday Workflows**:
   Merge `.github/workflows/tuesday_opening.yml`, `v2_shadow_logger.yml`, and `v3_shadow_logger.yml` into a single sequential workflow (or trigger shadow loggers via `workflow_run`) so GitHub Actions runner queue delays cannot cause v2/v3 shadow loggers to run before opening lines are captured.
2. **Fix `actions/cache@v4` Key Thrashing in `closing_line_monitor.yml`**:
   Remove `key: ${{ runner.os }}-odds-${{ github.run_id }}` from the 5-minute cron monitor so no-op runs do not upload a new cache archive every 5 minutes. Add `cache: "pip"` to `actions/setup-python@v5` across all workflows.
3. **Clean Up Repository & Dependencies**:
   - Add `scipy` and `joblib` explicitly to `requirements.txt`.
   - Replace the broken mock stub in `src/train_model.py` with the actual v1 Ridge training script (using `build_training_table` from `src/model_utils.py`).
   - Standardize metadata JSON keys (`feature_columns`) across `metadata.json`, `metadata_v2.json`, and `metadata_v3.json`.
   - Remove `chat-export-1793031333161.json` from the repository root.

---

## 4. High-Impact Strategic Improvements (Model v4 Roadmap)

### 4.1 Model v4 Feature Architecture
Integrate `src/qb_features.py` and `src/turnover_luck.py` orthogonally into the Model v4 feature matrix (`src/train_model_v4.py` / upgraded `v3`):
- **`diff_qb_shock`**: Difference between the confirmed Week $W$ starting QB's shrunk EPA/dropback ($k = 60$) and the team's rolling 6-game pass EPA/dropback, scaled to spread points ($\times 35 \times 0.38$). Captures starter injury/return shocks without collinearity against `diff_pass_epa`.
- **`diff_fumble_luck_per_gm`**: Full-team (offense + defense) Bayesian-shrunk net lucky fumble recoveries per game over the prior 6 games.
- **`diff_early_down_epa`**: Restored 1st/2nd-down matchup EPA differential.

### 4.2 Top 3 Alpha Enhancements
1. **Multi-Book Line Shopping & Consensus No-Vig Fair Probabilities (`src/odds_utils.py`)**:
   Use all sportsbooks already returned in the single Odds API response to compute consensus vig-free probabilities (`fair_home`, `fair_away`) while selecting the bookmaker offering the highest-EV line/price (`best_book`, `best_line`, `best_odds`).
2. **Totals- & Weather-Conditioned Margin Variance ($\sigma_i$) (`src/key_numbers.py`)**:
   Scale game-level margin standard deviation $\sigma_i = \sigma_{\text{base}} \sqrt{\text{clamp}(\text{total\_line}_i, 34, 58) / 44.5}$ so low-total/high-wind games have tighter margin distributions around key numbers `3` and `7`.
3. **Slate-Level Bankroll Cap & Unit Test Suite**:
   Add a simultaneous-slate Kelly cap (e.g., max `10.0u` total exposure per week) in `src/staking.py` and a `pytest` suite covering spread conventions, discrete cover probabilities, and Kelly sizing.

---

## 5. Phased Implementation Checklist

- [x] **Phase 1 (P0 — Immediate Production & Quant Hotfixes)**
  - [x] Fix `diff_early_down_epa` in `src/epa_features.py`.
  - [x] Fix `update_elo` winner/loser MOV autocorrelation denominator in `src/train_model_v3.py` and `src/backtest_epa.py`.
  - [x] Replace in-sample Platt calibration in `src/train_model_v3.py` and `src/backtest_epa.py` with out-of-sample walk-forward log-odds calibration (`C=1e6`, symmetric `HOME`/`AWAY` weights, `coef > 0` guard), and retrain `models/nfl_spread_model_v3.pkl` + `models/metadata_v3.json`.
  - [x] Fix `src/log_v3_shadow.py` to compute real point-in-time `elo_diff`, `diff_margin`, and `diff_rest`, and auto-refresh `data/team_game_epa.csv` when stale.
  - [x] Fix `src/compare_models.py` to include `"HIGH_CONVICTION"` decisions and pass unconditional win probability `v3_prob * (1.0 - p_push)` into `compute_push_aware_stake`.
  - [x] Replace `minimize_scalar` in `src/staking.py` with the exact closed-form push-aware Kelly formula and fix tie (`m = 0`) weighting in `src/key_numbers.py`.
- [x] **Phase 2 (P1 — Workflow, Quota & Guard Hardening)**
  - [x] Consolidate Tuesday GitHub Actions workflows and add `pip` caching.
  - [x] Fix UTC-midnight cache rollover in `src/odds_utils.py` and 5-minute cache thrashing in `closing_line_monitor.yml`.
  - [x] Update `src/fetch_schedule.py` to sync flexed `kickoff_time` values and use DST-safe localization.
  - [x] Pass live/updated lines into `audit_candidate_pick` so RLM detection is active.
- [x] **Phase 3 (P2 — Model v4 & Multi-Book Line Shopping)**
  - [x] Upgrade `src/qb_features.py` (`SHRINKAGE_K = 60.0`, dropback-weighted rates) and `src/turnover_luck.py` (include defensive fumble recoveries).
  - [x] Implement multi-book consensus fair pricing and best-line shopping in `src/odds_utils.py`.
  - [x] Integrate `diff_qb_shock`, `diff_fumble_luck_per_gm`, and totals-conditioned $\sigma_i$ into the model training and backtesting pipeline.
- [x] **Phase 4 (P3 — Testing & Live Promotion)**
  - [x] Replace broken `src/train_model.py` stub and unify `src/sheet_utils.py` + `src/v2_sheet_utils.py`.
  - [x] Add `pytest` suite covering quantitative modules and data pipelines.
