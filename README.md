# 🏈 NFL Paper Trader & Multi-Model Forecasting Pipeline

An autonomous, 100% free paper-trading pipeline and sports analytics platform that forecasts NFL point spreads against live consensus market odds, monitors closing line value (CLV), and benchmarks multi-model performance in Google Sheets via GitHub Actions.

---

## 📌 Architecture & Modeling Lineup

The repository maintains and compares three generations of quantitative spread models alongside an empirical safeguard engine:

| Model / Architecture | Data & Feature Engineering | Margin & Probability Engine | Execution Tab |
| :--- | :--- | :--- | :--- |
| **Model v1 (Baseline)** | Historical points scored/allowed | Continuous Gaussian Margin | `Master_Bets` (Live) |
| **Model v2 (Elo + Rest)** | Rolling Elo ratings, schedule rest differential, divisional flags | Continuous Gaussian Margin | `V2_Shadow` (Shadow) |
| **Model v3 (EPA Enhanced)** | `nflverse` Play-by-play Pass EPA, Rush EPA, Success Rate, and Early Down EPA | **Discrete Key Numbers Engine** (empirically weights 3, 7, 6, 10) + Platt Logistic Calibration | `V3_Shadow` (Shadow) |

### 🛡️ Pre-Bet Safeguard Engine (`src/injury_guard.py`)
Audits all candidate picks against market traps before logging:
- **Reverse Line Movement (RLM)**: Detects sharp syndicates moving lines $\ge 1.5$ points against public sides (`SUPPRESS_DANGER`).
- **Suspicious Market Divergence**: Flags outlier spreads $> 7.0$ points apart from consensus as unmodeled injuries or weather shocks.
- **Estimated Units Saved**: Tracked automatically in performance scorecards.

---

## 📊 Google Sheets Ecosystem

The system synchronizes with a central Google Sheet across four distinct views:

1. **`Master_Bets`**: The live trading ledger capturing opening lines, active paper bets, closing lines, game scores, CLV, and unit ROI.
2. **`V2_Shadow`**: Parallel shadow log of Model v2 evaluations.
3. **`V3_Shadow`**: Parallel shadow log of Model v3 EPA-enhanced evaluations with granular probability metrics and integrity audit flags.
4. **`Model_Scorecard`**: Executive dashboard with side-by-side win rates, net units, ROI %, CLV beat %, and the **Active Slate Head-to-Head Matrix**.

---

## ⚙️ Automated GitHub Actions Workflows

All operational tasks execute autonomously on GitHub Actions cron schedules:

```
Tuesday Morning / Afternoon:
 ├── 11:00 AM, 2:00 PM, 7:00 PM EDT: Tuesday Opening Capture (fetches odds & schedule)
 ├── 11:30 AM, 2:30 PM, 7:30 PM EDT: V2 Shadow Logger (logs v2 shadow evaluations)
 └── 11:35 AM, 2:35 PM, 7:35 PM EDT: V3 Shadow Logger (logs v3 shadow evaluations & updates scorecard)

Kickoff Days (Thu, Fri, Sat, Sun, Mon):
 └── Every 5 Minutes: Closing Line Monitor (captures consensus closing spreads 30-90m before kickoff)

Daily:
 └── 6:00 AM EDT (05:00 UTC): Nightly Results Settler (grades final scores, settles PnL, updates scorecard)
```

---

## 🛠️ Local Development & Quickstart

### Prerequisites
- Python 3.10+
- Virtual environment (`.venv`)

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Core Commands

* **Build EPA Features**:
  ```bash
  python -m src.build_features
  ```

* **Train Model v3 (Ridge + Platt Logistic Calibrator)**:
  ```bash
  python -m src.train_model_v3
  ```

* **Run Walk-Forward Backtester (Zero Lookahead Bias)**:
  ```bash
  python -m src.backtest_epa --test-seasons 2024 2025
  ```

* **Run Multi-Model Scorecard & Slate Comparator**:
  ```bash
  python -m src.compare_models
  ```

---

## 📈 Backtest Results (Out-of-Sample Walk-Forward)

Walk-forward backtest across seasons 2024–2025 (trained strictly on historical prior seasons):

| Strategy | Total Bets | Record | Win Rate | Total Units | ROI % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Favorite Baseline** | 544 | 274W - 265L - 5P | 50.8% | -16.56u | -3.04% |
| **Raw Points (v1 Baseline)** | 240 | 127W - 110L - 3P | 53.6% | +6.39u | +2.66% |
| **EPA-Enhanced (v3 Calibrated)** | 94 | 47W - 44L - 3P | 51.6% | +1.62u | +1.73% |
| **EPA High-Conviction ($\ge 3.5\%$ Edge)** | 8 | 5W - 2L - 1P | **71.4%** | **+2.99u** | **+37.39%** |

---

## 🔒 Security & Free-Tier Quota Integrity
- **100% Free**: Operates entirely within The Odds API free tier (500 monthly calls) via a 12-hour local odds cache with fallback bookmakers (`DraftKings` -> `FanDuel`).
- **Zero Secrets in Code**: All Google Service Account keys and API tokens are injected via GitHub Actions Secrets (`GSHEET_CREDENTIALS`, `SPREADSHEET_ID`, `ODDS_API_KEY`).
