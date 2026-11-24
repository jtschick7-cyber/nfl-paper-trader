# P5. Bottom-up `player_receptions` model

**Source:** owner's roadmap brief of 2026-11-24 (Step 5). The brief's Stage B assumed archived prop lines; on 2026-11-24 the owner decided **not** to buy The Odds API's historical prop odds, so the only prop prices are P1's 2026 archive. **Effort:** L, in two tasks: P5a (Stage A, offline) and P5b (Stage B, only after Stage A passes). **Branches:** `claude/p5a-receptions-stage-a`, `claude/p5b-receptions-stage-b`. **Depends on:** P4 merged (`src/props/prop_pricing.py`, `src/props/player_stats.py`, `normalize_name`), P3's `src/totals/features.py`.

Offline only. No Sheet tab, workflow, Worker, Odds API call or decision change.

## Question

Does a point-in-time player model (role, game script, teammate injuries, opponent) predict a pass-catcher's receptions better than his own recent average, and, if so, does it lean the right way against Pinnacle's Friday prop price before the line closes?

Stage A is a cheap kill test on free data. Stage B is the only test against prices and runs on 2026 captures. Without bought history, Stage B is small this season and probably carries into 2027.

## Pre-registration

- Every number and rule below is fixed at this spec's commit (2026-11-24, before the first `open` or `close` prop capture landed).
- The spec-time checks covered only the data's shape: column names of `stats_player_week_2025`, `snap_counts_2025`, `injuries_2025` and `players.parquet`, the `report_status` values, and that the 2016 and 2026 assets exist. No reception count, feature or model output was computed.
- Stage A runs **once**, after its code and tests are committed. Stage B code and tests are committed before it reads any prop capture together with model output.
- The builder may change code structure but not a number or rule. If one is unworkable, stop and ask the owner. Do not adjust it to the data.
- **If Stage A fails, Step 5 stops.** There is no Stage B and no live logging.

## Data (free, checked 2026-11-24)

From the `nflverse-data` GitHub releases, read with `pandas.read_parquet` and cached under `data/nflverse_cache/`. Do not upgrade `nfl_data_py`.

| Asset | Key columns | Use |
|---|---|---|
| `stats_player/stats_player_week_<S>.parquet` | `player_id` (GSIS), `team`, `opponent_team`, `game_id`, `season_type`, `receptions`, `targets`, `receiving_air_yards` | outcomes, role |
| `snap_counts/snap_counts_<S>.parquet` | `game_id`, `pfr_player_id`, `team`, `offense_snaps`, `offense_pct`, `game_type` | who played, snap share |
| `injuries/injuries_<S>.parquet` | `gsis_id`, `team`, `week`, `game_type`, `report_status` (`Out`, `Doubtful`, `Questionable`, blank) | teammate absences |
| `players/players.parquet` | `gsis_id`, `pfr_id`, `position_group` | ID map and positions |

- **PBP:** `data/epa_cache/pbp_<S>.parquet` (filled by `src.build_features`), through `src.totals.features.load_pbp` and `neutral_pace`.
- **Game lines:** `data/nflverse_cache/games.csv` (`spread_line`, `total_line`, `roof`, `wind`; `home_line = -spread_line`). These are single-source closing lines.
- **Seasons:** features from 2016, test seasons 2018–2025, gate on 2020–2025. Regular season only.
- `stats_player_week` has no row for a player who played but had no stat. The universe therefore comes from snap counts (`offense_snaps > 0`), and a missing stats row means 0 targets and 0 receptions.
- `snap_counts` uses PFR IDs and the rest use GSIS IDs; join through `players.parquet` (`pfr_id` ↔ `gsis_id`). A snap row with no `gsis_id` match is dropped and counted.

## Universe (frozen)

A player-game is in the sample when:

1. it is a regular-season game, and the player's `position_group` is WR, TE or RB;
2. `offense_snaps > 0` in that game (books void players who don't play, so the target is receptions **given** the player plays);
3. the player has ≥ `MIN_GAMES` = 3 played games in his prior `WINDOW` = 8 (across seasons, strictly before kickoff); and
4. his rolling 8-game mean receptions is ≥ `MIN_REC_MEAN` = 1.5 (a proxy for having a prop posted).

## Features (frozen; every window uses only games with `kickoff_utc` before the target)

Rolling windows: `WINDOW` = 8 and `SHORT_WINDOW` = 4 played games, at least `MIN_GAMES` = 3, carried across seasons. Team-level windows are the team's last 8 games.

1. **Game script:** `team_line` (the team's own spread, negative when favoured), `game_total`, `team_implied_total = (game_total − team_line) / 2`, `is_indoor`, `wind_mph` (indoor = 0; missing outdoor wind = training-season median), `is_high_wind` (≥ 15 mph). Reuse `src.totals.features` weather helpers.
2. **Role:**
   - `target_share_8` and `target_share_4`: player targets / team targets over the window.
   - `snap_pct_8` and `snap_pct_4`: mean `offense_pct`.
   - `targets_per_snap_8`.
   - `role_trend`: mean `offense_pct` over the last 2 played games minus the last 6.
   - `catch_rate_eb`: (receptions + k·prior) / (targets + k) over the window, with k = `EB_K` = 20 and `prior` = the training seasons' position-group catch rate.
   - `adot_eb`: air yards / targets, shrunk the same way (k = 20) to the training position mean.
   - `rec_mean_8`: the baseline's mean, also given to the model.
   - `depth_rank`: rank of `snap_pct_8` within team and position group (1, 2, 3 = 3 or lower).
   - Position group one-hot.
3. **Injuries (from that week's final report, never from realised inactives):**
   - `missing_same_pos_share`: summed `target_share_8` of teammates in the same position group with `report_status` Out or Doubtful.
   - `missing_team_share`: the same over all WR, TE and RB teammates.
   - `qb_out`: the team's QB with the most pass attempts over its last 8 games is listed Out or Doubtful.
   - `self_questionable`: the player himself is listed Questionable.
4. **Team and opponent:**
   - team rolling `neutral_pass_oe` and `neutral_sec_per_play` (from `neutral_pace`);
   - the opponent's rolling receptions allowed per game to the player's position group;
   - the opponent's rolling neutral `pass_oe` faced (its opponents' neutral PROE against it).
- **Fills:** a missing feature gets the training seasons' mean. Fills are computed per fold from training seasons only.

## Models (frozen)

- **Baseline:** Negative Binomial with mean `rec_mean_8` and dispersion `r` per position group.
- **Model:** `sklearn.linear_model.PoissonRegressor` on standardised features (`StandardScaler` fit on the training fold). `alpha` comes from `ALPHA_GRID` = {0.001, 0.01, 0.1, 1.0}, chosen by an inner split that leaves out each training season in turn (forward only, at least 2 earlier seasons). The predicted mean is paired with a Negative Binomial `r` per position group.
- **Dispersion:** fit `r` per position group by maximum likelihood on `[R_MIN, R_MAX]` = [1, 200], using the inner out-of-fold means (not in-sample fits), separately for the baseline and the model. If the fit reaches 200, use Poisson for that group.
- **PMF:** P(R = 0), …, P(R = 14), with P(R ≥ 15) lumped.
- Why not LightGBM or a fitted NB GLM, as the brief suggested: neither is in the pinned environment (numpy 1.26 / scikit-learn 1.7.2 for the pickles). A GLM with ~20 features gives the kill test a fair shot; a better learner can follow a pass.

## Stage A: walk-forward (P5a)

- For each test season S in 2018–2025, train on 2016..S−1 and predict S. Pooled evaluation over 2020–2025.
- **Synthetic lines:** `LINES` = {2.5, 3.5, 4.5, 5.5}. For each player-game, `LL_sum` = Σ over the lines of the binary log loss of 1{R > L} with p = P(R > L).
- **Primary metric:** Δ = mean over player-games of (`LL_sum`_model − `LL_sum`_baseline), with SE clustered by `game_id` (`topdown_replay.cluster_se`).
- **Calibration:** for each line, bin the model's P(over) into widths of 0.1. A bin with n ≥ `CAL_MIN_N` = 500 is calibrated when |observed rate − mean p| ≤ 2·√(p̄(1 − p̄)/n).
- **Injury subset:** player-games with `missing_same_pos_share` ≥ `INJ_SUBSET` = 0.10.

### Pre-registered gate (frozen)

- **Pass** when all four hold, pooled over 2020–2025:
  1. Δ < 0 by ≥ 3.0 game-clustered SE;
  2. every calibration bin with n ≥ 500, at every line, is calibrated;
  3. Δ < 0 in the injury subset (point estimate);
  4. Δ < 0 in at least 4 of the 6 seasons.
- **Fail** otherwise. One run, no retries with other settings.

### Report (printed and in `notes/p5a-receptions-stage-a.md`)

Δ and SE (pooled, per season, per position group, injury subset); full discrete log loss −log P(R = r) and CRPS beside the primary metric; calibration tables; chosen `alpha` and `r` per fold; coverage (snap rows unmatched, players excluded by each universe rule). Output `data/receptions_stage_a.csv` (add to `.gitignore` if no pattern covers it).

## Stage B: against 2026 prop prices (P5b, only after a Stage A pass)

- **Frozen model:** the Stage A procedure, trained on 2016–2025. 2026 features use 2026 games before the target kickoff.
- **Rows:** each `open`-phase capture (P1) × player with a two-sided Pinnacle quote, matched to the universe through P4's name matching. Player-games P4 grades VOID are dropped.
- **Game lines:** the median US-book `home_line` and total in the latest `snapshots/` file at or before the open's `captured_at`, within 72 h. With none, the row is excluded and counted.
- **Injury timing:** the final report is assumed public at **21:00 UTC two days before the kickoff's ET date, or one day before for a Thursday game** (Wednesday for Thursday and Friday games, Thursday for Saturday, Friday for Sunday, Saturday for Monday). An open captured before that time is excluded and counted, so the model never holds news the open price lacks.
- **Market:**
  - `lam_open` and `lam_close` from Pinnacle via P4's `implied_lambda`;
  - `raw_lean = mu_model − lam_open`; lean side Over when `raw_lean > 0`, else Under.
- **Best offer per side:** the `US_TARGETS` offer (any line) with the highest EV against `lam_open`. Its `close_ev` comes from P4's `price_prop_offer` against `lam_close`.
- **B1 (raw-lean CLV beats Always Under):** per player-game, d = `close_ev`(lean side) − `close_ev`(Under).
- **B2 (model EV):** the best `US_TARGETS` offer on either side by EV under the model's NB PMF; flagged when that EV ≥ `MODEL_EV` = 0.02. Its `close_ev` is measured against `lam_close`.

### Pre-registered gate (frozen)

- **Pass** when all hold:
  - B1: ≥ 200 player-games over ≥ 4 NFL weeks, mean d > 0 with game-clustered t ≥ 2.0;
  - B2: ≥ 100 flagged offers on ≥ 25 games, mean `close_ev` > 0 with game-clustered t ≥ 2.0, and > 0 with the 2 games of largest summed `close_ev` removed.
- **Fail:** once both sizes are reached, mean d ≤ 0 or B2 mean `close_ev` ≤ 0.
- **Inconclusive:** neither once week 18 has settled. In 2027 the same procedure, retrained through 2026, continues on 2027 captures with these rules unchanged.
- Weekly re-runs are descriptive. Do not stop early for a pass.
- **Descriptive only:** the slope of (`lam_close` − `lam_open`) on `raw_lean` with game-clustered SE; Over and Under leans separately; outcome W-L-P-Void and units of B2 offers.
- **Expected size:** about 15 games × 10–15 matched players a week, so B1's size is reachable by about week 16. B2 depends on how often the model clears 2%.
- **Only after a pass:** a spec for live `Props_Shadow` / `Props_Scorecard` tabs (locked owner decision 3).

## Files

- `src/props/data_sources.py`: download, cache and ID-join the four assets.
- `src/props/features.py`: universe and features (pure functions on frames).
- `src/props/train_receptions.py`: fit, `alpha` selection, NB dispersion, PMF.
- `src/props/backtest_receptions.py`: Stage A. CLI `.venv/bin/python -u -m src.props.backtest_receptions`.
- `src/props/receptions_vs_market.py` (P5b only): Stage B. CLI `.venv/bin/python -u -m src.props.receptions_vs_market [--source REF]`.

## Tests (write first)

`tests/test_receptions_features.py`, `tests/test_receptions_model.py` (P5a); `tests/test_receptions_vs_market.py` (P5b).

1. **Leakage:** a feature for game g never changes when every game at or after g's kickoff is altered or deleted, including team, opponent and injury features.
2. A player who played with no stats row enters with 0 receptions; a player with 0 offensive snaps is outside the universe.
3. `missing_same_pos_share` sums teammates listed Out or Doubtful, never Questionable, never the player himself; `qb_out` uses the last-8 attempts leader.
4. Fills and EB priors use training seasons only (a test-season value change leaves them unchanged).
5. **NB:** the PMF sums to 1 and its mean matches μ within 1e-6 before lumping; `r` = 200 falls back to Poisson; the ML fit recovers a known `r` on synthetic data within 10%.
6. `LL_sum`, game-clustered Δ and the calibration rule on hand-computed examples.
7. The Stage A gate returns pass or fail on synthetic summaries, one case per condition.
8. **Stage B (P5b):** the injury-timing exclusion (Monday-night game, Friday open excluded); the best-offer choice uses `lam_open`, never the close; d is 0 when the lean is Under; the B1/B2 gate on synthetic summaries.
9. A test asserting every frozen constant above equals its spec value.

The full suite stays green.

## Deliverables

- **P5a:** code and tests, one run, `notes/p5a-receptions-stage-a.md` with the verdict, and `PROGRESS.md` (about 10 bullets).
- **P5b:** only after a pass; adds `src.props.receptions_vs_market` to the weekly re-run line.

## Out of scope

Live logging, Sheet tabs, other prop markets, buying historical odds, using realised inactives or post-open news as features, and retraining any `models/*.pkl`.
