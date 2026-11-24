# P5a: receptions model, Stage A walk-forward: FAIL

Spec: `notes/implementation-plan/P5-receptions-model.md` (rules frozen 2026-11-24). Run once on 2026-11-24, after the tests (`9342779`) and the code (`03bcb5e`) were committed.

```bash
.venv/bin/python -u -m src.props.backtest_receptions   # writes data/receptions_stage_a.csv (gitignored)
```

## Verdict

**FAIL.** Three of the four gate conditions hold. Condition 2, calibration, does not: 15 of the 29 tested bins are outside the frozen band. Under the spec, Step 5 stops here. There is no Stage B (`src/props/receptions_vs_market.py` was not built) and no live logging.

| Gate condition (pooled 2020–2025) | Result | Holds? |
|---|---|---|
| 1. Δ < 0 by ≥ 3.0 game-clustered SE | Δ −0.0430, SE 0.0050 (Δ/SE −8.67) | yes |
| 2. every bin with n ≥ 500, at every line, calibrated | 15 of 29 tested bins miscalibrated | **no** |
| 3. Δ < 0 in the injury subset | −0.0571 (SE 0.0187, n 1,517) | yes |
| 4. Δ < 0 in ≥ 4 of the 6 seasons | 6 of 6 | yes |

The model is clearly sharper than the player's own 8-game mean on all three scores. Its P(over) is not calibrated, though. The misses run mostly one way: at 3.5, 4.5 and 5.5, and in the middle bins at 2.5, observed over-rates are above the predicted ones. The gate's calibration condition is absolute, so this is a FAIL however large Δ is. Recalibrating and re-running would be a retry with other settings, which the spec rules out.

## Owner decision before the run

The frozen alpha rule needs a held-out training season with at least 2 earlier training seasons. Test season 2018 trains on 2016–2017 only, so it has no inner fold, no alpha and no out-of-fold r. This came from the spec text alone, before any data was looked at. The owner chose **skip 2018** (2026-11-24). The test seasons are therefore 2019–2025. 2019 has one inner fold (2018), and the gate seasons 2020–2025 are unaffected. No constant changed.

## Report

### Δ LL_sum (model − baseline), game-clustered SE

| Slice | Δ | SE | n |
|---|---|---|---|
| Pooled 2020–2025 | −0.0430 | 0.0050 | 17,761 (1,615 games) |
| Injury subset (`missing_same_pos_share` ≥ 0.10) | −0.0571 | 0.0187 | 1,517 |
| 2019 (not in gate) | −0.0484 | 0.0122 | 2,818 |
| 2020 | −0.0437 | 0.0129 | 2,928 |
| 2021 | −0.0524 | 0.0118 | 3,098 |
| 2022 | −0.0358 | 0.0126 | 2,979 |
| 2023 | −0.0293 | 0.0116 | 2,999 |
| 2024 | −0.0430 | 0.0117 | 2,926 |
| 2025 | −0.0543 | 0.0124 | 2,831 |
| 2018 | skipped | | |
| WR (pooled) | −0.0378 | 0.0068 | 9,547 |
| TE (pooled) | −0.0443 | 0.0104 | 4,015 |
| RB (pooled) | −0.0537 | 0.0090 | 4,199 |

### Beside the primary metric (pooled 2020–2025, mean per player-game)

| Score | Model | Baseline | Δ (SE) |
|---|---|---|---|
| LL_sum (4 lines) | 1.9725 | 2.0156 | −0.0430 (0.0050) |
| Full discrete log loss −log P(R = r) | 2.0340 | 2.0587 | −0.0248 (0.0020) |
| CRPS | 1.1276 | 1.1548 | −0.0272 (0.0026) |

### Calibration of the model's P(over), pooled 2020–2025

Bins of width 0.1. A bin is tested when n ≥ 500, and it passes when |observed − mean p| ≤ 2·√(p̄(1 − p̄)/n).

| Line | Bin | n | Mean p | Observed | Result |
|---|---|---|---|---|---|
| 2.5 | [0.1,0.2) | 457 | 0.176 | 0.109 | untested |
| 2.5 | [0.2,0.3) | 2,232 | 0.257 | 0.233 | **miss** |
| 2.5 | [0.3,0.4) | 3,185 | 0.350 | 0.340 | ok |
| 2.5 | [0.4,0.5) | 3,018 | 0.450 | 0.459 | ok |
| 2.5 | [0.5,0.6) | 2,604 | 0.549 | 0.584 | **miss** |
| 2.5 | [0.6,0.7) | 2,248 | 0.647 | 0.684 | **miss** |
| 2.5 | [0.7,0.8) | 1,937 | 0.749 | 0.793 | **miss** |
| 2.5 | [0.8,0.9) | 1,519 | 0.847 | 0.858 | ok |
| 2.5 | [0.9,1.0) | 561 | 0.931 | 0.920 | ok |
| 3.5 | [0.0,0.1) | 905 | 0.079 | 0.062 | ok |
| 3.5 | [0.1,0.2) | 3,990 | 0.152 | 0.143 | ok |
| 3.5 | [0.2,0.3) | 3,571 | 0.249 | 0.269 | **miss** |
| 3.5 | [0.3,0.4) | 2,735 | 0.348 | 0.374 | **miss** |
| 3.5 | [0.4,0.5) | 2,120 | 0.447 | 0.483 | **miss** |
| 3.5 | [0.5,0.6) | 1,593 | 0.549 | 0.592 | **miss** |
| 3.5 | [0.6,0.7) | 1,319 | 0.647 | 0.687 | **miss** |
| 3.5 | [0.7,0.8) | 928 | 0.748 | 0.780 | **miss** |
| 3.5 | [0.8,0.9) | 513 | 0.844 | 0.825 | ok |
| 3.5 | [0.9,1.0) | 87 | 0.925 | 0.874 | untested |
| 4.5 | [0.0,0.1) | 4,989 | 0.062 | 0.064 | ok |
| 4.5 | [0.1,0.2) | 4,603 | 0.146 | 0.157 | **miss** |
| 4.5 | [0.2,0.3) | 2,914 | 0.248 | 0.265 | **miss** |
| 4.5 | [0.3,0.4) | 1,770 | 0.347 | 0.386 | **miss** |
| 4.5 | [0.4,0.5) | 1,370 | 0.447 | 0.472 | ok |
| 4.5 | [0.5,0.6) | 962 | 0.546 | 0.568 | ok |
| 4.5 | [0.6,0.7) | 644 | 0.647 | 0.683 | ok |
| 4.5 | [0.7,1.0) | 509 over three bins | | | untested |
| 5.5 | [0.0,0.1) | 9,032 | 0.046 | 0.048 | ok |
| 5.5 | [0.1,0.2) | 3,879 | 0.144 | 0.162 | **miss** |
| 5.5 | [0.2,0.3) | 1,902 | 0.247 | 0.275 | **miss** |
| 5.5 | [0.3,0.4) | 1,265 | 0.346 | 0.363 | ok |
| 5.5 | [0.4,0.5) | 770 | 0.447 | 0.412 | ok |
| 5.5 | [0.5,0.9) | 913 over four bins | | | untested |

The full per-bin listing, untested bins included, is in the run output and can be rebuilt from `data/receptions_stage_a.csv`.

### Folds: chosen alpha and NB r (model / baseline)

alpha is chosen by the mean Poisson deviance pooled over the inner held-out seasons.

| Test season | Train | Inner held-out | alpha | r WR | r TE | r RB | Train n | Test n |
|---|---|---|---|---|---|---|---|---|
| 2019 | 2016–2018 | 2018 | 0.01 | 10.03 / 8.03 | 8.20 / 7.10 | 4.22 / 3.80 | 7,825 | 2,818 |
| 2020 | 2016–2019 | 2018–2019 | 0.001 | 9.12 / 7.49 | 8.21 / 6.54 | 4.91 / 4.48 | 10,643 | 2,928 |
| 2021 | 2016–2020 | 2018–2020 | 0.001 | 8.74 / 7.39 | 8.80 / 6.65 | 4.81 / 4.24 | 13,571 | 3,098 |
| 2022 | 2016–2021 | 2018–2021 | 0.001 | 8.91 / 7.49 | 9.73 / 7.30 | 4.57 / 4.02 | 16,669 | 2,979 |
| 2023 | 2016–2022 | 2018–2022 | 0.001 | 8.43 / 7.11 | 9.17 / 7.10 | 4.64 / 4.09 | 19,648 | 2,999 |
| 2024 | 2016–2023 | 2018–2023 | 0.001 | 8.44 / 7.13 | 9.26 / 7.44 | 4.83 / 4.22 | 22,647 | 2,926 |
| 2025 | 2016–2024 | 2018–2024 | 0.001 | 8.51 / 7.22 | 8.57 / 7.08 | 5.20 / 4.46 | 25,573 | 2,831 |

No r fit reached the 200 cap, so no group fell back to Poisson. The inner deviances of the four alphas were within about 0.01 of each other in every fold, and 0.001, the bottom of the grid, won from 2020 on.

### Coverage

Shape checks were also run before the code commit (row counts, column names, join match rates, feature missing rates). No outcome was computed in them.

- Assets: `stats_player_week`, `snap_counts` and `injuries` for 2016–2025, plus `players.parquet`, downloaded from the nflverse-data releases into `data/nflverse_cache/`. `games.csv` came from `nflverse/nfldata` on GitHub, because `nfl_data_py.import_schedules`' host does not resolve from the container. PBP 2016–2025 was filled by `src.build_features`. Its side effect on the tracked `data/team_game_*.csv` was reverted, not committed.
- Games: 2,639 REG games, all with `spread_line` and `total_line`. The pace frame covers 5,278 team-games, all matched.
- Snap join (PFR → GSIS through `players.parquet`): 241,911 REG snap rows, 204 with no `gsis_id` match, dropped. By season: 2016 7, 2017 0, 2018 0, 2019 30, 2020 21, 2021 34, 2022 8, 2023 13, 2024 39, 2025 52.
- Universe rules, applied in order to the 241,707 joined rows:

| Rule | Excluded |
|---|---|
| Position group not WR/TE/RB | 180,108 |
| `offense_snaps` = 0 | 4,163 |
| Fewer than 3 played games in the prior 8 | 4,554 |
| `rec_mean_8` < 1.5 | 24,478 |
| **Universe** | **28,404** |

- Universe by season: 2016 2,251, 2017 2,759, 2018 2,815, 2019 2,818, 2020 2,928, 2021 3,098, 2022 2,979, 2023 2,999, 2024 2,926, 2025 2,831. Scored: 20,579 (2019–2025), of which 17,761 fall in the gate seasons.

## Builder choices where the spec is silent

All of these were fixed in the code's docstrings before the run.

- A played game is a REG snap row with `offense_snaps` > 0. A player's windows are his last 8 (or 4) played games for any team, across seasons. Team targets in a window are his team's targets in those same games.
- `role_trend` is the mean `offense_pct` of the last 2 played games minus that of the 6 before them, within the 8-game window.
- `depth_rank` ranks the player's `snap_pct_8` among his same-position teammates as of kickoff. A teammate counts if his last played game before kickoff was for this team and within its last 8 games, so realised participation in the target game is not used. Ties share the better rank, and the rank is capped at 3.
- Injured teammates come from the team's report for that season and week, with the position group from `players.parquet`. Each one contributes his `target_share_8` as of kickoff, or 0 with fewer than 3 played games.
- `qb_out` names the QB (position group) with the most attempts in the team's last 8 games, with ties broken by the lower `player_id`.
- The opponent features are the opponent's 8-game means (min 3) of the receptions its opponents' players of the position group made, and of its opponents' neutral `pass_oe`.
- The EB priors and fills come from the training-season universe rows. The catch prior is receptions / targets and the aDOT prior is air yards / targets.
- alpha is chosen by the pooled inner mean Poisson deviance (ties go to the smaller alpha). The baseline's r is fitted on the same inner held-out rows. A fit "reaches" 200 within 0.01.
- In CRPS, the lumped P(R ≥ 15) is placed at 15.

## What was not done

- Stage B (`src/props/receptions_vs_market.py`), live logging and Sheet tabs. Under the spec's rule, the FAIL ends Step 5.
- No re-run, no recalibration and no other alpha or r settings.
- No edits to `models/`, the workflows, the Worker, `src/config.py`, `src/market_consensus.py`, `src/topdown_replay.py`, `src/topdown_v2.py`, `src/total_pricing.py` or `src/totals/*`. No Odds API calls and no Sheet writes.
