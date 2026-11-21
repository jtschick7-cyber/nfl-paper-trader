# Audit P0-7: v4 QB shock fix and v4 feature study (offline, 2026-11-21)

Walk-forward backtest, price anchor A, test seasons 2014–2025 (3,150 decided games), closing lines,
flat 1u at the actual price. Reproduce with
`.venv/bin/python -u -m src.backtest_production --model v4 --anchor price --feature-set <set>`.
No artifact was retrained; the live v4 logger still uses the legacy features and override lookup.

| Model / set | Change | Log loss | Brier | RMSE | Mean β | Bets | Hit % | Units |
|---|---|---|---|---|---|---|---|---|
| v4 legacy | committed features | 0.69350 | 0.25017 | 12.8741 | 0.125 | 138 | 49.6 ± 4.3 | −3.3 |
| v4 qb | legacy EPA + fixed QB shock | 0.69368 | 0.25026 | 12.8754 | 0.160 | 244 | 52.3 ± 3.2 | +6.4 |
| v4 split_neutral | Phase 2 EPA fixes | 0.69349 | 0.25016 | 12.8702 | 0.100 | 82 | 40.7 ± 5.5 | −16.2 |
| v4 split_neutral_qb | + fixed QB shock | 0.69370 | 0.25027 | 12.8713 | 0.140 | 184 | 49.2 ± 3.7 | −6.8 |
| v4 split_neutral_qb_teams | + OAK/SD/STL codes | 0.69352 | 0.25018 | 12.8683 | 0.137 | 189 | 50.3 ± 3.7 | −2.5 |
| v3 split_neutral (Phase 2) | for reference | 0.69351 | 0.25018 | 12.8685 | 0.120 | 131 | 46.2 ± 4.4 | −12.4 |
| v3 split_neutral_teams | + OAK/SD/STL codes | 0.69340 | 0.25012 | 12.8665 | 0.111 | 98 | 47.4 ± 5.1 | −6.6 |

All seven: zero-disagreement probe 0 picks (max |p − q| 8e-13); every calibration bucket with
n ≥ 200 within 2 SE. β per fold is in `data/backtest_production_<model>_priceA_<set>_seasons.csv`
(for example, v4 legacy: 0.03, 0.06, 0.22, 0.34, 0.16, 0.13, 0.10, 0.10, 0.10, 0.13, 0.06, 0.07).
v4 legacy differs slightly from R5's recorded v4 because the builder now breaks kickoff ties by
game_id, as v3 does since Phase 2.

## The fixes (opt-in, defaults unchanged)

- `qb_version="fixed"`: the shock is the starter's rating minus the dropback-weighted rating of the
  QBs behind the team's last 6 games, both shrunk the same way (exactly 0 for an unchanged
  starter). A QB who was never his team's main passer shrinks toward −0.10, and a backup with no
  rows is rated −0.10 instead of a shock of 0. Training uses nflverse's actual starters
  (`home_qb_id`/`away_qb_id`), which is better information than the live Tuesday decision has.
- `fetch_pregame_starter_overrides(version="fixed", week=…)`: only that week's injury report
  (none on a Tuesday, so no automatic override); backup from an earlier season, else
  `REPLACEMENT_BACKUP`.
- New bug, not in the audit: the schedule uses OAK/SD/STL for 2012–2019, the EPA/QB/turnover tables
  LV/LAC/LA, so those teams' games (168 in the test seasons, about 7% of all rows) got zero EPA,
  QB and fumble features and no division flag. `canonical_teams=True` fixes it (v3 and v4).

## Reading

- The fixed QB shock removes the healthy-starter artifact: on games with no starter change its SD
  falls from 1.51 to 0.89 points, and games with |shock| > 1 pt fall from 1,194 to 415.
- But QB information does not beat the closing line. On the 628 test games with a starter change,
  the fixed shock's slope against the cover margin is +0.15 ± 0.27 (r = 0.02): the market already
  prices the starter. The fixed shock gives the ridge a larger coefficient and raises β, but cover
  log loss gets slightly worse (+0.0002, within noise). Its +6.4u is 0.8 SE and is not evidence.
- The team-code fix is the only change that improves every fit metric in both models (v3 log loss
  0.69351 → 0.69340, RMSE 12.8685 → 12.8665). It is still within noise but costs nothing.
- At its best, v4 (0.69349) does not beat v3 (0.69340). v4's extra terms (QB shock, fumble luck)
  add nothing measurable beyond v3.

## Recommendation: retire v4 at the off-season

- No mid-season retrain: no gate improves, and a retrain would restart the R3 raw-lean test.
- Keep logging v4 as a shadow until the season ends (it costs nothing, and R3 is collecting it),
  then retire it unless the live raw-lean test clearly favours it. If it is kept, retrain it with
  `split_neutral_qb_teams` and wire the fixed override lookup into the logger at the same time.
- Add `canonical_teams` to the off-season v3 retrain (`split_neutral_teams`).
