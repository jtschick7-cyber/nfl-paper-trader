# P3. Offline NFL totals kill test: result

**Spec:** [`implementation-plan/P3-totals-kill-test.md`](implementation-plan/P3-totals-kill-test.md) (rules frozen 2026-11-24). **Run:** 2026-11-24, one launch, after the code and tests were committed (`8506432`).

```bash
.venv/bin/python -u -m src.totals.backtest_totals   # writes data/totals_backtest_games.csv (gitignored)
```

## Verdict: KILL

Neither keep path is met on 2020–2025:

- **β path** needs β̂ > 0 with t ≥ 1.5 and Δ log loss < 0. Result: β̂ = +0.068 ± 0.100 (t = +0.68), and Δ = +0.00011 (the model is worse). Not met.
- **Wind path** needs mean (actual − t*) with |t| ≥ 2.0 on wind ≥ 15, and Δ log loss < 0 on that subset. Result: −1.60 ± 1.23 (t = −1.30), and Δ = +0.00098. Not met.

On the frozen features, a bottom-up totals model adds nothing to the price-adjusted closing total. No live totals pipeline gets built, and bottom-up totals are closed. The top-down totals work in P2 is a separate question and is not affected.

## Folds

`alpha` was chosen by a 5-fold `TimeSeriesSplit` inside the training seasons. β is the nested no-intercept slope on the walk-forward predictions inside those seasons, clipped to [0, 1] for the blend.

| Test season | Train games | alpha | β raw ± SE (n) | β used |
|---|---|---|---|---|
| 2014 | 512 | 100 | +0.286 ± 0.195 (425) | 0.286 |
| 2015 | 768 | 100 | +0.111 ± 0.169 (640) | 0.111 |
| 2016 | 1024 | 100 | +0.175 ± 0.152 (850) | 0.175 |
| 2017 | 1280 | 100 | +0.043 ± 0.130 (1065) | 0.043 |
| 2018 | 1536 | 100 | +0.046 ± 0.121 (1280) | 0.046 |
| 2019 | 1792 | 100 | −0.008 ± 0.110 (1490) | 0.000 |
| 2020 | 2048 | 100 | +0.037 ± 0.103 (1705) | 0.037 |
| 2021 | 2304 | 100 | −0.061 ± 0.096 (1920) | 0.000 |
| 2022 | 2576 | 100 | −0.054 ± 0.092 (2145) | 0.000 |
| 2023 | 2847 | 100 | −0.056 ± 0.084 (2370) | 0.000 |
| 2024 | 3119 | 100 | +0.014 ± 0.080 (2595) | 0.014 |
| 2025 | 3391 | 100 | +0.005 ± 0.077 (2825) | 0.005 |

Every fold picked the largest `alpha` on the grid, the most shrinkage allowed. Each in-sample β shrinks toward zero as the training seasons accumulate, the same pattern as the spread models after 2019.

## Pooled β̂ (out of sample)

A no-intercept OLS of (actual − t*) on (raw_mu − t*), using the unblended ridge prediction.

| Subset | 2014–2019: n | β̂ ± SE (t) | 2020–2025: n | β̂ ± SE (t) |
|---|---|---|---|---|
| All | 1536 | −0.046 ± 0.109 (−0.42) | 1615 | **+0.068 ± 0.100 (+0.68)** |
| Outdoor, wind ≥ 15 | 122 | +0.116 ± 0.278 (+0.42) | 114 | −0.814 ± 0.330 (−2.47) |
| Indoor | 361 | −0.011 ± 0.241 (−0.04) | 521 | +0.382 ± 0.181 (+2.11) |

## Over/under log loss, pushes excluded

Δ = model − market, with a paired SE over games. A negative Δ favours the model.

| Subset | Period | n | Model | Market | Δ ± SE |
|---|---|---|---|---|---|
| All | 2014–2019 | 1524 | 0.69379 | 0.69323 | +0.00056 ± 0.00070 |
| Wind ≥ 15 | 2014–2019 | 121 | 0.68936 | 0.68996 | −0.00061 ± 0.00393 |
| Indoor | 2014–2019 | 359 | 0.69208 | 0.69205 | +0.00003 ± 0.00129 |
| All | 2020–2025 | 1599 | 0.69320 | 0.69309 | **+0.00011 ± 0.00008** |
| Wind ≥ 15 | 2020–2025 | 113 | 0.69013 | 0.68915 | **+0.00098 ± 0.00044** |
| Indoor | 2020–2025 | 517 | 0.69471 | 0.69451 | +0.00020 ± 0.00013 |

The 2020–2025 blend used β ≤ 0.04, so its probabilities stay close to the market's. A Δ that is small and positive is what a model with nothing to add looks like.

## Wind check, no model: mean (actual − t*) ± SE

| Subset | 2014–2019: n | Mean ± SE (t) | 2020–2025: n | Mean ± SE (t) |
|---|---|---|---|---|
| Outdoor, wind ≥ 15 | 122 | −1.37 ± 1.11 (−1.23) | 114 | **−1.60 ± 1.23 (−1.30)** |
| Outdoor, known wind < 15 | 1027 | +0.04 ± 0.42 (+0.10) | 828 | +0.43 ± 0.44 (+0.97) |
| Indoor | 361 | +0.04 ± 0.72 (+0.05) | 521 | +1.92 ± 0.59 (+3.27) |

High-wind games came in under the closing total in both periods, by about 1.5 points. The gap is the sign a wind under-adjustment would give, but its t is −1.2 and −1.3, short of the 2.0 the rule needs.

## Descriptive only: record at the closing price

The model bet when `p_over` ≥ 0.55 or ≤ 0.45:

- 2014–2019: 8-10, −2.65 u in 18 bets.
- 2020–2025: no bets. With β near zero the blend never left the 0.45–0.55 band.

## Rows outside the rule (descriptive, not grounds to reopen)

- **Indoor games, 2020–2025:** they finished +1.92 ± 0.59 over t* (t = +3.27), and the model's indoor β̂ is +0.38 (t = +2.11). The model's indoor Δ log loss is still positive (+0.00020), and the rule has no indoor path.
- **Wind ≥ 15, 2020–2025:** the model's β̂ there is −0.81 (t = −2.47), the wrong sign. The model's wind term over-adjusts relative to the market.

Both rows come from a search across six subsets and two periods. Under the frozen rule neither changes the verdict, and neither is a reason for a new feature set or window.

## Caveats

- **Weather:** `games.csv` records wind and temperature at game time, not as a forecast. That flatters any weather effect, and a live system would have only the forecast. The weather terms still added nothing here, so the caveat strengthens the kill rather than weakening it.
- **Closing line:** the anchor is the nflverse closing total and its price from a single source, not a multi-book close. A noisier anchor would only make the market easier to beat, and it was not beaten.
- **Power:** the wind subset is underpowered. With residual SD ≈ 13.2, 114 games give an SE of about 1.2 points, so only a market bias of about 2.5 points or more could reach |t| ≥ 2. The −1.5-point gap seen in both periods is below that bar. A real bias of that size cannot be ruled out, and this sample cannot confirm one.
- **Sample count:** the spec quoted 1,155 outdoor or open-roof games and 121 with wind ≥ 15 for 2020–2025. `games.csv` holds 1,094 such regular-season games and 114 high-wind ones (26, 29, 13, 12, 15, 19 by season). The pipeline reproduces those raw counts exactly. The spec's figures probably included playoff games, which the frozen walk-forward excludes.

## Implementation choices the spec left open

These were fixed in code before the run and are recorded in the `src/totals/features.py` docstring:

- **Offensive plays** are pass or run plays, regular season only.
- **Neutral pace:** a gap counts only when both plays are neutral.
- **Red-zone trip:** a pass, run, field-goal, kneel or spike snap at `yardline_100` ≤ 20. It counts as a TD when the drive has an offensive touchdown.
- **Missing values:** an unknown temperature gives `cold` 0, and a missing roof counts as outdoor.
- **Calm subset:** the "wind < 15" row includes only outdoor games with a known wind.
