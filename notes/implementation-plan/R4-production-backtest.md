# R4. Backtest the production decision rule

**Review sections:** independent review §1 (coherence), b6, M4, M5. **Effort:** M. **Branch:** `claude/r4-production-backtest`. **Depends on:** nothing. Offline; touches no Sheet and no artifact.

## Why

`src/backtest_epa.py` and `src/backtest_model.py` do not use the market blend or the break-even edge, so nothing tests the rule that runs live. A backtest of the real rule would have shown before deployment that v3/v4 cannot bet and that a long price alone triggers a pick. Every later model change (R5 first) gets judged here.

## Step 1: one decision function, shared by loggers and backtest

New `src/decision.py`, pure, no I/O:

```python
def shadow_decision(raw_mu, home_line, home_odds, away_odds, *, sigma, market_beta,
                    calibrator, total_line=None, min_edge=MIN_EDGE_THRESHOLD) -> dict
```

Returns `mu_blend`, `p_home_discrete`, `p_push`, `p_home_cal`, `edge_home`, `edge_away`, `best_side`, `best_line`, `best_odds`, `best_edge`, `raw_lean_pts`, and `decision` (`PICK` or `PASS`, before the integrity guard).

It must reproduce today's behaviour exactly:

- v3: `log_v3_shadow.py:374-431` (blend if `market_beta` is not None, key numbers without total, calibrate, break-even edges).
- v4: `log_v4_shadow.py:430-478` (same, with `total_line`).
- v2: `log_v2_shadow.py:299-336` (`market_beta=None`).

Then make the three loggers call it. **This is a refactor: outputs must not change.** Before touching the loggers, write a characterization test that feeds fixed inputs through the current inline code path (copy the arithmetic into the test) and through `shadow_decision`, and asserts equality to 1e-9 for all three models, with and without a total, at integer and half-point lines.

## Step 2: `src/backtest_production.py`

Walk-forward by season, 2014–2025 test seasons (2012–2013 are the first training window), using the v3 pipeline; v4 behind a flag (its dataset build is slow).

For each test season S:

1. Build the dataset once with `build_v3_training_dataset` (`src/train_model_v3.py:105`) for 2012–2025.
2. Train on seasons < S: `StandardScaler`, `fit_constrained_ridge(alpha=15.0)`.
3. Fit β and the calibrator on walk-forward predictions **within the training seasons only** (the same nested procedure as `train_model_v3.py:262-302`). Do not reuse the committed artifact's β for past seasons: that leaks.
4. For each game in S, call `shadow_decision` with nflverse `home_line = -spread_line`, `home_spread_odds`, `away_spread_odds`, `total_line`.
5. Grade against the final score.

`scripts/review_repro/v3_beta_rederive.py` already does the dataset build and a single walk-forward β fit; start from it.

Report, per season and overall:

- games, bets, bets per season, hit rate with standard error, units at the actual price, ROI;
- how many bets had `abs(raw_lean_pts) < 0.5` (bets with no real model input);
- how many bets were at a price of +105 or longer;
- calibration table: predicted `p_home_cal` bucket against actual cover rate;
- **zero-disagreement probe:** re-run with `raw_mu = -home_line` for every game. Any `PICK` here is a price or line artifact. Print the count and the price of each.
- β and its standard error per fold.

Output goes to stdout and to `data/backtest_production_*.csv` (add the pattern to `.gitignore`). No model files are written. No Sheet access.

Limits to state in the docstring: nflverse lines are closing lines, so this tests the rule against the close, not against the in-week open the live system bets; and one 2012–2025 game has no spread price (skip it).

## Step 3: lock the current behaviour in tests

Using small fixtures, not the full backtest:

- With today's artifacts' parameters (v3: σ 12.8145, β 0.0369, slope 0.3614), `shadow_decision` at −110/−110 returns `PASS` for raw disagreements up to 60 points on a dead number. Mark this test with a comment that R5 is expected to change it.
- With zero disagreement at −3, +110/−130, it currently returns `PICK`. Write this as an `xfail(strict=True)` test asserting `PASS`, named for R5, so R5 turns it green.

## Expected results on the current rule

If the numbers differ much, look for a bug before believing them.

- v3: essentially no bets except where one side is priced about +109 or longer, and the zero-disagreement probe returns nearly the same set.
- β per fold near zero, with standard errors around 0.07–0.11.
- Raw-lean hit rate against the close near 50%.

## Tests

- `tests/test_decision.py`: the characterization test, plus the two locks above.
- `tests/test_backtest_production.py`: on a 3-season synthetic dataset, no row of a test season appears in its own training, β fitting, or calibration data (assert on game ids); the zero-disagreement probe is flagged when the fixture includes a +115 price.

## Done when

- `pytest tests/` green; the logger refactor changes no test expectation.
- `python -m src.backtest_production` runs in a few minutes and prints the report.
- README "Useful commands" lists it, and the sentence telling readers to judge models on the live scorecard alone is updated to mention it.
- The baseline report (current rule) is pasted into the PR description, so R5 has something to beat.

## Out of scope

- Changing any decision logic (that is R5).
- Backtesting v1 or v2 under their own training protocols. A flag for v2 is welcome if it is cheap; do not block on it.
- Retiring `src/backtest_epa.py` and `src/backtest_model.py`. Note in the README that they do not model the production rule.
