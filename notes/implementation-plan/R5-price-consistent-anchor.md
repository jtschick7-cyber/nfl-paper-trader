# R5. Price-consistent market anchor for v3/v4

**Review sections:** F3, M4, M5, b1, b2, a4. **Effort:** M. **Branch:** `claude/r5-price-anchor`. **Depends on:** R4 (`shadow_decision`, the backtest) and R2 (`market_implied_mean`).

Shadow tabs only. v1 is not touched. This is the only task that retrains artifacts.

## Problem

Today: `mu_blend = -home_line + β·(model_mu + home_line)`, then key numbers, then a logistic calibrator.

- The anchor sets the distribution's **mean** to the line and ignores the price. With no model input the cover probability at the market line ranges from 0.465 to 0.535 depending on the line, and a side priced +110 is treated as a 50% proposition.
- The calibrator on top of β is fitted mostly on that line-driven variation (96.5% of its input variance). Its slope is 0.36 ± 0.45.

## Target

With zero model disagreement, the model's probability equals the market's de-vigged probability at the offered line and price. Then the edge against break-even is about minus half the vig, at any line and any price, and can never trigger a pick.

## Design

For a game with `home_line`, opening `home_odds`, `away_odds`, optional `total_line`:

1. `q_home` = de-vigged home probability from the two prices. Missing or non-finite price: assume −110 both sides (`q_home = 0.5`) and flag the row.
2. `m_star = market_implied_mean(home_line, q_home, sigma, total_line)` (from R2, `src/key_numbers.py`).
3. `d = model_mu - m_star`.
4. `mu = m_star + β·d`.
5. `p_home = calculate_spread_probabilities(mu, sigma, home_line, total_line)["cond_home_win"]`.
6. No logistic calibrator for v3/v4: store `calibrator = ("identity", None)`. β is the shrinkage. `fit_calibrator` is still run on the walk-forward data as a **diagnostic**; store its slope and standard error in the metadata.
7. Edges and the 2% threshold are unchanged.

β is refitted under the new definition: no-intercept OLS of `actual_margin - m_star` on `model_mu - m_star` over walk-forward predictions, clipped to [0, 1], exactly as `src/market_blend.py:24-45` does now with `m_star` in place of `-home_line`.

### One open modelling choice, to settle in the R4 backtest, not by argument

`m_star` can sit more than a point from the line (−2.5 at even prices gives 1.465), because the engine puts so much mass on 3. That makes `d` partly a statement about the engine. Evaluate both in R4 and keep the better walk-forward fit:

- **A:** `d = model_mu - m_star` (as above, internally consistent).
- **B:** `d = model_mu - (-home_line)` with `mu = m_star + β·d` (disagreement measured against the line, anchor still price-consistent).

Both satisfy the target. Record the choice and the two β fits in the PR.

## Changes

- `src/market_blend.py`: add `blend_margin_anchored(model_mu, m_star, beta, line_mu=None)` and let `fit_market_beta` accept an explicit anchor array. Keep the old functions; the old artifacts and the legacy test still use them.
- `src/decision.py` (`shadow_decision`): branch on an artifact field `anchor = "price"`; absent means today's behaviour. v2 keeps today's behaviour.
- `src/train_model_v3.py`, `src/train_model_v4.py`: compute `m_star` per training row from nflverse `home_spread_odds`/`away_spread_odds`; fit β on the anchored quantities; write `anchor`, `anchor_variant` (A or B), `market_beta`, `market_beta_se`, `calibrator_diagnostic_slope`, `calibrator_diagnostic_se` to the artifact and metadata. Sigma: standard deviation of `actual_margin - mu` on the walk-forward rows, as now.
- `src/log_v3_shadow.py`, `src/log_v4_shadow.py`: nothing beyond what `shadow_decision` needs (they already pass both prices and the total). They pick up the new `artifact_id` automatically (R2).
- `src/compare_models.py`: nothing. R2's `artifact_id` already separates old and new rows; add the new artifact to the in-week block as its own line rather than merging it with rows from the old artifact.
- `README.md`: rewrite "Market blend" and "`no_edge` calibrator" for v3/v4.

## Validation in the R4 backtest, before any retrained artifact is committed

All must hold:

1. **Zero-disagreement probe returns no picks**, at any price, including +110/−130 and +115/−140, and `p_home` equals `q_home` to 1e-6.
2. **Calibration:** in walk-forward data, cover rate by `p_home` bucket is within two standard errors of predicted in every bucket with n ≥ 200. In particular the buckets that today's anchor gets wrong (the review found 52.4% predicted against 49.2% actual) are gone.
3. **Diagnostic calibrator slope** has a confidence interval that includes 1. If it clearly does not, stop and report rather than adding the calibrator back.
4. β and its standard error are reported for variants A and B. β near zero is an acceptable outcome; it is what the data has said so far.
5. Bet count and ROI are reported. Zero bets is acceptable. A burst of bets is a red flag: check the probe first.

## Deployment

- Retrain locally (`python -m src.train_model_v3`, `python -m src.train_model_v4`), commit the `.pkl` and metadata files in the PR.
- Merge so that the first run with the new artifacts is a Tuesday opening run, never mid-week.
- Old rows keep their old `artifact_id`. Nothing is re-logged or rewritten. Note that v3/v4 dedupe on `(game_id, home_line, home_odds)` (`log_v3_shadow.py:122-131`), so games already logged that week by the old artifact are not logged again; that is the reason for the Tuesday boundary.
- Requirements stay pinned; the pickles must still load under numpy 1.26 / scikit-learn 1.7.2 (see `requirements.txt`).

## Tests (write first)

- `tests/test_decision.py`: the `xfail` test from R4 (zero disagreement at −3, +110/−130 must be `PASS`) now passes; remove the marker. Add: for a grid of lines (−10 to +10 by 0.5) and prices (−130 to +115), zero disagreement gives `p_home == q_home` within 1e-6 and `decision == "PASS"`.
- `tests/test_market_blend.py`: anchored β fit recovers a known β on synthetic data for both variants; β = 0 returns `m_star`; β = 1 returns the model margin (variant A).
- A positive case: a 6-point disagreement with β = 0.5 at −110/−110 produces a pick on the model's side, so the rule is not simply incapable of betting.
- The R4 "cannot reach 2% up to 60 points" lock is updated to describe the new artifact, not deleted without replacement.
- `joblib.load` of each retrained artifact works and `shadow_decision` runs on it.

## Done when

- `pytest tests/` green.
- The R4 report for the new rule is in the PR next to the baseline report, and the five validation points are answered explicitly.
- `scripts/review_repro/anchor_and_price_checks.py` section 2 shows `PASS` on every row for v3 and v4 (v2 is unchanged and will still show its artifact picks; say so in the PR).
- After the first live run, V3/V4 rows carry the new `artifact_id`.

## Out of scope

- v2 (its calibrator mismatch and the excluded 2025 season are separate, lower-priority items in the review).
- Removing totals-conditioned sigma. Report whether calibration is better with or without it in the R4 run; act on it in a later task.
- Lowering the 2% threshold or promoting any model.
