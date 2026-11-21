# Audit Phase 2: EPA feature study (offline, 2026-11-21)

Walk-forward v3 backtest, price anchor A, test seasons 2014–2025 (3,150 decided games), closing
lines, flat 1u at the actual price. Reproduce with
`.venv/bin/python -m src.backtest_production --model v3 --anchor price --feature-set <set>`.
No artifact was retrained; the live loggers still use the legacy features.

| Feature set | Change | Log loss | Brier | Margin RMSE | Mean β | Bets | Hit % | Units |
|---|---|---|---|---|---|---|---|---|
| legacy | committed v3 features | 0.69356 | 0.25020 | 12.8728 | 0.134 | 151 | 51.7 ± 4.1 | +2.2 |
| sign | correct-sign EPA matchup (P0-1) | 0.69369 | 0.25026 | 12.8696 | 0.125 | 138 | 45.9 ± 4.3 | −13.3 |
| sign_scramble | + scrambles as passes only (P0-2) | 0.69366 | 0.25025 | 12.8694 | 0.116 | 114 | 44.6 ± 4.7 | −13.6 |
| split | + offence and defence as separate features | 0.69367 | 0.25025 | 12.8711 | 0.141 | 192 | 48.2 ± 3.6 | −10.4 |
| split_neutral | + neutral sites (Elo, home_field feature) (P1-2) | 0.69351 | 0.25018 | 12.8685 | 0.120 | 131 | 46.2 ± 4.4 | −12.4 |

All five: zero-disagreement probe 0 picks; every calibration bucket with n ≥ 200 within 2 SE.

## Reading

- The fixes are correct (unit tests lock the signs and the play split) but none of them changes
  out-of-sample cover prediction beyond noise: log-loss differences are ≤ 0.00018, against a
  per-game SD that makes ~0.0005 the smallest detectable gap at n = 3,071.
- Margin RMSE improves slightly with every fix (12.873 → 12.869), as the univariate study
  predicted, but the market already prices that information.
- Hit-rate and unit differences are all within 1–1.5 SE. The legacy +2.2u is not evidence that the
  bug helps.
- Split offence/defence: the ridge zeroes defensive pass EPA and defensive success rate and puts
  most weight on offensive success rate. The split raises β in every season 2018–2025
  (2025: 0.094 vs 0.066), the one consistent sign of extra independent information, but it does
  not show up in cover log loss.
- The legacy rerun differs a little from R5's recorded numbers (151 vs 142 bets, log loss
  0.69356 vs 0.69350) because the dataset now breaks kickoff ties by game_id (P1-4), which
  moves TimeSeriesSplit boundaries.

## Recommendation

Do not retrain mid-season for these fixes: no gate shows a betting improvement, and a retrain
would restart the R3 raw-lean test. Fold them in at the next planned retrain (off-season), using
split_neutral, which is best or tied-best on log loss, Brier and RMSE. Bigger gains need new
information (Phase 3 market structure, Phase 4 features), not these corrections.

## Not done in this study

- P0-7 (v4 QB shock) and the v4 backtest: v4 shares the EPA features; its QB term is next.
- P1-5 vectorised builder: each dataset build takes about 2 minutes (15 minutes per full run with
  five in parallel), slow but workable.
