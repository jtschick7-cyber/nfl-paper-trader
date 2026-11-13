# Independent second review of `notes/shadow-model-findings.md`

Reviewed 2026-11-13 at `main` = `99dae0c`. Read-only: no source file was changed. Checks were run against the code, the model artifacts in `models/`, and the live Sheet (through `scripts/read_sheet.py`, which is view-only).

The repro scripts are in `scripts/review_repro/`. The build spec for recommendations R1–R5 is in [`notes/implementation-plan/`](notes/implementation-plan/README.md).

Issue labels: **W1–W3** are the three "What's wrong" items, **F1–F5** the five "Fix plan" items, **S** the repo-state and scorecard sections, **D** the decision point. The findings doc gives no severities, so "re-rank" compares against the priority its ordering implies.

## Summary

| # | Claim in the findings doc | Verdict | Severity (mine) | Agreed fix? |
|---|---|---|---|---|
| S | Repo state and scorecard table | Partially agree: numbers right, repo state stale, table needs caveats | Medium | n/a |
| W1 | The edge bar now includes the vig, so shadow models can't reach it | Partially agree: facts right, cause wrong | High (up) | n/a |
| W2 | v3/v4 barely move off the market; calibrators report `no_edge` | Partially agree: β right, `no_edge` claim false | High (same) | n/a |
| W3 | v1 still measures edge against the no-vig price | **Disagree**: fixed on Nov 2 | Low (down) | n/a |
| F1 | Track raw-lean CLV for v2–v4 | Partially agree: right idea, would crash two loggers as written | High | Y, with changes |
| F2 | Move v1's edge to the after-vig measure | **Disagree**: already done | none | N |
| F3 | Leave the β blend and 2% threshold alone | Partially agree: keep the bar, fix the anchor | Medium | Y, with changes |
| F4 | Unit tests for the raw lean | Agree, incomplete | Low | Y, extended |
| F5 | Commit `scripts/read_sheet.py` on a `claude/` branch | Stale: merged in PR #12 | none | N |
| D | Decide at week 14–15 on 40+ leans | Partially agree: no usable decision rule | Medium | Y, with changes |

The five things the original review missed matter more than anything in it. In order:

1. **v1's baseline has look-ahead bias.** Opening lines are captured a median 9 days before kickoff, and later runs add picks at that old line using results the line pre-dates. This applies to 11 of the 23 week 8–10 picks I could time (M1, M2).
2. **Every v2 and v3 bet on the scorecard is an artifact**, not a model opinion (M4, M7).
3. **v3/v4 probabilities come from the line number, not the model.** 96.5% of the variance the calibrator is fitted on is the spread itself (M5).
4. **The rest-day feature is wrong for every look-ahead evaluation**, in all four models (M3).
5. **The promotion rule is too weak at 40 bets.** A model with zero edge clears the +0.5% CLV bar about 28% of the time (M9).

## Per-issue detail

### S. Repo state and scorecard

**Verdict: partially agree.**

- **Repo state is stale.** The doc says "Head is `b555cc8`" and "Not committed: `scripts/read_sheet.py`". Both stopped being true in the commit that added the doc: `32c8215` ("Add read-only Sheet reader and shadow-model findings note"), merged as `99dae0c`.
- **Scorecard numbers match the Sheet.** `Model_Scorecard` (updated 2026-11-13 13:04 UTC) shows v1 18 bets, 7–11, 61.5%, −0.52%; v2 6, 2–4, 25.0%, −1.18%; v3 4, 3–1, 0.0%, −2.94%; v4 0.
- **The table is presented without three caveats that change its meaning:**
  - The v2 and v3 rows are artifacts (M4, M7).
  - At least 9 of v1's 17 closed bets used results the line pre-dated (M2).
  - "CLV Beat 61.5%" drops ties from the denominator (8 of 13), next to an average CLV of 0.00 points and −0.52% (`src/compare_models.py:247-251`).

**Severity: Medium.** The table is the evidence base for everything else in the doc.

### W1. "The edge bar now includes the vig, so the shadow models can't reach it"

**Verdict: partially agree.** The facts are right. The explanation, that honest edges fell 2.4 points below the bar, is wrong: there were no honest edges before the change.

**Evidence that the facts hold.**
- `src/config.py:20`: `MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", "0.02"))`.
- `src/log_v2_shadow.py:318`: `edge_home = p_home_cal - american_to_implied_prob_scalar(home_odds)`. The same form is at `src/log_v3_shadow.py:413` and `src/log_v4_shadow.py:460`.
- At −110 the break-even is 52.38%, so a bet needs 54.38%.
- Average edges from the Sheet, first row per game: v2 +1.03% (weeks 8–9) to −1.14% (weeks 10–11); v3 +1.24% to −1.33%; best weeks 10–11 edge +0.54%. The doc's figures are within rounding.

**Evidence that the cause is wrong.**
- **All six v2 bets were price artifacts.** The model probability was within 1.05 points of 50% every time, and every bet was on the cheaper-priced side. The old measure subtracted the no-vig fair price of that side (0.468–0.489), which manufactured a 2% "edge".

  | Game | Side, odds | Model p | Fair p | Old edge | Edge after vig |
  |---|---|---|---|---|---|
  | 08 CAR@GB | HOME −105 | 0.5105 | 0.4892 | +2.13% | −0.17% |
  | 08 MIN@DET | AWAY +100 | 0.4986 | 0.4783 | +2.03% | −0.14% |
  | 08 LV@NYJ | AWAY +105 | 0.4895 | 0.4675 | +2.20% | +0.17% |
  | 09 CIN@ATL | HOME +100 | 0.5053 | 0.4783 | +2.70% | +0.53% |
  | 09 DAL@IND | HOME −105 | 0.5103 | 0.4892 | +2.11% | −0.19% |
  | 09 DET@MIA | HOME −105 | 0.5096 | 0.4892 | +2.04% | −0.26% |

- **All four v3 bets came from a model that has since been replaced, and bet against its own raw view.** They were logged 10-26 with sigma 13.2147 and an inverted calibrator. Raw probability for the side bet: 0.414, 0.477, 0.425, 0.381.
- **"Can't reach it" is literally true.** With the committed artifacts, at −110/−110 on a dead number:
  - v2 needs a raw disagreement of at least 16.25 points with the market.
  - v3 and v4 do not reach a 2% edge at any disagreement up to 60 points.

  Command: `.venv/bin/python scripts/review_repro/anchor_and_price_checks.py`, section 3.

**Severity: High, re-ranked up.** The doc files this under "deliberate, honest changes, not bugs". The change was correct, but the consequence is that the v2/v3 history is void and the README promotion rule (`README.md:37-44`) cannot be met by any shadow model as built.

### W2. "v3 and v4 barely move off the market line"

**Verdict: partially agree.** The β numbers are exact. One sentence is false and the section understates the shrinkage.

**Evidence that reproduces.**
- I rebuilt v3's training set in memory and re-ran the walk-forward fit: `beta=0.0369 se=0.0710 n=3050`, identical to `models/metadata_v3.json`. Command: `.venv/bin/python scripts/review_repro/v3_beta_rederive.py` (60 s, writes nothing to the repo). I did not re-derive v4's β.
- Mean distance of the blended margin from the line in the Sheet: v3 0.083 points, v4 0.092 points. The doc says 0.09 for v4.
- `src/market_blend.py:48-51`:
  ```python
  def blend_margin(model_mu, home_line, beta):
      m = market_mu(home_line)
      return m + float(beta) * (float(model_mu) - m)
  ```

**What is false.** "The calibrators for both models also report `no_edge`."
- `models/metadata_v3.json` has `"calibrator_method": "logistic"`, slope 0.3614. v4 has 0.3873. The `.pkl` files agree.
- `no_edge` was true for 53 minutes: at `bf3e4aa` (slopes −0.110 and −0.133, 816 rows), until `97761e1` retrained on 2012–2025. The doc was written eleven days later.
- With `no_edge` both sides of a −110/−110 game would show exactly −2.38%. v3's row for MIA@IND shows −1.23% and −3.53%.

**What is understated.**
- β is followed by the logistic calibrator, so the model's disagreement is shrunk twice: 0.037 × 0.361 ≈ 1.3% of it survives in v3, 1.6% in v4. "96% the market's number" should read about 98.5%.
- The calibrator slope is noise: **0.361 ± 0.453** (my refit, same data). See M5 for what it is actually fitted on.
- β by era: 2014–2019 +0.098 ± 0.107; 2020–2025 **−0.015 ± 0.095**.
- Historical record of the raw lean against the closing line, 2012–2025 out of sample:

  | Lean size | Games | Cover % | SE |
  |---|---|---|---|
  | any | 2,974 | 49.83 | 0.92 |
  | ≥ 1.5 pts | 1,843 | 50.62 | 1.16 |
  | ≥ 3 pts | 963 | 51.92 | 1.61 |
  | ≥ 5 pts | 315 | 51.75 | 2.82 |

  None is distinguishable from 50%, and none reaches the 52.38% break-even.

**Severity: High, same rank.** This is a finding about the models, not a defect. The false `no_edge` sentence on its own is Low.

### W3. "v1 is measured on the easier scale"

**Verdict: disagree.** The claim was true until Nov 2 and is false at HEAD and for every pick since.

**Evidence.**
- `src/generate_picks.py:181-182`, changed in `bf3e4aa`:
  ```python
  edge_home = p_home - american_to_implied_prob(home_odds)
  edge_away = p_away - american_to_implied_prob(away_odds)
  ```
- Sheet, week 10: WAS@NYG has `model_probability` 0.5637, `pick_odds` −105 (break-even 0.5122), `edge` 0.0515 = 0.5637 − 0.5122. MIA@IND: 0.6024 − 0.5238 = 0.0786. Both match the after-vig measure.
- Weeks 8–9 rows were written before the change and are on the old scale: MIN@DET `edge` 0.0374 = 0.5592 − 0.5217.
- The likely source of the misreading is the column `implied_probability_no_vig`, which still stores the fair price next to the new edge (`src/generate_picks.py:316`).

**What survives.** v1 is overconfident, and the evidence is stronger than the doc says: on the same scale as the others it shows edges of 2.1–9.5% on 17 of 27 week 10–11 games, with no calibration and no market input.

**Severity: Low, re-ranked down.** The real residue is that `Master_Bets.edge` mixes two definitions across the Nov 2 boundary.

### F1. Track raw-lean CLV for v2, v3 and v4

**Verdict: partially agree.** This is the right kind of fix, because the shadow models will produce no bets to grade. As written it is incomplete and would break two loggers.

**Is it correct, complete and safe?**
1. **It would crash the v2 and v3 loggers.** The plan says "Add new columns at the end of each shadow tab". Neither `ensure_headers` writes new header cells:
   - `src/log_v2_shadow.py:129` only checks `if first[0] != HEADERS[0]:` and then raises.
   - `src/log_v3_shadow.py:144` rewrites headers only when the first header differs.
   - Two unlabelled columns make the header row contain duplicate blanks, and gspread 6.2.1 `get_all_records()` raises `"the header row in the worksheet contains duplicates"`. Both loggers call it (`log_v2_shadow.py:207`, `log_v3_shadow.py:280`).
   - v4 is safe: `src/log_v4_shadow.py:143` compares the whole row.
2. **The existing "best side" cannot stand in for the lean.** It disagrees with the raw-lean side in 15 of 27 v3 rows and 24 of 42 v4 rows, because it is driven by price and line number (M4, M5). The doc's definition, raw margin minus market margin, is the right one.
3. **Leans already logged can be backfilled; waiting is unnecessary.** Raw lean = (logged margin + home line) / β.
   - v3 logs the blended margin to 4 decimals (`log_v3_shadow.py:105`), so the backfill is exact.
   - v4 rounds to 2 decimals (`log_v4_shadow.py:556`), giving ±0.12 points; small leans lose their sign. Log the raw margin from now on.
   - v2 has no blend, so "before the market blend" does not apply; its logged margin is already raw.
4. **The measurement is much noisier than the doc assumes.**
   - Leans are taken at a look-ahead line a median 9.4 days before kickoff. Open-to-close movement has SD 2.58 points across the 31 games with both lines.
   - Per-bet CLV SD is 5.5 points of win probability (v1, n = 17).
   - The three models share features, so 14 leans a week per model is not 42 independent observations.
5. **It needs baselines.** The v3 raw lean is 0.52 points toward the home team on average in the walk-forward data. A drift in look-ahead lines toward home teams or favourites would read as model skill. Report "always home" and "always underdog" alongside.
6. **The CLV maths ignores price.** `clv_win_probability` says "closing price for the pick side is not stored" (`compare_models.py:115`), but `closing_odds` and `closing_away_odds` are in `Master_Bets`. A lean at −3 +100 that closes −3 −120 scores as a tie.

**Better fix.** Do F1 with these changes, after the timing fix (R1): one shared header-migration helper for all three loggers, raw margin logged unrounded, backfill, price-aware CLV, two baselines, and a test rule written down before the data comes in (see D).

**Severity: High.** Keep it near the top, but second to R1.

### F2. Measure v1's edge after the vig

**Verdict: disagree, already implemented** (`bf3e4aa`, Nov 2; see W3). "Log the stricter edge in a new column first" is moot. v1 has bet at 2% under the stricter measure for two weeks: 10 picks in week 10, 7 in week 11.

**Remaining work.** Annotate the weeks 6–9 rows as old-scale. Do not recompute them in place.

**Severity: none.**

### F3. Leave the β blend and the 2% threshold alone

**Verdict: partially agree.** Not lowering the bar is right. "Leave alone" also freezes two defects the doc did not see.

- **A price alone still triggers a pick.** With zero disagreement between model and market, all three shadow models return PICK when a side is priced +110 or longer:
  ```
  model  home_line  home_odds  away_odds  p_home_cal  edge    decision
  v3     -3.0       110        -130       0.5025      0.0263  PICK
  v4     -3.0       110        -130       0.5027      0.0265  PICK
  v2     -3.0       110        -130       0.5004      0.0242  PICK
  ```
  A book hanging +110 on a −3 is saying that side covers about 45.7%. The blend anchors on the spread and ignores the price. Pre-game openers captured so far top out at +105, so exposure is low but not zero.
- **Stacking the calibrator on β is double shrinkage fitted on noise** (W2, M5).

**Better fix.** Keep the 2% bar. Replace the anchor with one that is consistent with the price (R3). Say in the README that under the current design the shadow models cannot bet.

**Severity: Medium.**

### F4. Tests

**Verdict: agree, incomplete.** `pytest tests/` passes today (`75 passed in 3.41s`). Beyond lean sign and CLV grading, add:
- header migration on a tab with the old header row;
- zero model disagreement never returns PICK at any price;
- backfilled v3 lean equals a freshly computed one;
- `capture_opening` ignores games outside the decision window.

**Severity: Low.**

### F5. Commit `scripts/read_sheet.py`

**Verdict: stale.** Done in `32c8215`, PR #12. Nothing to do.

### D. Decision point around week 14–15

**Verdict: partially agree.** A checkpoint is right. "Positive CLV over 40 or more games" is not a rule that can decide anything.

- With a per-bet SD of 5.5 points, the standard error at 40 leans is 0.87 points. A zero-edge model shows positive CLV half the time.
- There are six comparisons (three models, two lean-size cuts) on correlated leans.
- "Look at raising β" from 40 in-season leans is fitting β to 40 points. β should move only through the walk-forward fit.
- About 109 regular-season games remain (weeks 12–18). At SD 5.5 that gives a standard error of 0.53 points, enough to see an effect of 1.1 points or more, not 0.5. Fixing the timing (R1) should cut the SD substantially; my guess is about half.
- "Betting at open versus near close" is listed as a fallback. It is the first-order issue (M1).

**Better rule.** Fix before looking: one primary test (v3, all leans, price-aware CLV), a threshold of t ≥ 2 against the always-home baseline, and a minimum of 100 lean-games after the timing fix. Treat this season as a measurement, with decisions in the offseason.

**Severity: Medium.**

## Missed by the original review

Same area of the code as the findings, ordered by impact.

### M1. Opening lines are look-ahead lines taken 8–13 days out (High)

- `src/capture_opening.py:68-100` captures an opener for every future `PENDING` game the Odds API returns. There is no horizon.
- Sheet: week 10 openers were stamped 11-04 23:38, before week 9 was played. Week 11 openers were stamped 11-10 21:18, before week 10. Median lead across games with both lines is 9.4 days, range 3.4–13.0.
- The models were trained on features through the previous game and, for β, against the closing line. Live, they are evaluated once (the loggers dedupe per game) with a full game week missing.
- One week-9 game moved 11 points between "open" and close (JAX@BAL, −2.5 to +8.5). That is what the CLV column is mostly measuring.

### M2. Later runs add v1 picks at the old line using newer results (Critical for the baseline)

**Mechanism.**
- The workflow runs five times a week (`tuesday_opening.yml:5-6`).
- `src/generate_picks.py:248` takes `picks = scored[:MAX_PICKS]` per run, and `:105` skips games already picked. Each run therefore adds up to five more.
- Each pick is priced at `opening_line` (`:126`), whenever that was captured. There is no pick timestamp.

**Evidence.** I recomputed v1's margin from the results available at each 3-hour step between line capture and kickoff and matched it to the `mu=` value in `notes` (tolerance 0.015). Command: `.venv/bin/python scripts/review_repro/v1_pick_timing.py data/sheet_dump` (needs a local Sheet dump; see the script's docstring).

| Week | Picks | Timed | Made after prior-week results | Note |
|---|---|---|---|---|
| 8 | 8 | 8 | **8** | Lines from 10-23; picks first reproducible 10-25 20:00 to 10-27 02:00 |
| 9 | 8 | 5 | 0 | 3 unmatched |
| 10 | 10 | 10 | **3** | WAS@NYG, KC@ATL, BUF@NYJ, at lines from 11-04 |

- That is 11 of 23 timed picks. Among bets with a close, at least 9 of 17 are affected.
- Week 8 is v1's only positive CLV week (+1.24%). Week 9, which is clean on this test, is −2.34%. Eight bets is not proof, but the direction is what the bias predicts.

**Limits.** Three week-9 picks matched neither time, and week 11 cannot be separated yet.

### M3. Rest-day feature is wrong on look-ahead evaluations (High)

- All four models compute rest from each team's last *completed* game as of now, not its last game before the kickoff in question (`generate_picks.py:146-152`; the same pattern in `model_v2_core.py:415`, `log_v3_shadow.py:224-229`, `log_v4_shadow.py:185-190`).
- Examples from the reconstruction, as (home, away) days: HOU@LAC computed (21, 14), true (7, 14). BUF@MIN computed (15, 22), true (8, 8).
- v1's coefficient is 0.29 points per day, so a 14-day error moves the margin about 4 points.
- It is visible in the shadow tabs too: v3 logged `diff_rest` = 7 for WAS@NYG (a Thursday game evaluated on 11-04), where both teams had 4 days.

### M4. Shadow "picks" are produced by price, not by the model (High)

See the tables in W1 and F3. Before Nov 2 the artifact came from subtracting the no-vig price. After Nov 2 it needs +110 or longer. In both regimes the bet lands on the side the book has priced as *less* likely, which is consistent with v2's and v3's negative CLV.

### M5. The key-number engine makes the line itself look like an edge (High)

- `discrete_margin_probabilities` forces the distribution's *mean* onto the input margin (`src/key_numbers.py:121-137`). The spikes at 3 and 7 then leave the cover probability at the market line away from 50%.
- With model equal to market (same `anchor_and_price_checks.py` run as in W1, section 4):

  | Home line | −7.5 | −3.5 | −2.5 | +3.5 | +7.5 |
  |---|---|---|---|---|---|
  | P(home covers) | 0.465 | 0.481 | 0.532 | 0.519 | 0.535 |

- History does not support it. 2012–2025 closing lines: where this anchor gives the home side more than 51.5% (n = 925, mean 52.4%), home teams covered **49.2% ± 1.6**. Where it gives less than 48.5% (n = 1,320, mean 47.6%), they covered 48.5% ± 1.4.
- In the v3 calibration data, the line alone explains **96.5%** of the variance of the calibrator's input. Refitting the calibrator on the line with no model input gives 0.305 ± 0.457, against 0.361 ± 0.453 with the model. The calibrator is grading the key-number assumption, not the model.
- Sheet example: BUF@NYJ, home +7.5, blended margin −7.58 (a 0.08-point lean to the *away* side), `p_home_discrete` 0.5324.

### M6. β is fitted against the close and applied to a look-ahead open (Medium)

- `src/train_model_v3.py:131` and `:187` use nflverse `spread_line` as the market. I am treating that as a closing line; I did not check upstream documentation.
- Live, the anchor is `opening_line` (`log_v3_shadow.py:316`, `:378`).
- β ≈ 0.04 says the model adds nothing to the close. It does not measure what the model adds to a line nine days early, which is the bet actually being simulated. Only F1 with R1 can answer that.

### M7. The scorecard pools different models under one name (High)

- `MODEL_VERSION` defaults are still `"v3_epa_enhanced_2023_2025"` and `"v4_epa_qb_luck_totals_2023_2025"` (`log_v3_shadow.py:45`, `log_v4_shadow.py:57`). The artifacts are trained on 2012–2025 with a market blend.
- v3's week 8–9 rows were logged 10-26 to 10-28 by the pre-fix artifact. `QUANT_AUDIT_AND_ROADMAP.md` lists that artifact's calibrator slope as −0.3194 and its context features as hardcoded to 0.0. The Sheet confirms it: all 30 of those rows have `elo_diff`, `diff_margin` and `diff_rest` equal to 0.
- They were never re-logged: the dedupe key is `(game_id, home_line, home_odds)` (`log_v3_shadow.py:122-131`).
- v3's four scorecard bets are all from that artifact. V3_Shadow also carries each of those games five times; `first_priced_decision` hides that for grading.

### M8. "Guard suppressed" counts rows that were never bets (Low)

- v3 and v4 assign `SUPPRESS_DANGER` before checking the edge threshold (`log_v3_shadow.py:461-470`, `log_v4_shadow.py:498-507`). v2 checks the edge first.
- Three of the six "suppressed danger bets" had edges below 2%: NE@MIA +1.53%, JAX@BAL −2.0%, DAL@IND −0.29%.

### M9. The promotion rule is too weak at 40 bets (Medium)

- Condition 2 is Avg CLV Win Prob ≥ +0.5% over 40 bets (`README.md:40`).
- With the observed per-bet SD of 5.5 points, the standard error is 0.87 points. A zero-edge model clears +0.5% about 28% of the time.
- Condition 3 counts a one-bet week as a week.
- CLV ignores price (F1, point 6).

## Independent review

Fresh pass, findings doc set aside. Not reviewed in depth: `external_trigger/`, `src/staking.py`, `src/turnover_luck.py`, and the two backtest scripts beyond confirming what they do not cover.

### 1. Model quality

**v1 (live).**
- Ridge on five rolling box-score differences, 672 games from 2023–2025, in-sample sigma 13.43.
- No calibration and no market information. Its probabilities are not calibrated probabilities; it reports 2–9.5% edges on most of the slate.
- Features are season-only rolling four games, so it has no view before week 4.
- Useful as a naive control. Its CLV record is contaminated (M2).

**v2 (shadow).**
- **Discards the 2025 season at inference.** `log_v2_shadow.py:188-192` passes `exclude_seasons={2025}`, so 2026 Elo starts at 0.5² = 25% of end-2024 ratings (`model_v2_core.py:300`). That protects a holdout nobody is evaluating and costs a season of information.
- **Train/serve mismatch.** The calibrator is fitted on normal-CDF probabilities (`model_v2_core.py:596`) and served key-number conditional probabilities (`log_v2_shadow.py:304-308`).
- Calibration slope 0.0908: outputs sit in 0.486–0.519. In effect it is a constant.

**v3 (shadow).**
- Feature construction is leak-free on my read. Elo, margin and rest state update after each game, and EPA snapshots use `kickoff_utc < as_of` (`epa_features.py:165`).
- The walk-forward split for β is sound.
- The model itself has no measurable information beyond the close (W2).
- The probability stage is the weak part: mean-anchored key numbers, then a calibrator that is fitted on line artifacts (M5).

**v4 (shadow).**
- v3 plus QB shock and fumble luck. `diff_fumble_luck_per_gm` has coefficient −0.0, pinned at its bound.
- RMSE 13.234 against v3's 13.242; β 0.042 against 0.037. It is v3 again, and its leans will be almost perfectly correlated with v3's.
- QB starter is "last game's top passer" in training and injury-report overrides live (`qb_features.py:313`). That is a mismatch, not a leak.
- Totals-conditioned sigma was shipped without validation. At β ≈ 0.04 its only effect is to rescale the M5 artifact.

**Coherence of the whole approach.**
- Margin, to discrete distribution, to conditional cover probability, to edge against break-even is coherent.
- Two links break it: the anchor ignores price and uses the mean, and the calibrator is stacked on β.
- No backtest runs the production decision rule. `backtest_epa.py` and `backtest_model.py` use neither `market_blend` nor the break-even edge. Such a backtest would have shown "zero bets possible" and the price-triggered picks before deployment.

### 2. Pipeline correctness

**Sound.**
- Row-index writes are serialized by the `sheet-writer` group. The closing loop re-checks `game_id` before writing (`capture_closing.py:480-491`).
- Settling is status-gated and idempotent.
- The live-market guard (`mkt_lu > kickoff`) is in place.
- V3 and V4 rows do append after existing data: confirmed in the Sheet for the 11-04 and 11-10 runs.

**Defects.** See the classification below. On races specifically:
- The closing loop and the nightly job can both rewrite `notes` and `status` for the same row from stale reads. Last writer wins. Low impact.
- `log_v4_shadow.py:283-296`: if `get_all_records()` fails after the worksheet handle is set, the dedupe set stays empty and every game is appended again.
- The loggers' `append_rows` and `get_all_records` calls are not retried, and workflow steps have no `continue-on-error`. One transient Sheets error in the v2 step skips v3, v4 and the scorecard for that run. Later runs heal it.

### 3. Classification

**(a) Bugs and defects**

| # | Defect | Where | Severity |
|---|---|---|---|
| a1 | Picks added on later runs at a stale line; cap is per run; no pick timestamp | `generate_picks.py:105,126,248` | Critical |
| a2 | No horizon on opening capture | `capture_opening.py:68-100` | High |
| a3 | Rest computed from last completed game, not last game before kickoff | four call sites (M3) | High |
| a4 | A price of +110 or longer triggers a shadow pick with no model input | `log_v2/3/4_shadow.py` | High |
| a5 | Scorecard mixes artifacts under one `MODEL_VERSION` | `log_v3_shadow.py:45`, `log_v4_shadow.py:57` | High |
| a6 | Opening quote may be up to 12 h old but stamped now; book not recorded | `odds_utils.py:122`, `capture_opening.py:26,131` | Medium |
| a7 | v4 logger swallows Sheet errors and can append duplicates | `log_v4_shadow.py:283-296,636` | Medium |
| a8 | Guard statistic counts sub-threshold rows | `log_v3_shadow.py:461`, `log_v4_shadow.py:498` | Low |
| a9 | `current_line or open_line` treats a pick'em (0.0) as missing | `injury_guard.py:140` | Low |
| a10 | `Master_Bets.clv_points` uses the single-book close; the scorecard uses consensus | `update_results.py:149-160` vs `compare_models.py:123-126` | Low |

**(b) Methodological weaknesses**

| # | Weakness | Severity |
|---|---|---|
| b1 | Anchor matches the mean, not the price, at the market line (M5) | High |
| b2 | Calibrator stacked on β; slope 0.36 ± 0.45 | High |
| b3 | β measured against the close, used against a look-ahead open (M6) | Medium |
| b4 | Promotion rule underpowered; CLV ignores price (M9) | Medium |
| b5 | v2 drops 2025 at inference; calibrator train/serve mismatch | Medium |
| b6 | No backtest of the production rule | Medium |
| b7 | v1 has no calibration or market input | Medium |
| b8 | v4 duplicates v3; one dead feature; unvalidated totals sigma | Low |

**(c) Opportunistic improvements**

- Record the best available line across books as a second CLV series. The data is already parsed in `odds_utils.py`.
- Capture a second "decision line" on Thursday so open-versus-late timing can be compared directly.
- Retire v4 or fold its one live feature into v3, to halve the comparisons.
- Remove `src/generate_picks_v2.py`. Nothing references it.

### 4. Recommendations

Five, in order. None is on `QUANT_AUDIT_AND_ROADMAP.md`, where every item is already ticked.

**R1. Fix decision timing. Effort S. Now.**
- Limit opening capture, v1 picks and shadow evaluation to games kicking off within about 6.5 days.
- Price a pick only at a line captured in the same run.
- Make `MAX_PICKS` a weekly cap.
- Write `pick_made_at`, the quote timestamp and the book.
- Compute rest from the schedule (a3).
- **Why:** removes the look-ahead in the baseline, puts live inputs on the footing the models were trained on, and should cut CLV noise substantially.
- **Catch:** this does disturb the v1 baseline, on purpose; the current baseline is not valid evidence. Start a new regime at week 12 and report it separately. If look-ahead openers are wanted, keep capturing them in their own column and do not bet them. This is your call before anything is built.

**R2. Scorecard hygiene. Effort S. Now (reporting only).**
- Stamp rows with the artifact's training date or hash.
- Exclude v2/v3 rows from before Nov 2 from any promotion sample.
- Check the edge before `SUPPRESS_DANGER`.
- Show ties in CLV Beat %.
- Use closing odds in CLV.

**R3. Raw-lean tracking: F1 as amended. Effort M. Now (evaluation only).**
- Shared header migration, raw margin unrounded, backfill weeks 10–11, baselines, pre-registered test.
- Expect little: the historical record of raw leans against the close is 49.8%.

**R4. Backtest the production rule. Effort M. Now (offline).**
- One walk-forward script that calls the same functions as the loggers (blend, key numbers, calibrator, break-even edge, 2% bar) on 2012–2025 with nflverse closing prices.
- It should report bet count, hit rate and the zero-disagreement case. Every later model change gets judged here first.

**R5. Price-consistent anchor. Effort M. Build and validate in R4 now; deploy to shadow tabs at a week boundary under a new version string.**
- Solve for the anchor mean at which the engine's cover probability at the offered line equals the de-vigged market probability. Add β times the disagreement. Drop the calibrator stage for v3/v4.
- Zero disagreement then gives the fair probability and an edge of minus half the vig at any price or line. That removes a4, b1 and b2 together.
- No effect on v1.

**Wait for more data before:** changing β or the 2% bar, replacing v1, or promoting anything. About 109 games remain this season.

**Roadmap items I would challenge.**
- **§2.1.1 "Mean shift distortion".** The fix enforced mean = μ. For a market-anchored model that is the wrong invariant; it is the source of M5.
- **§4.2.2 Totals-conditioned sigma.** Marked done with no evidence it improves calibration. Validate in R4 or remove.
- **§4.1 QB shock and fumble luck.** Marked done. Fumble luck fitted to exactly zero and v4 gained 0.008 RMSE points.
- **§1.1 Bug 5, RLM guard.** On look-ahead lines it fires on the previous week's results, not on "injury or sharp fade" (TEN@CIN, 3.5 points).
- **Phase 4 "Testing & Live Promotion".** Ticked, while promotion is unreachable by construction.

## Overall assessment

**Is the findings document trustworthy as a backlog? Partly.**
- **What holds up.** The symptom is diagnosed correctly: the shadow models cannot bet. The numbers I could check are accurate: β, average edges, the scorecard, v4's 0.09 points.
- **What does not.**
  - One of three problems is wrong (W3).
  - One contains a false statement that a single file read would have caught (`no_edge`).
  - Two of five fix items were already done when it was written (F2, F5).
  - Its main fix would crash two loggers as specified (F1).
- **What it never questions.** The scorecard it opens with. It never asks whether the bets on it were real or whether the lines were bettable when the picks were made.
- **How to use it.** Take F1, F3 and F4 with the amendments above. Drop F2 and F5. Replace the decision point with a pre-registered test.

**Which of my recommendations to act on first.**
1. **R1**, because every CLV number collected before it is compromised, and each week of delay costs about 15 of the 109 remaining games.
2. **R2**, the same day. It is small and stops the scorecard overstating what is known.
3. **R3**, so weeks 12–18 produce something gradable.
4. **R4, then R5**, offline, on no deadline.

**The honest reading of the evidence so far:** no model here has shown an edge. This season's realistic output is a clean measurement, not a promotion.
