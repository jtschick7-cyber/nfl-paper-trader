# P4. Prop pricer and top-down `player_receptions` scanner

**Source:** owner's roadmap brief of 2026-11-24 (Step 4), changed after P1's probe the same day. **Effort:** M. **Branch:** `claude/p4-prop-scanner`. **Depends on:** P1 (merged; the archive under `odds-data:props/`).

Offline only. No Sheet tab, workflow, Worker or decision change.

## Question

Do soft US books' Friday `player_receptions` prices, when they disagree with Pinnacle, beat the pre-kickoff close? This is P2's sharp-vs-soft test applied to props, where the market is younger, thinner and moves on injury news.

## Pre-registration

- Every number and rule below is fixed at this spec's commit (2026-11-24).
- At that point the only prop data anyone had looked at was P1's probe file (GB @ LA, phase `probe`), which is excluded.
- So all `open` and `close` captures count as evidence. There is no design sample.
- The builder may change code structure but not a number or rule. If one is unworkable, stop and ask the owner.

## What the probe showed (2026-11-24, GB @ LA, 28 h before kickoff)

| Book | Players | Two-sided |
|---|---|---|
| `pinnacle`, `draftkings`, `fanduel`, `betonlineag`, `bovada` | 15 | yes |
| `betmgm` | 12 | yes |
| `betrivers` | 12 | **Over only** |
| `williamhill_us`, `betus`, `mybookieag` | 0 | — |

The outcome format, unchanged from the Odds API, is `{"name": "Over"|"Under", "description": <player name>, "point": 5.5, "price": -135}`. All probe lines were x.5.

## Design

### Book classes (frozen)

```python
ANCHOR_BOOK = "pinnacle"                    # the only sharp prop book returned
ALT_ANCHOR_BOOK = "betonlineag"             # sensitivity anchor only, never flagged
US_TARGETS = {"draftkings", "fanduel", "betmgm", "betrivers", "williamhill_us",
              "espnbet", "fanatics", "ballybet", "hardrockbet"}
OFFSHORE = {"bovada", "betus", "mybookieag"}  # reported, never in the gate
FLAG_EV = 0.02
NB_SENSITIVITY_R = 8.0
```

- **Threshold.** `FLAG_EV` is 2%, above P2's 1.5%. Pinnacle's prop markets take smaller limits than its game lines, so its prop price is a noisier anchor and needs a wider buffer.
- **Anchor and targets.** Pinnacle and BetOnline are never flagged. Only `US_TARGETS` and `OFFSHORE` offers are priced.
- **Unknown books.** A book in no class is ignored and listed once in the run log.

### Pricing: `src/props/prop_pricing.py`

- **`normalize_name(s)`:** lower case; strip `.`, `'` and `’`; drop a trailing `jr`, `sr`, `ii`, `iii`, `iv` or `v`; collapse whitespace. Players are keyed by `(event_id, normalized name)`.
- **`prop_quotes(response)`:** one row per `(book, player, line)` with `over_odds` and `under_odds` (either may be missing) and the market's `last_update`. If a book lists several lines for one player, each line is its own row.
- **Count model (frozen): Poisson.** For a player with mean receptions `λ`, `P(R > L) = 1 − F(floor(L))`; at an integer `L`, `P(R = L)` is a push.
  - Why Poisson, not the brief's position-fitted Negative Binomial: given an accurate mean, receptions are close to Poisson, it has no parameter to fit, and the choice only matters when comparing books on different lines. Fitting dispersion on historical stats would mix the error of a rolling-mean proxy into `r`.
  - **Sensitivity:** every EV is also computed under a Negative Binomial with `r = NB_SENSITIVITY_R`. That column is reported, not used for flags.
- **`implied_lambda(line, q_over)`:** the `λ` at which `P(over) / (P(over) + P(under)) = q_over`, by bisection on `[0.05, 20]` to 1e-6. `q_over` comes from `total_pricing.devig_two_way`, which P2 merged.
- **`price_prop_offer(lam, side, line, odds)`** returns `(p_win, p_push, ev)` via `key_numbers.calculate_expected_value`.
- **One-sided offers** (BetRivers posts Overs only) are priced the same way: the offered side against the anchor's `λ`. A one-sided quote can never be an anchor.

### Replay: `src/props/topdown_props.py`

- **Sightings:** every `open`-phase snapshot. Each `(event, player)` with a two-sided Pinnacle quote gets `λ*_open` from Pinnacle. Every target offer on that player, at any line, is priced against it.
  - Flag when `ev >= FLAG_EV`.
  - With no Pinnacle quote, the player is skipped in the primary table. The `ALT_ANCHOR_BOOK` table repeats everything with BetOnline as the anchor.
- **Close:** the matching `close`-phase snapshot. `λ*_close` comes from Pinnacle at the close, and `close_ev` is the offer priced against it (the primary CLV).
  - Sensitivity: `close_ev_median`, against the median of every two-sided book's `λ*` at the close.
  - If Pinnacle is missing at the close, the row has no `close_ev` and is counted under "no close" in the report.
- **Columns per row:**
  - identifiers: `event_id`, `kickoff`, `player`, `book`, `book_class`, `side`;
  - pricing: `line`, `odds`, `one_sided`, `lam_open`, `ev`, `ev_nb`, `flagged`;
  - close: `lam_close`, `close_ev`, `close_ev_nb`, `close_ev_median`;
  - timing: `hours_to_kickoff`, `quote_age_min`;
  - results: `receptions`, `result`, `units`.
- **Output:** write `data/topdown_props_offers.csv`, and add it to `.gitignore` if no pattern covers it.

CLI: `.venv/bin/python -u -m src.props.topdown_props [--source REF]`.

### Grading: `src/props/player_stats.py`

- **Data:** read the nflverse 2026 weekly player stats and snap counts with `pandas.read_parquet` from the `nflverse-data` GitHub releases. Cache them under `data/nflverse_cache/` (already gitignored).
  - The builder confirms the exact asset names on the releases page (likely `stats_player_week_2026.parquet` and `snap_counts_2026.parquet`) and records them in the module docstring.
  - Do not upgrade `nfl_data_py`.
- **Matching:** join on `normalize_name` plus team plus game. Any player the join can't resolve is listed in the report and left ungraded, never guessed.
- **Void:** a player with zero offensive snaps (or no snap-count row) gets `result = "VOID"` and 0 units, and is excluded from W-L. Books void props on players who don't play.
- **Results:** `WIN`, `LOSS` or `PUSH` at the offered line and price.

### Report (printed)

1. **Pre-registered gate group:** flagged US-target offers with a `close_ev`. Show offers, games, players, mean EV at sighting, mean `close_ev` ± game-clustered SE (`topdown_replay.cluster_se` on `event_id`), beat-close %, and W-L-P-Void with units.
2. The same group with the two games with the largest summed `close_ev` removed.
3. **Breakdowns:**
   - per book, and Over against Under;
   - one-sided against two-sided;
   - same line as Pinnacle against a different line;
   - `close_ev_nb` and `close_ev_median` beside `close_ev`;
   - the `ALT_ANCHOR_BOOK` table;
   - offshore offers.
4. **Drift check:** mean `close_ev` of **all** US-target Over offers and of all Under offers. If either side moves systematically between Friday and the close, flagged offers on that side look better than they are.
5. Coverage: players or books seen, unmatched names, missing closes, voids.

## Pre-registered test (frozen)

Gate group as in report item 1.

- **Pass:** ≥ 150 flagged offers on ≥ 30 games, mean `close_ev` > 0 by ≥ 2.0 game-clustered SE, **and** > 0 with the top 2 games removed, **and** the flagged side's mean `close_ev` above that side's all-offer mean from the drift check.
- **Fail:** mean `close_ev` ≤ 0 once the sample reaches 150 offers on 30 games.
- **Inconclusive:** neither once week 18 has settled. The rules carry unchanged into 2027.
- Weekly re-runs are descriptive only. Do not stop early for a pass.
- **Sample:** about 15 games a week, about 15 players a game and 4 US books give roughly 900 US offers a week, so the size is reachable by week 18 if even 3% are flagged.
- **Only after a pass:** a spec for live `Props_Ledger` / `Props_Scorecard` tabs (locked owner decision 3 in the brief).

## Tests (write first): `tests/test_prop_pricing.py`, `tests/test_topdown_props.py`

1. **`normalize_name`:** "Ja'Marr Chase" and "JaMarr Chase" are equal, and "Michael Pittman Jr." equals "Michael Pittman". When two players on one team share a normalized name, both are left ungraded and listed, never merged.
2. **Pricing:**
   - Poisson round trip `λ → q_over → implied_lambda` within 1e-6 at lines 2.5, 4.5 and 6.5;
   - with zero disagreement at −110/−110 on the same line, EV is about −4.5% on both sides;
   - an integer line gives a push mass equal to the Poisson pmf.
3. **Cross-line:** an anchor at 4.5 (−130/+100) prices a 3.5 Over and a 5.5 Under consistently, so the EVs follow from one `λ`.
4. **One-sided:** a BetRivers Over-only quote is priced and can be flagged, but is never used as an anchor.
5. Pinnacle and BetOnline are never flagged. An offer with no Pinnacle quote is absent from the primary table and present in the alternate-anchor table.
6. **Close matching:** `close_ev` uses the close-phase Pinnacle quote. When Pinnacle is missing at the close, the row is marked "no close". A `probe`-phase file is ignored.
7. **Grading:** win, loss and push; zero snaps gives VOID with 0 units, excluded from W-L; an unmatched name is ungraded and listed.
8. The gate function returns pass, fail or inconclusive on synthetic summaries, including the top-2-games and drift rules.
9. A test asserting every frozen constant above equals its spec value.

The full suite stays green (1694 Python and 10 Worker tests at `2f23441`, plus whatever P3 adds).

## Deliverables

- The code and tests above, plus `notes/p4-prop-scanner.md`: the first run's tables (likely only week 12's captures) with the rules restated.
- `PROGRESS.md`: about 10 bullets, plus a "re-run weekly" line beside the Phase 3 and P2 ones.

## Out of scope

Live logging, any Sheet tab, the bottom-up receptions model (Step 5), other prop markets, replacing books in P1's list.
