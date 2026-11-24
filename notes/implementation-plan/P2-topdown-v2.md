# P2. Sharp-vs-soft top-down replay v2 (spreads and totals)

**Source:** owner's roadmap brief of 2026-11-24 (Step 1), with the changes agreed in review the same day. **Effort:** M. **Branch:** `claude/p2-topdown-v2`. **Depends on:** nothing (P1 may land first; no shared files except `PROGRESS.md`).

Offline only. No Sheet tab, workflow, Worker or decision change.

## Why the rules are frozen here, before the build

v2's changes come from reading Phase 3's 34 flagged offers: drop LowVig as a target, treat FanDuel and BetRivers as the stale books, and DraftKings flags lost. Scoring v2 on the same snapshots would test it on the data it was designed from. So:

- every rule below is fixed as of this spec's commit (2026-11-24);
- evidence comes only from the **evaluation sample**: games kicking off at or after **2026-11-25 00:00 UTC**, using only snapshots taken at or after that time;
- earlier snapshots are the **design sample**. They are reported in a separate table headed "design sample, not evidence", and nothing may be tuned on them after this commit.

The builder may change code structure but not a number or rule in this file. If a rule turns out to be unworkable, stop and ask the owner. Do not adjust it to the data.

## Leave alone

`src/market_consensus.py`, `src/topdown_replay.py` and their constants keep the pre-registered Phase 3 test clean. v2 **imports** from them and never edits them. Also untouched: `src/config.py`, `src/odds_utils.py`, every workflow, `models/`.

## Facts about the archive (checked 2026-11-24, 52 snapshots)

- Books present: `betonlineag`, `lowvig`, `fanduel`, `draftkings`, `betmgm`, `betrivers`, `betus`, `mybookieag` (all 52), `bovada` (51). No `pinnacle`, `circasports`, `williamhill_us`, `espnbet` or `fanatics`.
- Both `spreads` and `totals` are present for almost every book and event.
- LowVig is BetOnline's reduced-vig brand on the same feed. The two are **one** source, not two sharp books agreeing.
- Snapshots cluster on game days. A typical week is about 20, of which only the Tuesday–Thursday runs fall inside 144 h of most games.

## Design

### Book classes (frozen)

```python
SHARP_GROUPS = [            # one vote per group; first book present in the group is used
    (("pinnacle",), 3.0),
    (("circasports",), 2.0),
    (("betonlineag", "lowvig"), 2.0),
]
US_TARGETS = {"fanduel", "draftkings", "betmgm", "betrivers", "williamhill_us",
              "espnbet", "fanatics", "ballybet", "hardrockbet"}
OFFSHORE = {"betus", "mybookieag", "bovada"}   # reported, never in the gate
FLAG_EV = 0.015
PERSIST_EV = 0.005
PERSIST_HOURS = 6
IN_WEEK_HOURS = 144
SPREAD_SIGMA = 12.8          # = market_consensus.DEFAULT_SIGMA
TOTAL_SIGMA = 13.2           # SD of (total - closing total_line), 2012-2025 REG, n = 3,663: 13.24
EVAL_START = "2026-11-25T00:00:00Z"
```

- **Anchor.** The weighted median of the sharp-group votes present. On today's archive only the BetOnline group is present, so the anchor is BetOnline's implied mean; `lowvig` is used only when `betonlineag` is missing. Record `anchor_groups` (how many groups voted) on every row. If no group is present, the event is skipped in that snapshot.
- **Never flag a sharp-group book.** Only `US_TARGETS` and `OFFSHORE` offers are priced and flagged. A book in neither set is ignored and listed once in the run log.

### Spreads

Per book: `implied_mean(quote, SPREAD_SIGMA, median_total)` and `price_offer(...)` from `src/market_consensus.py`, with the sharp anchor in place of the leave-one-out consensus. Keep `crosses_key` as a reported column.

### Totals: new module `src/total_pricing.py`

P3 imports this module too. Whichever of P2 and P3 is built first creates it exactly as specified here, with its tests.

- `total_probabilities(mu, line, sigma=TOTAL_SIGMA)` returns `(p_over, p_under, p_push)` for an integer total `T ~ discretised Normal(mu, sigma)` with continuity correction:
  - integer `line`: `p_push = Φ((L+0.5−mu)/σ) − Φ((L−0.5−mu)/σ)` and `p_over = 1 − Φ((L+0.5−mu)/σ)`;
  - x.5 lines: no push.
  - No key-number reweighting. The most common totals each hold 3.4–3.9% of games, close to the Normal's 3.0% at the mode, unlike the spread's 3 and 7.
- `devig_two_way(odds_a, odds_b)` is the same multiplicative de-vig as `price_anchor.devig_home`, with −110/−110 assumed when a price is missing.
- `total_implied_mean(line, q_over, sigma)` returns the `mu` at which `p_over / (p_over + p_under) = q_over`, found by bisection on `[line − 30, line + 30]` to 1e-6.
- `price_total_offer(mu, side, line, odds, sigma)` returns `(p_win, p_push, ev)`, using `key_numbers.calculate_expected_value`.
- `total_quotes(event)` returns one dict per book with `book, line, over_odds, under_odds, last_update`. It uses the `totals` market and skips a book whose Over and Under points differ.

### Replay `src/topdown_v2.py`

Mirror `topdown_replay.replay`:

- **Close.** An event's close is its last snapshot strictly before kickoff in which the anchor exists. Only events that kicked off by the archive's latest snapshot count. Reuse the logic of Phase 3's `closing_state`; do not import a changed copy.
- **First sightings.** For every earlier snapshot, take the first sighting of each `(event_id, market, book, side, line, price)`.
- **Columns per row:**
  - identifiers: `market` (`spread`/`total`), `book_class` (`us`/`offshore`), `side`;
  - pricing: `line`, `odds`, `anchor_mean`, `anchor_groups`, `ev`, `flagged` (`ev >= FLAG_EV`);
  - close: `close_anchor_mean`, `close_ev` (the offer priced against the closing anchor: the primary CLV), and `close_ev_allbooks` (against the plain median of all books' implied means at the close: a sensitivity check);
  - timing: `snapshot_ts`, `hours_to_kickoff`, `in_week` (`hours_to_kickoff <= 144`);
  - freshness: `quote_age_min` (snapshot time minus the book's market `last_update`) and `anchor_fresher` (the anchor book's `last_update` is later than the target's);
  - persistence: `persisted_next_snap`. True if the next snapshot within 6 h still shows the same book, market, side, line and price with EV ≥ 0.005 against that snapshot's anchor. False if it doesn't. NaN if no snapshot falls within 6 h.
  - `sample` (`design`/`evaluation`, by `EVAL_START`, applied to both kickoff and snapshot time).
- **Grading** at the offered line and price from `data/nflverse_cache/games.csv`. Spreads as in `attach_results`. Totals: over, under, or push on `home_score + away_score`.
- **Output.** Write `data/topdown_v2_offers.csv` and add it to `.gitignore` if no pattern covers it.

CLI: `.venv/bin/python -u -m src.topdown_v2 [--source REF]`.

### Report (printed)

1. **Evaluation sample, pre-registered gate group:** flagged, US targets, in-week, spreads and totals combined. Show offers, games, mean EV at sighting, mean `close_ev` ± game-clustered SE (`topdown_replay.cluster_se` on `event_id`; spreads and totals on one game share a cluster), beat-close %, W-L-P and units.
2. The same group with the two games with the largest summed `close_ev` removed.
3. Breakdowns: market × horizon × book class; per book; `persisted_next_snap` share; `anchor_fresher` slice; `close_ev_allbooks` beside `close_ev`.
4. **Design sample:** the same tables, headed "design sample, not evidence".

## Pre-registered test (frozen)

On the evaluation sample, gate group as in report item 1:

- **Pass:** ≥ 80 flagged offers on ≥ 35 games, mean `close_ev` > 0 by ≥ 2.0 game-clustered SE, **and** mean `close_ev` > 0 with the top 2 games removed.
- **Fail:** mean `close_ev` ≤ 0 once the sample reaches 80 offers on 35 games.
- **Inconclusive:** neither, once week 18 has settled. The rules carry unchanged into 2027, with no re-tuning.
- Weekly re-runs are descriptive only. Do not stop early for a pass.
- Only after a pass: a spec for a live `TopDown_Shadow` tab (owner decision).

**Expect inconclusive this season.** Phase 3 flagged 7 in-week offers in 24 days at a 1% threshold, and v2 raises it to 1.5% and drops sharp-book flags. At the current snapshot cadence the 80/35 size will probably not be reached before week 18. See the owner question below.

## Tests (write first) `tests/test_topdown_v2.py`, `tests/test_total_pricing.py`

1. A sharp-group book is never flagged, even with EV ≥ 1.5% against the anchor.
2. With only `betonlineag` and `lowvig` present, the anchor equals `betonlineag`'s implied mean. With `betonlineag` missing, it equals `lowvig`'s.
3. The Phase 3 `NE @ DET` pattern on a synthetic event: stale DK HOME −1.5 plus LowVig AWAY +2.5. v2 flags DK only, and never both sides of one market in one snapshot.
4. Totals:
   - round trip `mu → q_over → total_implied_mean` within 1e-6 for lines 41, 43.5 and 47;
   - with zero disagreement at −110/−110, EV is about −4.5% on both sides;
   - push mass at `mu = line = 44` is between 0.029 and 0.031.
5. `total_quotes` skips mismatched Over and Under points and reads `last_update`.
6. `persisted_next_snap`: True, False and NaN cases (next snapshot after 6 h).
7. Horizon boundary at exactly 144 h counts as in-week. A row is in the evaluation sample only when both its kickoff and its snapshot are at or after `EVAL_START`.
8. Grading of totals at an integer line (push) and an x.5 line.
9. A gate function returns pass, fail or inconclusive on synthetic summaries, including the top-2-games rule.
10. A test asserting every frozen constant above equals its spec value, so tuning shows up in review.

The full suite stays green (1630 Python, 8 Worker tests at `aa677fb`, plus whatever P1 adds).

## Deliverables

- The code and tests above, plus `notes/p2-topdown-v2.md`: first run's design-sample tables and the (likely empty) evaluation sample, with the rules restated.
- `PROGRESS.md`: about 10 bullets, plus a "re-run weekly" line next to the Phase 3 one.

## Owner question (does not block the build)

| Question | Default |
|---|---|
| Add mid-week game-odds snapshots to give the gate a chance this season? For example, Wed/Thu/Sat 17:47 UTC through a small new capture job. Each call costs about 2 credits (spreads and totals), about 6 a week, taking the total to about 310/500. This changes no existing workflow. | No change this season; decide after the first evaluation-sample week |
