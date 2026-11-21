# Audit Phase 3: multi-book consensus pricer and archive replay (2026-11-21)

Offline. No Sheet tab, workflow or decision-rule change. Reproduce with
`.venv/bin/python -u -m src.topdown_replay` (reads `origin/odds-data`; run `git fetch origin odds-data`
first, and refresh `data/nflverse_cache/games.csv` for grading).

## What was built

- `src/market_consensus.py`: each book's implied mean margin m*_book (R5's `anchor_mean` on its
  de-vigged price, σ 12.8, median total), a leave-one-out consensus (median ± 1.5 pts outlier cut,
  then a weighted median with lowvig, betonlineag, pinnacle and circa weighted higher), every
  book's offer priced against it (p_win, p_push, EV per 1u), a stale flag at EV ≥ 1%, and the key
  number a better line newly covers.
- `src/load_odds_archive.py`: reads the gzipped snapshots from the branch with `git show` (or
  from a checkout directory).
- `src/topdown_replay.py`: for each snapshot before an event's last pre-kickoff snapshot,
  first sightings of each (event, book, side, line, price), priced against the consensus then
  and against the closing consensus (price-aware CLV), and graded at the offered line and price.

## Result on the archive (40 snapshots, 1–20 Nov 2026, 63 games, 9 US books)

| Group | Offers | Games | EV when seen | Closing EV ± SE (by game) | Beat close | W-L | Units |
|---|---|---|---|---|---|---|---|
| All offers | 4,288 | 63 | −4.1% | −4.1% ± 0.0% | 18% | 1522-1518 | −121.4 |
| DraftKings | 508 | 63 | −4.4% | −4.4% ± 0.0% | 19% | 167-167 | −15.6 |
| Flagged (EV ≥ 1%) | 38 | 21 | +2.1% | **+2.4% ± 1.0%** | 68% | 15-14 | +0.3 |
| Flagged, key-number cross | 6 | 5 | +1.8% | +0.8% ± 2.6% | 67% | 0-5 | −5.0 |
| Flagged, ≥ 48 h out | 34 | 19 | +2.2% | +2.5% ± 1.1% | 65% | 13-12 | +0.2 |

- Flagged offers against DraftKings' offer on the same game, side and snapshot: closing EV
  +5.1% ± 0.6% better (14 pairs).
- 68% of flagged offers' books had moved against that side by the close (they corrected).
- Flagged by book: lowvig 11, draftkings 8, betrivers 7, fanduel 5, betus 4, mybookieag 3.

## Reading

- The detector finds prices the market later corrects: about 2.5 SE above zero closing EV, with
  the SE clustered by game. That is the expected sign, but 38 offers on 21 games is too small to
  rely on, and the threshold and weights were chosen before any look at this result, not tuned.
- Win/loss on 29 graded tickets says nothing (±9% SE).
- Only US books are archived, so the "sharp" anchor is lowvig/betonlineag, not Pinnacle. Adding
  `eu` to `REGIONS` would cost more Odds API quota per call (owner decision).
- Snapshots only exist when workflows run (Tuesday open, game-day closing captures), so most
  offers are seen ≥ 48 h out; mid-week staleness is not sampled.

## Pre-registered test (no change to these rules after today)

Rules fixed as committed: σ 12.8, 1.5-pt outlier cut, weights above, EV ≥ 1%, first sighting.
Re-run the replay on the growing archive. Pass: at least 100 flagged offers on at least 40 games
with mean closing EV > 0 by at least 2 game-clustered SE. Fail: mean closing EV ≤ 0 at that size.
Only after a pass: a live `TopDown_Shadow` tab and best-book vs primary-book CLV on
`Model_Scorecard` (audit 3.1 item 1).

## Not done

- Live `TopDown_Shadow` logging, scorecard columns, a Thursday decision snapshot: these need
  live workflow and Sheet changes; held until the pre-registered test passes or the owner asks.
- Model backtests against archived openers: the archive starts 1 Nov 2026, too short to backtest
  a model; re-assess at season end.
