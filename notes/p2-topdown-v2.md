# P2: sharp-vs-soft top-down replay v2 (spreads and totals)

Spec and frozen rules: `notes/implementation-plan/P2-topdown-v2.md`. Code: `src/total_pricing.py`, `src/topdown_v2.py`. Tests: `tests/test_total_pricing.py`, `tests/test_topdown_v2.py`. Offline only: no Sheet, workflow or decision change. `src/market_consensus.py` and `src/topdown_replay.py` are imported, not edited.

```bash
git fetch origin odds-data
.venv/bin/python -u -m src.topdown_v2            # add --season-settled once week 18 has settled
```

## Rules (restated, frozen 2026-11-24)

- **Anchor.** The weighted median of sharp-group votes, one vote per group: Pinnacle (3), Circa (2), BetOnline/LowVig (2, `betonlineag` first, `lowvig` only when it is missing). No group means the market is skipped in that snapshot. The anchor books are never priced or flagged.
- **Priced books.** US targets (FanDuel, DraftKings, BetMGM, BetRivers, Caesars, ESPN Bet, Fanatics, Bally, Hard Rock) and offshore books (BetUS, MyBookie, Bovada). Offshore books are reported but never in the gate. A book in neither set is ignored and listed once in the run log.
- **Pricing.** Spreads use `implied_mean` and `price_offer` with σ = 12.8. Totals use a discretised Normal with σ = 13.2, a continuity correction and no key-number reweighting. An offer is flagged at EV ≥ 1.5%.
- **Close.** The close is the market's last snapshot strictly before kickoff that has an anchor. Only games that kicked off by the latest snapshot count. `close_ev` prices the offer against the closing anchor. `close_ev_allbooks` prices it against the plain median of every book at the close, as a sensitivity check.
- **Samples.** A row is in the evaluation sample only if both its kickoff and its snapshot are at or after 2026-11-25 00:00 UTC. Every earlier row is design sample, not evidence.
- **Gate.** The gate group is flagged offers from US targets, in-week (≤ 144 h), spreads and totals together. The result is:
  - **Pass:** at least 80 offers on at least 35 games, mean `close_ev` at least 2.0 game-clustered SE above 0, and still above 0 with the top 2 games removed.
  - **Fail:** mean `close_ev` ≤ 0 once the gate group reaches that size.
  - **Inconclusive:** neither, once week 18 has settled.

  Weekly re-runs are descriptive only.

### Choices the spec left open (code structure only, no rule changed)

- `gate()` returns `pending` before week 18 settles when there is neither a pass nor a fail. The spec says weekly re-runs are descriptive only, so mid-season there is no final verdict. The CLI's `--season-settled` switch turns `pending` into `inconclusive`.
- The close is found per (game, market), because the spread anchor and the total anchor can exist in different snapshots.
- `persisted_next_snap` looks at the archive's next snapshot. It is NaN if that snapshot is more than 6 h later. It is False if the book, line or price changed, if the book is missing, if there is no anchor, or if EV is below 0.5%.
- If more than one sharp group votes, `anchor_fresher` uses the latest `last_update` among the voting books, and `anchor_line` is the median of their lines. On today's archive only one group ever votes, so neither choice matters yet.
- Spread rows give `line` from the home side, as Phase 3 does. Totals give the total.

## First run, 2026-11-24 (archive `origin/odds-data` at 6466ba8)

52 snapshots, from 2026-11-01 15:57 to 2026-11-24 15:18 UTC. All 52 are before `EVAL_START`. The run gave 6,052 first sightings on 47 played games, 3,038 spread and 3,014 total. Only the BetOnline group was present, so `anchor_groups` = 1 on every row. No book fell outside the classes. Grading used the main checkout's `data/nflverse_cache/games.csv`, dated 11-24.

### Evaluation sample (evidence)

No rows. No snapshot has been taken at or after 2026-11-25 00:00 UTC yet. The gate status is **pending**, with 0 of 80 offers on 0 of 35 games.

### Design sample, not evidence

**Gate group** (flagged, US, in-week, spreads + totals):

| group | offers | games | mean EV | close_ev | SE (game) | close_ev_allbooks | beat close | W-L-P | units |
|---|---|---|---|---|---|---|---|---|---|
| gate group | 9 | 9 | +2.26% | +2.06% | 1.01% | +1.65% | 89% | 5-4-0 | +0.62 |
| top 2 games removed | 7 | 7 | +2.14% | +0.91% | 0.80% | +0.69% | 86% | 4-3-0 | +0.75 |

**Flagged offers by market × horizon × book class:**

| group | offers | games | mean EV | close_ev | SE (game) | close_ev_allbooks | beat close | W-L-P | units |
|---|---|---|---|---|---|---|---|---|---|
| spread / >144h / offshore | 4 | 4 | +3.62% | +3.44% | 3.51% | +3.66% | 75% | 3-1-0 | +1.82 |
| spread / >144h / us | 14 | 10 | +2.51% | +4.54% | 2.90% | +5.11% | 71% | 10-4-0 | +5.19 |
| spread / in-week / us | 2 | 2 | +2.34% | +2.14% | 0.03% | +1.97% | 100% | 0-2-0 | −2.00 |
| total / >144h / offshore | 1 | 1 | +3.28% | +3.28% | — | +1.01% | 100% | 1-0-0 | +0.91 |
| total / >144h / us | 10 | 8 | +3.48% | −3.29% | 4.92% | −2.97% | 30% | 5-5-0 | −0.41 |
| total / in-week / us | 7 | 7 | +2.23% | +2.04% | 1.32% | +1.55% | 86% | 5-2-0 | +2.62 |

**Flagged offers by book:**

| book | offers | games | mean EV | close_ev | SE (game) | beat close | W-L-P | units |
|---|---|---|---|---|---|---|---|---|
| betrivers | 13 | 12 | +2.89% | +2.46% | 2.07% | 77% | 7-6-0 | +0.39 |
| fanduel | 10 | 7 | +3.02% | +4.84% | 4.70% | 60% | 7-3-0 | +3.55 |
| draftkings | 10 | 10 | +2.26% | −3.11% | 2.70% | 50% | 6-4-0 | +1.46 |
| mybookieag | 3 | 3 | +4.16% | +1.92% | 4.48% | 67% | 2-1-0 | +0.82 |
| betus | 2 | 2 | +2.64% | +5.64% | 2.36% | 100% | 2-0-0 | +1.91 |

**Gate-group persistence** (next snapshot within 6 h, EV ≥ 0.5%): 3 true, 2 false, 4 with no snapshot within 6 h.

**Gate group by `anchor_fresher`:** False: 6 offers, close_ev +2.37% ± 1.53%. True: 3 offers, +1.44% ± 0.46%.

**Context:** all 2,782 US in-week first sightings, flagged or not, average close_ev −4.41%. That is the vig, since both sides of every quote are included.

### Reading

- In-week flags are rare. There are 9 in 24 days, against the 80 the gate needs. The design sample is 9 games, and the test needs 35. As the spec expected, the gate will probably still be pending at week 18 at the current snapshot cadence. That is the owner question in the spec, about mid-week snapshots.
- The design sample suggests a positive in-week close_ev (+2.1% ± 1.0%) that halves without the top 2 games. Under the frozen rules this is not evidence and must not be used to tune anything.
- Long-horizon (>144 h) total flags lost against the close (−3.3%). They are outside the gate.

## Re-running

Re-run weekly, after the Tuesday odds snapshot, next to the Phase 3 `src.topdown_replay` re-run. Report the evaluation-sample gate group and status. Never stop early for a pass, and never change a number or rule in the spec.
