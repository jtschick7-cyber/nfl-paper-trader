# P4: top-down `player_receptions` scanner, first run

Spec: [`implementation-plan/P4-prop-scanner.md`](implementation-plan/P4-prop-scanner.md) (rules frozen 2026-11-24). Code: `src/props/prop_pricing.py`, `src/props/topdown_props.py`, `src/props/player_stats.py`. Tests: `tests/test_prop_pricing.py`, `tests/test_topdown_props.py`, committed (`7141f0c`) before the code and before any run.

```bash
.venv/bin/python -u -m src.props.topdown_props                 # origin/odds-data, fetch it first
.venv/bin/python -u -m src.props.topdown_props --no-refresh    # grade from the cached nflverse files
```

Writes `data/topdown_props_offers.csv` (gitignored). No Odds API call, no Sheet write.

## Rules, restated (frozen)

- **Books.** Anchor `pinnacle`; sensitivity anchor `betonlineag` (never flagged). Priced targets: US `draftkings`, `fanduel`, `betmgm`, `betrivers`, `williamhill_us`, `espnbet`, `fanatics`, `ballybet`, `hardrockbet`; offshore `bovada`, `betus`, `mybookieag` (reported, never in the gate). Books in no class are ignored and listed once.
- **Pricing.** Receptions ~ Poisson(λ). P(over L) = 1 − F(floor L); an integer line pushes with P(R = L). λ* is the λ at which the no-push over share equals the de-vigged Pinnacle `q_over` (`total_pricing.devig_two_way`), by bisection on [0.05, 20] to 1e-6. EV comes from `key_numbers.calculate_expected_value`. Every EV is also computed under NB(r = 8), reported only.
- **Sightings.** Every `open`-phase snapshot; `probe` files are excluded. Each target offer (any line, one-sided included) on a player with a two-sided Pinnacle quote is priced against λ*_open. **Flag at EV ≥ 2%.** No Pinnacle quote: the player is skipped in the primary table and appears only in the BetOnline table.
- **Close.** The event's latest `close` snapshot before kickoff. `close_ev` = the offer priced against Pinnacle's λ*_close (primary CLV). `close_ev_median` uses the median λ* of every two-sided classed book at the close. If Pinnacle is missing at the close, or there is no close snapshot, the row has no `close_ev` (`close_status` says which).
- **Grading.** nflverse `stats_player_week_2026.parquet` and `snap_counts_2026.parquet`, joined on normalized name + team + game. Zero offensive snaps or no snap-count row: VOID, 0 units, out of W-L. Unmatched or ambiguous names are listed and never graded.
- **Gate group.** Flagged US-target offers with a `close_ev`, Pinnacle anchor.
- **Pass:** ≥ 150 offers on ≥ 30 games, mean `close_ev` > 0 by ≥ 2.0 game-clustered SE (`topdown_replay.cluster_se`), > 0 with the top 2 games removed, and on each side in the gate group the flagged mean `close_ev` above that side's all-US-offer mean (drift check). **Fail:** mean ≤ 0 once the size is reached. **Inconclusive:** neither once week 18 has settled; the rules carry into 2027. Weekly re-runs are descriptive; never stop early for a pass.

Builder choices where the spec is silent (structure, not rules):

- A book quoting several two-sided lines for one player gives its λ* from the line whose de-vigged `q_over` is nearest 0.5 (ties: the lower line). This applies to the anchor at open and close, and to each book in the close median.
- Several open snapshots for one event each yield sightings (P1 captures one, so normally one).
- A player with offensive snaps but no stats row for the game is graded as 0 receptions (`graded_zero`) when the name matches exactly one stats-file player on the same team elsewhere in the season. The nflverse stats file only holds players who recorded a stat, so this is usually a real 0 catches, and dropping those rows would tilt the W-L toward Overs. Otherwise the player is listed as `no_stats_row` and left ungraded, since the name may not match between the two files. (Changed on owner approval, 2026-11-24, before any open or close capture existed.)
- The report's coverage section counts how many Pinnacle sightings had several anchor lines (so the nearest-50/50 rule fired). It also lists games still `pending` 3+ days after kickoff (`STALE_PENDING_DAYS`, a report-only setting), so a stuck nflverse download gets noticed.
- A game with no rows yet in either nflverse file is `pending`. Such players are not graded and not listed as unmatched.
- The CSV adds `anchor`, `anchor_line`, `same_line`, `anchor_multi_line`, `close_status`, `grade_status`, `snapshot_ts`, `close_ts`, `home_team`, `away_team`, `player_name` to the spec's columns.

## First run, 2026-11-24 (archive `origin/odds-data`, props/ as of ~21:50 UTC)

The archive holds one file, P1's probe (GB @ LA, phase `probe`), which is excluded. **The evaluation is empty.**

| Item | Value |
|---|---|
| Prop snapshots | 1 (`probe`: 1, `open`: 0, `close`: 0) |
| Priced offers | 0 on 0 games |
| Gate group | 0 offers on 0 games; need 150 on 30 |
| Gate | **pending** |
| Breakdowns, alternate anchor, offshore, drift check | no rows |
| Unmatched names, voids, missing closes | none (no rows) |

Smoke check, not evidence: with the probe file relabelled `open` in memory, the replay priced 126 Pinnacle-anchored and 126 BetOnline-anchored offers from `betmgm`, `betrivers`, `bovada`, `draftkings` and `fanduel`. That included BetRivers' 24 one-sided Overs. No book fell outside the classes, and grading returned `pending` (week 12 isn't played yet). No EVs from the probe were looked at or recorded.

The GB @ LA (Wed 11-25) open and close files should land by about 01:30 UTC Thu 11-26, with the Thanksgiving and Black Friday games after that. The weekly re-run fills this table. Add each re-run below as a dated section; do not edit this one.
