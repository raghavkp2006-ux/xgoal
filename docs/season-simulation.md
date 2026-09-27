# Season simulation

The on-demand job `python -m jobs.simulate_season` fits Dixon-Coles using data
available at the current information cut. It builds a match-bootstrap parameter
ensemble and samples one ensemble parameter vector for each simulation. Future
fixture rates are calculated for the entire run at once; scorelines are drawn in
a vectorized NumPy tensor and corrected with the Dixon-Coles low-score `tau`
factors. Final tables use the pure `rank_standings` function in
`backend/app/standings.py`.
The five criteria follow Appendix B in the buildplan and match the repository's
published 2018/19 and 2021/22 final-table references. The official RFEF site
returned HTTP 403 during this implementation, so current regulation text could
not be independently re-fetched here; that source check remains outstanding.

## Current-season fixture limitation

The 2026/27 DB season currently has 69 completed fixtures and no unplayed fixture
rows. The job derives the remaining fixtures as the set difference between those
played ordered home/away pairs and the 20-club, 380-pair double round robin. The
derived schedule has 311 fixtures. Those fixtures have no real kickoff dates or
matchday numbers; the job documents this on the stored run and uses
`as_of_matchday=0` to mean unknown, not a real matchday.

## Matchday-20 back-test

Historical SP1 match rows have no `matchday` values. The back-test starts by
sorting by `kickoff_utc` and checking per-club appearances in the first 200
fixtures. Where any club differs from 20, it uses the latest of the clubs'
individual twentieth-match kickoffs as one consistent as-of cut and includes
all results through that time. This preserves complete, balanced fixtures and
avoids training on results after the simulated remainder begins; some clubs
may consequently have played more than 20 matches in the adjusted snapshot.
The test remains a chronological matchday-20 proxy, not an official round
assignment.

## Stored output and independent checks

Each run appends a `simulation_runs` row with the model version, seed,
simulation count, as-of marker, per-team champion/top-four/top-six/relegation
probabilities, the full finish-position distribution, expected final points,
and raw per-run finish positions, team points, and draw counts.
`python -m jobs.verify_simulation --season-id 1734` independently reloads that
raw payload and recomputes the finish-position probability sums and exact
points-conservation check without importing simulator logic.

The historical gate is run with
`python -m jobs.backtest_simulation --sims 10000`; it reports each season's
actual champion, the top two teams by simulated title probability, and whether
the champion was among them.

## Verified run (2026-09-27)

Current season 2026/27 (`season_id=1734`) was run with production
`model_version_id=4`, 10,000 simulations, seed 42 and 100 match-bootstrap fits.
The 69 played plus 311 derived fixtures formed all 380 ordered pairs.
The 100 bootstrap fits took **7.865 seconds**; vectorized goal generation and
batch ranking took **1.533 seconds** (about **9.4 seconds** combined).
The stored raw payload is `simulation_runs.id=3`.

The independent script recomputed:

- All 20 team finish-position distributions sum to `1.000000`.
- All 20 position probabilities across teams sum to `1.000000`.
- Exact points conservation holds in `10,000/10,000` simulations, with `0`
  mismatches. Observed and expected total-points ranges were both `1016..1079`;
  the draw-count range was `61..124`.

The five-season historical gate was **4/5**, meeting the buildplan threshold of
at least 4/5. It did not pass every season:

| Season | Actual champion | Top two by simulated title probability | Gate |
| --- | --- | --- | --- |
| 2020/21 | Atlético Madrid | Atlético Madrid (0.749), Barcelona (0.160) | PASS |
| 2021/22 | Real Madrid | Real Madrid (0.900), Sevilla (0.087) | PASS |
| 2022/23 | Barcelona | Barcelona (0.904), Real Madrid (0.096) | PASS |
| 2023/24 | Real Madrid | Real Madrid (0.763), Barcelona (0.119) | PASS |
| 2024/25 | Barcelona | Real Madrid (0.711), Atlético Madrid (0.167) | **FAIL** |

For the first-200 chronological-slice check, 2022/23 and 2024/25 each had all
20 clubs at exactly 20 appearances. Three seasons needed the per-club check:

- 2020/21: first-200 appearance counts ranged 18–21; personal 20th-match dates
  ranged from 2021-01-22 to 2021-02-08. The coherent latest-20th-match cutoff
  included 215 fixtures.
- 2021/22: counts ranged 19–21; personal dates ranged from 2022-01-02 to
  2022-01-10. The coherent cutoff included 201 fixtures.
- 2023/24: counts ranged 19–21; personal dates ranged from 2024-01-12 to
  2024-01-22. The coherent cutoff included 207 fixtures.

The common cutoff includes matches after an earlier club's personal twentieth
fixture, so some clubs have more than 20 results in the snapshot. This keeps the
back-test state chronologically coherent; matchday numbers themselves are
missing from the source data.
