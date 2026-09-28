# La Liga season simulator

`python -m jobs.simulate_season --comp SP1` is the nightly baseline job. It
simulates 10,000 completions of the current La Liga schedule with a seeded NumPy
generator, applies the fitted Dixon–Coles score model, ranks the resulting
tables, and appends the results to `simulation_runs`. The existing simulation
table already contains the season, model version, simulation count, seed,
as-of matchday, and JSON result fields; no schema migration is needed.

## Model parameters and uncertainty

The baseline run loads the registered production `dixon_coles` artifact. It
does not select or promote a model version and does not refit or update the
production model. It checks that the matches in the artifact's recorded
training-date range match its recorded training-set size.

Parameter uncertainty uses 100 nonparametric match-bootstrap refits by default.
Each refit samples the same number of matches with replacement from that
training set. The ensemble consists of those vectors plus the unmodified
production vector. A vector is selected once per simulated season and reused
for every remaining fixture in that season. The job prints bootstrap fit count,
training sample size, and elapsed bootstrap time separately from the simulation
runtime.

For a selected vector, expected goal rates are
`lambda_home = exp(attack_home - defence_away + home_advantage)` and
`lambda_away = exp(attack_away - defence_home)`. One NumPy Poisson draw creates
the `(n_simulations, n_remaining_fixtures, 2)` score tensor. Dixon–Coles
dependence is imposed by vectorized rejection sampling: the four 0/1 goal
cells receive their exact `tau(0,0)`, `tau(0,1)`, `tau(1,0)`, and `tau(1,1)`
factors; each candidate is accepted with probability `tau / max(1, tau cells)`.
Rejected cells alone are redrawn in vectorized batches. This samples the
corrected distribution without an approximate post-hoc scoreline adjustment.

## Ranking and returned results

All completed and scheduled fixtures are passed through
`app.standings.rank_standings_many`, the vectorized counterpart of the tested
`rank_standings` implementation. Completed head-to-head records are applied
within point-tied groups, followed by overall goal difference, goals scored,
and a stable team-name ordering. Parity tests exercise generated final tables
with both two-way and three-way ties against the reference ranker.

Each team result includes title, top-four, top-six, and bottom-three
probabilities; probabilities for all 20 finish positions; expected final
points; and the 2.5th/97.5th percentile interval for final points.
The engine checks finish-position sums and exact total-points conservation for
every simulated season. Raw per-simulation arrays are checked in memory but
omitted from the cached JSON to keep the read-only API response small. Same
inputs and seed produce the same result.

`GET /api/v1/simulation?season=2026/27` reads the newest cached baseline run;
it does not start a simulation. The separate `simulation.yml` workflow runs
daily at 07:00 UTC and can also be started manually; it has no push trigger.

## Historical backtest

Use an explicit season and matchday to run a point-in-time backtest without
writing to `simulation_runs`:

```powershell
python -m jobs.simulate_season --comp SP1 --season 2024/25 --as-of-matchday 20 --backtest
```

Backtests refit only on match results available before the selected historical
cutoff, simulate the fixtures after it, and report the eventual champion's
title probability and rank. This mode intentionally does not use the current
production artifact, which may contain data from after a historical cutoff.
When stored matchday labels are absent, the job uses an explicitly labeled
chronological fixture-count proxy (20 clubs imply 10 fixtures per round) and
records that limitation in its output. Such a proxy is not an exact
matchday-20 reconstruction.

To exercise the engine locally with a bounded bootstrap cost, pass
`--parameter-bootstrap 20`; the normal nightly job uses the default 100.
