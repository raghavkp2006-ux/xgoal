# xgoal — La Liga Analytics Platform

A portfolio project: La Liga analytics powered by a Dixon-Coles scoreline model, with predictions, live standings, and a Monte Carlo season simulator.

## Architecture

```
GitHub repo (monorepo)
  ├── GitHub Actions: CI (pytest, ruff, mypy) on every push
  ├── GitHub Actions: cron jobs ──────────► API-Football ──► Neon Postgres
  │     • pre-matchday prediction writer — daily 06:00 UTC (predictions.yml, live)
  │     • live poller (matchday windows only) — planned
  │     • nightly reconcile + standings rebuild — planned
  │     • weekly player snapshot — planned
  │     • nightly season simulation — planned
  ├── Render free web service (FastAPI) ──► Neon Postgres  [reads only]
  └── Vercel (Next.js) ───────────────────► FastAPI
```

## Stack

- **Backend:** FastAPI, SQLAlchemy 2.x, Alembic, Python 3.12
- **Database:** Neon Postgres (free tier — scales to zero, doesn't expire)
- **Frontend:** Next.js App Router, TypeScript, Tailwind, TanStack Query, Recharts
- **Scheduled jobs:** GitHub Actions cron
- **ML:** scikit-learn / XGBoost (comparison), hand-rolled Dixon-Coles (primary)

## Data Sources

| Source | Use | Coverage |
|---|---|---|
| football-data.co.uk | Historical training data | 20+ divisions, 10+ seasons |
| API-Football (v3) | Current season fixtures, live scores, players | La Liga (league 140) |

## Model

**Primary:** Dixon-Coles Poisson scoreline model with:
- Per-team attack/defence ratings
- Global home advantage
- Low-score correction (τ)
- Exponential time decay

**Comparison:** XGBoost / multinomial logistic regression on point-in-time features

**Baselines:** Elo, base-rate, uniform, de-vigged closing odds

**Evaluation:** Walk-forward across 5 seasons. Log loss (primary), RPS, Brier, ECE, and
pick accuracy against the 51-54% target band.

### Walk-forward results — La Liga, 2020/21 to 2024/25 (1,900 out-of-sample matches)

| model | log loss | RPS | Brier | ECE | accuracy |
|---|---|---|---|---|---|
| closing odds (market) | 0.9693 | 0.1912 | 0.5756 | 0.0373 | 54.3% |
| **Dixon-Coles** | **0.9916** | **0.1983** | **0.5908** | **0.0194** | **52.3%** |
| Elo | 0.9957 | 0.1979 | 0.5932 | 0.0291 | 53.3% |
| XGBoost | 1.0583 | 0.2156 | 0.6351 | 0.0524 | 46.7% |
| base rate | 1.0736 | 0.2260 | 0.6493 | 0.0157 | 44.3% |
| uniform | 1.0986 | 0.2328 | 0.6667 | 0.1093 | 44.3% |

Acceptance checks (`evaluation.target_lines()`, printed by `jobs/train_model.py`):

```
[PASS] beats base rate on all 5 seasons - 5/5
[PASS] beats Elo on >= 4 of 5 seasons - 5/5
[FAIL] within 0.03 log loss of closing odds - worst gap +0.0355 (2024/25), mean +0.0223
[PASS] pick accuracy inside 51-54% - pooled 52.26%, 3/5 seasons in band
```

The closing-line target is a **marginal miss, reported as such**: 4 of 5 seasons sit
inside 0.03 (mean gap +0.0223) and 2024/25 misses by 0.0055. A half-life sweep
(`jobs/sweep_half_life.py`) confirms it cannot be closed by tuning the time decay —
the validation-selected half-life (1095 days, chosen on 2019/20 alone) is *worse* on
the test window (0.9926) than the untuned default (540 days, 0.9915), and the whole
grid spans 0.0013 log loss. The unfavourable number stands rather than being tuned away.

**Accuracy note:** the primary model lands in the intended band at 52.3% pooled. It is
reported, not optimised — the plan explicitly does not chase a higher headline number,
and Elo's slightly higher accuracy comes with worse calibration (ECE 0.0291 vs 0.0194).

**Calibration:** Isotonic regression fitted on a dedicated split (earlier out-of-sample
seasons), reported with reliability tables and ECE before/after
(`python -m jobs.calibrate_forecasts`). It pays off for the tree model and not for
Dixon-Coles, which is already calibrated — see the report's pooled verdict row.

**Leakage:** A 200-match point-in-time test asserts that a fixture's features cannot
change when its own result or any later result is rewritten, and that the harness refits
each season only on strictly earlier matches (`tests/test_ml_leakage.py`).

## Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & skeleton deploy | ✅ Complete |
| 1 | Data foundation (schema, ingestion, validation) | ✅ Complete — Phase 1 data foundation gate closed (57/0); security follow-up: rotate the API key exposed in commit 5030750 |
| 2 | Prediction model (Dixon-Coles, XGBoost, evaluation) | ✅ Complete — walk-forward results above |
| 3 | Live pipeline (poller, standings, reconciliation) | 🔜 |
| 4 | Frontend (dashboard, fixtures, model scoreboard) | 🔜 |
| 5 | Season simulator (Monte Carlo, what-if mode) | 🔜 |
| 6 | Hardening, docs, writeup | 🔜 |

## Limitations

- No xG data (shots-on-target used as proxy)
- No lineup, injury, or suspension data
- Player stats are current-season-only, updated weekly
- Free-tier cold start on Render (~30-60s on first request after idle)
- API-Football free tier: 100 requests/day

## Attribution

- Historical match data from [football-data.co.uk](https://www.football-data.co.uk/) (free for personal use)
- Current season data from [API-Football](https://www.api-football.com/)
- Dixon-Coles model based on Dixon & Coles (1997), *Applied Statistics* 46(2)