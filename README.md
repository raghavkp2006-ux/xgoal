# xgoal — La Liga Analytics Platform

A portfolio project: La Liga analytics powered by a Dixon-Coles scoreline model, with predictions, live standings, and a Monte Carlo season simulator.

## Architecture

```
GitHub repo (monorepo)
  ├── GitHub Actions: CI (pytest, ruff, mypy) on every push
  ├── GitHub Actions: cron jobs ──────────► API-Football ──► Neon Postgres
  │     • live poller (matchday windows only)
  │     • nightly reconcile + standings rebuild
  │     • weekly player snapshot
  │     • pre-matchday prediction writer
  │     • nightly season simulation
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

**Evaluation:** Walk-forward across 5 seasons. Log loss (primary), RPS, Brier, ECE.

## Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & skeleton deploy | ✅ Complete |
| 1 | Data foundation (schema, ingestion, validation) | 🔜 |
| 2 | Prediction model (Dixon-Coles, XGBoost, evaluation) | 🔜 |
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