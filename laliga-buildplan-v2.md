# La Liga Analytics Platform - Build Plan v2

## Phase 1: Foundation & Data Ingestion

### 1.1 Database Schema (Alembic Migrations)
- [x] Initial schema with all 13 tables (schema holds 14 application tables; `0001_initial_schema`)
- [x] Confirm all tables created — verified 2026-09-16 against Neon via `python scripts/check_db.py`: 15 tables present (14 + `alembic_version`)
- [ ] Verify `alembic upgrade head` against a *fresh* Neon database — the seeded DB is migrated, but no from-scratch run has been recorded

### 1.2 API-Football Integration
- [x] Verify API key works against `/status` and `/leagues?id=140` — `python scripts/verify_api.py` exits 0 (account + quota returned). Caveat: `/leagues?id=140` returns an empty `seasons` array
- [ ] Implement data ingestion service — `app/ingestion.py` + `jobs/ingest_api.py` exist but are superseded by the CSV pipeline and have not been exercised end to end since
- [x] Create API request logging — `api_request_log` is written by `app/api_client.py` (quota header captured per call)
- [ ] **Blocker:** revoke the API key exposed in commit `5030750`. That value still authenticates (checked 2026-09-16, HTTP 200 + account block); rotation must happen in the API-Football dashboard, then the new value goes into `backend/.env`

### 1.3 Data Models & Pydantic Schemas
- [ ] Complete all Pydantic schemas — `app/schemas.py` covers competitions/seasons/teams/matches/standings; players and simulations unverified
- [ ] CRUD operations for all entities — read endpoints exist for competitions/matches/standings/teams; no players or simulations routers; `mypy app/` reports 47 errors, all in `app/routers/*` + `app/main.py`

### 1.4 Seed Data
- [x] Seed competitions (La Liga) — 19 competitions ingested
- [x] Seed initial teams — 529 teams, 548 alias entries in `db/aliases/football_data_co_uk.yaml`
- [ ] Seed initial players — `players` 20 rows, `player_season_stats` 23 rows from a single API pull; no seed script

### 1.5 Validation
- [x] `jobs/validate.py` suite runs: 57 checks pass, 0 fail (2026-09-16)
- [x] Fixed the 9 rows where half-time goals exceeded full-time goals (root-caused 2026-09-16: historical alias mapping `La Coruna`/`Deportivo` → `Deportivo Alaves` put Deportivo La Coruña fixtures under the Alavés team id, then `jobs/enrich_match_stats.py` overwrote the HT columns from the genuine Alavés fixture sharing that identity). Applied via `scripts/repair_ht_gt_ft.py --apply` (dry-run by default, snapshots to `data/repairs/`), then re-ran `jobs/train_model` so the registered artifact matches the corrected data
- [x] Widened the HT check to the away side (it had reported only 5 of the 9 offenders)
- [x] Final tables verified position-by-position, tiebreak chain included, for 3 seasons — `jobs/validate.py` section 8 compares 2018/19 and 2021/22 against `db/reference/la_liga_standings.json` (published Wikipedia tables) and 2024/25 against `standings_snapshots`; all 20 positions match in each
- [x] Idempotency proof — full ingestion re-run inserted 0 rows and the match-table content hash was unchanged (`scripts/db_fingerprint.py`: `5bbcc5e29c891e8eeca497b6aaa213cd` before and after)

### Phase 1 data foundation gate — CLOSED (2026-09-17)

All five acceptance gates pass:

- [x] `jobs/validate.py` exits 0: 57 checks passed, 0 failed
- [x] Dataset volume: 3,800 SP1 (La Liga) matches over 10 seasons; 65,921 matches across 19 competitions
- [x] Final-table reconstruction matches published ordering, including tiebreaks, for 2018/19, 2021/22, and 2024/25
- [x] Full-ingestion rerun is idempotent: 0 rows inserted and the matches content hash is unchanged
- [x] Team-name mapping is complete: 548 aliases and 0 unmapped targets

API-Football key rotation remains tracked in section 1.2 and does not block this
data foundation gate.

## Phase 2: Data Pipeline
- [ ] Match ingestion
- [ ] Player stats ingestion
- [ ] Standings computation
- [ ] Data freshness tracking

## Phase 3: Prediction Engine
- [ ] Model versioning
- [ ] Prediction generation
- [ ] Simulation runs

## Phase 4: API & Frontend
- [ ] REST API endpoints
- [ ] Next.js frontend
- [ ] Dashboard views