# La Liga Analytics Platform - Build Plan v2

## Phase 1: Foundation & Data Ingestion

### 1.1 Database Schema (Alembic Migrations)
- [x] Initial schema with all 13 tables
- [ ] Verify `alembic upgrade head` against real Neon DB
- [ ] Confirm all tables created

### 1.2 API-Football Integration
- [ ] Verify API key works against /status and /leagues?id=140
- [ ] Implement data ingestion service
- [ ] Create API request logging

### 1.3 Data Models & Pydantic Schemas
- [ ] Complete all Pydantic schemas
- [ ] CRUD operations for all entities

### 1.4 Seed Data
- [ ] Seed competitions (La Liga)
- [ ] Seed initial teams
- [ ] Seed initial players

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