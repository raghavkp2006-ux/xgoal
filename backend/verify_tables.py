"""Verify all 14 expected tables were created in the database."""
from app.database import engine
from sqlalchemy import text

expected = {
    "competitions", "seasons", "teams", "team_aliases", "matches",
    "match_events", "players", "player_season_stats", "standings_snapshots",
    "model_versions", "predictions", "simulation_runs", "api_request_log",
    "data_freshness",
}

with engine.connect() as conn:
    result = conn.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    )
    actual = {r[0] for r in result}

missing = expected - actual
extra = actual - expected - {"alembic_version"}

print(f"Total tables found: {len(actual)}")
print(f"Tables: {sorted(actual)}")
if missing:
    print(f"\nMISSING ({len(missing)}): {sorted(missing)}")
else:
    print(f"\nAll {len(expected)} expected tables present ✓")

if extra:
    print(f"Extra tables: {sorted(extra)}")