"""Clear old API-seeded data to make way for CSV ingestion."""
from app.database import SessionLocal
from app.models import Match, PlayerSeasonStat, StandingsSnapshot, ApiRequestLog, DataFreshness

db = SessionLocal()
try:
    count = db.query(Match).filter(Match.source == "api-football").delete()
    print(f"Deleted {count} API-football matches")
    db.commit()
finally:
    db.close()