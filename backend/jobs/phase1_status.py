#!/usr/bin/env python3
"""Print a one-screen summary of the Phase 1 data state."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text
from app.database import SessionLocal
from ingest_direct import load_aliases

db = SessionLocal()
aliases = load_aliases()
print("=" * 72)
print("  XGoal Phase 1 - data state")
print("=" * 72)

print(f"  competitions : {db.execute(text('SELECT COUNT(*) FROM competitions')).scalar()}")
print(f"  seasons      : {db.execute(text('SELECT COUNT(*) FROM seasons')).scalar()}")
print(f"  teams        : {db.execute(text('SELECT COUNT(*) FROM teams')).scalar()}")
print(f"  aliases yaml : {len(aliases)}")
print(f"  team_aliases : {db.execute(text('SELECT COUNT(*) FROM team_aliases')).scalar()}")
print(f"  matches      : {db.execute(text('SELECT COUNT(*) FROM matches')).scalar()}")
print(f"  seasons w/data: "
      f"{db.execute(text('SELECT COUNT(DISTINCT season_id) FROM matches')).scalar()}")

print("\n  per competition (matches | seasons | teams used)")
for r in db.execute(text("""
        SELECT c.code,
               COUNT(m.id) AS ms,
               COUNT(DISTINCT m.season_id) AS seas,
               COUNT(DISTINCT t.id) AS nteams
        FROM competitions c
        LEFT JOIN matches m ON m.competition_id = c.id
        LEFT JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
        GROUP BY c.code ORDER BY c.code""")).fetchall():
    print(f"    {r.code:<6} {r.ms:>6} | {r.seas:>2} | {r.nteams:>3}")

unmapped = sorted({v for v in aliases.values()})
names = {n for (n,) in db.execute(text("SELECT canonical_name FROM teams")).fetchall()}
missing = [u for u in unmapped if u not in names]
print(f"\n  alias targets not in DB: {len(missing)}"
      + (f" -> {missing[:8]}" if missing else ""))

dupes = db.execute(text("""
    SELECT lower(canonical_name), COUNT(*) c FROM teams
    GROUP BY lower(canonical_name) HAVING COUNT(*) > 1""")).fetchall()
print(f"  duplicate team names: {len(dupes)}")
print("=" * 72)
db.close()
