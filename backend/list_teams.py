"""List current canonical teams in DB."""
from app.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    r = conn.execute(text("SELECT id, canonical_name, api_football_id FROM teams ORDER BY canonical_name"))
    print("Current canonical teams in DB:")
    for row in r:
        print(f"  {row.id:3d} | {row.canonical_name:30s} | api={row.api_football_id}")
    print(f"---\nTotal: {r.rowcount}")