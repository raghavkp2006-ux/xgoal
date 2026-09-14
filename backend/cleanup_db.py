"""Clean up orphaned match_status type from partial migration."""
from app.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("DROP TYPE IF EXISTS match_status CASCADE"))
    conn.commit()
print("Cleaned up match_status type")