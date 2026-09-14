"""Verify data counts after seeding."""
from app.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    queries = {
        "Competitions": "SELECT COUNT(*) FROM competitions",
        "Seasons": "SELECT COUNT(*) FROM seasons",
        "Teams": "SELECT COUNT(*) FROM teams",
        "Team Aliases": "SELECT COUNT(*) FROM team_aliases",
        "Matches": "SELECT COUNT(*) FROM matches",
        "Players": "SELECT COUNT(*) FROM players",
        "Player Season Stats": "SELECT COUNT(*) FROM player_season_stats",
        "Standings Snapshots": "SELECT COUNT(*) FROM standings_snapshots",
        "Data Freshness": "SELECT COUNT(*) FROM data_freshness",
        "API Request Logs": "SELECT COUNT(*) FROM api_request_log",
    }
    print("=== Data Counts After Seed ===")
    for name, sql in queries.items():
        result = conn.execute(text(sql))
        count = result.scalar()
        print(f"  {name:25s}: {count}")