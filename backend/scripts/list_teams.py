#!/usr/bin/env python3
"""List the canonical teams in the database.

Quick inspection utility: what is actually in ``teams`` right now, with the
optional API-Football id so an unmerged or API-only row stands out.

Usage:
    python scripts/list_teams.py
    python scripts/list_teams.py --filter deportivo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text  # noqa: E402

from app.database import engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="List canonical teams in the database.")
    parser.add_argument("--filter", default=None, help="only names containing this text")
    args = parser.parse_args()

    query = "SELECT id, canonical_name, api_football_id FROM teams"
    params: dict[str, str] = {}
    if args.filter:
        query += " WHERE canonical_name ILIKE :pattern"
        params["pattern"] = f"%{args.filter}%"
    query += " ORDER BY canonical_name"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    print(f"canonical teams: {len(rows)}")
    for row in rows:
        print(f"  {row.id:4d} | {row.canonical_name:<32} | api={row.api_football_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
