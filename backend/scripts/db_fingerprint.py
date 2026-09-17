#!/usr/bin/env python3
"""Print a stable fingerprint of the ingested data (idempotency gate).

The Phase 1 gate requires proof that re-running the full ingestion changes zero
rows. Row counts alone are too weak (a row deleted and another inserted cancels
out), so this hashes the ordered content of the match table and prints the row
count of every table. Run it before and after an ingest run; the two outputs must
be identical.

Usage:
    python scripts/db_fingerprint.py
    python scripts/db_fingerprint.py --csv before.csv
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text  # noqa: E402

from app.database import engine  # noqa: E402

TABLES = (
    "competitions",
    "seasons",
    "teams",
    "team_aliases",
    "matches",
    "match_events",
    "players",
    "player_season_stats",
    "standings_snapshots",
    "model_versions",
    "predictions",
    "simulation_runs",
    "api_request_log",
    "data_freshness",
)

MATCH_COLUMNS = (
    "id",
    "competition_id",
    "season_id",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "status",
    "home_goals",
    "away_goals",
    "home_goals_ht",
    "away_goals_ht",
    "closing_p_home",
    "closing_p_draw",
    "closing_p_away",
    "source",
    "external_id",
)


def match_digest(conn) -> tuple[str, int]:
    """(md5 of every match row in id order, row count)."""
    digest = hashlib.md5()
    rows = 0
    query = f"SELECT {', '.join(MATCH_COLUMNS)} FROM matches ORDER BY id"
    for row in conn.execution_options(stream_results=True).execute(text(query)):
        cells = "|".join("" if value is None else str(value) for value in row)
        digest.update(cells.encode("utf-8"))
        digest.update(b"\n")
        rows += 1
    return digest.hexdigest(), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fingerprint the ingested data.")
    parser.add_argument("--csv", default=None, help="also write the counts to this CSV")
    args = parser.parse_args()

    with engine.connect() as conn:
        digest, rows = match_digest(conn)
        counts = {
            name: conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar() for name in TABLES
        }

    print(f"matches rows   : {rows}")
    print(f"matches md5    : {digest}")
    for name, count in counts.items():
        print(f"  {name:<24} {count:>8}")

    if args.csv:
        path = Path(args.csv)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["table", "rows"])
            writer.writerow(["matches_md5", digest])
            for name, count in counts.items():
                writer.writerow([name, count])
        print(f"written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
