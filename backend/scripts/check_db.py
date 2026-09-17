#!/usr/bin/env python3
"""Check the database schema and row counts (Phase 1.1 acceptance check).

Two read-only checks in one pass:

* every table the schema contract expects actually exists — the build plan's
  "confirm all tables created" step after ``alembic upgrade head``;
* row counts per table, so a failed, partial or half-migrated ingest is obvious
  at a glance.

Exit code is 1 when an expected table is missing, which makes it usable as a
deploy gate; ``--counts-only`` skips the schema check.

Usage:
    python scripts/check_db.py
    python scripts/check_db.py --counts-only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text  # noqa: E402

from app.database import engine  # noqa: E402

# The 14 tables of the Phase 1 schema (see app/models.py). Kept as an explicit
# list rather than derived from the models, so the check is independent of them.
EXPECTED_TABLES = (
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify tables exist and report row counts.")
    parser.add_argument("--counts-only", action="store_true", help="skip the schema check")
    args = parser.parse_args()

    with engine.connect() as conn:
        actual = {
            name
            for (name,) in conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            )
        }
        missing = [name for name in EXPECTED_TABLES if name not in actual]
        unexpected = sorted(actual - set(EXPECTED_TABLES) - {"alembic_version"})

        if not args.counts_only:
            print(f"tables present: {len(actual)}")
            if missing:
                print(f"  MISSING ({len(missing)}): {missing}")
            else:
                print(f"  all {len(EXPECTED_TABLES)} expected tables present")
            if unexpected:
                print(f"  unexpected: {unexpected}")

        print("\nrow counts")
        for name in EXPECTED_TABLES:
            if name not in actual:
                print(f"  {name:<24} {'-':>8}")
                continue
            count = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            print(f"  {name:<24} {count:>8}")

    if missing and not args.counts_only:
        print(f"\nFAILED: {len(missing)} expected table(s) missing")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
