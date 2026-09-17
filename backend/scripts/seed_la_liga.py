#!/usr/bin/env python3
"""Seed La Liga (API-Football league 140) data into the database.

Usage:
    python scripts/seed_la_liga.py [--season 2025]
"""
import argparse
import os
import sys

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingestion import full_ingest


def main():
    parser = argparse.ArgumentParser(description="Seed La Liga data")
    parser.add_argument("--season", type=int, default=None, help="Season year (e.g. 2025)")
    args = parser.parse_args()

    print("=== Seeding La Liga (ID 140) ===")
    full_ingest(api_league_id=140, season_year=args.season)


if __name__ == "__main__":
    main()
