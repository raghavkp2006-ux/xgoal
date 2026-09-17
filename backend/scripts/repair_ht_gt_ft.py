#!/usr/bin/env python3
"""Repair matches whose half-time goals exceed their full-time goals.

The affected rows are an artifact of a historical alias mapping
(``La Coruna``/``Deportivo`` -> ``Deportivo Alaves``): each row was created for a
Deportivo La Coruna fixture (so it carries La Coruna's kickoff and full-time
score) under the Alaves team id, and ``jobs/enrich_match_stats.py`` later
overwrote its half-time columns from the genuine Alaves fixture that shares the
same (home, away) identity.

This script re-derives each row's true values from the cached
football-data.co.uk CSV of that division and season and rewrites kickoff date,
full-time and half-time goals in place. It never inserts or deletes rows, and it
refuses to touch a row whose fixture cannot be found in the cache.

Dry-run is the default. ``--apply`` writes, after snapshotting the previous
values to ``data/repairs/ht_gt_ft_<utc>.json`` so the change can be reverted by
hand.

Usage:
    python scripts/repair_ht_gt_ft.py
    python scripts/repair_ht_gt_ft.py --apply
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, time, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.database import SessionLocal  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJ / "data" / "csv_cache"
ALIAS_FILE = PROJ.parent / "db" / "aliases" / "football_data_co_uk.yaml"
SNAPSHOT_DIR = PROJ / "data" / "repairs"
KICKOFF_TIME = time(20, 0)  # matches the convention of the rows already stored

CANDIDATES = """
SELECT m.id, c.code AS comp, s.label AS season, s.start_year, m.kickoff_utc,
       ht.canonical_name AS home, at.canonical_name AS away,
       m.home_goals, m.away_goals, m.home_goals_ht, m.away_goals_ht
FROM matches m
JOIN seasons s ON s.id = m.season_id
JOIN competitions c ON c.id = m.competition_id
JOIN teams ht ON ht.id = m.home_team_id
JOIN teams at ON at.id = m.away_team_id
WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
  AND ((m.home_goals_ht IS NOT NULL AND m.home_goals_ht > m.home_goals)
    OR (m.away_goals_ht IS NOT NULL AND m.away_goals_ht > m.away_goals))
ORDER BY m.kickoff_utc
"""


def load_aliases() -> dict[str, str]:
    """{raw CSV name: canonical DB name} from the shared alias YAML."""
    with open(ALIAS_FILE, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("football_data_co_uk", {}) if isinstance(data, dict) else {}


def parse_date(raw: str) -> datetime | None:
    """football-data dates are dd/mm/yy in older files and dd/mm/yyyy in newer."""
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_int(raw: object) -> int | None:
    """CSV cells arrive as strings; anything non-numeric means 'no data'."""
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def csv_fixtures(code: str, comp: str, aliases: dict[str, str]) -> dict[tuple[str, str], dict]:
    """{(home, away) canonical: row} for one cached division/season CSV."""
    path = CACHE_DIR / f"{code}_{comp}.csv"
    found: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return found
    with path.open(encoding="latin-1") as handle:
        for item in csv.DictReader(handle):
            home = aliases.get((item.get("HomeTeam") or "").strip())
            away = aliases.get((item.get("AwayTeam") or "").strip())
            if home and away:
                found.setdefault((home, away), item)
    return found


def proposals(db, aliases: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """(fixable, unresolved) row proposals derived from the cached CSVs."""
    rows = db.execute(text(CANDIDATES)).fetchall()
    cache: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
    fixable: list[dict] = []
    unresolved: list[dict] = []
    for row in rows:
        code = f"{row.start_year % 100:02d}{(row.start_year + 1) % 100:02d}"
        key = (code, row.comp)
        if key not in cache:
            cache[key] = csv_fixtures(code, row.comp, aliases)
        source = cache[key].get((row.home, row.away))
        entry: dict = {
            "id": row.id,
            "comp": row.comp,
            "season": row.season,
            "fixture": f"{row.home} vs {row.away}",
            "kickoff": row.kickoff_utc,
            "ft": (row.home_goals, row.away_goals),
            "ht": (row.home_goals_ht, row.away_goals_ht),
        }
        if source is None:
            unresolved.append(entry)
            continue
        date = parse_date(source.get("Date") or "")
        home_goals = parse_int(source.get("FTHG"))
        away_goals = parse_int(source.get("FTAG"))
        home_ht = parse_int(source.get("HTHG"))
        away_ht = parse_int(source.get("HTAG"))
        if None in (date, home_goals, away_goals, home_ht, away_ht):
            unresolved.append(entry)
            continue
        assert date is not None
        entry["new_kickoff"] = date.replace(
            hour=KICKOFF_TIME.hour, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        entry["new_ft"] = (home_goals, away_goals)
        entry["new_ht"] = (home_ht, away_ht)
        entry["source"] = (
            f"{code}_{row.comp}.csv {source.get('Date')} "
            f"{source.get('HomeTeam')} vs {source.get('AwayTeam')}"
        )
        fixable.append(entry)
    return fixable, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite matches whose HT goals exceed their FT goals."
    )
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    aliases = load_aliases()
    db = SessionLocal()
    try:
        fixable, unresolved = proposals(db, aliases)
        print(
            f"candidates {len(fixable) + len(unresolved)}  fixable {len(fixable)}  "
            f"unresolved {len(unresolved)}"
        )
        for entry in fixable:
            print(f"\n  id={entry['id']} [{entry['comp']} {entry['season']}] {entry['fixture']}")
            print(
                f"    current : kickoff {entry['kickoff']:%Y-%m-%d} "
                f"FT {entry['ft'][0]}-{entry['ft'][1]} HT {entry['ht'][0]}-{entry['ht'][1]}"
            )
            print(
                f"    proposed: kickoff {entry['new_kickoff']:%Y-%m-%d} "
                f"FT {entry['new_ft'][0]}-{entry['new_ft'][1]} "
                f"HT {entry['new_ht'][0]}-{entry['new_ht'][1]}"
            )
            print(f"    source  : {entry['source']}")
        for entry in unresolved:
            print(
                f"\n  id={entry['id']} [{entry['comp']} {entry['season']}] "
                f"{entry['fixture']}  -> NO SOURCE ROW, left untouched"
            )

        if not args.apply:
            print("\nDRY RUN - re-run with --apply to write the changes")
            return 0

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot = SNAPSHOT_DIR / f"ht_gt_ft_{stamp}.json"
        snapshot.write_text(json.dumps(fixable, indent=2, default=str), encoding="utf-8")
        for entry in fixable:
            db.execute(
                text(
                    "UPDATE matches SET kickoff_utc = :kickoff, home_goals = :hg, "
                    "away_goals = :ag, home_goals_ht = :hhg, away_goals_ht = :hag "
                    "WHERE id = :id"
                ),
                {
                    "kickoff": entry["new_kickoff"],
                    "hg": entry["new_ft"][0],
                    "ag": entry["new_ft"][1],
                    "hhg": entry["new_ht"][0],
                    "hag": entry["new_ht"][1],
                    "id": entry["id"],
                },
            )
        db.commit()
        print(f"\nAPPLIED {len(fixable)} update(s); previous values saved to {snapshot}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
