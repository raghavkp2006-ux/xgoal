#!/usr/bin/env python3
"""Rewrite the football-data.co.uk alias YAML so it points at the merged
canonical team names produced by jobs/dedup_teams.py.

Usage:
    python -m jobs.fix_alias_yaml            # dry-run
    python -m jobs.fix_alias_yaml --apply    # rewrite the file
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from app.database import SessionLocal
from app.models import Team

ALIAS_FILE = (Path(__file__).resolve().parent.parent.parent
              / "db" / "aliases" / "football_data_co_uk.yaml")

# old canonical value (pre-merge) -> new canonical value (post-merge)
VALUE_RENAMES = {
    "Alcorcon": "Alcorcón",
    "Almeria": "Almería",
    "Atletico Madrid": "Atlético Madrid",
    "Castellon": "Castellón",
    "Cadiz": "Cádiz",
    "Deportivo Alaves": "Deportivo Alavés",
    "La Coruna": "Deportivo La Coruña",
    "Gimnastic Tarragona": "Gimnàstic Tarragona",
    "Leganes": "Leganés",
    "Mirandes": "Mirandés",
    "Malaga": "Málaga",
    "Sporting Gijon": "Sporting Gijón",
    "Logrones": "UD Logroñés",
    "Valladolid": "Real Valladolid",
    "Andorra": "FC Andorra",
    "Cartagena": "FC Cartagena",
    "Lorca": "Lorca FC",
    "Espanol": "Espanyol",
    "Ath Bilbao": "Athletic Club",
}

# raw names that must be (re)pointed explicitly, regardless of current value
RAW_OVERRIDES = {
    "Alaves": "Deportivo Alavés",
    "La Coruna": "Deportivo La Coruña",
    "Deportivo": "Deportivo La Coruña",
    "Alcorcon": "Alcorcón",
    "Almeria": "Almería",
    "Castellon": "Castellón",
    "Cadiz": "Cádiz",
    "Deportivo Alaves": "Deportivo Alavés",
    "Gimnastic": "Gimnàstic Tarragona",
    "Gimnastic Tarragona": "Gimnàstic Tarragona",
    "Leganes": "Leganés",
    "Mirandes": "Mirandés",
    "Malaga": "Málaga",
    "Sp Gijon": "Sporting Gijón",
    "Sporting": "Sporting Gijón",
    "Sporting Gijon": "Sporting Gijón",
    "Logrones": "UD Logroñés",
    "UD Logrones": "UD Logroñés",
    "Valladolid": "Real Valladolid",
    "Andorra": "FC Andorra",
    "Cartagena": "FC Cartagena",
    "Lorca": "Lorca FC",
    "Espanol": "Espanyol",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atlético Madrid",
    "Atletico Madrid": "Atlético Madrid",
    # present in D2 2425 but missing from the alias file
    "Preußen Münster": "Preussen Muenster",
}

HEADER = (
    "# Team alias mappings: football-data.co.uk raw names -> canonical team names.\n"
    "# SP1/SP2: manually reviewed (accented Spanish club names).\n"
    "# Other divisions: auto-mapped (CSV name = canonical name).\n"
    "\n"
    "football_data_co_uk:\n"
)


def main():
    apply = "--apply" in sys.argv
    data = yaml.safe_load(open(ALIAS_FILE, encoding="utf-8")) or {}
    aliases = dict(data.get("football_data_co_uk", {}))
    before = len(aliases)

    changes = []
    for raw, canon in list(aliases.items()):
        if canon in VALUE_RENAMES:
            aliases[raw] = VALUE_RENAMES[canon]
            changes.append(f"  value {raw!r}: {canon!r} -> {aliases[raw]!r}")

    added = []
    for raw, canon in RAW_OVERRIDES.items():
        if aliases.get(raw) != canon:
            added.append(f"  {raw!r}: {aliases.get(raw)!r} -> {canon!r}")
            aliases[raw] = canon

    # validate every alias resolves to a team that exists in the DB
    db = SessionLocal()
    try:
        names = {t.canonical_name for t in db.query(Team).all()}
        unresolved = sorted({v for v in aliases.values() if v not in names})
    finally:
        db.close()

    print(f"aliases before={before} after={len(aliases)}")
    print(f"\nvalue renames applied ({len(changes)}):")
    for c in changes:
        print(c)
    print(f"\nraw overrides applied ({len(added)}):")
    for a in added:
        print(a)

    if unresolved:
        print(f"\n!! {len(unresolved)} alias value(s) do NOT resolve to a DB team:")
        for u in unresolved[:40]:
            raws = [r for r, v in aliases.items() if v == u]
            print(f"   {u!r} <- {raws[:5]}")
    else:
        print("\nOK: every alias value resolves to an existing DB team")

    if not apply:
        print("\nDRY RUN - re-run with --apply to rewrite the file")
        return

    lines = [HEADER]
    for raw, canon in sorted(aliases.items(), key=lambda kv: kv[0].lower()):
        lines.append(f'  "{raw}": "{canon}"\n')
    ALIAS_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"\nWROTE {ALIAS_FILE} ({len(aliases)} aliases)")


if __name__ == "__main__":
    main()
