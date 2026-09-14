"""Insert missing canonical teams referenced in the YAML alias file
into the database with source='manual'."""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import yaml
from app.database import SessionLocal
from app.models import Team

ALIAS_FILE = os.path.join(os.path.dirname(__file__), "..", "db", "aliases", "football_data_co_uk.yaml")

# Load YAML and extract all target canonical names
with open(ALIAS_FILE, "r", encoding="utf-8") as f:
    parsed = yaml.safe_load(f)
    aliases = parsed.get("football_data_co_uk", {})

# Collect all unique target canonical names
targets = set(aliases.values())

# Additional teams that we know should exist from unmapped CSV names
# (these may have been manually added to YAML later)
extra_teams = {
    "Ath Bilbao B", "Cordoba", "Extremadura UD", "Cultural Leonesa",
    "Llagostera", "UD Logroñés", "Lorca FC", "Rayo Majadahonda",
    "Reus Deportiu",
}

all_targets = targets | extra_teams
print(f"Total target canonical names: {len(all_targets)}")

db = SessionLocal()
try:
    existing = {t.canonical_name: t for t in db.query(Team).all()}
    print(f"Existing teams in DB: {len(existing)}")

    inserted = 0
    for name in sorted(all_targets):
        if name not in existing:
            team = Team(
                canonical_name=name,
                short_name=name[:10],
                created_at=datetime.now(timezone.utc),
            )
            db.add(team)
            inserted += 1
            print(f"  INSERT: {name}")
        else:
            print(f"  EXISTS: {name} (id={existing[name].id})")

    if inserted:
        db.commit()
        print(f"\n✅ Inserted {inserted} new teams")
    else:
        print("\n✅ All teams already exist")

finally:
    db.close()