#!/usr/bin/env python3
"""Fast bootstrap: download 1 season per division, extract names, insert teams + aliases."""
import csv, io, os, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from app.database import SessionLocal
from app.models import Team

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
ALIAS_FILE = (BASE.parent.parent / "db" / "aliases" / "football_data_co_uk.yaml").resolve()
DIVS = ["SP1","SP2","E0","E1","E2","E3","D1","D2","I1","I2","F1","F2","N1","P1","T1","B1","G1","SC0","SC1"]

def download(div, sc="2425"):
    try:
        with urllib.request.urlopen(f"https://www.football-data.co.uk/mmz4281/{sc}/{div}.csv", timeout=30) as r:
            c = r.read().decode("latin-1")
        if c.startswith("\ufeff"): c = c[1:]
        return list(csv.DictReader(io.StringIO(c)))
    except: return None

prev = {}
if ALIAS_FILE.exists():
    try:
        t = ALIAS_FILE.read_text(encoding="utf-8", errors="replace")
        p = yaml.safe_load(t)
        if isinstance(p, dict) and "football_data_co_uk" in p: prev = p["football_data_co_uk"]
    except: pass
print(f"Existing aliases: {len(prev)}")

print("Downloading 1 season per division to extract team names...")
div_names = {}
for d in DIVS:
    rows = download(d)
    if rows is None:
        print(f"  {d}: not available (2425)")
        div_names[d] = set()
        continue
    names = set()
    for r in rows:
        ht = (r.get("HomeTeam") or "").strip()
        at = (r.get("AwayTeam") or "").strip()
        if ht: names.add(ht)
        if at: names.add(at)
    div_names[d] = names
    print(f"  {d}: {len(names)} names")

db = SessionLocal()
try:
    existing = {t.canonical_name: t for t in db.query(Team).all()}
    print(f"Existing DB teams: {len(existing)}")

    new_aliases = dict(prev)
    to_add = {}

    for d in DIVS:
        for raw_name in div_names[d]:
            if raw_name in prev: continue
            new_aliases[raw_name] = raw_name
            if raw_name not in existing: to_add[raw_name] = d

    inserted = 0
    for name in sorted(to_add):
        db.add(Team(canonical_name=name, created_at=datetime.now(timezone.utc)))
        inserted += 1
    if inserted:
        db.commit()
        print(f"Inserted {inserted} new teams")
    else:
        print("No new teams needed")

    now_set = {t.canonical_name for t in db.query(Team).all()}
    bad = [(r, c) for r, c in new_aliases.items() if c not in now_set]
    if bad:
        print(f"WARNING: {len(bad)} unmapped aliases:")
        for r, c in bad[:10]: print(f"  '{r}' -> '{c}'")
    else:
        print(f"All {len(new_aliases)} aliases resolve to DB teams")

    lines = [
        "# Team alias mappings: football-data.co.uk raw names -> canonical team names.\n",
        "# SP1/SP2: manually reviewed.\n",
        "# Other divisions: auto-mapped (CSV name = canonical name).\n",
        "\n", "football_data_co_uk:\n",
    ]
    for raw, canon in sorted(new_aliases.items(), key=lambda x: x[0].lower()):
        lines.append(f'  "{raw}": "{canon}"\n')
    ALIAS_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(new_aliases)} aliases")

    final = db.query(Team).count()
    print(f"Final team count: {final}")
finally: db.close()
print("Done.")
