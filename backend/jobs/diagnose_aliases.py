#!/usr/bin/env python3
"""Diagnostic: scan cached CSVs for unresolved / self-playing aliases.

Writes results to backend/diag_out.txt as UTF-8 (avoids Windows console encoding
issues with accented team names).
"""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from app.database import SessionLocal
from app.models import Team
from ingest_direct import fetch_cached, parse_csv, DIVS, season_codes, ALIAS_FILE

OUT = Path(__file__).resolve().parent.parent / "diag_out.txt"
buf = []


def say(msg=""):
    buf.append(str(msg))


aliases = (yaml.safe_load(open(ALIAS_FILE, encoding="utf-8")) or {}).get(
    "football_data_co_uk", {})

db = SessionLocal()
teams = {t.canonical_name: t for t in db.query(Team).all()}
say(f"teams={len(teams)} aliases={len(aliases)}")

no_alias, no_team, selfplay = {}, {}, {}
missing_csv = []
for div in DIVS:
    for sc in season_codes(10):
        content = fetch_cached(sc, div)
        if not content:
            missing_csv.append(f"{div} {sc}")
            continue
        for row in parse_csv(content):
            rh = (row.get("HomeTeam") or "").strip()
            ra = (row.get("AwayTeam") or "").strip()
            if not rh or not ra:
                continue
            ch = aliases.get(rh)
            ca = aliases.get(ra)
            if not ch:
                no_alias.setdefault(rh, set()).add(f"{div} {sc}")
            if not ca:
                no_alias.setdefault(ra, set()).add(f"{div} {sc}")
            if ch and not teams.get(ch):
                no_team.setdefault(ch, set()).add(f"{div} {sc}")
            if ca and not teams.get(ca):
                no_team.setdefault(ca, set()).add(f"{div} {sc}")
            if ch and ca and ch == ca:
                selfplay.setdefault((rh, ra, ch), set()).add(f"{div} {sc}")

say(f"\n=== MISSING CSVs ({len(missing_csv)}) ===")
for m in missing_csv:
    say(f"  {m}")

say(f"\n=== UNMAPPED RAW NAMES ({len(no_alias)}) ===")
for k in sorted(no_alias):
    say(f"  {k!r}  <- {sorted(no_alias[k])[:4]}")

say(f"\n=== CANONICAL NOT IN DB ({len(no_team)}) ===")
for k in sorted(no_team):
    say(f"  {k!r}  <- {sorted(no_team[k])[:4]}")

say(f"\n=== SELF-PLAY (same canonical on both sides) ({len(selfplay)}) ===")
for k in sorted(selfplay):
    say(f"  {k[0]!r} vs {k[1]!r} -> {k[2]!r}  <- {sorted(selfplay[k])[:4]}")

# Existing DB teams that look related to the problem names
say("\n=== RELATED DB TEAMS ===")
keys = ["alav", "coru", "deportivo", "munster", "preu", "espa", "osasu"]
for t in db.query(Team).all():
    n = t.canonical_name
    if any(k in n.lower() for k in keys):
        say(f"  id={t.id} {n!r}")

db.close()
OUT.write_text("\n".join(buf), encoding="utf-8")
print(f"wrote {OUT} ({len(buf)} lines)")
