"""Generate alias entries for all non-Spanish divisions by scanning CSVs.
For SP1/SP2 the aliases are already reviewed. For other divisions the CSV
team names are typically exact matches or trivially mappable.
"""
import csv
import io
import os
import sys
import urllib.request

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

ALIAS_FILE = os.path.join(os.path.dirname(__file__), "..", "db", "aliases", "football_data_co_uk.yaml")

DIVISIONS = [
    "E0", "E1", "E2", "E3",
    "D1", "D2",
    "I1", "I2",
    "F1", "F2",
    "N1", "P1", "T1", "B1", "G1", "SC0", "SC1",
]

def season_codes(n=10):
    codes = []
    for i in range(n):
        start = 25 - 1 - i
        codes.append(f"{(start) % 100:02d}{(start + 1) % 100:02d}")
    return codes

# Load existing aliases
with open(ALIAS_FILE, "r", encoding="utf-8") as f:
    parsed = yaml.safe_load(f)
aliases = parsed.get("football_data_co_uk", {})

# Extract all team names from non-Spanish divisions
all_names = set()
codes = season_codes()
for code in codes:
    for div in DIVISIONS:
        url = f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read()
        except Exception:
            continue
        content = raw.decode("latin-1")
        if content.startswith("\ufeff"):
            content = content[1:]
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            ht = (row.get("HomeTeam") or "").strip()
            at = (row.get("AwayTeam") or "").strip()
            if ht:
                all_names.add(ht)
            if at:
                all_names.add(at)

print(f"Found {len(all_names)} distinct team names across non-Spanish divisions")

# Add mappings — for non-Spanish leagues, team names in CSV are typically
# the same as the canonical name (e.g. "Liverpool", "Bayern Munich", etc.)
added = 0
for name in sorted(all_names):
    if name not in aliases:
        aliases[name] = name
        added += 1

print(f"Added {added} new alias mappings (auto: name == canonical)")
print(f"Total aliases: {len(aliases)}")

# Write back
header = (
    "# Team alias mappings: football-data.co.uk raw names → canonical team names.\n"
    "# SP1/SP2: manually reviewed.\n"
    "# Other divisions: auto-mapped (CSV name = canonical name, which holds for\n"
    "# English, German, Italian, French, Dutch, Portuguese, Turkish, Belgian,\n"
    "# Greek, and Scottish leagues).\n\n"
    "football_data_co_uk:\n"
)

lines = [header]
for raw, canon in sorted(aliases.items(), key=lambda x: x[0].lower()):
    lines.append(f'  "{raw}": "{canon}"\n')

with open(ALIAS_FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"✅ Written to {ALIAS_FILE}")