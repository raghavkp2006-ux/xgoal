#!/usr/bin/env python3
"""
Phase 1.2 — Team identity resolution (football-data.co.uk → canonical).

1. Download SP1 + SP2 CSVs for N seasons (default 10, 2015/16–2024/25).
2. Extract every distinct HomeTeam/AwayTeam string across all CSVs.
3. Fuzzy-match (rapidfuzz) against canonical team names from DB.
4. Write proposed mappings to db/aliases/football_data_co_uk.yaml.
5. STOP — user reviews and corrects before any ingestion uses it.

Usage:
    python -m jobs.resolve_teams [--seasons 10]
"""

import argparse
import csv
import io
import os
import sys
import urllib.request
from collections import OrderedDict

import yaml
from rapidfuzz import fuzz, process as fuzz_process

# Ensure package discovery
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import engine
from sqlalchemy import text

DIVISIONS = ["SP1", "SP2"]
ALIAS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "db", "aliases", "football_data_co_uk.yaml")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Season codes: SSSS = two-digit start year + two-digit end year
def season_codes(n: int):
    """Generate the last N season codes in descending order.
    E.g. for n=1 and current year 2025 → "2526".
    Uses 2025-26 as the latest complete season.
    """
    # Latest complete season ending in 2025
    end_year = 2025
    codes = []
    for i in range(n):
        start = end_year - 1 - i
        end_short = (start + 1) % 100
        codes.append(f"{start % 100:02d}{end_short:02d}")
    return codes


def download_csv(season_code: str, div: str) -> list[dict]:
    """Download a CSV from football-data.co.uk and return parsed rows."""
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{div}.csv"
    print(f"  Downloading {url} ...", end=" ")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"FAILED: {e}")
        return []

    # Decode, strip BOM
    content = raw.decode("latin-1")
    if content.startswith("\ufeff"):
        content = content[1:]

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    print(f"{len(rows)} rows")
    return rows


def extract_team_names(rows: list[dict]) -> set[str]:
    """Extract all distinct HomeTeam and AwayTeam strings."""
    names: set[str] = set()
    for r in rows:
        ht = (r.get("HomeTeam") or "").strip()
        at = (r.get("AwayTeam") or "").strip()
        if ht:
            names.add(ht)
        if at:
            names.add(at)
    return names


def get_canonical_teams() -> dict[str, int]:
    """Return {canonical_name: id} from DB."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, canonical_name FROM teams ORDER BY canonical_name")
        )
        return {r.canonical_name: r.id for r in rows}


def propose_mappings(
    raw_names: set[str],
    canonical: dict[str, int],
    threshold: float = 75.0,
) -> OrderedDict:
    """Fuzzy-match each raw name against canonical list. Return ordered
    dict of {raw_name: proposed_canonical_name}.
    """
    canonical_names = list(canonical.keys())
    proposals = OrderedDict()

    # Manual overrides for known tricky cases (prevent common fuzzy-match traps)
    manual_overrides = {
        "Ath Bilbao": "Athletic Club",
        "Ath Madrid": "Atlético Madrid",
        "Ath. Bilbao": "Athletic Club",
        "Ath. Madrid": "Atlético Madrid",
        "Athletic Bilbao": "Athletic Club",
        "Atletico Madrid": "Atlético Madrid",
        "Sociedad": "Real Sociedad",  # ← explicit override for the trap
        "Betis": "Real Betis",
        "Vallecano": "Rayo Vallecano",
        "Rayo": "Rayo Vallecano",
        "Celta": "Celta Vigo",
        "Espanol": "Espanyol",
        "Dep La Coruna": "Deportivo La Coruña",
        "La Coruna": "Deportivo La Coruña",
        "Sp Gijon": "Sporting Gijón",
        "Sporting": "Sporting Gijón",
        "Alaves": "Deportivo Alavés",
        "Malaga": "Málaga",
        "Cadiz": "Cádiz",
        "Almeria": "Almería",
        "Leganes": "Leganés",
        "Gimnastic": "Gimnàstic Tarragona",
        "Racing Santander": "Racing Santander",
        "Santander": "Racing Santander",
        "Huelva": "Recreativo Huelva",
        "Recreativo": "Recreativo Huelva",
        "Zaragoza": "Real Zaragoza",
        "Oviedo": "Real Oviedo",
        "Valldolid": "Real Valladolid",
        "Villareal": "Villarreal",
        "Mirandes": "Mirandés",
        "Alcorcon": "Alcorcón",
        "Castellon": "Castellón",
        "Eldense": "Eldense",
        "Ferrol": "Racing Ferrol",
        "Sociedad B": "Real Sociedad B",
        "Villarreal B": "Villarreal B",
        "Ibiza": "Ibiza",
        "Amorebieta": "Amorebieta",
        "Andorra": "FC Andorra",
        "Cartagena": "FC Cartagena",
        "Yeclano": "Yeclano",
    }

    for raw in sorted(raw_names):
        # Check manual override first
        if raw in manual_overrides:
            proposals[raw] = manual_overrides[raw]
            continue

        # Exact match
        if raw in canonical_names:
            proposals[raw] = raw
            continue

        # Fuzzy match
        result = fuzz_process.extractOne(
            raw, canonical_names, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        if result:
            proposals[raw] = result[0]
        else:
            proposals[raw] = None  # Unmapped — needs manual handling

    return proposals


def update_alias_file(proposals: OrderedDict, existing_file: str = ALIAS_FILE):
    """Update the YAML alias file with proposed mappings, preserving
    existing manual entries and adding new proposals.
    """
    # Load existing file
    existing = {}
    if os.path.exists(existing_file):
        with open(existing_file, "r", encoding="utf-8") as f:
            content = f.read()
            parsed = yaml.safe_load(content) or {}
            existing = parsed.get("football_data_co_uk", {}) if isinstance(parsed, dict) else {}

    # Merge: keep existing entries, add new proposals for unmapped names
    merged = dict(existing)
    for raw, canon in proposals.items():
        if raw not in merged and canon is not None:
            merged[raw] = canon

    # Write back with header
    header = (
        "# Team alias mappings: football-data.co.uk raw names → canonical team names.\n"
        "# Built by Phase 1.2 fuzzy matching + manual review.\n"
        "# Every name across all ingested CSVs must be mapped here.\n"
        "# Unmapped names cause ingestion to fail loudly.\n\n"
        "football_data_co_uk:\n"
    )

    lines = [header]
    for raw, canon in sorted(merged.items(), key=lambda x: x[0].lower()):
        if canon is None:
            # Keep but commented as TODO
            lines.append(f"  # TODO: \"{raw}\" → ??? — review needed\n")
        else:
            lines.append(f'  "{raw}": "{canon}"\n')

    with open(existing_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n✅ Wrote {len(merged)} mappings to {existing_file}")


def main():
    parser = argparse.ArgumentParser(description="Resolve football-data.co.uk team names")
    parser.add_argument("--seasons", type=int, default=10, help="Number of seasons to scan (default 10)")
    args = parser.parse_args()

    codes = season_codes(args.seasons)
    print(f"Scanning {len(codes)} seasons ({codes[0]}–{codes[-1]}) for divisions {DIVISIONS}")

    all_names: set[str] = set()
    for code in codes:
        for div in DIVISIONS:
            rows = download_csv(code, div)
            if rows:
                names = extract_team_names(rows)
                all_names.update(names)
                print(f"    -> {len(names)} distinct names so far")

    print(f"\nTotal distinct team names found: {len(all_names)}")
    for n in sorted(all_names):
        print(f"  '{n}'")

    # Get canonical teams
    canonical = get_canonical_teams()
    print(f"\nCanonical teams in DB: {len(canonical)}")
    for name in canonical:
        print(f"  {name}")

    # Propose mappings
    proposals = propose_mappings(all_names, canonical)
    unmapped = {k: v for k, v in proposals.items() if v is None}

    print(f"\n=== Proposed Mappings ===")
    for raw, canon in proposals.items():
        if canon:
            print(f"  '{raw}' → '{canon}'")
        else:
            print(f"  ⚠️  '{raw}' → ??? UNMAPPED")

    if unmapped:
        print(f"\n⚠️  {len(unmapped)} names could not be mapped automatically:")
        for n in unmapped:
            print(f"  - '{n}'")

    # Write to alias file
    update_alias_file(proposals)

    print("\n" + "=" * 60)
    print("Phase 1.2 complete. The alias file has been generated.")
    print("PLEASE REVIEW AND CORRECT THE MAPPINGS BEFORE CONTINUING.")
    print("=" * 60)


if __name__ == "__main__":
    main()