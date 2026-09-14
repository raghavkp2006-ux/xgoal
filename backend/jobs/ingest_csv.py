#!/usr/bin/env python3
"""
Phase 1.3 — Bulk historical match ingestion from football-data.co.uk CSVs.

Downloads, parses, and upserts matches across all configured divisions and
seasons. Handles every CSV gotcha in the spec.

Usage:
    python -m jobs.ingest_csv [--season 2425] [--div SP1]
    python -m jobs.ingest_csv                   # full run
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
from collections import OrderedDict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal, engine
from app.models import Competition, Match, Season, Team, TeamAlias
from sqlalchemy import text

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
ALIAS_FILE = BASE_DIR.parent / "db" / "aliases" / "football_data_co_uk.yaml"

DIVISIONS_CONFIG = BASE_DIR / "data" / "football_data_divisions.py"

# season codes: SSSS = two-digit start-year + two-digit end-year
def season_codes(n: int = 10):
    """Return the last N season codes, ending at 2425 (2024-25)."""
    codes = []
    for i in range(n):
        start = 25 - 1 - i
        codes.append(f"{(start) % 100:02d}{(start + 1) % 100:02d}")
    return codes


# ── CSV parsing helpers ───────────────────────────────────────────────────

def load_aliases() -> dict[str, str]:
    """Load {raw_name: canonical_name} from YAML."""
    with open(ALIAS_FILE, "r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f)
    return parsed.get("football_data_co_uk", {})


def parse_date_cell(val: str) -> Optional[date]:
    """Try multiple date formats that football-data.co.uk uses."""
    if not val or not val.strip():
        return None
    val = val.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(val, fmt).date()
        except ValueError:
            continue
    return None


def parse_time_cell(val: str) -> Optional[str]:
    """Return HH:MM string or None."""
    if not val or not val.strip():
        return None
    val = val.strip()
    # Some have format "15:00:00" — truncate
    if val.count(":") == 2:
        val = val.rsplit(":", 1)[0]
    try:
        datetime.strptime(val, "%H:%M")
        return val
    except ValueError:
        return None


def parse_int(val) -> Optional[int]:
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def parse_float(val) -> Optional[float]:
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


DE_VIG_KEYS = [
    ("B365CH", "B365CD", "B365CA"),
    ("BSCH", "BSCD", "BSCA"),
    ("BWCH", "BWCD", "BWCA"),
    ("IWH", "IWD", "IWA"),
    ("PSCH", "PSCD", "PSCA"),
    ("WHH", "WHD", "WHA"),
    ("VCH", "VCD", "VCA"),
    ("MaxH", "MaxD", "MaxA"),
    ("AvgH", "AvgD", "AvgA"),
]


def de_vig(odds_h: float, odds_d: float, odds_a: float) -> tuple[float, float, float]:
    """Simple de-vig: normalise inverse odds."""
    inv_h = 1.0 / odds_h if odds_h else 0.0
    inv_d = 1.0 / odds_d if odds_d else 0.0
    inv_a = 1.0 / odds_a if odds_a else 0.0
    total = inv_h + inv_d + inv_a
    if total == 0:
        return (0.0, 0.0, 0.0)
    return (inv_h / total, inv_d / total, inv_a / total)


def extract_closing_probs(row: dict) -> dict:
    """Extract the best available de-vigged closing odds."""
    best = None
    best_max = 0.0
    for hk, dk, ak in DE_VIG_KEYS:
        h = parse_float(row.get(hk))
        d = parse_float(row.get(dk))
        a = parse_float(row.get(ak))
        if h and d and a and h > 0 and d > 0 and a > 0:
            total = h + d + a
            if total > best_max:
                best = (h, d, a)
                best_max = total
    if best:
        ph, pd, pa = de_vig(*best)
        return {
            "closing_p_home": round(Decimal(str(ph)), 5),
            "closing_p_draw": round(Decimal(str(pd)), 5),
            "closing_p_away": round(Decimal(str(pa)), 5),
        }
    return {}


# ── Competition upsert ───────────────────────────────────────────────────

def upsert_competitions(db) -> dict[str, Competition]:
    """Insert/update competition rows from divisions config, return {code: comp}."""
    # Import divisions dynamically
    divs = []
    exec(open(str(DIVISIONS_CONFIG), encoding="utf-8").read(), globals := {})
    # Use hardcoded list instead
    div_data = [
        ("SP1", "La Liga", "Spain", 1),
        ("SP2", "Segunda División", "Spain", 2),
        ("E0", "Premier League", "England", 1),
        ("E1", "Championship", "England", 2),
        ("E2", "League One", "England", 3),
        ("E3", "League Two", "England", 4),
        ("D1", "Bundesliga", "Germany", 1),
        ("D2", "2. Bundesliga", "Germany", 2),
        ("I1", "Serie A", "Italy", 1),
        ("I2", "Serie B", "Italy", 2),
        ("F1", "Ligue 1", "France", 1),
        ("F2", "Ligue 2", "France", 2),
        ("N1", "Eredivisie", "Netherlands", 1),
        ("P1", "Primeira Liga", "Portugal", 1),
        ("T1", "Süper Lig", "Turkey", 1),
        ("B1", "Jupiler Pro League", "Belgium", 1),
        ("G1", "Super League", "Greece", 1),
        ("SC0", "Scottish Premiership", "Scotland", 1),
        ("SC1", "Scottish Championship", "Scotland", 2),
    ]
    result = {}
    for code, name, country, tier in div_data:
        comp = db.query(Competition).filter(Competition.code == code).first()
        if not comp:
            comp = Competition(code=code, name=name, country=country, tier=tier)
            db.add(comp)
            db.flush()
        result[code] = comp
    db.commit()
    return result


def upsert_seasons(db, comp_id: int, code: str, years: list[int]) -> dict[int, Season]:
    """Ensure season rows exist for given years. Return {start_year: Season}."""
    result = {}
    for yr in years:
        existing = db.query(Season).filter(
            Season.competition_id == comp_id, Season.start_year == yr
        ).first()
        if not existing:
            label = f"{yr}/{yr + 1 - 2000}"
            existing = Season(
                competition_id=comp_id,
                label=label,
                start_year=yr,
                start_date=date(yr, 8, 1),
                end_date=date(yr + 1, 5, 31),
                is_current=(yr == 2026),
            )
            db.add(existing)
            db.flush()
        result[yr] = existing
    return result


# ── Team resolution ──────────────────────────────────────────────────────

class UnknownTeamError(Exception):
    """Raised when a CSV team name has no alias."""

    def __init__(self, raw_name: str):
        self.raw_name = raw_name
        super().__init__(f"No alias mapping for team '{raw_name}'")


def resolve_team(db, raw_name: str, aliases: dict[str, str]) -> Team:
    """Look up raw_name via alias -> canonical -> team row."""
    canonical = aliases.get(raw_name)
    if not canonical:
        raise UnknownTeamError(raw_name)
    team = db.query(Team).filter(Team.canonical_name == canonical).first()
    if not team:
        raise UnknownTeamError(f"{raw_name} -> {canonical} (not in DB)")
    return team


# ── Single-CSV ingestion ─────────────────────────────────────────────────

def download_csv(season_code: str, div: str) -> Optional[list[dict]]:
    """Download & return parsed rows, or None on failure."""
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{div}.csv"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
    except Exception:
        return None
    content = raw.decode("latin-1")
    if content.startswith("\ufeff"):
        content = content[1:]
    # Strip trailing empty rows
    lines = content.strip().split("\n")
    # Remove trailing blank/empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    content = "\n".join(lines)

    # Cache to disk
    (RAW_DIR / season_code).mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / season_code / f"{div}.csv"
    with open(cache_path, "wb") as f:
        f.write(content.encode("latin-1"))

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    return rows


def ingest_csv(
    db,
    season_code: str,
    div: str,
    comp: Competition,
    season: Season,
    aliases: dict[str, str],
) -> dict:
    """Ingest rows from one CSV. Returns counts dict."""
    rows = download_csv(season_code, div)
    if not rows:
        return {"downloaded": False, "inserted": 0, "updated": 0, "skipped": 0, "errors": []}

    has_time_col = "Time" in rows[0]
    inserted = updated = skipped = 0
    errors = []
    matched_home = set()

    for row in rows:
        # Skip blank rows where Div/Date/HomeTeam are all missing
        if not row.get("Div") and not row.get("Date") and not row.get("HomeTeam"):
            continue

        raw_home = (row.get("HomeTeam") or "").strip()
        raw_away = (row.get("AwayTeam") or "").strip()
        if not raw_home or not raw_away:
            skipped += 1
            continue

        # Resolve teams — raises UnknownTeamError for unmapped
        try:
            home_team = resolve_team(db, raw_home, aliases)
            away_team = resolve_team(db, raw_away, aliases)
        except UnknownTeamError as e:
            errors.append(str(e))
            skipped += 1
            continue

        # Date parsing
        raw_date = (row.get("Date") or "").strip()
        match_date = parse_date_cell(raw_date)
        if not match_date:
            skipped += 1
            continue

        # Time
        match_time = None
        if has_time_col:
            match_time = parse_time_cell(row.get("Time") or "")

        # Kickoff
        if match_time:
            try:
                h, m = match_time.split(":")
                kickoff = datetime(
                    match_date.year, match_date.month, match_date.day,
                    int(h), int(m), tzinfo=timezone.utc,
                )
            except ValueError:
                kickoff = datetime(
                    match_date.year, match_date.month, match_date.day, 20, 0,
                    tzinfo=timezone.utc,
                )
        else:
            kickoff = datetime(
                match_date.year, match_date.month, match_date.day, 20, 0,
                tzinfo=timezone.utc,
            )

        # Scores
        fthg = parse_int(row.get("FTHG"))
        ftag = parse_int(row.get("FTAG"))
        hthg = parse_int(row.get("HTHG"))
        htag = parse_int(row.get("HTAG"))

        # Status
        if fthg is not None and ftag is not None:
            status = "FT"
        else:
            status = "NS"

        # Other stats
        hs = parse_int(row.get("HS"))
        as_ = parse_int(row.get("AS"))
        hst = parse_int(row.get("HST"))
        ast = parse_int(row.get("AST"))
        hc = parse_int(row.get("HC"))
        ac = parse_int(row.get("AC"))
        hf = parse_int(row.get("HF"))
        af = parse_int(row.get("AF"))
        hy = parse_int(row.get("HY"))
        ay = parse_int(row.get("AY"))
        hr = parse_int(row.get("HR"))
        ar = parse_int(row.get("AR"))

        # Referee
        referee = (row.get("Referee") or "").strip() or None

        # Closing odds → de-vigged probabilities
        closing = extract_closing_probs(row)

        matchday = None

        # Upsert
        existing = db.query(Match).filter(
            Match.season_id == season.id,
            Match.home_team_id == home_team.id,
            Match.away_team_id == away_team.id,
        ).first()

        match_data = {
            "season_id": season.id,
            "competition_id": comp.id,
            "matchday": matchday,
            "kickoff_utc": kickoff,
            "home_team_id": home_team.id,
            "away_team_id": away_team.id,
            "status": status,
            "home_goals": fthg,
            "away_goals": ftag,
            "home_goals_ht": hthg,
            "away_goals_ht": htag,
            "home_shots": hs,
            "away_shots": as_,
            "home_shots_on_tgt": hst,
            "away_shots_on_tgt": ast,
            "home_corners": hc,
            "away_corners": ac,
            "home_fouls": hf,
            "away_fouls": af,
            "home_yellows": hy,
            "away_yellows": ay,
            "home_reds": hr,
            "away_reds": ar,
            "referee": referee,
            "source": "football-data.co.uk",
            "external_id": None,
        }
        match_data.update(closing)

        if existing:
            # Update in place
            for k, v in match_data.items():
                setattr(existing, k, v)
            existing.updated_at = datetime.now(timezone.utc)
            updated += 1
        else:
            match_data["source"] = "football-data.co.uk"
            match_data["ingested_at"] = datetime.now(timezone.utc)
            match_data["updated_at"] = datetime.now(timezone.utc)
            m = Match(**match_data)
            db.add(m)
            inserted += 1

    db.commit()
    return {
        "downloaded": True,
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
    }


# ── Full run ────────────────────────────────────────────────────────────

def hash_table(db, table: str) -> str:
    """Compute sha256 of all rows in a table for idempotency checks."""
    rows = db.execute(text(f"SELECT * FROM {table} ORDER BY id")).fetchall()
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r._mapping).encode("utf-8"))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Bulk CSV ingestion from football-data.co.uk")
    parser.add_argument("--season", type=str, default=None, help="Single season code (e.g. 2425)")
    parser.add_argument("--div", type=str, default=None, help="Single division code (e.g. SP1)")
    args = parser.parse_args()

    aliases = load_aliases()
    print(f"Loaded {len(aliases)} team aliases")

    codes = [args.season] if args.season else season_codes()
    divs_to_run = [args.div] if args.div else [
        "SP1", "SP2", "E0", "E1", "E2", "E3",
        "D1", "D2", "I1", "I2", "F1", "F2",
        "N1", "P1", "T1", "B1", "G1", "SC0", "SC1",
    ]

    years = [int(c[:2]) + 2000 for c in codes]

    db = SessionLocal()
    try:
        # 1. Competitions
        comps = upsert_competitions(db)
        print(f"Upserted {len(comps)} competitions")

        # 2. Seasons
        season_map = {}
        for code, comp in comps.items():
            if args.div and code != args.div:
                continue
            seas = upsert_seasons(db, comp.id, code, years)
            season_map[code] = seas
        print(f"Seasons ensured")

        # 3. Ingest CSVs
        total_inserted = total_updated = 0
        errors_all = []
        for code in divs_to_run:
            comp = comps[code]
            for sc in codes:
                yr = int(sc[:2]) + 2000
                season = season_map[code].get(yr)
                if not season:
                    continue
                result = ingest_csv(db, sc, code, comp, season, aliases)
                if result["downloaded"]:
                    print(f"  {code} {sc}: +{result['inserted']} ~{result['updated']} "
                          f"skipped={result['skipped']} errors={len(result['errors'])}")
                    total_inserted += result["inserted"]
                    total_updated += result["updated"]
                    errors_all.extend(result["errors"])
                else:
                    print(f"  {code} {sc}: not available")

        print(f"\n=== CSV Ingestion Summary ===")
        print(f"  Total inserted: {total_inserted}")
        print(f"  Total updated:  {total_updated}")
        if errors_all:
            print(f"  Errors ({len(errors_all)}):")
            for e in set(errors_all):
                print(f"    - {e}")

        # Hash for idempotency baseline
        h = hash_table(db, "matches")
        print(f"  matches table hash: {h[:16]}...")

    finally:
        db.close()

    print("\nDone.")


if __name__ == "__main__":
    main()