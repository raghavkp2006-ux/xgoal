#!/usr/bin/env python3
"""Phase 2 prerequisite: backfill match statistics and closing odds.

Phase 1's bulk ingest wrote fixtures only (teams, kickoff, final score). The raw
football-data.co.uk CSVs cached in ``backend/data/csv_cache/`` carry much more,
and two groups matter for the prediction models:

* ``HS/AS/HST/AST`` — shots and shots-on-target (the xG proxy used by the
  point-in-time feature builder); present in every cached season.
* ``B365CH/B365CD/B365CA`` — Bet365 *closing* decimal odds (the de-vigged market
  baseline); present from 2019/20 onwards.

The job UPDATEs existing match rows in place (it never inserts), joining on
(season, home team, away team) after alias resolution, so it is idempotent and
safe to re-run. Closing odds are converted from decimal odds to implied
probabilities and renormalised (overround removed) because the DB columns
(``closing_p_*``) are probabilities, not prices.

Usage:
    python -m jobs.enrich_match_stats
    python -m jobs.enrich_match_stats --div SP1
    python -m jobs.enrich_match_stats --div SP1 --season 2425
"""
import argparse
import csv
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import yaml  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.models import Competition, DataFreshness, Match, Season, Team  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
ALIAS_FILE = PROJ.parent / "db" / "aliases" / "football_data_co_uk.yaml"
CACHE_DIR = PROJ / "data" / "csv_cache"

DIVS = ["SP1", "SP2", "E0", "E1", "E2", "E3", "D1", "D2", "I1", "I2",
        "F1", "F2", "N1", "P1", "T1", "B1", "G1", "SC0", "SC1"]

# DB column -> CSV column, for the integer match statistics.
STAT_COLUMNS = {
    "home_goals_ht": "HTHG",
    "away_goals_ht": "HTAG",
    "home_shots": "HS",
    "away_shots": "AS",
    "home_shots_on_tgt": "HST",
    "away_shots_on_tgt": "AST",
    "home_corners": "HC",
    "away_corners": "AC",
    "home_fouls": "HF",
    "away_fouls": "AF",
    "home_yellows": "HY",
    "away_yellows": "AY",
    "home_reds": "HR",
    "away_reds": "AR",
}

# Closing-odds triplets, in order of preference (decimal odds).
ODDS_TRIOS = (("B365CH", "B365CD", "B365CA"), ("PSCH", "PSCD", "PSCA"))


def load_aliases():
    """{raw_name: canonical_name} from the shared alias YAML."""
    with open(ALIAS_FILE, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("football_data_co_uk", {}) if isinstance(data, dict) else {}


def season_codes(n=10):
    """2015/16 .. 2024/25 expressed as football-data.co.uk season codes."""
    return [f"{(25 - 1 - i) % 100:02d}{(25 - i) % 100:02d}" for i in range(n)]


def read_csv_text(path):
    """Decode a cached CSV (UTF-8 for recent seasons, cp1252 for older ones)."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_int(value):
    """CSV cell -> int, or None when blank / unparseable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value):
    """CSV cell -> float, or None when blank / unparseable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def closing_odds_probs(row):
    """De-vigged (home, draw, away) probabilities from closing decimal odds.

    Returns ``None`` when no closing price is available for the fixture, which
    is the case for every season before 2019/20.
    """
    for home_col, draw_col, away_col in ODDS_TRIOS:
        odds = [parse_float(row.get(home_col)),
                parse_float(row.get(draw_col)),
                parse_float(row.get(away_col))]
        if not all(o is not None and o > 1.0 for o in odds):
            continue
        implied = [1.0 / o for o in odds if o is not None]
        total = sum(implied)
        if total <= 0.0:
            continue
        return [round(p / total, 4) for p in implied]
    return None


def enrich_season(db, div, code, aliases, teams):
    """Update stats/odds columns for one (division, season); returns a summary."""
    comp = db.query(Competition).filter(Competition.code == div).one_or_none()
    if comp is None:
        return None
    season = (db.query(Season)
              .filter(Season.competition_id == comp.id,
                      Season.start_year == int(code[:2]) + 2000)
              .one_or_none())
    if season is None:
        return None
    csv_file = CACHE_DIR / f"{code}_{div}.csv"
    if not csv_file.exists():
        return None

    rows = list(csv.DictReader(io.StringIO(read_csv_text(csv_file))))
    matches = {(m.home_team_id, m.away_team_id): m
               for m in db.query(Match).filter(Match.season_id == season.id).all()}

    stat_cells = 0
    odds_rows = 0
    unmatched = 0
    for row in rows:
        home_name = aliases.get((row.get("HomeTeam") or "").strip())
        away_name = aliases.get((row.get("AwayTeam") or "").strip())
        if not home_name or not away_name:
            continue
        home_team = teams.get(home_name)
        away_team = teams.get(away_name)
        if home_team is None or away_team is None:
            continue
        match = matches.get((home_team.id, away_team.id))
        if match is None:
            unmatched += 1
            continue

        for column, csv_column in STAT_COLUMNS.items():
            value = parse_int(row.get(csv_column))
            if value is not None and getattr(match, column) != value:
                setattr(match, column, value)
                stat_cells += 1

        probs = closing_odds_probs(row)
        if probs is not None:
            stored = [match.closing_p_home, match.closing_p_draw, match.closing_p_away]
            current = [round(float(v), 4) if v is not None else None for v in stored]
            if current != probs:
                match.closing_p_home = probs[0]
                match.closing_p_draw = probs[1]
                match.closing_p_away = probs[2]
                odds_rows += 1

    db.commit()
    return {
        "div": div,
        "season": code,
        "csv_rows": len(rows),
        "matched": len(rows) - unmatched,
        "stat_cells": stat_cells,
        "odds_rows": odds_rows,
    }


def record_freshness(db, source, rows):
    """Upsert a data_freshness row, mirroring app.ingestion's bookkeeping."""
    now = datetime.now(timezone.utc)
    row = db.query(DataFreshness).filter(DataFreshness.source == source).first()
    if row is None:
        row = DataFreshness(source=source)
        db.add(row)
    row.last_attempt_at = now
    row.last_success_at = now
    row.last_error = None
    row.rows_affected = rows
    db.commit()


def main():
    parser = argparse.ArgumentParser(description="Backfill match stats and closing odds.")
    parser.add_argument("--div", type=str, help="single division code, e.g. SP1")
    parser.add_argument("--season", type=str, help="single season code, e.g. 2425")
    args = parser.parse_args()

    divs = [args.div] if args.div else DIVS
    codes = [args.season] if args.season else season_codes(10)
    aliases = load_aliases()
    db = SessionLocal()
    print("=" * 72)
    print("  XGoal match statistics / closing-odds backfill")
    print("=" * 72)
    print(f"  divisions: {len(divs)}   seasons: {len(codes)}   aliases: {len(aliases)}",
          flush=True)
    teams = {t.canonical_name: t for t in db.query(Team).all()}

    touched = 0
    total_cells = 0
    total_odds = 0
    try:
        for div in divs:
            for code in codes:
                try:
                    result = enrich_season(db, div, code, aliases, teams)
                except Exception as exc:  # noqa: BLE001 - keep going across files
                    db.rollback()
                    print(f"  {div} {code}: FAILED {type(exc).__name__}: {str(exc)[:140]}",
                          flush=True)
                    continue
                if result is None:
                    continue
                touched += 1
                total_cells += result["stat_cells"]
                total_odds += result["odds_rows"]
                print(f"  {result['div']} {result['season']}: "
                      f"{result['matched']}/{result['csv_rows']} rows matched, "
                      f"{result['stat_cells']} stat cells, "
                      f"{result['odds_rows']} odds rows", flush=True)
        record_freshness(db, "match_stats_backfill", total_cells + total_odds)
    finally:
        db.close()

    print("-" * 72)
    print(f"  competition-seasons touched : {touched}")
    print(f"  stat cells written          : {total_cells}")
    print(f"  rows with closing odds      : {total_odds}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())