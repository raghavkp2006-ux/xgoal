#!/usr/bin/env python3
"""Reconcile the matches table against the source CSVs.

Every ingested fixture must correspond to a real row in its (division, season)
CSV. Rows that no longer map to a CSV row are stale leftovers (for example
fixtures written before an alias mapping was corrected) and are deleted.

Usage:
    python -m jobs.reconcile_matches            # dry-run report
    python -m jobs.reconcile_matches --apply    # delete stale rows
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text

from app.database import SessionLocal
from ingest_direct import (DIVS, fetch_cached, load_aliases, parse_csv,
                           season_codes)

OUT = Path(__file__).resolve().parent.parent / "reconcile_report.txt"
report = []


def say(m=""):
    report.append(str(m))
    print(m)


def main():
    apply = "--apply" in sys.argv
    say("=" * 66)
    say(f"  Match reconciliation against source CSVs "
        f"({'APPLY' if apply else 'DRY RUN'})")
    say("=" * 66)

    aliases = load_aliases()
    db = SessionLocal()
    try:
        rows = db.execute(text("SELECT id, canonical_name FROM teams")).fetchall()
        teams = {r.canonical_name: r.id for r in rows}

        seasons = db.execute(text("""
            SELECT s.id, c.code, s.label
            FROM seasons s JOIN competitions c ON c.id = s.competition_id
        """)).fetchall()
        season_index = defaultdict(dict)      # code -> label -> season_id
        for s in seasons:
            season_index[s.code][s.label] = s.id

        total_stale = 0
        for div in DIVS:
            for sc in season_codes(10):
                sid = season_index.get(div, {}).get(
                    f"{int(sc[:2]) + 2000}/{int(sc[:2]) + 2001 - 2000}")
                if sid is None:
                    continue
                content = fetch_cached(sc, div)
                if not content:
                    continue

                # (home, away) canonical pairs the CSV legitimately contains
                csv_pairs = set()
                for r in parse_csv(content):
                    rh = (r.get("HomeTeam") or "").strip()
                    ra = (r.get("AwayTeam") or "").strip()
                    ch, ca = aliases.get(rh), aliases.get(ra)
                    if ch and ca and ch in teams and ca in teams:
                        csv_pairs.add((teams[ch], teams[ca]))

                db_rows = db.execute(text("""
                    SELECT id, home_team_id, away_team_id
                    FROM matches WHERE season_id = :sid
                """), {"sid": sid}).fetchall()

                stale = [m for m in db_rows
                         if (m.home_team_id, m.away_team_id) not in csv_pairs]
                if stale:
                    total_stale += len(stale)
                    say(f"\n{div} {sc}: db={len(db_rows)} csv={len(csv_pairs)} "
                        f"stale={len(stale)}")
                    ids = [m.id for m in stale[:6]]
                    names = db.execute(text("""
                        SELECT m.id, th.canonical_name, ta.canonical_name
                        FROM matches m
                        JOIN teams th ON th.id = m.home_team_id
                        JOIN teams ta ON ta.id = m.away_team_id
                        WHERE m.id = ANY(:ids)
                    """), {"ids": ids}).fetchall()
                    for n in names:
                        say(f"   stale #{n[0]}: {n[1]} vs {n[2]}")
                    if len(stale) > 6:
                        say(f"   ... and {len(stale) - 6} more")
                    if apply:
                        db.execute(text("DELETE FROM matches WHERE id = ANY(:ids)"),
                                   {"ids": [m.id for m in stale]})

        if apply:
            db.commit()

        say("\n" + "=" * 66)
        if total_stale == 0:
            say("  No stale fixtures - matches table matches the source CSVs")
        elif apply:
            say(f"  Deleted {total_stale} stale fixture(s)")
        else:
            say(f"  {total_stale} stale fixture(s) found. "
                f"Re-run with --apply to delete.")
        say("=" * 66)
    finally:
        db.close()

    OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport written to {OUT}")


if __name__ == "__main__":
    main()
