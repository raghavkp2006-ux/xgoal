#!/usr/bin/env python3
"""Consolidate the duplicate La Liga competition rows.

Two bootstrap paths created the same competition twice:
  * `scripts/seed_la_liga.py` (API-Football league 140) -> code 'LA_LIGA'
  * the football-data.co.uk CSV bootstrap               -> code 'SP1'

'LA_LIGA' holds no matches; 'SP1' holds the ingested fixtures. This job folds
them into one row: the API-Football league id moves onto the surviving
competition, reference standings snapshots are re-pointed, and the empty
duplicate (competition + seasons) is removed.

Usage:
    python -m jobs.consolidate_competitions            # dry-run
    python -m jobs.consolidate_competitions --apply
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from app.database import SessionLocal

OUT = Path(__file__).resolve().parent.parent / "consolidate_report.txt"
report = []


def say(m=""):
    report.append(str(m))
    print(m)


def season_refs(db):
    """(table, column) pairs whose foreign key targets seasons(id)."""
    return db.execute(text("""
        SELECT DISTINCT tc.table_name AS t, kcu.column_name AS c
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.constraint_schema = tc.constraint_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.constraint_schema = tc.constraint_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND ccu.table_name = 'seasons'
        ORDER BY 1, 2
    """)).fetchall()


def unique_keys(db, table):
    """Unique/primary key column tuples declared on `table`."""
    rows = db.execute(text("""
        SELECT tc.constraint_name AS n, kcu.column_name AS c
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.constraint_schema = tc.constraint_schema
        WHERE tc.table_name = :t
          AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
    """), {"t": table}).fetchall()
    keys = {}
    for r in rows:
        keys.setdefault(r.n, []).append(r.c)
    return list(keys.values())


def repoint_season(db, table, col, from_sid, to_sid, pk="id"):
    """Move rows of `table` from one season to another, dropping collisions."""
    keys = unique_keys(db, table)
    rows = db.execute(text(f"SELECT * FROM {table} WHERE {col} = :s"),
                      {"s": from_sid}).mappings().all()
    moved = dropped = 0
    for row in rows:
        clash = False
        for k in keys:
            if col not in k or len(k) < 2:
                continue
            others = [c for c in k if c != col]
            conds = " AND ".join(
                f"{c} IS NOT DISTINCT FROM :v{i}" for i, c in enumerate(others))
            params = {f"v{i}": row[c] for i, c in enumerate(others)}
            params["s"] = to_sid
            if db.execute(text(
                    f"SELECT 1 FROM {table} WHERE {col} = :s AND {conds}"),
                    params).first():
                clash = True
                break
        if clash:
            db.execute(text(f"DELETE FROM {table} WHERE {pk} = :i"),
                       {"i": row[pk]})
            dropped += 1
        else:
            db.execute(text(f"UPDATE {table} SET {col} = :t WHERE {pk} = :i"),
                       {"t": to_sid, "i": row[pk]})
            moved += 1
    return moved, dropped


def main():
    apply = "--apply" in sys.argv
    say("=" * 66)
    say(f"  Competition consolidation ({'APPLY' if apply else 'DRY RUN'})")
    say("=" * 66)

    db = SessionLocal()
    try:
        comps = db.execute(text(
            "SELECT id, code, name, country, tier, api_football_id, "
            "(SELECT COUNT(*) FROM matches m WHERE m.competition_id = c.id) ms "
            "FROM competitions c ORDER BY id")).fetchall()

        groups = {}
        for c in comps:
            groups.setdefault((c.name, c.country, c.tier), []).append(c)

        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        if not dupes:
            say("\nNo duplicate competitions found.")
            return

        for key, members in sorted(dupes.items()):
            say(f"\n[{key[0]!r} / {key[1]}]")
            for m in members:
                say(f"   id={m.id:>3} code={m.code!r:<10} api={m.api_football_id} "
                    f"matches={m.ms}")
            keeper = max(members, key=lambda m: (m.ms, m.api_football_id is not None))
            drop = [m for m in members if m.id != keeper.id]
            say(f"   KEEP   id={keeper.id} code={keeper.code!r} ({keeper.ms} matches)")
            for d in drop:
                say(f"   REMOVE id={d.id} code={d.code!r} ({d.ms} matches)")

            if not apply:
                continue

            for d in drop:
                if d.ms:
                    say(f"   SKIP {d.code!r}: still holds {d.ms} matches")
                    continue
                for s in db.execute(text(
                        "SELECT id, start_year FROM seasons "
                        "WHERE competition_id = :c"), {"c": d.id}).fetchall():
                    target = db.execute(text(
                        "SELECT id FROM seasons WHERE competition_id = :c "
                        "AND start_year = :y"),
                        {"c": keeper.id, "y": s.start_year}).first()

                    # move every dependent row onto the matching keeper season
                    for ref_t, ref_c in season_refs(db):
                        n = db.execute(text(
                            f"SELECT COUNT(*) FROM {ref_t} WHERE {ref_c} = :s"),
                            {"s": s.id}).scalar()
                        if not n:
                            continue
                        if target is None:
                            say(f"   WARN {ref_t}.{ref_c}: {n} row(s) for "
                                f"{s.start_year} but no {keeper.code} season")
                            continue
                        m, drop_n = repoint_season(db, ref_t, ref_c, s.id, target.id)
                        say(f"   {ref_t}.{ref_c} {s.start_year}: +{m}/-{drop_n}")

                left = db.execute(text("""
                    SELECT COUNT(*) FROM standings_snapshots ss
                    JOIN seasons s ON s.id = ss.season_id
                    WHERE s.competition_id = :c
                """), {"c": d.id}).scalar()
                stuck = []
                for ref_t, ref_c in season_refs(db):
                    n = db.execute(text(
                        f"SELECT COUNT(*) FROM {ref_t} r JOIN seasons s "
                        f"ON s.id = r.{ref_c} WHERE s.competition_id = :c"),
                        {"c": d.id}).scalar()
                    if n:
                        stuck.append(f"{ref_t}.{ref_c}={n}")
                if left or stuck:
                    say(f"   SKIP {d.code!r}: still referenced ({left} snapshots, "
                        f"{stuck})")
                    continue
                db.execute(text("DELETE FROM seasons WHERE competition_id = :c"),
                           {"c": d.id})
                db.execute(text("DELETE FROM competitions WHERE id = :i"), {"i": d.id})
                say(f"   removed competition {d.id} ({d.code!r}) and its seasons")

            # carry the API-Football identity over once the duplicate is gone
            if keeper.api_football_id is None:
                api = next((d.api_football_id for d in drop
                            if d.api_football_id is not None), None)
                if api is not None:
                    db.execute(text("UPDATE competitions SET api_football_id = :a "
                                    "WHERE id = :i"), {"a": api, "i": keeper.id})
                    say(f"   api_football_id {api} -> competition {keeper.id} "
                        f"({keeper.code!r})")

        if apply:
            db.commit()
            say("\n" + "=" * 66)
            say("  Consolidation applied")
            for c in db.execute(text(
                    "SELECT id, code, api_football_id, "
                    "(SELECT COUNT(*) FROM seasons s WHERE s.competition_id = c.id) seas, "
                    "(SELECT COUNT(*) FROM matches m WHERE m.competition_id = c.id) ms "
                    "FROM competitions c ORDER BY id")).fetchall():
                say(f"   id={c.id:>3} {c.code!r:<10} api={c.api_football_id} "
                    f"seasons={c.seas} matches={c.ms}")
            say("=" * 66)
        else:
            db.rollback()
            say("\nDRY RUN - re-run with --apply to consolidate")
    finally:
        db.close()

    OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport written to {OUT}")


if __name__ == "__main__":
    main()
