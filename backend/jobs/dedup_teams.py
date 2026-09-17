#!/usr/bin/env python3
"""Merge duplicate teams created by two overlapping bootstrap passes.

The original Spanish bootstrap created properly accented club names (e.g.
'Alcorcón'), while the football-data.co.uk bootstrap auto-mapped raw CSV names
to themselves, creating ASCII twins (e.g. 'Alcorcon'). The CSV twins ended up
holding all the matches.

This job fuses each duplicate group into ONE team:
  * the member holding the most matches survives,
  * the survivor is renamed to the preferred (accented) canonical name,
  * matches and aliases are re-pointed, colliding fixtures/aliases are dropped.

Usage:
    python -m jobs.dedup_teams            # dry-run report (default)
    python -m jobs.dedup_teams --apply    # actually merge
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import or_, text

from app.database import SessionLocal
from app.models import Match, Team, TeamAlias

OUT = Path(__file__).resolve().parent.parent / "dedup_report.txt"

report = []


def say(msg=""):
    report.append(str(msg))
    print(msg)

# preferred canonical name -> every current name for that same club
MERGE_GROUPS = {
    "Atlético Madrid": ["Atletico Madrid", "Atlético Madrid"],
    "Leganés": ["Leganes", "Leganés"],
    "Alcorcón": ["Alcorcon", "Alcorcón"],
    "Almería": ["Almeria", "Almería"],
    "Cádiz": ["Cadiz", "Cádiz"],
    "Castellón": ["Castellon", "Castellón"],
    "Deportivo Alavés": ["Deportivo Alaves", "Deportivo Alavés", "Alaves"],
    "Deportivo La Coruña": ["Deportivo La Coruña", "La Coruna"],
    "Gimnàstic Tarragona": ["Gimnastic Tarragona", "Gimnàstic Tarragona"],
    "Mirandés": ["Mirandes", "Mirandés"],
    "Málaga": ["Malaga", "Málaga"],
    "Sporting Gijón": ["Sporting Gijon", "Sporting Gijón"],
    "UD Logroñés": ["Logrones", "UD Logroñés"],
    "Real Valladolid": ["Real Valladolid", "Valladolid"],
    "FC Andorra": ["FC Andorra", "Andorra"],
    "FC Cartagena": ["FC Cartagena", "Cartagena"],
    "Lorca FC": ["Lorca FC", "Lorca"],
    "Espanyol": ["Espanyol", "Espanol"],
    "Athletic Club": ["Athletic Club", "Ath Bilbao"],
}

def match_counts(db):
    rows = db.execute(text(
        "SELECT t.id, "
        "(SELECT COUNT(*) FROM matches m "
        " WHERE m.home_team_id = t.id OR m.away_team_id = t.id) AS nm "
        "FROM teams t"
    )).fetchall()
    return {r.id: r.nm for r in rows}


def merge_matches(db, loser_id, winner_id):
    """Re-point the loser's fixtures at the winner; drop anything that collides."""
    win_ms = db.query(Match).filter(
        or_(Match.home_team_id == winner_id, Match.away_team_id == winner_id)
    ).all()
    keys = {(m.season_id, m.home_team_id, m.away_team_id): m for m in win_ms}

    lose_ms = db.query(Match).filter(
        or_(Match.home_team_id == loser_id, Match.away_team_id == loser_id)
    ).all()

    moved = dropped = 0
    for m in lose_ms:
        h = winner_id if m.home_team_id == loser_id else m.home_team_id
        a = winner_id if m.away_team_id == loser_id else m.away_team_id
        if h == a:                      # would become self-play -> discard
            db.delete(m)
            dropped += 1
            continue
        key = (m.season_id, h, a)
        twin = keys.get(key)
        if twin is not None:            # same fixture already on the survivor
            # keep whichever copy actually has a score
            if twin.home_goals is None and m.home_goals is not None:
                twin.home_goals = m.home_goals
                twin.away_goals = m.away_goals
                twin.status = m.status
            db.delete(m)
            dropped += 1
            continue
        m.home_team_id, m.away_team_id = h, a
        keys[key] = m
        moved += 1
    return moved, dropped


# tables (other than matches/team_aliases) that reference teams(id):
#   (table, fk column, unique-key columns excluding the fk column)
SIMPLE_REFS = [
    ("match_events", "team_id",
     ("match_id", "minute", "event_type", "player_id", "detail")),
    ("player_season_stats", "team_id",
     ("player_id", "season_id", "snapshot_date")),
    ("standings_snapshots", "team_id",
     ("season_id", "computed_at")),
]


def repoint_simple(db, table, fk, uniq, loser_id, winner_id):
    """Re-point loser_id -> winner_id in table.fk.

    Rows that would violate the table's unique key against an existing winner
    row are deleted instead (NULL-safe comparison via IS NOT DISTINCT FROM).
    """
    cols = ", ".join(uniq)
    rows = db.execute(text(
        f"SELECT id, {cols} FROM {table} WHERE {fk} = :l"), {"l": loser_id}).fetchall()
    moved = dropped = 0
    for r in rows:
        rid = r[0]
        conds = " AND ".join(
            f"{c} IS NOT DISTINCT FROM :v{i}" for i, c in enumerate(uniq))
        params = {f"v{i}": r[i + 1] for i in range(len(uniq))}
        params["w"] = winner_id
        clash = db.execute(text(
            f"SELECT 1 FROM {table} WHERE {fk} = :w AND {conds}"), params).first()
        if clash:
            db.execute(text(f"DELETE FROM {table} WHERE id = :i"), {"i": rid})
            dropped += 1
        else:
            db.execute(text(f"UPDATE {table} SET {fk} = :w WHERE id = :i"),
                       {"w": winner_id, "i": rid})
            moved += 1
    return moved, dropped


def remaining_refs(db, loser_id):
    """Any rows still pointing at loser_id? (safety net before DELETE)."""
    left = {}
    for table, fk, _uniq in SIMPLE_REFS:
        n = db.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {fk} = :l"),
                       {"l": loser_id}).scalar()
        if n:
            left[f"{table}.{fk}"] = n
    n = db.execute(text(
        "SELECT COUNT(*) FROM matches WHERE home_team_id = :l OR away_team_id = :l"),
        {"l": loser_id}).scalar()
    if n:
        left["matches"] = n
    n = db.execute(text(
        "SELECT COUNT(*) FROM team_aliases WHERE team_id = :l"),
        {"l": loser_id}).scalar()
    if n:
        left["team_aliases"] = n
    return left


def merge_aliases(db, loser_id, winner_id):
    """Re-point the loser's alias rows at the winner; drop duplicates."""
    existing = {
        (a.source, a.raw_name)
        for a in db.query(TeamAlias).filter(TeamAlias.team_id == winner_id).all()
    }
    rows = db.query(TeamAlias).filter(TeamAlias.team_id == loser_id).all()
    moved = dropped = 0
    for a in rows:
        key = (a.source, a.raw_name)
        if key in existing:
            db.delete(a)
            dropped += 1
            continue
        a.team_id = winner_id
        existing.add(key)
        moved += 1
    return moved, dropped


def main():
    apply = "--apply" in sys.argv
    say("=" * 66)
    say(f"  Team de-duplication  ({'APPLY' if apply else 'DRY RUN'})")
    say("=" * 66)

    db = SessionLocal()
    try:
        counts = match_counts(db)
        by_name = {t.canonical_name: t for t in db.query(Team).all()}
        say(f"\nStarting teams: {len(by_name)}   teams with matches: "
            f"{sum(1 for v in counts.values() if v)}")

        renamed = {}        # old name -> new name (for YAML rewrite)
        merged = deleted = moved_m = drop_m = moved_a = drop_a = 0
        skipped = []

        for target, members in MERGE_GROUPS.items():
            present = [n for n in members if n in by_name]
            if len(present) < 2:
                skipped.append(f"{target}: only {present} present")
                continue

            cand = [by_name[n] for n in present]
            # survivor: most matches, then the one already named `target`, then lowest id
            cand.sort(key=lambda t: (-counts.get(t.id, 0),
                                     t.canonical_name != target,
                                     t.id))
            winner = cand[0]
            losers = cand[1:]

            say(f"\n[{target}]")
            for t in cand:
                tag = "KEEP" if t is winner else "MERGE"
                say(f"   {tag}  id={t.id:>4} matches={counts.get(t.id, 0):>5} "
                    f"{t.canonical_name!r}")

            for loser in losers:
                if not apply:
                    say(f"   (dry) would move {counts.get(loser.id, 0)} matches "
                        f"from {loser.canonical_name!r} -> {winner.canonical_name!r}")
                    continue
                m_moved, m_drop = merge_matches(db, loser.id, winner.id)
                a_moved, a_drop = merge_aliases(db, loser.id, winner.id)
                extra = []
                for table, fk, uniq in SIMPLE_REFS:
                    mv, dr = repoint_simple(db, table, fk, uniq, loser.id, winner.id)
                    if mv or dr:
                        extra.append(f"{table} +{mv}/-{dr}")
                        moved_m += mv
                        drop_m += dr
                if loser.canonical_name not in renamed:
                    renamed[loser.canonical_name] = target

                db.flush()          # ensure re-pointed rows are visible to the check
                left = remaining_refs(db, loser.id)
                if left:
                    say(f"   CANNOT delete {loser.canonical_name!r}: still referenced "
                        f"by {left}")
                    continue

                db.delete(loser)
                db.flush()
                merged += 1
                moved_a += a_moved
                drop_a += a_drop
                say(f"   merged {loser.canonical_name!r}: "
                    f"matches +{m_moved}/-{m_drop}, aliases +{a_moved}/-{a_drop}"
                    + (f", {', '.join(extra)}" if extra else ""))

            if apply and winner.canonical_name != target:
                occupied = [
                    t.canonical_name for t in db.query(Team).all()
                    if t.canonical_name == target and t.id != winner.id
                ]
                if occupied:
                    say(f"   SKIP rename: target {target!r} still occupied")
                else:
                    old = winner.canonical_name
                    winner.canonical_name = target
                    renamed[old] = target
                    say(f"   renamed {old!r} -> {target!r}")

        if apply:
            db.commit()
        else:
            db.rollback()

        say("\n" + "=" * 66)
        if apply:
            say(f"  Merged {merged} team(s); matches moved={moved_m} dropped={drop_m}; "
                f"aliases moved={moved_a} dropped={drop_a}")
        else:
            say("  Dry run - nothing changed. Re-run with --apply to merge.")
        if skipped:
            say(f"  Skipped groups: {skipped}")
        say("=" * 66)

        if renamed:
            say("\nRENAMED (old -> new)  [use to rewrite the alias YAML]")
            for o, n in sorted(renamed.items()):
                say(f"  {o!r} -> {n!r}")
    finally:
        db.close()

    OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport written to {OUT}")


if __name__ == "__main__":
    main()

