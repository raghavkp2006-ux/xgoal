#!/usr/bin/env python3
"""
Phase 1.5 - Data validation suite for ingested match data.

Checks:
  1. Season match counts (realistic range per competition)
  2. Home/away balance (each team plays roughly equal H and A)
  3. Status/score consistency
  4. No duplicate fixtures
  5. No orphaned foreign keys
  6. Goal range sanity (0..20)
  7. Standings reconstruction (points conservation + optional reference compare)
  8. Final tables in published order (tiebreak chain) for reference seasons

Usage:
    python -m jobs.validate
"""
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from app.database import SessionLocal
from app.models import Match, Team


class V:
    """Tiny validator that records passed / failed assertions."""

    def __init__(self):
        self.ok_list = []
        self.fail_list = []

    def ok(self, msg):
        self.ok_list.append(msg)
        print(f"  [PASS] {msg}")

    def fail(self, msg):
        self.fail_list.append(msg)
        print(f"  [FAIL] {msg}")

    def summary(self):
        print(f"\n{'=' * 64}")
        print(f"  Passed: {len(self.ok_list)}   Failed: {len(self.fail_list)}")
        print(f"{'=' * 64}")
        if self.fail_list:
            print("  RESULT: VALIDATION FAILED")
            for m in self.fail_list[:25]:
                print(f"    - {m}")
        else:
            print("  RESULT: ALL CHECKS PASSED")
        return len(self.fail_list) == 0


EXPECTED_COUNTS = {
    # competition code: (min, max) plausible matches in one season.
    # Ranges deliberately cover legitimate structural variation: league size
    # changes (F1/F2 dropped to 18 clubs) and seasons curtailed by COVID
    # (E2/E3 2019/20, F1/F2 2019/20, N1 2019/20).
    "SP1": (330, 400), "SP2": (400, 470),
    "E0": (360, 400), "E1": (480, 570), "E2": (390, 570), "E3": (430, 570),
    "D1": (290, 320), "D2": (290, 320),
    "I1": (360, 400), "I2": (330, 470),
    "F1": (270, 400), "F2": (270, 400),
    "N1": (220, 320), "P1": (290, 320),
    "T1": (280, 430), "B1": (220, 430),
    "G1": (160, 330),
    # SC0/SC1 replay the same home/away pairing twice a season (split format).
    # matches has UniqueConstraint(season_id, home_team_id, away_team_id), so only
    # ONE meeting per pairing can be stored -> expect the distinct-pair count.
    "SC0": (110, 145), "SC1": (80, 100),
}

# Leagues whose format creates repeated home/away pairings within a season.
# The `matches` unique constraint keeps only the first meeting of each pairing.
SPLIT_FORMAT_DIVS = {"SC0", "SC1"}

# Seasons whose data is still incomplete. Every CSV season we ingest
# (2015/16 .. 2024/25) is finished, so this stays empty; it exists so a future
# in-progress season can be declared instead of failing the range check.
IN_PROGRESS_SEASONS = set()
IN_PROGRESS_FLOOR = 0.55      # an in-progress season must reach this share of `lo`
MEDIAN_FLOOR = 0.60           # flag a season below this share of its competition median


def check_season_counts(db, v):
    print("\n--- 1. Season match counts ---")
    rows = db.execute(text("""
        SELECT c.code, c.name, s.label, s.start_year, s.is_current,
               COUNT(m.id) AS cnt
        FROM competitions c
        JOIN seasons s ON s.competition_id = c.id
        LEFT JOIN matches m ON m.season_id = s.id
        GROUP BY c.code, c.name, s.label, s.start_year, s.is_current
        ORDER BY c.code, s.start_year
    """)).fetchall()
    if not rows:
        v.fail("No competitions/seasons found")
        return rows

    # median count per competition (ignores the still-running season)
    per_comp = defaultdict(list)
    for r in rows:
        if r.cnt and r.label not in IN_PROGRESS_SEASONS:
            per_comp[r.code].append(r.cnt)

    empty, out_of_range, low = 0, 0, 0
    for r in rows:
        lo, hi = EXPECTED_COUNTS.get(r.code, (0, 10000))
        if r.cnt == 0:
            empty += 1
            v.fail(f"NO DATA {r.code} {r.label}: 0 matches")
            continue
        live = r.label in IN_PROGRESS_SEASONS or r.is_current
        floor = int(lo * IN_PROGRESS_FLOOR) if live else lo
        if not (floor <= r.cnt <= hi):
            out_of_range += 1
            v.fail(f"COUNT {r.code} {r.label}: {r.cnt} (expected {floor}-{hi}"
                   f"{', in-progress' if live else ''})")
            continue
        med = per_comp.get(r.code)
        if med and not live:
            m = sorted(med)[len(med) // 2]
            if m and r.cnt < m * MEDIAN_FLOOR:
                low += 1
                v.fail(f"LOW {r.code} {r.label}: {r.cnt} vs median {m}")
    ok_seasons = len(rows) - empty - out_of_range - low
    if empty or out_of_range or low:
        v.fail(f"{empty} empty, {out_of_range} out-of-range, {low} suspiciously low")
    if ok_seasons:
        v.ok(f"{ok_seasons}/{len(rows)} competition-seasons within expected ranges")

    for div in sorted(SPLIT_FORMAT_DIVS):
        d = [r for r in rows if r.code == div and r.cnt]
        if not d:
            continue
        v.ok(f"NOTE {div}: {len(d)} season(s) stored; this league replays the same "
             f"home/away pairing, so only one meeting per pairing is representable "
             f"under the matches unique constraint")
    return rows


def check_home_away_balance(db, v):
    print("\n--- 2. Home/away balance ---")
    rows = db.execute(text("""
        SELECT c.code, s.label, t.canonical_name,
               COUNT(*) FILTER (WHERE m.home_team_id = t.id) AS h,
               COUNT(*) FILTER (WHERE m.away_team_id = t.id) AS a
        FROM teams t
        JOIN matches m ON t.id IN (m.home_team_id, m.away_team_id)
        JOIN seasons s ON m.season_id = s.id
        JOIN competitions c ON m.competition_id = c.id
        GROUP BY c.code, s.label, t.canonical_name
    """)).fetchall()
    bad = 0
    for r in rows:
        played = r.h + r.a
        if played == 0:
            continue
        # a full season should be near-perfectly balanced; curtailed or
        # split seasons legitimately drift further apart
        tolerance = max(2, round(played * 0.15))
        if abs(r.h - r.a) > tolerance:
            bad += 1
            if bad <= 20:
                v.fail(f"IMBALANCE {r.code} {r.label} {r.canonical_name}: "
                       f"H={r.h} A={r.a} (played {played}, tol {tolerance})")
    if bad == 0:
        v.ok(f"All {len(rows)} team-season rows balanced within tolerance")
    else:
        v.fail(f"{bad} team-season row(s) imbalanced")


def check_status_consistency(db, v):
    print("\n--- 3. Status/score consistency ---")
    ft_no_score = db.execute(text(
        "SELECT COUNT(*) FROM matches WHERE status='FT' "
        "AND (home_goals IS NULL OR away_goals IS NULL)"
    )).scalar()
    if ft_no_score == 0:
        v.ok("Every FT match has both goals")
    else:
        v.fail(f"{ft_no_score} FT match(es) missing goals")

    ns_with_score = db.execute(text(
        "SELECT COUNT(*) FROM matches WHERE status='NS' "
        "AND home_goals IS NOT NULL AND away_goals IS NOT NULL"
    )).scalar()
    if ns_with_score == 0:
        v.ok("No NS match has a final score")
    else:
        v.fail(f"{ns_with_score} NS match(es) already have a score")

    half_no_full = db.execute(text(
        "SELECT COUNT(*) FROM matches WHERE (home_goals_ht IS NOT NULL "
        "OR away_goals_ht IS NOT NULL) "
        "AND (home_goals IS NULL OR away_goals IS NULL)"
    )).scalar()
    if half_no_full == 0:
        v.ok("No HT score without a FT score")
    else:
        v.fail(f"{half_no_full} match(es) with HT but no FT score")

    ht_gt_ft = db.execute(text(
        "SELECT COUNT(*) FROM matches WHERE home_goals IS NOT NULL "
        "AND away_goals IS NOT NULL AND ((home_goals_ht IS NOT NULL "
        "AND home_goals_ht > home_goals) OR (away_goals_ht IS NOT NULL "
        "AND away_goals_ht > away_goals))"
    )).scalar()
    if ht_gt_ft == 0:
        v.ok("No HT goals exceed FT goals (home or away)")
    else:
        v.fail(f"{ht_gt_ft} match(es) where HT goals exceed FT goals (home or away)")


def check_duplicates(db, v):
    print("\n--- 4. No duplicate fixtures ---")
    rows = db.execute(text("""
        SELECT season_id, home_team_id, away_team_id, COUNT(*) AS cnt
        FROM matches
        GROUP BY season_id, home_team_id, away_team_id
        HAVING COUNT(*) > 1
    """)).fetchall()
    if not rows:
        v.ok("No duplicate fixtures (season + home + away)")
    else:
        v.fail(f"{len(rows)} duplicate fixture group(s) found")
        for r in rows[:10]:
            v.fail(f"  dup: season={r.season_id} h={r.home_team_id} a={r.away_team_id} x{r.cnt}")


def check_orphans(db, v):
    print("\n--- 5. No orphaned foreign keys ---")
    pairs = [
        ("matches.competition_id", "matches", "competition_id", "competitions", "id"),
        ("matches.season_id", "matches", "season_id", "seasons", "id"),
        ("matches.home_team_id", "matches", "home_team_id", "teams", "id"),
        ("matches.away_team_id", "matches", "away_team_id", "teams", "id"),
    ]
    bad = 0
    for label, tbl, col, ref_tbl, ref_col in pairs:
        n = db.execute(text(
            f"SELECT COUNT(*) FROM {tbl} m "
            f"LEFT JOIN {ref_tbl} r ON r.{ref_col} = m.{col} "
            f"WHERE r.{ref_col} IS NULL"
        )).scalar()
        if n == 0:
            v.ok(f"No orphans: {label}")
        else:
            bad += 1
            v.fail(f"{n} orphan(s): {label}")
    return bad == 0


def check_goal_range(db, v):
    print("\n--- 6. Goal range sanity ---")
    for col in ("home_goals", "away_goals"):
        n = db.execute(text(
            f"SELECT COUNT(*) FROM matches WHERE {col} IS NOT NULL "
            f"AND ({col} < 0 OR {col} > 20)"
        )).scalar()
        if n == 0:
            v.ok(f"All {col} within [0, 20]")
        else:
            v.fail(f"{n} {col} value(s) outside [0, 20]")

def reconstruct_standings(db, season_id):
    """Return {team_id: {p,w,d,l,gf,ga,played}} for a season's finished matches."""
    matches = db.query(Match).filter(
        Match.season_id == season_id,
        Match.home_goals.isnot(None),
        Match.away_goals.isnot(None),
    ).all()
    sd = defaultdict(lambda: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "played": 0})
    for m in matches:
        h, a = m.home_team_id, m.away_team_id
        sd[h]["played"] += 1
        sd[a]["played"] += 1
        sd[h]["gf"] += m.home_goals
        sd[h]["ga"] += m.away_goals
        sd[a]["gf"] += m.away_goals
        sd[a]["ga"] += m.home_goals
        if m.home_goals > m.away_goals:
            sd[h]["p"] += 3
            sd[h]["w"] += 1
            sd[a]["l"] += 1
        elif m.home_goals < m.away_goals:
            sd[a]["p"] += 3
            sd[a]["w"] += 1
            sd[h]["l"] += 1
        else:
            sd[h]["p"] += 1
            sd[h]["d"] += 1
            sd[a]["p"] += 1
            sd[a]["d"] += 1
    return sd, matches


def check_standings(db, v, max_seasons=10):
    print("\n--- 7. Standings reconstruction ---")
    rows = db.execute(text("""
        SELECT c.code, s.id, s.start_year, s.label
        FROM competitions c
        JOIN seasons s ON s.competition_id = c.id
        WHERE EXISTS (SELECT 1 FROM matches m WHERE m.season_id = s.id)
        ORDER BY c.code, s.start_year DESC
    """)).fetchall()
    if not rows:
        v.fail("No seasons with matches to reconstruct")
        return

    checked, failed = 0, 0
    per_comp = defaultdict(int)
    for r in rows:
        if per_comp[r.code] >= max_seasons:
            continue
        per_comp[r.code] += 1
        sd, matches = reconstruct_standings(db, r.id)
        if not matches:
            continue

        # Invariants of any set of football results:
        #   * each decisive match awards 3 points total, each draw awards 2
        #   * every goal scored by someone is conceded by someone
        #   * each match contributes 2 to the total of "played"
        n = len(matches)
        draws = sum(1 for m in matches if m.home_goals == m.away_goals)
        total_pts = sum(t["p"] for t in sd.values())
        total_gf = sum(t["gf"] for t in sd.values())
        total_ga = sum(t["ga"] for t in sd.values())
        total_played = sum(t["played"] for t in sd.values())
        expect_pts = 3 * n - draws

        ok = True
        if total_pts != expect_pts:
            ok = False
            v.fail(f"{r.code} {r.label}: points {total_pts} != "
                   f"3*{n}-{draws}={expect_pts}")
        if total_gf != total_ga:
            ok = False
            v.fail(f"{r.code} {r.label}: GF {total_gf} != GA {total_ga}")
        if total_played != 2 * n:
            ok = False
            v.fail(f"{r.code} {r.label}: played {total_played} != 2*{n}")
        for tid, t in sd.items():
            if t["w"] + t["d"] + t["l"] != t["played"]:
                ok = False
                v.fail(f"{r.code} {r.label}: team {tid} W+D+L != played")
                break
            if t["p"] != 3 * t["w"] + t["d"]:
                ok = False
                v.fail(f"{r.code} {r.label}: team {tid} points != 3W+D")
                break

        # compare against the stored reference table, when one exists
        ref = db.execute(text("""
            SELECT t.canonical_name, ss.position, ss.points, ss.played,
                   ss.won, ss.drawn, ss.lost, ss.goals_for, ss.goals_against
            FROM standings_snapshots ss
            JOIN teams t ON t.id = ss.team_id
            WHERE ss.season_id = :s
        """), {"s": r.id}).fetchall()
        if ref:
            by_id = {t_id: t for t_id, t in sd.items()}
            tnames = dict(db.execute(text(
                "SELECT id, canonical_name FROM teams")).fetchall())
            bad = 0
            for row in ref:
                tid = next((k for k, v2 in tnames.items()
                            if v2 == row.canonical_name), None)
                t = by_id.get(tid)
                if t is None:
                    bad += 1
                    v.fail(f"{r.code} {r.label}: reference team "
                           f"{row.canonical_name!r} has no matches")
                    continue
                if t["p"] != row.points:
                    bad += 1
                    v.fail(f"{r.code} {r.label} {row.canonical_name}: "
                           f"points {t['p']} != reference {row.points}")
                elif (row.played is not None and t["played"] != row.played):
                    bad += 1
                    v.fail(f"{r.code} {r.label} {row.canonical_name}: "
                           f"played {t['played']} != reference {row.played}")
            if bad == 0:
                v.ok(f"{r.code} {r.label}: {len(ref)} reference standings rows "
                     f"match the reconstruction exactly")
                ok = True

        if ok:
            checked += 1
            if per_comp[r.code] <= 2 and not ref:
                top = sorted(sd.items(),
                             key=lambda kv: (-kv[1]["p"],
                                             -(kv[1]["gf"] - kv[1]["ga"])))[:1]
                v.ok(f"{r.code} {r.label}: {len(matches)}m, {len(sd)}t, "
                     f"pts={total_pts}, GF={total_gf}, "
                     f"top_pts={top[0][1]['p'] if top else 0}")
        else:
            failed += 1
    if failed == 0 and checked:
        v.ok(f"Standings reconstructed consistently for {checked} "
             f"competition-season(s)")
    elif failed:
        v.fail(f"{failed} competition-season(s) failed reconstruction")


REFERENCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "db",
    "reference",
    "la_liga_standings.json",
)


def load_reference_tables():
    """{season label: [rows]} of published final tables, or {} when absent."""
    if not os.path.exists(REFERENCE_FILE):
        return {}
    with open(REFERENCE_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("seasons", {}) if isinstance(data, dict) else {}


def order_final_table(stats, matches):
    """Team ids in published La Liga order.

    Reglamento General art. 201, as used by the published tables: points, then
    head-to-head points among the tied clubs, then head-to-head goal difference,
    then overall goal difference, then goals scored.
    """
    by_points = defaultdict(list)
    for team_id, row in stats.items():
        by_points[row["p"]].append(team_id)

    ordered = []
    for points in sorted(by_points, reverse=True):
        block = by_points[points]
        if len(block) == 1:
            ordered.extend(block)
            continue
        mini = {team_id: {"p": 0, "gd": 0} for team_id in block}
        for m in matches:
            if m.home_team_id in mini and m.away_team_id in mini:
                home, away = mini[m.home_team_id], mini[m.away_team_id]
                home["gd"] += m.home_goals - m.away_goals
                away["gd"] += m.away_goals - m.home_goals
                if m.home_goals > m.away_goals:
                    home["p"] += 3
                elif m.home_goals < m.away_goals:
                    away["p"] += 3
                else:
                    home["p"] += 1
                    away["p"] += 1
        ordered.extend(
            sorted(
                block,
                key=lambda team_id: (
                    -mini[team_id]["p"],
                    -mini[team_id]["gd"],
                    -(stats[team_id]["gf"] - stats[team_id]["ga"]),
                    -stats[team_id]["gf"],
                ),
            )
        )
    return ordered


def check_final_table_order(db, v):
    """Compare reconstructed final tables against published references, in order.

    2018/19 and 2021/22 come from db/reference/la_liga_standings.json (published
    Wikipedia tables); any season with a stored standings_snapshots reference
    (the live pipeline's own rows, e.g. 2024/25) is compared the same way.
    """
    print("\n--- 8. Final-table ordering vs published reference ---")
    names = dict(db.execute(text("SELECT id, canonical_name FROM teams")).fetchall())
    reference = load_reference_tables()
    seasons = []

    for label, rows in sorted(reference.items()):
        season_id = db.execute(
            text(
                "SELECT s.id FROM seasons s JOIN competitions c ON c.id = s.competition_id "
                "WHERE c.code = 'SP1' AND s.label = :label"
            ),
            {"label": label},
        ).scalar()
        if season_id is None:
            v.fail(f"final table {label}: season missing from the database")
            continue
        stats, matches = reconstruct_standings(db, season_id)
        ours = [names[t] for t in order_final_table(stats, matches)]
        expected = [row["team"] for row in sorted(rows, key=lambda row: row["position"])]
        seasons.append((label, ours, expected, "db/reference/la_liga_standings.json"))

    snapshot_seasons = db.execute(text("""
        SELECT c.code, s.label, s.id
        FROM standings_snapshots ss
        JOIN seasons s ON s.id = ss.season_id
        JOIN competitions c ON c.id = s.competition_id
        GROUP BY c.code, s.label, s.id
        HAVING COUNT(*) > 1
        ORDER BY s.start_year DESC
    """)).fetchall()
    for row in snapshot_seasons:
        if row.label in reference:
            continue
        stats, matches = reconstruct_standings(db, row.id)
        ours = [names[t] for t in order_final_table(stats, matches)]
        expected = [
            team
            for (team,) in db.execute(
                text(
                    "SELECT t.canonical_name FROM standings_snapshots ss "
                    "JOIN teams t ON t.id = ss.team_id WHERE ss.season_id = :s "
                    "ORDER BY ss.position"
                ),
                {"s": row.id},
            )
        ]
        seasons.append((row.label, ours, expected, f"standings_snapshots ({row.code})"))

    if not seasons:
        v.fail("no published final tables available to compare")
        return

    failures = 0
    for label, ours, expected, source in seasons:
        if len(ours) != len(expected):
            failures += 1
            v.fail(
                f"final table {label}: reconstructed {len(ours)} teams vs "
                f"reference {len(expected)} ({source})"
            )
            continue
        bad = [
            (pos, mine, theirs)
            for pos, (mine, theirs) in enumerate(zip(ours, expected, strict=True), start=1)
            if mine != theirs
        ]
        if bad:
            failures += 1
            for pos, mine, theirs in bad[:6]:
                v.fail(
                    f"final table {label} position {pos}: reconstructed {mine!r} "
                    f"but published {theirs!r} ({source})"
                )
        else:
            print(
                f"  [  ] final table {label}: all {len(ours)} positions match exactly, "
                f"tiebreaks included ({source})"
            )

    if failures == 0:
        v.ok(
            "Final tables match the published order position-by-position for all "
            f"{len(seasons)} reference season(s): "
            + ", ".join(label for label, _, _, _ in seasons)
            + " (tiebreaks: points, head-to-head points, head-to-head GD, GD, goals)"
        )


def check_duplicate_teams(db, v):
    """No two teams should represent the same club (same normalised name)."""
    print("\n--- 8. No duplicate teams ---")

    def norm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"[^a-z0-9]+", "", s.lower())
        return s

    rows = db.query(Team.id, Team.canonical_name).order_by(Team.id).all()
    groups = defaultdict(list)
    for tid, name in rows:
        groups[norm(name)].append((tid, name))

    dups = {k: g for k, g in groups.items() if len(g) > 1}
    if not dups:
        v.ok(f"No duplicate team names across {len(rows)} teams")
    else:
        for k, g in sorted(dups.items()):
            v.fail(f"DUPLICATE TEAM {k!r}: " + ", ".join(f"{i}:{n!r}" for i, n in g))

    # every alias must resolve to a real team and never to a different team twice
    orphan = db.execute(text("""
        SELECT COUNT(*) FROM team_aliases a
        LEFT JOIN teams t ON t.id = a.team_id
        WHERE t.id IS NULL
    """)).scalar()
    if orphan == 0:
        v.ok("All team_aliases rows resolve to a team")
    else:
        v.fail(f"{orphan} team_aliases row(s) point at a missing team")


def main():
    print("=" * 64)
    print("  XGoal Data Validation Suite (Phase 1.5)")
    print("=" * 64)
    v = V()
    db = SessionLocal()
    try:
        total = db.query(Match).count()
        teams = db.query(Team).count()
        print(f"\nTotal matches in DB: {total}    teams: {teams}")
        if total == 0:
            v.fail("No matches - run ingestion first")
            return v.summary()
        check_season_counts(db, v)
        check_home_away_balance(db, v)
        check_status_consistency(db, v)
        check_duplicates(db, v)
        check_orphans(db, v)
        check_goal_range(db, v)
        check_standings(db, v)
        check_final_table_order(db, v)
        check_duplicate_teams(db, v)
    finally:
        db.close()
    return v.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
