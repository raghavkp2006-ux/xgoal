#!/usr/bin/env python3
"""Write a basic, match-derived standings snapshot for one season.

This is deliberately a one-off bridge until Phase 3.3 implements La Liga's
full head-to-head tiebreak chain. Rows are ordered only by points, overall goal
difference, goals scored, and team id (the final key makes the ordering stable).

Usage:
    python -m jobs.snapshot_basic_standings --season-id 1734
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from app.database import SessionLocal
from app.models import Match, MatchStatus, Season, StandingsSnapshot

FINISHED_STATUSES = (MatchStatus.FT, MatchStatus.AET, MatchStatus.PEN)


@dataclass
class TableRow:
    team_id: int
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


def apply_result(row: TableRow, goals_for: int, goals_against: int) -> str:
    """Accumulate one team's result and return its form character."""
    row.played += 1
    row.goals_for += goals_for
    row.goals_against += goals_against
    if goals_for > goals_against:
        row.won += 1
        row.points += 3
        return "W"
    if goals_for < goals_against:
        row.lost += 1
        return "L"
    row.drawn += 1
    row.points += 1
    return "D"


def compute_basic_standings(matches: list[Match]) -> tuple[list[TableRow], dict[int, str]]:
    """Compute basic W/D/L standings; intentionally excludes head-to-head rules."""
    rows: dict[int, TableRow] = {}
    forms: defaultdict[int, list[str]] = defaultdict(list)
    for match in matches:
        if match.home_goals is None or match.away_goals is None:
            continue
        home = rows.setdefault(match.home_team_id, TableRow(team_id=match.home_team_id))
        away = rows.setdefault(match.away_team_id, TableRow(team_id=match.away_team_id))
        forms[match.home_team_id].append(apply_result(home, match.home_goals, match.away_goals))
        forms[match.away_team_id].append(apply_result(away, match.away_goals, match.home_goals))

    ranked = sorted(
        rows.values(),
        key=lambda row: (-row.points, -row.goal_difference, -row.goals_for, row.team_id),
    )
    return ranked, {team_id: "".join(results[-5:]) for team_id, results in forms.items()}


def write_snapshot(season_id: int) -> list[TableRow]:
    """Persist one timestamp-aligned snapshot and return its ranked rows."""
    db = SessionLocal()
    try:
        season = db.get(Season, season_id)
        if season is None:
            raise ValueError(f"Season {season_id} not found")
        matches = (
            db.query(Match)
            .filter(
                Match.season_id == season_id,
                Match.status.in_(FINISHED_STATUSES),
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
            )
            .order_by(Match.kickoff_utc, Match.id)
            .all()
        )
        if not matches:
            raise ValueError(f"Season {season_id} has no completed matches")

        ranked, forms = compute_basic_standings(matches)
        stamp = datetime.now(timezone.utc)
        latest_matchday = max(
            (match.matchday for match in matches if match.matchday is not None),
            default=None,
        )
        for position, row in enumerate(ranked, start=1):
            db.add(
                StandingsSnapshot(
                    season_id=season_id,
                    computed_at=stamp,
                    as_of_matchday=latest_matchday,
                    position=position,
                    team_id=row.team_id,
                    played=row.played,
                    won=row.won,
                    drawn=row.drawn,
                    lost=row.lost,
                    goals_for=row.goals_for,
                    goals_against=row.goals_against,
                    points=row.points,
                    form=forms.get(row.team_id),
                )
            )
        db.commit()
        print(
            f"Wrote {len(ranked)} rows for {season.label} from {len(matches)} completed matches "
            f"at {stamp.isoformat()} (basic points/GD/GS ordering only)."
        )
        return ranked
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a basic standings snapshot from matches.")
    parser.add_argument("--season-id", type=int, required=True)
    args = parser.parse_args()
    write_snapshot(args.season_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
