"""Historical and synthetic tests for La Liga's standings tie-breakers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from app.database import SessionLocal
from app.models import Competition, Match, Season, Team
from app.standings import rank_standings, rank_standings_many


@dataclass(frozen=True)
class Fixture:
    season_id: int
    home_team_id: str
    away_team_id: str
    home_goals: int | None
    away_goals: int | None


@pytest.mark.parametrize("label", ["2018/19", "2021/22"])
def test_real_completed_tiebreak_seasons_match_published_order(label: str) -> None:
    """Historical DB seasons cover two-way and three-way completed ties."""
    root = Path(__file__).resolve().parents[2]
    references = json.loads(
        (root / "db" / "reference" / "la_liga_standings.json").read_text(
            encoding="utf-8"
        )
    )["seasons"]
    with SessionLocal() as db:
        competition = db.query(Competition).filter_by(code="SP1").one()
        season = (
            db.query(Season)
            .filter_by(competition_id=competition.id, label=label)
            .one()
        )
        matches = db.query(Match).filter_by(season_id=season.id).all()
        names = {team.id: team.canonical_name for team in db.query(Team).all()}
        ranked = rank_standings(matches, season.id)

    assert len(matches) == 380
    assert [row.position for row in ranked] == list(range(1, 21))
    assert [names[row.team_id] for row in ranked] == [
        row["team"] for row in references[label]
    ]


def test_three_way_tie_uses_the_complete_mini_table() -> None:
    """Head-to-head mini-table beats overall GD for a tied three-club group."""
    season_id = 7
    fixtures = [
        Fixture(season_id, "A", "B", 2, 0),
        Fixture(season_id, "B", "A", 1, 0),
        Fixture(season_id, "A", "C", 1, 0),
        Fixture(season_id, "C", "A", 0, 1),
        Fixture(season_id, "B", "C", 1, 0),
        Fixture(season_id, "C", "B", 0, 1),
        Fixture(season_id, "A", "D", 1, 0),
        Fixture(season_id, "A", "E", 1, 0),
        Fixture(season_id, "B", "E", 1, 0),
        Fixture(season_id, "B", "F", 1, 0),
        Fixture(season_id, "C", "G", 10, 0),
        Fixture(season_id, "C", "H", 10, 0),
        Fixture(season_id, "C", "I", 10, 0),
        Fixture(season_id, "C", "J", 10, 0),
        Fixture(season_id, "C", "K", 10, 0),
    ]

    ranked = rank_standings(fixtures, season_id)
    rows = {row.team_id: row for row in ranked}

    assert [row.team_id for row in ranked[:3]] == ["A", "B", "C"]
    assert rows["A"].points == rows["B"].points == rows["C"].points == 15
    assert rows["C"].goal_difference > rows["A"].goal_difference


def test_incomplete_head_to_head_falls_back_to_overall_goal_difference() -> None:
    season_id = 12
    fixtures = [
        Fixture(season_id, "A", "B", 1, 0),
        Fixture(season_id, "B", "A", None, None),
        Fixture(season_id, "C", "A", 1, 0),
        Fixture(season_id, "B", "D", 5, 0),
    ]

    ranked = rank_standings(fixtures, season_id)

    assert next(row.position for row in ranked if row.team_id == "B") < next(
        row.position for row in ranked if row.team_id == "A"
    )


def test_vectorized_ranker_matches_the_pure_single_season_ranker() -> None:
    season_id = 24
    teams = ["A", "B", "C", "D"]
    fixtures = [
        Fixture(season_id, home, away, None, None)
        for home in teams
        for away in teams
        if home != away
    ]
    home_goals = np.asarray(
        [
            [2, 0, 1, 0, 3, 0, 1, 2, 0, 2, 1, 0],
            [0, 2, 1, 2, 0, 1, 2, 0, 1, 0, 0, 2],
        ],
        dtype=np.int64,
    )
    away_goals = np.asarray(
        [
            [0, 1, 0, 1, 0, 0, 0, 1, 2, 0, 2, 1],
            [0, 0, 0, 0, 1, 2, 0, 2, 0, 1, 2, 0],
        ],
        dtype=np.int64,
    )

    positions, points = rank_standings_many(
        fixtures, season_id, teams, home_goals, away_goals
    )
    for run_index in range(home_goals.shape[0]):
        scored_fixtures = [
            Fixture(
                fixture.season_id,
                fixture.home_team_id,
                fixture.away_team_id,
                int(home_goals[run_index, fixture_index]),
                int(away_goals[run_index, fixture_index]),
            )
            for fixture_index, fixture in enumerate(fixtures)
        ]
        scalar = rank_standings(scored_fixtures, season_id)
        scalar_positions = {
            row.team_id: row.position for row in scalar
        }
        scalar_points = {row.team_id: row.points for row in scalar}
        assert positions[run_index].tolist() == [
            scalar_positions[team_id] for team_id in teams
        ]
        assert points[run_index].tolist() == [
            scalar_points[team_id] for team_id in teams
        ]
