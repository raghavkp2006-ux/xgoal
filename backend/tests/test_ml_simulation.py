"""Vectorized simulation engine regression tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.ml.simulation import (
    Fixture,
    derive_missing_round_robin_fixtures,
    simulate_season,
)
from app.models import Match
from app.standings import rank_standings, rank_standings_many
from jobs.simulate_season import historical_cutoff


def test_current_season_fixture_derivation_closes_every_round_robin_pair() -> None:
    teams = list(range(20))
    all_pairs = [
        (home, away)
        for home in teams
        for away in teams
        if home != away
    ]
    played = [
        Fixture(3, home, away, index % 3, (index // 3) % 3)
        for index, (home, away) in enumerate(all_pairs[:69])
    ]

    remaining = derive_missing_round_robin_fixtures(
        teams, played, season_id=3
    )

    assert len(played) == 69
    assert len(remaining) == 311
    complete = {(fixture.home_team_id, fixture.away_team_id) for fixture in played}
    complete.update(remaining)
    assert len(complete) == 380
    assert all(sum(home == team for home, _ in complete) == 19 for team in teams)
    assert all(sum(away == team for _, away in complete) == 19 for team in teams)


def test_vectorized_simulation_stores_balanced_finish_and_points_outputs() -> None:
    teams = list(range(4))
    names = {team: f"Team {team}" for team in teams}
    fixtures = [
        Fixture(8, home, away, None, None)
        for home in teams
        for away in teams
        if home != away
    ]
    parameters = [
        [0.0] * 10,
        [0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.05],
    ]

    result = simulate_season(
        season_id=8,
        team_ids=teams,
        team_names=names,
        fixtures=fixtures,
        parameter_ensemble=np.asarray(parameters, dtype=float),
        n_simulations=100,
        seed=42,
    )

    positions = result["raw_simulations"]["finish_positions"]
    final_points = result["raw_simulations"]["final_points"]
    draws = result["raw_simulations"]["draws"]
    assert len(positions) == len(final_points) == len(draws) == 100
    for run_positions in positions:
        assert sorted(run_positions) == [1, 2, 3, 4]
    assert len(result["teams"]) == 4
    assert all(
        abs(sum(row["finish_position_distribution"].values()) - 1.0) < 0.001
        for row in result["teams"]
    )
    for position in range(1, 5):
        assert sum(
            row["finish_position_distribution"][str(position)]
            for row in result["teams"]
        ) == 1.0
    assert all(
        row["expected_final_points_95ci"]["level"] == 0.95
        and row["expected_final_points_95ci"]["lower"]
        <= row["expected_final_points"]
        <= row["expected_final_points_95ci"]["upper"]
        for row in result["teams"]
    )
    for run_points, draw_count in zip(final_points, draws, strict=True):
        assert sum(run_points) == 3 * (12 - draw_count) + 2 * draw_count


def test_same_seed_produces_identical_simulations() -> None:
    teams = list(range(4))
    names = {team: f"Team {team}" for team in teams}
    fixtures = [
        Fixture(9, home, away, None, None)
        for home in teams
        for away in teams
        if home != away
    ]
    parameters = np.asarray([[0.0] * 10, [0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.05]])

    first = simulate_season(
        season_id=9,
        team_ids=teams,
        team_names=names,
        fixtures=fixtures,
        parameter_ensemble=parameters,
        n_simulations=100,
        seed=2718,
    )
    second = simulate_season(
        season_id=9,
        team_ids=teams,
        team_names=names,
        fixtures=fixtures,
        parameter_ensemble=parameters,
        n_simulations=100,
        seed=2718,
    )

    assert first == second


def test_historical_cutoff_uses_labeled_chronological_proxy_when_rounds_missing() -> None:
    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    matches = [
        Match(id=index + 1, kickoff_utc=start + timedelta(days=index), matchday=None)
        for index in range(380)
    ]

    cutoff, method, proxy_target = historical_cutoff(
        matches, as_of_matchday=20, fixtures_per_matchday=10
    )

    assert cutoff == start + timedelta(days=199, microseconds=1)
    assert method == "chronological_fixture_count_proxy"
    assert proxy_target == 200


def test_vectorized_ranking_matches_reference_with_two_and_three_way_ties() -> None:
    teams = list(range(6))
    fixtures = [
        Fixture(10, home, away, None, None)
        for home in teams
        for away in teams
        if home != away
    ]
    rng = np.random.default_rng(732)
    home_goals = rng.poisson(1.5, size=(500, len(fixtures))).astype(np.int64)
    away_goals = rng.poisson(1.5, size=(500, len(fixtures))).astype(np.int64)

    positions, points = rank_standings_many(
        fixtures, 10, teams, home_goals, away_goals
    )
    tie_sizes: set[int] = set()
    for run_index in range(len(home_goals)):
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
        reference = rank_standings(scored_fixtures, 10)
        reference_positions = {row.team_id: row.position for row in reference}
        reference_points = {row.team_id: row.points for row in reference}
        assert positions[run_index].tolist() == [
            reference_positions[team_id] for team_id in teams
        ]
        assert points[run_index].tolist() == [
            reference_points[team_id] for team_id in teams
        ]
        point_groups: dict[int, list[int]] = {}
        for row in reference:
            point_groups.setdefault(row.points, []).append(row.team_id)
        tie_sizes.update(
            len(group) for group in point_groups.values() if len(group) > 1
        )

    assert 2 in tie_sizes
    assert 3 in tie_sizes
