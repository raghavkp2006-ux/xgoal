"""Vectorized simulation engine regression tests."""

from __future__ import annotations

import numpy as np

from app.ml.simulation import (
    Fixture,
    derive_missing_round_robin_fixtures,
    simulate_season,
)


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
    for run_points, draw_count in zip(final_points, draws, strict=True):
        assert sum(run_points) == 3 * (12 - draw_count) + 2 * draw_count
