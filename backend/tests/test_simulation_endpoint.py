from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.simulation import Fixture, simulate_season
from app.models import Match, MatchStatus, Season
from app.routers.simulation import (
    WhatIfRequest,
    WhatIfValidationError,
    get_simulation,
    paired_team_deltas,
    router,
    validate_forced_results,
)


class FakeSession:
    def __init__(self, row: SimpleNamespace) -> None:
        self.row = row
        self.statements: list[Any] = []

    def execute(self, statement: Any) -> FakeSession:
        self.statements.append(statement)
        return self

    def one_or_none(self) -> SimpleNamespace:
        return self.row


def test_cached_simulation_uses_one_limited_query_and_cache_headers() -> None:
    run = SimpleNamespace(
        id=7,
        season_id=1734,
        model_version_id=4,
        n_simulations=10_000,
        random_seed=42,
        as_of_matchday=7,
        run_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
        results={"teams": [{"team": "Example", "p_champion": 0.5}]},
    )
    db = FakeSession(run)

    response = get_simulation(
        season="2026/27", if_none_match=None, db=cast(Session, db)
    )

    assert len(db.statements) == 1
    assert "LIMIT" in str(db.statements[0]).upper()
    assert isinstance(response, Response)
    assert response.status_code == 200
    assert response.headers["cache-control"] == (
        "public, max-age=300, stale-while-revalidate=3600"
    )
    assert response.headers["etag"]


def test_matching_etag_returns_not_modified_without_response_body() -> None:
    run = SimpleNamespace(
        id=7,
        season_id=1734,
        model_version_id=4,
        n_simulations=10_000,
        random_seed=42,
        as_of_matchday=7,
        run_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
        results={"teams": [{"team": "Example", "p_champion": 0.5}]},
    )
    first_response = get_simulation(
        season="2026/27",
        if_none_match=None,
        db=cast(Session, FakeSession(run)),
    )

    cached_response = get_simulation(
        season="2026/27",
        if_none_match=first_response.headers["etag"],
        db=cast(Session, FakeSession(run)),
    )

    assert cached_response.status_code == 304
    assert cached_response.body == b""
    assert cached_response.headers["etag"] == first_response.headers["etag"]
    assert cached_response.headers["cache-control"] == (
        "public, max-age=300, stale-while-revalidate=3600"
    )


def test_whatif_rejects_unknown_and_finished_matches_without_database() -> None:
    matches = [
        Match(id=1, season_id=1734, status=MatchStatus.NS),
        Match(id=2, season_id=1734, status=MatchStatus.FT),
        Match(id=3, season_id=99, status=MatchStatus.NS),
    ]
    for match_id, message in [(9, "current season"), (2, "not NS"), (3, "current season")]:
        request = WhatIfRequest.model_validate(
            {"forced_results": [{"match_id": match_id, "home_goals": 1, "away_goals": 0}]}
        )
        with pytest.raises(WhatIfValidationError, match=message):
            validate_forced_results(request.forced_results, 1734, matches)


@pytest.mark.parametrize("score", [-1, 21, 1.5, "2", True])
def test_whatif_rejects_goals_outside_strict_integer_range(score: object) -> None:
    with pytest.raises(ValidationError):
        WhatIfRequest.model_validate(
            {"forced_results": [{"match_id": 1, "home_goals": score, "away_goals": 0}]}
        )


def test_whatif_rejects_more_than_fifty_results() -> None:
    with pytest.raises(ValidationError):
        WhatIfRequest.model_validate(
            {"forced_results": [
                {"match_id": index, "home_goals": 0, "away_goals": 0}
                for index in range(1, 52)
            ]}
        )


def test_whatif_validation_uses_problem_json_without_live_database() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: None
    response = TestClient(app).post(
        "/api/v1/simulation/whatif",
        json={"forced_results": [{"match_id": 1, "home_goals": 21, "away_goals": 0}]},
    )
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["status"] == 422


def test_whatif_rejects_unknown_and_finished_ids_as_problem_json() -> None:
    class FakeRoutingSession:
        def query(self, model: type[Any]) -> FakeRoutingSession:
            self.model = model
            return self

        def filter(self, *args: Any) -> FakeRoutingSession:
            return self

        def order_by(self, *args: Any) -> FakeRoutingSession:
            return self

        def first(self) -> Season:
            return Season(id=1734, label="2026/27", is_current=True)

        def all(self) -> list[Match]:
            return [
                Match(id=1, season_id=1734, status=MatchStatus.NS),
                Match(id=2, season_id=1734, status=MatchStatus.FT),
            ]

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = FakeRoutingSession
    client = TestClient(app)
    for match_id, detail in [(9, "current season"), (2, "not NS")]:
        response = client.post(
            "/api/v1/simulation/whatif",
            json={"forced_results": [{"match_id": match_id, "home_goals": 1, "away_goals": 0}]},
        )
        assert response.status_code == 422
        assert response.headers["content-type"] == "application/problem+json"
        assert detail in response.json()["detail"]


def test_whatif_forced_score_is_used_in_every_sim_and_positions_sum_to_one() -> None:
    teams = list(range(4))
    names = {team: f"Team {team}" for team in teams}
    fixtures = [
        Fixture(1734, home, away, None, None)
        for home in teams for away in teams if home != away
    ]
    result = simulate_season(
        season_id=1734,
        team_ids=teams,
        team_names=names,
        fixtures=fixtures,
        parameter_ensemble=np.asarray([[0.0] * 10], dtype=float),
        n_simulations=2_000,
        seed=42,
        forced_results={"Team 0 vs Team 1": (3, 1)},
    )
    raw = result["raw_simulations"]
    assert raw["forced_fixture_scores"]["0"] == [[3, 1]] * 2_000
    for team in result["teams"]:
        assert abs(sum(team["finish_position_distribution"].values()) - 1.0) <= 0.001


def _paired_synthetic_runs(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    teams = list(range(4))
    names = {team: f"Team {team}" for team in teams}
    fixtures = [
        Fixture(1734, home, away, None, None)
        for home in teams for away in teams if home != away
    ]
    options = {
        "season_id": 1734,
        "team_ids": teams,
        "team_names": names,
        "fixtures": fixtures,
        "parameter_ensemble": np.asarray([[0.0] * 10], dtype=float),
        "n_simulations": 2_000,
        "seed": seed,
    }
    unforced = simulate_season(**options)
    forced = simulate_season(
        **options, forced_results={"Team 0 vs Team 1": (2, 1)}
    )
    return unforced, forced


def test_paired_control_has_exactly_zero_deltas() -> None:
    unforced, _ = _paired_synthetic_runs(seed=42)
    control = simulate_season(
        season_id=1734,
        team_ids=list(range(4)),
        team_names={team: f"Team {team}" for team in range(4)},
        fixtures=[
            Fixture(1734, home, away, None, None)
            for home in range(4) for away in range(4) if home != away
        ],
        parameter_ensemble=np.asarray([[0.0] * 10], dtype=float),
        n_simulations=2_000,
        seed=42,
        forced_results={},
    )
    assert control["teams"] == unforced["teams"]
    deltas = paired_team_deltas(control["teams"], unforced["teams"])
    assert all(all(value == 0 for value in (
        row["p_champion"], row["p_top4"], row["p_top6"],
        row["p_relegation"], row["expected_final_points"],
    )) for row in deltas)


@pytest.mark.parametrize("seed", [1, 42, 1337])
def test_forced_home_win_does_not_lower_home_expected_points(seed: int) -> None:
    unforced, forced = _paired_synthetic_runs(seed)
    delta = paired_team_deltas(forced["teams"], unforced["teams"])
    home = next(row for row in delta if row["team_id"] == 0)
    assert home["expected_final_points"] >= 0


def test_paired_probability_delta_columns_sum_to_zero() -> None:
    unforced, forced = _paired_synthetic_runs(seed=42)
    deltas = paired_team_deltas(forced["teams"], unforced["teams"])
    for field in ("p_champion", "p_top4", "p_top6", "p_relegation"):
        assert abs(sum(row[field] for row in deltas)) < 0.001
