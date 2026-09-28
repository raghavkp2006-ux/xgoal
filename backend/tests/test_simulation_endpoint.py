from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

from fastapi import Response
from sqlalchemy.orm import Session

from app.routers.simulation import get_simulation


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
