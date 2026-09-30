"""Lockdown tests use mocked sessions and never write to the project database."""

from collections import Counter
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app, limiter


@pytest.fixture
def client():
    db = MagicMock()
    db.query.return_value.order_by.return_value.all.return_value = []
    db.query.return_value.filter.return_value.first.return_value = None
    app.dependency_overrides[get_db] = lambda: db
    limiter.reset()
    try:
        yield TestClient(app, client=("198.51.100.10", 50000))
    finally:
        app.dependency_overrides.clear()
        limiter.reset()


def test_only_intentional_post_routes_are_registered():
    posts = {path for path, methods in app.openapi()["paths"].items() if "post" in methods}
    assert posts == {
        "/api/v1/predictions/hypothetical",
        "/api/v1/simulation/whatif",
    }


def test_default_limit_covers_included_router(client):
    responses = [client.get("/api/v1/competitions") for _ in range(65)]
    assert Counter(response.status_code for response in responses) == {200: 60, 429: 5}
    blocked = responses[-1]
    assert blocked.headers["content-type"] == "application/problem+json"
    assert int(blocked.headers["retry-after"]) > 0
    assert blocked.json()["status"] == 429
    other_ip = TestClient(app, client=("198.51.100.11", 50000))
    assert other_ip.get("/api/v1/competitions").status_code == 200


def test_health_is_exempt(client, monkeypatch):
    db = MagicMock()
    db.query.return_value.all.return_value = []
    monkeypatch.setattr("app.main.SessionLocal", lambda: db)
    assert all(client.get("/health").status_code == 200 for _ in range(65))


@pytest.mark.parametrize(
    ("path", "body", "allowed_status"),
    [
        ("/api/v1/simulation/whatif", {"forced_results": []}, 404),
        ("/api/v1/predictions/hypothetical", {"home_team_id": 1, "away_team_id": 1}, 400),
    ],
)
def test_expensive_posts_have_stricter_limit(client, path, body, allowed_status):
    # Missing cached data / same-team validation avoid invoking model computation.
    responses = [client.post(path, json=body) for _ in range(12)]
    assert [response.status_code for response in responses] == [allowed_status] * 10 + [429] * 2
    assert responses[-1].headers["content-type"] == "application/problem+json"
    assert int(responses[-1].headers["retry-after"]) > 0
