"""HTTP caching contracts without a connection to the project database."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from app.caching import LONG_CACHE, SHORT_CACHE, SIMULATION_CACHE, ReadCacheMiddleware
from app.database import get_db
from app.main import app, limiter


@pytest.fixture
def client():
    db = MagicMock()
    db.query.return_value.order_by.return_value.all.return_value = []
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
    filtered = db.query.return_value.filter.return_value
    filtered.options.return_value.order_by.return_value.all.return_value = []
    db.query.return_value.group_by.return_value.all.return_value = []
    joined = db.query.return_value.join.return_value.filter.return_value
    joined.options.return_value.order_by.return_value.all.return_value = []
    app.dependency_overrides[get_db] = lambda: db
    limiter.reset()
    try:
        yield TestClient(app, client=("198.51.100.40", 50000))
    finally:
        app.dependency_overrides.clear()
        limiter.reset()


@pytest.mark.parametrize(
    "path,policy",
    [
        ("/api/v1/competitions", LONG_CACHE),
        ("/api/v1/competitions/1/seasons", LONG_CACHE),
        ("/api/v1/teams", LONG_CACHE),
        ("/api/v1/model-versions", LONG_CACHE),
        ("/api/v1/matches", SHORT_CACHE),
        ("/api/v1/standings?season_id=1", SHORT_CACHE),
        ("/api/v1/predictions?match_ids=1", SHORT_CACHE),
    ],
)
def test_every_router_has_stable_etag_and_empty_304(client, path, policy):
    first = client.get(path)
    second = client.get(path)
    assert first.status_code == second.status_code == 200
    assert first.headers["etag"] == second.headers["etag"]
    assert first.headers["cache-control"] == policy
    for validator in (first.headers["etag"], f'"other", W/{first.headers["etag"]}', "*"):
        conditional = client.get(path, headers={"If-None-Match": validator})
        assert conditional.status_code == 304
        assert conditional.content == b""
        assert conditional.headers["etag"] == first.headers["etag"]
        assert conditional.headers["cache-control"] == policy
        assert "content-length" not in conditional.headers


def test_changed_body_changes_etag():
    fixture = FastAPI()
    fixture.add_middleware(ReadCacheMiddleware)
    payload = {"value": 1}

    @fixture.get("/api/v1/teams")
    def read():
        return payload

    client = TestClient(fixture)
    first = client.get("/api/v1/teams")
    payload["value"] = 2
    second = client.get("/api/v1/teams", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 200
    assert second.headers["etag"] != first.headers["etag"]


def test_health_never_caches_and_simulation_keeps_its_policy():
    fixture = FastAPI()
    fixture.add_middleware(ReadCacheMiddleware)

    @fixture.get("/health")
    def health():
        return {"status": "ok"}

    @fixture.get("/api/v1/simulation")
    def simulation():
        return {"n_simulations": 10000}

    client = TestClient(fixture)
    health_response = client.get("/health", headers={"If-None-Match": "*"})
    assert health_response.status_code == 200
    assert health_response.headers["cache-control"] == "no-store"
    assert "etag" not in health_response.headers
    response = client.get("/api/v1/simulation")
    assert response.headers["cache-control"] == SIMULATION_CACHE


@pytest.mark.parametrize("path", ["/api/v1/predictions", "/api/v1/standings"])
def test_get_errors_are_not_cacheable(client, path):
    response = client.get(path, headers={"If-None-Match": "*"})
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert "etag" not in response.headers


@pytest.mark.parametrize("path", ["/api/v1/predictions/hypothetical", "/api/v1/simulation/whatif"])
def test_actual_post_errors_have_no_cache_headers(client, path):
    response = client.post(path, json={})
    assert response.status_code == 422
    assert "etag" not in response.headers
    assert "cache-control" not in response.headers


@pytest.mark.parametrize("path", ["/api/v1/predictions/hypothetical", "/api/v1/simulation/whatif"])
def test_successful_posts_bypass_cache_middleware(path):
    fixture = FastAPI()
    fixture.add_middleware(ReadCacheMiddleware)

    @fixture.post(path)
    def write():
        return JSONResponse({"result": "ok"})

    response = TestClient(fixture).post(path, headers={"If-None-Match": "*"})
    assert response.status_code == 200
    assert "etag" not in response.headers
    assert "cache-control" not in response.headers
