"""Tests for the health check endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


class FakeSession:
    def query(self, _model: object) -> FakeSession:
        return self

    def all(self) -> list[object]:
        return []

    def close(self) -> None:
        pass


def test_health_returns_200(monkeypatch):
    """Health endpoint should return 200 and a status field."""
    monkeypatch.setattr(main, "SessionLocal", FakeSession)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "db" in data
    assert "sources" in data
