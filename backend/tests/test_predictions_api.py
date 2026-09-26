"""Tests for stored and on-demand prediction endpoints."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app
from app.ml.data import ScorelinePrediction
from app.routers import predictions


def _client_with_db(db: MagicMock) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def test_get_prediction_returns_latest_stored_prediction():
    db = MagicMock()
    match_query = MagicMock()
    prediction_query = MagicMock()
    db.query.side_effect = [match_query, prediction_query]
    match_query.filter.return_value.first.return_value = SimpleNamespace(id=42)
    prediction_query.join.return_value.filter.return_value.order_by.return_value.first.return_value = (
        SimpleNamespace(
            match_id=42,
            model_version=SimpleNamespace(name="dixon_coles", version="v2"),
            as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
            p_home=0.5,
            p_draw=0.3,
            p_away=0.2,
            expected_home_goals=1.75,
            expected_away_goals=0.9,
            score_matrix={"max_goals": 1, "cells": [[0.2, 0.3], [0.3, 0.2]]},
        )
    )

    try:
        response = _client_with_db(db).get("/api/v1/predictions/42")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "match_id": 42,
        "model_name": "dixon_coles",
        "model_version": "v2",
        "as_of": "2026-01-02T00:00:00Z",
        "p_home": 0.5,
        "p_draw": 0.3,
        "p_away": 0.2,
        "expected_home_goals": 1.75,
        "expected_away_goals": 0.9,
        "score_matrix": {"max_goals": 1, "cells": [[0.2, 0.3], [0.3, 0.2]]},
        "is_hypothetical": False,
    }


def test_get_prediction_returns_404_when_match_has_no_prediction():
    db = MagicMock()
    match_query = MagicMock()
    prediction_query = MagicMock()
    db.query.side_effect = [match_query, prediction_query]
    match_query.filter.return_value.first.return_value = SimpleNamespace(id=42)
    prediction_query.join.return_value.filter.return_value.order_by.return_value.first.return_value = None

    try:
        response = _client_with_db(db).get("/api/v1/predictions/42")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "No prediction found for match 42"


def test_hypothetical_prediction_uses_production_model(monkeypatch):
    db = MagicMock()
    home_query = MagicMock()
    away_query = MagicMock()
    db.query.side_effect = [home_query, away_query]
    home_query.filter.return_value.first.return_value = SimpleNamespace(canonical_name="Home FC")
    away_query.filter.return_value.first.return_value = SimpleNamespace(canonical_name="Away FC")
    model = MagicMock()
    model.predict.return_value = ScorelinePrediction(
        p_home=0.4,
        p_draw=0.35,
        p_away=0.25,
        exp_home_goals=1.2,
        exp_away_goals=0.8,
        max_goals=1,
        matrix=[[0.3, 0.2], [0.2, 0.3]],
    )
    monkeypatch.setattr(
        predictions,
        "load_version_model",
        lambda _db, version: (model, SimpleNamespace(name="dixon_coles", version="v3")),
    )

    try:
        response = _client_with_db(db).post(
            "/api/v1/predictions/hypothetical",
            json={"home_team_id": 1, "away_team_id": 2, "as_of": "2026-01-02T12:00:00Z"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["is_hypothetical"] is True
    assert response.json()["match_id"] is None
    assert response.json()["model_version"] == "v3"
    model.predict.assert_called_once_with("Home FC", "Away FC")


def test_hypothetical_prediction_rejects_same_team():
    response = TestClient(app).post(
        "/api/v1/predictions/hypothetical",
        json={"home_team_id": 1, "away_team_id": 1},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "home_team_id and away_team_id must differ"
