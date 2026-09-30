"""Latest prediction selection against an isolated SQLite database."""

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import JSON, MetaData, create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app, limiter
from app.models import Competition, Match, ModelVersion, Prediction, Season, Team


@pytest.fixture
def fixture_client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    metadata = MetaData()
    for model in (Competition, Season, Team, Match, ModelVersion, Prediction):
        table = model.__table__.to_metadata(metadata)
        for column in table.columns:
            if isinstance(column.type, JSONB):
                column.type = JSON()
    metadata.create_all(engine)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(Competition(id=1, code="SP1", name="La Liga", country="Spain", tier=1))
        db.add(
            Season(
                id=1,
                competition_id=1,
                label="2026/27",
                start_year=2026,
                start_date=date(2026, 8, 1),
                end_date=date(2027, 5, 31),
            )
        )
        for team_id in (1, 2, 3, 4):
            db.add(Team(id=team_id, canonical_name=f"Team {team_id}", created_at=now))
        for version_id in range(1, 5):
            db.add(
                ModelVersion(
                    id=version_id,
                    name="test",
                    version=str(version_id),
                    git_sha="test",
                    random_seed=42,
                    trained_at=now,
                    train_start=now.date(),
                    train_end=now.date(),
                    n_train_matches=1,
                    hyperparameters={},
                    eval_metrics={},
                    artifact_path="unused",
                    is_production=version_id == 1,
                )
            )
        for match_id in (1, 2, 3):
            db.add(
                Match(
                    id=match_id,
                    competition_id=1,
                    season_id=1,
                    home_team_id=1,
                    away_team_id=match_id + 1,
                    kickoff_utc=now,
                    status="NS",
                    source="test",
                    ingested_at=now,
                    updated_at=now,
                )
            )
            # ID 4 wins an exact timestamp tie despite being non-production.
            for version_id in range(1, 5):
                db.add(
                    Prediction(
                        id=match_id * 10 + version_id,
                        match_id=match_id,
                        model_version_id=version_id,
                        as_of=now - timedelta(days=1 if version_id == 1 else 0),
                        predicted_at=now - timedelta(hours=1 if version_id == 2 else 0),
                        p_home=0.4 + version_id / 100,
                        p_draw=0.3,
                        p_away=0.3 - version_id / 100,
                        feature_hash="test",
                    )
                )
        db.commit()
        statements = []
        event.listen(engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
        app.dependency_overrides[get_db] = lambda: db
        limiter.reset()
        try:
            yield TestClient(app, client=("198.51.100.30", 50000)), statements
        finally:
            app.dependency_overrides.clear()
            limiter.reset()
    engine.dispose()


def test_bulk_equals_single_across_versions_and_ties_in_one_query(fixture_client):
    client, statements = fixture_client
    response = client.get("/api/v1/predictions?match_ids=1,2,3")
    assert response.status_code == 200
    assert len(statements) == 1
    for row in response.json():
        single = client.get(f"/api/v1/predictions/{row['match_id']}")
        assert row == single.json()
        assert row["model_version"] == "4"
        assert row["p_home"] == 0.44


def test_bulk_omits_unknown_ids(fixture_client):
    client, _ = fixture_client
    response = client.get("/api/v1/predictions?match_ids=1,999999")
    assert response.status_code == 200
    assert [row["match_id"] for row in response.json()] == [1]
    assert client.get("/api/v1/predictions?match_ids=999999").json() == []


@pytest.mark.parametrize(
    "ids", [",".join(map(str, range(1, 102))), "", "1,,2", "abc", "0", "-1", "2147483648"]
)
def test_bulk_rejects_invalid_ids(fixture_client, ids):
    client, statements = fixture_client
    assert client.get("/api/v1/predictions", params={"match_ids": ids}).status_code == 422
    assert statements == []


def test_scoreboard_still_requires_a_model_version(fixture_client):
    client, statements = fixture_client
    response = client.get("/api/v1/predictions")
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "missing",
                "loc": ["query", "model_version_id"],
                "msg": "Field required",
                "input": None,
            }
        ]
    }
    assert statements == []


def test_scoreboard_selection_is_unchanged(fixture_client):
    client, _ = fixture_client
    response = client.get("/api/v1/predictions?model_version_id=1&resolved_only=false")
    assert response.status_code == 200
    assert len(response.json()) == 3
    assert all(row["model_version_id"] == 1 for row in response.json())
