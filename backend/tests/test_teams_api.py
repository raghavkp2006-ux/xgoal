"""Team API regressions against an isolated in-memory database."""

from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app, limiter
from app.models import Competition, Match, MatchStatus, Season, Team


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    for model in (Competition, Season, Team, Match):
        model.__table__.create(engine)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(Competition(id=1, code="SP1", name="La Liga", country="Spain", tier=1))
        for season_id, year in ((10, 2025), (20, 2026)):
            db.add(Season(
                id=season_id, competition_id=1, label=f"{year}/{year + 1}",
                start_year=year, start_date=date(year, 8, 1), end_date=date(year + 1, 5, 31),
            ))
        for team_id in range(1, 5):
            db.add(Team(id=team_id, canonical_name=f"Team {team_id}", created_at=now))
        for match_id, season_id, home, away in ((1, 10, 3, 4), (2, 20, 1, 2), (3, 20, 2, 1)):
            db.add(Match(
                id=match_id, competition_id=1, season_id=season_id,
                home_team_id=home, away_team_id=away, kickoff_utc=now,
                status=MatchStatus.NS, source="test", ingested_at=now, updated_at=now,
            ))
        db.commit()
        app.dependency_overrides[get_db] = lambda: db
        limiter.reset()
        try:
            yield TestClient(app, client=("198.51.100.20", 50000))
        finally:
            app.dependency_overrides.clear()
            limiter.reset()
    engine.dispose()


def test_missing_team_returns_http_404(client):
    response = client.get("/api/v1/teams/999999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Team not found"}


def test_season_filter_includes_both_sides_and_deduplicates(client):
    response = client.get("/api/v1/teams?season=2026")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [1, 2]


def test_no_season_returns_all_teams(client):
    response = client.get("/api/v1/teams")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [1, 2, 3, 4]


def test_unknown_season_returns_empty_list(client):
    response = client.get("/api/v1/teams?season=2099")
    assert response.status_code == 200
    assert response.json() == []
