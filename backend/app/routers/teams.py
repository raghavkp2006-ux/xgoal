"""Teams API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Team

router = APIRouter(prefix="/api/v1/teams", tags=["teams"])


@router.get("")
def list_teams(season: int | None = None, db: Session = Depends(get_db)):
    """List all teams, optionally filtered by season."""
    query = db.query(Team).order_by(Team.canonical_name)
    teams = query.all()
    return [
        {
            "id": t.id,
            "name": t.canonical_name,
            "short_name": t.short_name,
            "logo_url": t.logo_url,
        }
        for t in teams
    ]


@router.get("/{team_id}")
def get_team(team_id: int, db: Session = Depends(get_db)):
    """Get a single team by ID."""
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        return {"error": "not_found"}, 404
    return {
        "id": team.id,
        "name": team.canonical_name,
        "short_name": team.short_name,
        "api_football_id": team.api_football_id,
        "founded": team.founded,
        "venue_name": team.venue_name,
        "logo_url": team.logo_url,
    }