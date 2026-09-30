"""Teams API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Match, Season, Team

router = APIRouter(prefix="/api/v1/teams", tags=["teams"])


@router.get("")
def list_teams(
    season: int | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    """List all teams, optionally filtered by season."""
    query = db.query(Team).order_by(Team.canonical_name)
    if season is not None:
        # Part 5 uses the season's starting year, not the database season ID.
        home_teams = (
            select(Match.home_team_id)
            .join(Season, Match.season_id == Season.id)
            .where(Season.start_year == season)
        )
        away_teams = (
            select(Match.away_team_id)
            .join(Season, Match.season_id == Season.id)
            .where(Season.start_year == season)
        )
        query = query.filter(Team.id.in_(home_teams.union(away_teams)))
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
def get_team(
    team_id: int, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Get a single team by ID."""
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return {
        "id": team.id,
        "name": team.canonical_name,
        "short_name": team.short_name,
        "api_football_id": team.api_football_id,
        "founded": team.founded,
        "venue_name": team.venue_name,
        "logo_url": team.logo_url,
    }
