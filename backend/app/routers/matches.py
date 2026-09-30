"""Matches, match events, and player stats API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from app.database import get_db
from app.models import Match, Player, PlayerSeasonStat
from app.schemas import (
    MatchResponse,
    PlayerResponse,
)

router = APIRouter(prefix="/api/v1/matches", tags=["matches"])


# ── Matches ───────────────────────────────────────────────────────────────

@router.get("", response_model=list[MatchResponse])
def list_matches(
    season_id: int | None = None,
    team_id: int | None = None,
    status: str | None = Query(None, max_length=10),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Match]:
    q = db.query(Match)
    if season_id:
        q = q.filter(Match.season_id == season_id)
    if team_id:
        q = q.filter((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
    if status:
        q = q.filter(Match.status == status)
    return q.order_by(Match.kickoff_utc.desc()).limit(limit).all()


@router.get("/{match_id}", response_model=MatchResponse)
def get_match(match_id: int, db: Session = Depends(get_db)) -> Match:
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, detail="Match not found")
    return match


# ── Players ───────────────────────────────────────────────────────────────

@router.get("/players/list", response_model=list[PlayerResponse])
def list_players(
    season_id: int | None = None,
    team_id: int | None = None,
    db: Session = Depends(get_db),
) -> list[Player]:
    q = db.query(Player)
    if season_id or team_id:
        filters: list[ColumnElement[bool]] = []
        if season_id is not None:
            filters.append(PlayerSeasonStat.season_id == season_id)
        if team_id is not None:
            filters.append(PlayerSeasonStat.team_id == team_id)
        q = q.filter(Player.id.in_(db.query(PlayerSeasonStat.player_id).filter(*filters)))
    return q.order_by(Player.name).all()


@router.get("/players/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, db: Session = Depends(get_db)) -> Player:
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(404, detail="Player not found")
    return player
