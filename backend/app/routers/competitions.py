"""Competitions and seasons API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Competition, Season
from app.schemas import CompetitionResponse, SeasonResponse

router = APIRouter(prefix="/api/v1/competitions", tags=["competitions"])


# ── Competitions ──────────────────────────────────────────────────────────

@router.get("", response_model=list[CompetitionResponse])
def list_competitions(db: Session = Depends(get_db)) -> list[Competition]:
    return db.query(Competition).order_by(Competition.tier, Competition.name).all()


@router.get("/{competition_id}", response_model=CompetitionResponse)
def get_competition(competition_id: int, db: Session = Depends(get_db)) -> Competition:
    comp = db.query(Competition).filter(Competition.id == competition_id).first()
    if not comp:
        raise HTTPException(404, detail="Competition not found")
    return comp


# ── Seasons ───────────────────────────────────────────────────────────────

@router.get("/{competition_id}/seasons", response_model=list[SeasonResponse])
def list_seasons(competition_id: int, db: Session = Depends(get_db)) -> list[Season]:
    return (
        db.query(Season)
        .filter(Season.competition_id == competition_id)
        .order_by(Season.start_year.desc())
        .all()
    )
