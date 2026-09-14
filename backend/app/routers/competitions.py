"""Competitions and seasons API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Competition, Season
from app.schemas import CompetitionCreate, CompetitionResponse, SeasonCreate, SeasonResponse

router = APIRouter(prefix="/api/v1/competitions", tags=["competitions"])


# ── Competitions ──────────────────────────────────────────────────────────

@router.get("", response_model=list[CompetitionResponse])
def list_competitions(db: Session = Depends(get_db)):
    return db.query(Competition).order_by(Competition.tier, Competition.name).all()


@router.get("/{competition_id}", response_model=CompetitionResponse)
def get_competition(competition_id: int, db: Session = Depends(get_db)):
    comp = db.query(Competition).filter(Competition.id == competition_id).first()
    if not comp:
        raise HTTPException(404, detail="Competition not found")
    return comp


@router.post("", response_model=CompetitionResponse, status_code=201)
def create_competition(body: CompetitionCreate, db: Session = Depends(get_db)):
    existing = db.query(Competition).filter(
        (Competition.code == body.code) | (Competition.api_football_id == body.api_football_id)
    ).first()
    if existing:
        raise HTTPException(409, detail="Competition already exists")
    comp = Competition(**body.model_dump())
    db.add(comp)
    db.commit()
    db.refresh(comp)
    return comp


# ── Seasons ───────────────────────────────────────────────────────────────

@router.get("/{competition_id}/seasons", response_model=list[SeasonResponse])
def list_seasons(competition_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Season)
        .filter(Season.competition_id == competition_id)
        .order_by(Season.start_year.desc())
        .all()
    )


@router.post("/{competition_id}/seasons", response_model=SeasonResponse, status_code=201)
def create_season(competition_id: int, body: SeasonCreate, db: Session = Depends(get_db)):
    comp = db.query(Competition).filter(Competition.id == competition_id).first()
    if not comp:
        raise HTTPException(404, detail="Competition not found")
    if body.competition_id != competition_id:
        raise HTTPException(400, detail="competition_id mismatch")
    season = Season(**body.model_dump())
    db.add(season)
    db.commit()
    db.refresh(season)
    return season