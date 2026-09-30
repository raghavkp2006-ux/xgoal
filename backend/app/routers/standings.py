"""Standings snapshots API endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import StandingsSnapshot
from app.schemas import StandingsSnapshotResponse

router = APIRouter(prefix="/api/v1/standings", tags=["standings"])


@router.get("", response_model=list[StandingsSnapshotResponse])
def list_standings(
    season_id: int,
    matchday: int | None = None,
    db: Session = Depends(get_db),
) -> list[StandingsSnapshot]:
    subq = (
        db.query(StandingsSnapshot.season_id, StandingsSnapshot.computed_at)
        .filter(StandingsSnapshot.season_id == season_id)
        .order_by(StandingsSnapshot.computed_at.desc())
        .limit(1)
        .subquery()
    )
    q = db.query(StandingsSnapshot).filter(
        StandingsSnapshot.season_id == subq.c.season_id,
        StandingsSnapshot.computed_at == subq.c.computed_at,
    )
    if matchday:
        q = q.filter(StandingsSnapshot.as_of_matchday == matchday)
    return q.order_by(StandingsSnapshot.position).all()
