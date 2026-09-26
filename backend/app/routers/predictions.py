"""Stored and on-demand match prediction API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Match, ModelVersion, Prediction, Team
from app.schemas import HypotheticalPredictionRequest, PredictionResponse
from jobs.predict_matchday import load_version_model

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.get("/{match_id}", response_model=PredictionResponse)
def get_prediction(match_id: int, db: Session = Depends(get_db)):
    """Return the newest stored prediction for a fixture."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, detail="Match not found")

    prediction = (
        db.query(Prediction)
        .join(ModelVersion)
        .filter(Prediction.match_id == match_id)
        .order_by(Prediction.as_of.desc(), Prediction.predicted_at.desc())
        .first()
    )
    if not prediction:
        raise HTTPException(404, detail=f"No prediction found for match {match_id}")

    return _stored_response(prediction)


@router.post("/hypothetical", response_model=PredictionResponse)
def create_hypothetical_prediction(
    body: HypotheticalPredictionRequest,
    db: Session = Depends(get_db),
):
    """Generate, but do not persist, a forecast for an arbitrary pairing."""
    if body.home_team_id == body.away_team_id:
        raise HTTPException(400, detail="home_team_id and away_team_id must differ")

    home_team = db.query(Team).filter(Team.id == body.home_team_id).first()
    if not home_team:
        raise HTTPException(404, detail="Home team not found")
    away_team = db.query(Team).filter(Team.id == body.away_team_id).first()
    if not away_team:
        raise HTTPException(404, detail="Away team not found")

    try:
        model, version = load_version_model(db, version=None)
    except SystemExit as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    prediction = model.predict(home_team.canonical_name, away_team.canonical_name)
    return PredictionResponse(
        model_name=version.name,
        model_version=version.version,
        as_of=body.as_of or datetime.now(timezone.utc),
        p_home=prediction.p_home,
        p_draw=prediction.p_draw,
        p_away=prediction.p_away,
        expected_home_goals=prediction.exp_home_goals,
        expected_away_goals=prediction.exp_away_goals,
        score_matrix={"max_goals": prediction.max_goals, "cells": prediction.matrix},
        is_hypothetical=True,
    )


def _stored_response(prediction: Prediction) -> PredictionResponse:
    """Map a persisted prediction and its joined model version to the public shape."""
    return PredictionResponse(
        match_id=prediction.match_id,
        model_name=prediction.model_version.name,
        model_version=prediction.model_version.version,
        as_of=prediction.as_of,
        p_home=float(prediction.p_home),
        p_draw=float(prediction.p_draw),
        p_away=float(prediction.p_away),
        expected_home_goals=(
            float(prediction.expected_home_goals)
            if prediction.expected_home_goals is not None
            else None
        ),
        expected_away_goals=(
            float(prediction.expected_away_goals)
            if prediction.expected_away_goals is not None
            else None
        ),
        score_matrix=prediction.score_matrix,
    )
