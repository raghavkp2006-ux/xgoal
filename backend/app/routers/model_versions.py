"""Model version listing — lets callers see which fitted models have logged
predictions before picking one to score, instead of pooling every version's
predictions together in aggregate (they are never comparable)."""
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ModelVersion, Prediction
from app.schemas import ModelVersionResponse

router = APIRouter(prefix="/api/v1/model-versions", tags=["model-versions"])


@router.get("", response_model=list[ModelVersionResponse])
def list_model_versions(db: Session = Depends(get_db)) -> list[ModelVersionResponse]:
    counts: dict[int, int] = {
        model_version_id: count
        for model_version_id, count in db.query(
            Prediction.model_version_id, func.count(Prediction.id)
        ).group_by(Prediction.model_version_id).all()
    }
    versions = db.query(ModelVersion).order_by(ModelVersion.trained_at.desc()).all()
    return [
        ModelVersionResponse(
            id=version.id,
            name=version.name,
            version=version.version,
            is_production=version.is_production,
            trained_at=version.trained_at,
            train_start=version.train_start,
            train_end=version.train_end,
            n_train_matches=version.n_train_matches,
            prediction_count=counts.get(version.id, 0),
        )
        for version in versions
    ]
