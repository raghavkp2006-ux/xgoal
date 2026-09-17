"""Persistence helpers for model versions, artifacts and predictions.

Artifacts are written to the repo-root ``ml/artifacts/`` directory (already
gitignored), so any environment that can see the checkout loads the same
model; the database stores only the relative path.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy.orm import Session

from app.ml.data import ScorelinePrediction
from app.models import ModelVersion, Prediction, SimulationRun

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"


@dataclass(frozen=True)
class PendingPrediction:
    """A prediction that is ready to be written to the ``predictions`` table."""

    match_id: int
    prediction: ScorelinePrediction
    feature_hash: str


def feature_hash(parts: Any) -> str:
    """Stable 16-char digest of any JSON-describable payload (an audit key)."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def git_sha() -> str:
    """Short SHA of HEAD, or ``"unknown"`` outside a git checkout."""
    env_sha = os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_artifact(
    name: str,
    version: str,
    payload: Mapping[str, Any],
    trained_at: datetime | None = None,
) -> Path:
    """Write ``payload`` as JSON under ``ml/artifacts`` and return the file path."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = (trained_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACT_DIR / f"{name}_{version}_{stamp}.json"
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def artifact_reference(path: Path) -> str:
    """Repo-relative POSIX path, i.e. what goes into ``artifact_path``."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_artifact(reference: str) -> dict[str, Any]:
    """Load an artifact by repo-relative or absolute path."""
    path = Path(reference)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact {path} must contain a JSON object")
    return payload


def upsert_model_version(
    db: Session,
    *,
    name: str,
    version: str,
    train_start: date,
    train_end: date,
    n_train_matches: int,
    hyperparameters: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    artifact_path: str,
    random_seed: int = 42,
    is_production: bool = False,
) -> ModelVersion:
    """Create or refresh the ``model_versions`` row for ``(name, version)``."""
    now = datetime.now(timezone.utc)
    row = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == name, ModelVersion.version == version)
        .one_or_none()
    )
    if row is None:
        row = ModelVersion(
            name=name,
            version=version,
            git_sha=git_sha(),
            random_seed=random_seed,
            trained_at=now,
            train_start=train_start,
            train_end=train_end,
            n_train_matches=n_train_matches,
            hyperparameters=dict(hyperparameters),
            eval_metrics=dict(eval_metrics),
            artifact_path=artifact_path,
            is_production=is_production,
        )
        db.add(row)
    else:
        row.git_sha = git_sha()
        row.random_seed = random_seed
        row.trained_at = now
        row.train_start = train_start
        row.train_end = train_end
        row.n_train_matches = n_train_matches
        row.hyperparameters = dict(hyperparameters)
        row.eval_metrics = dict(eval_metrics)
        row.artifact_path = artifact_path
        row.is_production = is_production
    if is_production:
        (
            db.query(ModelVersion)
            .filter(ModelVersion.name == name, ModelVersion.version != version)
            .update({ModelVersion.is_production: False})
        )
    db.commit()
    db.refresh(row)
    return row


def _rounded_probs(prediction: ScorelinePrediction) -> tuple[float, float, float]:
    """Round to the 5 decimals of ``predictions.p_*`` and rebalance the residual.

    The table has a CHECK constraint that the three probabilities sum to 1
    within 0.001, so any rounding residual is pushed onto the largest class.
    """
    values = [round(value, 5) for value in prediction.probs]
    residual = round(1.0 - sum(values), 5)
    largest = max(range(len(values)), key=lambda index: values[index])
    values[largest] = round(values[largest] + residual, 5)
    return (values[0], values[1], values[2])


def write_predictions(
    db: Session,
    model_version: ModelVersion,
    pending: Sequence[PendingPrediction],
    as_of: datetime,
    predicted_at: datetime | None = None,
) -> int:
    """Insert or refresh ``predictions`` rows for one version at one ``as_of`` cut."""
    stamp = predicted_at or datetime.now(timezone.utc)
    written = 0
    for item in pending:
        row = (
            db.query(Prediction)
            .filter(
                Prediction.match_id == item.match_id,
                Prediction.model_version_id == model_version.id,
                Prediction.as_of == as_of,
            )
            .one_or_none()
        )
        if row is None:
            row = Prediction(
                match_id=item.match_id,
                model_version_id=model_version.id,
                as_of=as_of,
                predicted_at=stamp,
                feature_hash=item.feature_hash,
                p_home=0.0,
                p_draw=0.0,
                p_away=0.0,
            )
            db.add(row)
        p_home, p_draw, p_away = _rounded_probs(item.prediction)
        row.predicted_at = stamp
        row.p_home = Decimal(str(p_home))
        row.p_draw = Decimal(str(p_draw))
        row.p_away = Decimal(str(p_away))
        row.expected_home_goals = Decimal(str(round(item.prediction.exp_home_goals, 2)))
        row.expected_away_goals = Decimal(str(round(item.prediction.exp_away_goals, 2)))
        row.score_matrix = {
            "max_goals": item.prediction.max_goals,
            "cells": item.prediction.matrix,
        }
        row.feature_hash = item.feature_hash
        written += 1
    db.commit()
    return written


def write_simulation_run(
    db: Session,
    *,
    season_id: int,
    model_version_id: int,
    n_simulations: int,
    random_seed: int,
    as_of_matchday: int,
    results: Mapping[str, Any],
    forced_results: Mapping[str, Any] | None = None,
    run_at: datetime | None = None,
) -> SimulationRun:
    """Append a Monte Carlo row (history is kept; newest wins by ``run_at``)."""
    row = SimulationRun(
        season_id=season_id,
        model_version_id=model_version_id,
        run_at=run_at or datetime.now(timezone.utc),
        n_simulations=n_simulations,
        random_seed=random_seed,
        as_of_matchday=as_of_matchday,
        forced_results=dict(forced_results) if forced_results is not None else None,
        results=dict(results),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
