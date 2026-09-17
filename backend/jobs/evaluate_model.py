#!/usr/bin/env python3
"""Model scoreboard — compare the registered models on the walk-forward metrics.

``jobs/train_model.py`` stores the whole walk-forward payload in
``model_versions.eval_metrics``; this job is the read-only side of it: it ranks
the forecasters, prints the per-season log loss and, with ``--rerun``, replays
the protocol live (handy for a version registered with ``--skip-eval``).

Usage:
    python -m jobs.evaluate_model
    python -m jobs.evaluate_model --version v1
    python -m jobs.evaluate_model --rerun --test-seasons 2023/24,2024/25 --no-xgboost
"""
import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import get_competition, load_labeled_matches, load_team_names  # noqa: E402
from app.ml.evaluation import PRIMARY_METRIC, walk_forward  # noqa: E402
from app.ml.features import FEATURE_NAMES  # noqa: E402
from app.models import ModelVersion  # noqa: E402

FALLBACK_SEASONS = 5


def load_versions(db: Session, version: str | None) -> list[ModelVersion]:
    """Every registered model version, optionally filtered to one label."""
    query = db.query(ModelVersion)
    if version:
        query = query.filter(ModelVersion.version == version)
    return query.order_by(ModelVersion.name, ModelVersion.version).all()


def score_lines(pooled: dict[str, Any], coverage: dict[str, Any]) -> list[str]:
    """Ranking table for the pooled walk-forward metrics."""
    lines = [
        f"      {'model':<12} {'log_loss':>9} {'rps':>7} {'brier':>7} {'ece':>7} {'n':>6}",
        "      " + "-" * 52,
    ]
    for name, metrics in sorted(pooled.items(), key=lambda item: item[1][PRIMARY_METRIC]):
        lines.append(
            f"      {name:<12} {metrics['log_loss']:>9.4f} {metrics['rps']:>7.4f} "
            f"{metrics['brier']:>7.4f} {metrics['ece']:>7.4f} "
            f"{coverage.get(name, 0):>6d}"
        )
    return lines


def season_lines(per_season: dict[str, Any], best: str) -> list[str]:
    """Per-season log loss: the season winner plus our primary model's score."""
    lines = [f"      {'season':<9} {'winner':<12} {'log loss':>9}   {best}", "      " + "-" * 50]
    for season in sorted(per_season):
        models = per_season[season]
        winner, metrics = min(models.items(), key=lambda item: item[1][PRIMARY_METRIC])
        ours = models.get(best, {}).get(PRIMARY_METRIC, float("nan"))
        lines.append(
            f"      {season:<9} {winner:<12} {metrics[PRIMARY_METRIC]:>9.4f}   {ours:.4f}"
        )
    return lines


def version_lines(row: ModelVersion) -> list[str]:
    """Everything stored about one registered model version."""
    payload: dict[str, Any] = row.eval_metrics or {}
    pooled: dict[str, Any] = payload.get("pooled") or {}
    coverage: dict[str, Any] = payload.get("coverage") or {}
    lines = [
        f"  {row.name} {row.version}  (id {row.id}, "
        f"{'production' if row.is_production else 'non-production'})",
        f"      git={row.git_sha}  seed={row.random_seed}",
        f"      trained_at={row.trained_at:%Y-%m-%d %H:%M:%S}  "
        f"train {row.train_start} .. {row.train_end}",
        f"      n_train={row.n_train_matches}",
        f"      artifact={row.artifact_path}",
    ]
    if not pooled:
        lines.append("      (no walk-forward metrics stored)")
        return lines
    lines.append(
        f"      protocol={payload.get('protocol')} "
        f"primary={payload.get('primary_metric')} "
        f"test_seasons={','.join(payload.get('test_seasons') or [])} "
        f"n_test={payload.get('n_test_matches')}"
    )
    lines.extend(score_lines(pooled, coverage))
    best = min(pooled.items(), key=lambda item: item[1][PRIMARY_METRIC])[0]
    lines.append(f"      best by {PRIMARY_METRIC}: {best}")
    lines.extend(season_lines(payload.get("per_season") or {}, best))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare registered model versions.")
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument("--version", default=None, help="only this version label")
    parser.add_argument("--rerun", action="store_true", help="replay the walk-forward live")
    parser.add_argument("--test-seasons", default=None, help="comma-separated season labels")
    parser.add_argument("--no-xgboost", action="store_true", help="skip XGBoost in --rerun")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        rows = load_versions(db, args.version)
        print("=" * 78)
        print(f"  XGoal model scoreboard - {competition.code} ({competition.name})")
        print("=" * 78)
        if not rows:
            raise SystemExit("no model versions registered - run jobs.train_model first")
        for row in rows:
            for line in version_lines(row):
                print(line)
            print()

        if args.rerun:
            names = load_team_names(db)
            labels, matches = load_labeled_matches(db, competition.id, names)
            seasons = [
                part.strip() for part in (args.test_seasons or "").split(",") if part.strip()
            ]
            if not seasons:
                seasons = sorted(set(labels))[-FALLBACK_SEASONS:]
            print(f"  replaying the walk-forward protocol over {', '.join(seasons)} ...")
            result = walk_forward(
                labels,
                matches,
                seasons,
                include_xgboost=not args.no_xgboost,
            )
            print(
                f"  {result.n_test_matches} test matches, "
                f"{len(FEATURE_NAMES)} XGBoost features"
            )
            for line in result.summary_lines():
                print(line)
            print()
            for line in result.season_lines():
                print(line)
            print("\n  per-season metrics (log loss / rps / brier / ece / n)")
            for line in result.table_lines():
                print(line)
            print("\n  per-season pick accuracy (plan target band 51-54%)")
            for line in result.accuracy_lines():
                print(line)
            print("\n  Phase 2 acceptance targets")
            for line in result.target_lines():
                print(line)
            print(f"\n  best model: {result.best()}")
    finally:
        db.close()

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
