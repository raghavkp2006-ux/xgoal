#!/usr/bin/env python3
"""Walk-forward training and registration of the Phase 2 models.

Runs the walk-forward protocol from :mod:`app.ml.evaluation` over the requested
test seasons, then fits the final models on *all* completed matches, writes their
artifacts under ``ml/artifacts/`` and records both in ``model_versions``:

* ``dixon_coles`` — the primary model, promoted to production
* ``xgboost``     — the comparison model, recorded but never production

Re-running with the same ``--version`` refreshes that row and writes a fresh
timestamped artifact, so the job is safe to put on a schedule.

Usage:
    python -m jobs.train_model
    python -m jobs.train_model --test-seasons 2023/24,2024/25 --version v2
    python -m jobs.train_model --skip-eval --no-xgboost
"""
import argparse
import os
import sys
from datetime import date, datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import get_competition, load_labeled_matches, load_team_names  # noqa: E402
from app.ml.dixon_coles import DixonColesModel  # noqa: E402
from app.ml.evaluation import walk_forward  # noqa: E402
from app.ml.features import FEATURE_NAMES  # noqa: E402
from app.ml.store import (  # noqa: E402
    artifact_reference,
    upsert_model_version,
    write_artifact,
)
from app.ml.xgb_model import XGBoostModel  # noqa: E402

MODEL_NAME = "dixon_coles"
COMPARISON_NAME = "xgboost"
DEFAULT_TEST_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25")
MIN_TRAIN_MATCHES = 500


def parse_seasons(raw: str | None) -> list[str]:
    """Comma-separated season labels, or the README's five-season default."""
    if not raw:
        return list(DEFAULT_TEST_SEASONS)
    return [part.strip() for part in raw.split(",") if part.strip()]


def register_model(
    db: Session,
    *,
    name: str,
    version: str,
    model: Any,
    train_start: date,
    train_end: date,
    eval_metrics: dict[str, Any],
    is_production: bool,
) -> str:
    """Write the artifact, upsert the ``model_versions`` row, return a log line."""
    trained_at = datetime.now(timezone.utc)
    path = write_artifact(name, version, model.to_artifact(), trained_at=trained_at)
    row = upsert_model_version(
        db,
        name=name,
        version=version,
        train_start=train_start,
        train_end=train_end,
        n_train_matches=int(model.n_train_matches),
        hyperparameters=model.hyperparameters(),
        eval_metrics=eval_metrics,
        artifact_path=artifact_reference(path),
        is_production=is_production,
    )
    return (
        f"{row.name} {row.version} (id {row.id}) "
        f"n_train={row.n_train_matches} -> {row.artifact_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and register the Phase 2 models.")
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument("--test-seasons", default=None, help="comma-separated season labels")
    parser.add_argument(
        "--half-life", type=float, default=540.0, help="time-decay half-life in days"
    )
    parser.add_argument("--max-goals", type=int, default=10, help="score matrix size - 1")
    parser.add_argument("--window", type=int, default=6, help="rolling feature window")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--version", default="v1", help="model version label")
    parser.add_argument("--no-xgboost", action="store_true", help="skip the comparison model")
    parser.add_argument("--skip-eval", action="store_true", help="skip walk-forward evaluation")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        names = load_team_names(db)
        labels, matches = load_labeled_matches(db, competition.id, names)

        print("=" * 78)
        print(f"  XGoal model training - {competition.code} ({competition.name})")
        print("=" * 78)
        print(f"  completed matches: {len(matches)}   seasons: {len(set(labels))}")
        if len(matches) < MIN_TRAIN_MATCHES:
            raise SystemExit(
                f"need at least {MIN_TRAIN_MATCHES} completed matches - run ingestion first"
            )

        result = None
        if not args.skip_eval:
            seasons = parse_seasons(args.test_seasons)
            known = set(labels)
            unknown = [season for season in seasons if season not in known]
            if unknown:
                print(f"  ! season label(s) with no data: {unknown}")
            result = walk_forward(
                labels,
                matches,
                seasons,
                half_life_days=args.half_life,
                max_goals=args.max_goals,
                random_seed=args.seed,
                window=args.window,
                include_xgboost=not args.no_xgboost,
            )
            print(
                f"\n  walk-forward: {len(result.test_seasons)} season(s), "
                f"{result.n_test_matches} test matches (primary metric: log loss)"
            )
            for line in result.summary_lines():
                print(line)
            print(f"\n  best model: {result.best()}")
            print("\n  per-season metrics (log loss / rps / brier / ece / n)")
            for line in result.table_lines():
                print(line)
            print("  per-season log loss (compact)")
            for line in result.season_lines():
                print(line)
            print("\n  per-season pick accuracy (plan target band 51-54%)")
            for line in result.accuracy_lines():
                print(line)
            print("\n  Phase 2 acceptance targets")
            for line in result.target_lines():
                print(line)

        eval_metrics: dict[str, Any] = (
            result.to_metrics_payload() if result is not None else {"protocol": "none"}
        )
        train_start = min(match.kickoff for match in matches).date()
        train_end = max(match.kickoff for match in matches).date()

        print("\n  fitting the production Dixon-Coles model on all completed matches ...")
        dc_model = DixonColesModel(
            half_life_days=args.half_life, max_goals=args.max_goals
        ).fit(matches)
        dc_metrics = dict(eval_metrics)
        dc_metrics["final_fit"] = {
            "n_train_matches": dc_model.n_train_matches,
            "n_teams": len(dc_model.attack),
            "home_adv": round(dc_model.home_adv, 5),
            "rho": round(dc_model.rho, 5),
            "log_likelihood": round(dc_model.log_likelihood, 4),
        }
        print(
            "  "
            + register_model(
                db,
                name=MODEL_NAME,
                version=args.version,
                model=dc_model,
                train_start=train_start,
                train_end=train_end,
                eval_metrics=dc_metrics,
                is_production=True,
            )
        )
        print(
            f"  home_adv={dc_model.home_adv:.3f}  rho={dc_model.rho:.4f}  "
            f"teams={len(dc_model.attack)}  log-likelihood={dc_model.log_likelihood:.1f}"
        )

        if not args.no_xgboost:
            print("\n  fitting the XGBoost comparison model on all completed matches ...")
            booster = XGBoostModel(random_seed=args.seed, window=args.window).fit(matches)
            xgb_metrics = dict(eval_metrics)
            importances = booster.feature_importance(top=len(FEATURE_NAMES))
            xgb_metrics["feature_importance"] = importances
            print(
                "  "
                + register_model(
                    db,
                    name=COMPARISON_NAME,
                    version=args.version,
                    model=booster,
                    train_start=train_start,
                    train_end=train_end,
                    eval_metrics=xgb_metrics,
                    is_production=False,
                )
            )
            top = ", ".join(f"{key}={value:.3f}" for key, value in list(importances.items())[:5])
            print(f"  top features: {top}")
    finally:
        db.close()

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
