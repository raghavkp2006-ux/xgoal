#!/usr/bin/env python3
"""Pre-matchday prediction writer — the job that fills the ``predictions`` table.

Two modes:

``--upcoming`` (default)
    Predict every not-yet-played fixture in the next ``--days`` days, using a
    model fitted on every completed match up to the run time. The cut written to
    ``predictions.as_of`` is the current hour, so re-running the job in a later
    hour adds a new observation instead of overwriting the old one.

``--backfill N``
    Write point-in-time predictions for the last N completed fixtures. The model
    is fitted once on everything that kicked off before the window, so no
    fixture in the window ever informs another one (deliberately conservative,
    and cheap enough to run inside a cron job). That fit is registered as
    ``backfill-YYYYMMDD`` and stays non-production, so a scheduled backfill never
    demotes the trained production model; pass ``--version`` to pin a label
    instead (which does promote it).

Usage:
    python -m jobs.predict_matchday
    python -m jobs.predict_matchday --days 14
    python -m jobs.predict_matchday --backfill 380
    python -m jobs.predict_matchday --backfill 380 --version v1 --quiet
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import (  # noqa: E402
    get_competition,
    load_completed_matches,
    load_recent_completed,
    load_team_names,
    load_upcoming_matches,
)
from app.ml.dixon_coles import DixonColesModel  # noqa: E402
from app.ml.store import (  # noqa: E402
    PendingPrediction,
    artifact_reference,
    feature_hash,
    read_artifact,
    upsert_model_version,
    write_artifact,
    write_predictions,
)
from app.models import Competition, ModelVersion  # noqa: E402

MODEL_NAME = "dixon_coles"


def register_version(
    db: Session,
    model: DixonColesModel,
    version: str,
    is_production: bool = True,
) -> ModelVersion:
    """Persist the artifact and the ``model_versions`` row for a fitted model."""
    trained_at = datetime.now(timezone.utc)
    path = write_artifact(MODEL_NAME, version, model.to_artifact(), trained_at=trained_at)
    train_start: date = model.train_start.date() if model.train_start else date(2000, 1, 1)
    train_end: date = model.train_end.date() if model.train_end else date(2000, 1, 1)
    return upsert_model_version(
        db,
        name=MODEL_NAME,
        version=version,
        train_start=train_start,
        train_end=train_end,
        n_train_matches=model.n_train_matches,
        hyperparameters=model.hyperparameters(),
        eval_metrics={
            "trained_at": trained_at.isoformat(),
            "n_train_matches": model.n_train_matches,
            "home_adv": round(model.home_adv, 5),
            "rho": round(model.rho, 5),
            "log_likelihood": round(model.log_likelihood, 4),
        },
        artifact_path=artifact_reference(path),
        is_production=is_production,
    )


def load_version_model(db: Session, version: str | None) -> tuple[DixonColesModel, ModelVersion]:
    """Load a fitted model from ``model_versions`` (production, or the given version)."""
    query = db.query(ModelVersion).filter(ModelVersion.name == MODEL_NAME)
    if version:
        query = query.filter(ModelVersion.version == version)
    else:
        query = query.filter(ModelVersion.is_production.is_(True))
    row = query.order_by(ModelVersion.trained_at.desc()).first()
    if row is None:
        row = (
            db.query(ModelVersion)
            .filter(ModelVersion.name == MODEL_NAME)
            .order_by(ModelVersion.trained_at.desc())
            .first()
        )
    if row is None:
        raise SystemExit(
            "no model_versions row found - run jobs.train_model or "
            "jobs.predict_matchday --retrain first"
        )
    payload = read_artifact(row.artifact_path)
    if payload.get("model") != "dixon_coles":
        raise SystemExit(f"artifact {row.artifact_path} is not a Dixon-Coles model")
    return DixonColesModel.from_artifact(payload), row


def predict_upcoming(
    db: Session,
    competition: Competition,
    names: dict[int, str],
    days: int,
    version: str | None,
    retrain: bool,
    quiet: bool,
) -> int:
    """Predict the next ``days`` days of unplayed fixtures for a competition."""
    now = datetime.now(timezone.utc)
    as_of = now.replace(minute=0, second=0, microsecond=0)
    fixtures = load_upcoming_matches(db, competition.id, now + timedelta(days=days))
    if not fixtures:
        print(f"  no unplayed fixtures for {competition.code} in the next {days} day(s)")
        return 0

    if retrain:
        history = load_completed_matches(db, competition.id, names, before=as_of)
        if not history:
            print("  no completed matches to train on")
            return 0
        model = DixonColesModel().fit(history)
        version_row = register_version(db, model, version or "v1")
        print(
            f"  retrained on {len(history)} matches -> "
            f"{version_row.name} {version_row.version} (id {version_row.id})"
        )
    else:
        model, version_row = load_version_model(db, version)
        print(
            f"  using {version_row.name} {version_row.version} "
            f"({version_row.n_train_matches} training matches)"
        )

    fixtures_by_id = {fixture.id: fixture for fixture in fixtures}
    pending: list[PendingPrediction] = []
    for fixture in fixtures:
        home = names[fixture.home_team_id]
        away = names[fixture.away_team_id]
        prediction = model.predict(home, away)
        pending.append(
            PendingPrediction(
                match_id=fixture.id,
                prediction=prediction,
                feature_hash=feature_hash(
                    {
                        "model": MODEL_NAME,
                        "version": version_row.version,
                        "home": home,
                        "away": away,
                        "as_of": as_of.isoformat(),
                    }
                ),
            )
        )
        if not quiet:
            print(
                f"  #{fixture.id:<6} {fixture.kickoff_utc:%Y-%m-%d %H:%M} "
                f"{home[:22]:<22} vs {away[:22]:<22} "
                f"H {prediction.p_home:.3f}  D {prediction.p_draw:.3f}  "
                f"A {prediction.p_away:.3f}  xG {prediction.exp_home_goals:.2f}-"
                f"{prediction.exp_away_goals:.2f}"
            )

    written = write_predictions(db, version_row, pending, as_of=as_of)
    print(f"  as_of cut {as_of.isoformat()} -> {written} row(s) for "
          f"{len(fixtures_by_id)} fixture(s)")
    return written


def backfill_predictions(
    db: Session,
    competition: Competition,
    names: dict[int, str],
    count: int,
    version: str | None,
    quiet: bool,
) -> int:
    """Point-in-time predictions for the last ``count`` finished fixtures."""
    window = load_recent_completed(db, competition.id, count)
    if not window:
        print("  no completed fixtures to backfill")
        return 0
    window_start = window[0].kickoff_utc
    history = load_completed_matches(db, competition.id, names, before=window_start)
    if not history:
        print(f"  no history available before {window_start:%Y-%m-%d}")
        return 0

    model = DixonColesModel().fit(history)
    label = version or f"backfill-{window_start:%Y%m%d}"
    version_row = register_version(db, model, label, is_production=version is not None)
    print(
        f"  fitted on {len(history)} matches before {window_start:%Y-%m-%d} -> "
        f"{version_row.name} {version_row.version} (id {version_row.id}, "
        f"{'production' if version_row.is_production else 'non-production'})"
    )

    grouped: dict[datetime, list[PendingPrediction]] = defaultdict(list)
    for fixture in window:
        home = names[fixture.home_team_id]
        away = names[fixture.away_team_id]
        cut = fixture.kickoff_utc.replace(minute=0, second=0, microsecond=0)
        prediction = model.predict(home, away)
        grouped[cut].append(
            PendingPrediction(
                match_id=fixture.id,
                prediction=prediction,
                feature_hash=feature_hash(
                    {
                        "model": MODEL_NAME,
                        "version": version_row.version,
                        "home": home,
                        "away": away,
                        "as_of": cut.isoformat(),
                    }
                ),
            )
        )
        if not quiet:
            actual = f"{fixture.home_goals}-{fixture.away_goals}"
            print(
                f"  #{fixture.id:<6} {fixture.kickoff_utc:%Y-%m-%d} "
                f"{home[:20]:<20} vs {away[:20]:<20} actual {actual:<5} "
                f"H {prediction.p_home:.3f}  D {prediction.p_draw:.3f}  "
                f"A {prediction.p_away:.3f}"
            )

    written = sum(
        write_predictions(db, version_row, items, as_of=cut)
        for cut, items in grouped.items()
    )
    print(f"  {written} row(s) across {len(grouped)} kick-off cut(s)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write La Liga predictions for a model version."
    )
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument("--days", type=int, default=7, help="upcoming horizon in days")
    parser.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="write predictions for the last N completed fixtures instead",
    )
    parser.add_argument("--version", default=None, help="model version (default: production)")
    parser.add_argument("--retrain", action="store_true", help="refit before predicting")
    parser.add_argument("--quiet", action="store_true", help="suppress per-fixture output")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        names = load_team_names(db)
        print("=" * 78)
        print(f"  XGoal prediction writer - {competition.code} ({competition.name})")
        print("=" * 78)
        if args.backfill:
            written = backfill_predictions(
                db, competition, names, args.backfill, args.version, args.quiet
            )
        else:
            written = predict_upcoming(
                db, competition, names, args.days, args.version, args.retrain, args.quiet
            )
    finally:
        db.close()

    print(f"\n  predictions written: {written}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
