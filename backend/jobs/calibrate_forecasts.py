#!/usr/bin/env python3
"""Phase 2.5 — isotonic calibration of the Dixon-Coles and XGBoost forecasts.

Replays the walk-forward harness, then for every test season fits isotonic
regression on the *dedicated calibration split* (all earlier test seasons, i.e.
out-of-sample forecasts the model never trained on) and scores the current
season before and after calibration.

Reports ECE and log loss before/after per season, pooled totals per model, and
ASCII reliability diagrams; ``--save`` writes the fitted calibrators under
``ml/artifacts`` for serving.

Usage:
    python -m jobs.calibrate_forecasts
    python -m jobs.calibrate_forecasts --test-seasons 2023/24,2024/25 --no-xgboost
    python -m jobs.calibrate_forecasts --shrink 0.5 --bins 8
    python -m jobs.calibrate_forecasts --save
"""
import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402


from app.database import SessionLocal  # noqa: E402
from app.ml.calibration import IsotonicCalibrator, reliability_lines  # noqa: E402
from app.ml.data import MatchInput, Probs  # noqa: E402
from app.ml.dataset import get_competition, load_labeled_matches, load_team_names  # noqa: E402
from app.ml.evaluation import walk_forward  # noqa: E402
from app.ml.metrics import evaluate_predictions  # noqa: E402
from app.ml.store import artifact_reference, write_artifact  # noqa: E402

DEFAULT_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25")
MODEL_NAMES = ("dixon_coles", "xgboost")
SHRINK_GRID = (0.0, 0.2, 0.3, 0.5, 0.75, 1.0)
DEFAULT_SHRINK = 0.3


def parse_seasons(raw: str | None) -> list[str]:
    """Comma-separated season labels, or the five-season Phase 2 window."""
    if not raw:
        return list(DEFAULT_SEASONS)
    return [part.strip() for part in raw.split(",") if part.strip()]


def split(items: list[tuple[MatchInput, Probs]]) -> tuple[list[Probs], list[int]]:
    """(probabilities, outcome indices) of a forecast list."""
    return [probs for _, probs in items], [match.outcome for match, _ in items]


def score(
    items: list[tuple[MatchInput, Probs]], calibrated: list[Probs] | None = None
) -> dict[str, float]:
    """Metrics of a forecast list, optionally replaced by calibrated probs."""
    if calibrated is None:
        probs, outcomes = split(items)
    else:
        probs, outcomes = calibrated, [match.outcome for match, _ in items]
    return evaluate_predictions(probs, outcomes)


def table_header() -> tuple[str, str]:
    """Header and rule for the per-season calibration table."""
    header = (
        f"  {'season':<9} {'n':>5} {'cal n':>6} "
        f"{'ece before':>11} {'ece after':>10} "
        f"{'ll before':>10} {'ll after':>9}"
    )
    return header, "  " + "-" * (len(header) - 2)


def table_row(
    season: str, n_test: int, n_cal: int, before: dict[str, float], after: dict[str, float]
) -> str:
    """One per-season row of the calibration report."""
    return (
        f"  {season:<9} {n_test:>5} {n_cal:>6} "
        f"{before['ece']:>11.4f} {after['ece']:>10.4f} "
        f"{before['log_loss']:>10.4f} {after['log_loss']:>9.4f}"
    )


def pooled_row(label: str, n_test: int, before: dict[str, float], after: dict[str, float]) -> str:
    """One pooled row of the calibration report."""
    return (
        f"  {label:<9} {n_test:>5} {'-':>6} "
        f"{before['ece']:>11.4f} {after['ece']:>10.4f} "
        f"{before['log_loss']:>10.4f} {after['log_loss']:>9.4f}"
    )


def sweep_lines(
    model: str,
    seasons: list[str],
    forecasts: dict[str, dict[str, list[tuple[MatchInput, Probs]]]],
) -> list[str]:
    """ECE and log loss over the calibrated seasons for each shrinkage factor.

    ``shrink`` mixes the isotonic map back towards the raw forecast; the sweep is
    what justifies the default the job then reports in detail.
    """
    lines = [f"    {model}", "    " + "-" * 26]
    for shrink in SHRINK_GRID:
        pooled_probs: list[Probs] = []
        pooled_outcomes: list[int] = []
        for index, season in enumerate(seasons):
            items = forecasts.get(season, {}).get(model, [])
            history = [
                pair
                for earlier in seasons[:index]
                for pair in forecasts.get(earlier, {}).get(model, [])
            ]
            if not items or not history:
                continue
            calibrator = IsotonicCalibrator(shrink=shrink).fit(*split(history))
            pooled_probs.extend(calibrator.transform(split(items)[0]))
            pooled_outcomes.extend(match.outcome for match, _ in items)
        if not pooled_probs:
            continue
        metrics = evaluate_predictions(pooled_probs, pooled_outcomes)
        lines.append(
            f"    {shrink:>7.2f} {metrics['ece']:>8.4f} {metrics['log_loss']:>9.4f}"
        )
    return lines


def calibrate_model(
    model: str,
    seasons: list[str],
    forecasts: dict[str, dict[str, list[tuple[MatchInput, Probs]]]],
    bins: int,
    shrink: float,
) -> dict[str, Any]:
    """Walk-forward calibration for one model: fit on earlier seasons, score the next."""
    header, rule = table_header()
    print(f"\n  {model}: isotonic (shrink {shrink:.2f}) fitted on the earlier test seasons")
    print(header)
    print(rule)

    pooled_before: list[tuple[MatchInput, Probs]] = []
    pooled_after: list[tuple[MatchInput, Probs]] = []
    last: tuple[str, IsotonicCalibrator, list[tuple[MatchInput, Probs]]] | None = None

    for index, season in enumerate(seasons):
        items = forecasts.get(season, {}).get(model, [])
        if not items:
            continue
        history = [
            pair
            for earlier in seasons[:index]
            for pair in forecasts.get(earlier, {}).get(model, [])
        ]
        if not history:
            print(f"  {season:<9} {len(items):>5} {'-':>6} no calibration split yet")
            continue
        calibrator = IsotonicCalibrator(shrink=shrink).fit(*split(history))
        calibrated = calibrator.transform(split(items)[0])
        print(
            table_row(season, len(items), len(history), score(items), score(items, calibrated))
        )
        pooled_before.extend(items)
        pooled_after.extend(
            zip([match for match, _ in items], calibrated, strict=True)
        )
        last = (season, calibrator, items)

    print(rule)
    print(pooled_row("pooled", len(pooled_before), score(pooled_before), score(pooled_after)))

    if last is not None:
        season, calibrator, items = last
        calibrated = calibrator.transform(split(items)[0])
        outcomes = [match.outcome for match, _ in items]
        print()
        for line in reliability_lines(split(items)[0], outcomes, bins, f"{model} {season} before"):
            print(line)
        print()
        for line in reliability_lines(calibrated, outcomes, bins, f"{model} {season} after"):
            print(line)

    return {
        "model": model,
        "pooled_before": score(pooled_before),
        "pooled_after": score(pooled_after),
        "calibrator": None if last is None else last[1],
        "n_pooled": len(pooled_before),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Isotonic calibration report (Phase 2.5).")
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument("--test-seasons", default=None, help="comma-separated season labels")
    parser.add_argument("--no-xgboost", action="store_true", help="skip the comparison model")
    parser.add_argument("--bins", type=int, default=5, help="reliability-diagram bins")
    parser.add_argument(
        "--shrink",
        type=float,
        default=DEFAULT_SHRINK,
        help="how much of the isotonic map to apply (0 = raw, 1 = plain isotonic)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="write the calibrators (fitted on all but the last season) to ml/artifacts",
    )
    args = parser.parse_args()
    if args.bins < 2:
        raise SystemExit("--bins must be at least 2")
    if not 0.0 <= args.shrink <= 1.0:
        raise SystemExit("--shrink must be between 0.0 and 1.0")

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        names = load_team_names(db)
        labels, matches = load_labeled_matches(db, competition.id, names)
        seasons = parse_seasons(args.test_seasons)
        print("=" * 78)
        print(f"  XGoal calibration report - {competition.code} ({competition.name})")
        print("=" * 78)
        print(
            f"  walk-forward over {', '.join(seasons)} "
            f"(xgboost {'off' if args.no_xgboost else 'on'})"
        )
        result = walk_forward(labels, matches, seasons, include_xgboost=not args.no_xgboost)
        for line in result.summary_lines():
            print(line)

        scored_seasons = [season for season in seasons if season in result.per_season]
        available = [name for name in MODEL_NAMES if name in result.coverage]
        print("\n  shrinkage sweep (calibrated seasons only, pooled)")
        for name in available:
            for line in sweep_lines(name, scored_seasons, result.per_season_forecasts):
                print(line)

        summaries = [
            calibrate_model(
                name, scored_seasons, result.per_season_forecasts, args.bins, args.shrink
            )
            for name in available
        ]

        print("\n  === pooled effect of isotonic calibration ===")
        print(
            f"  {'model':<12} {'n':>5} {'ece before':>11} {'ece after':>10} "
            f"{'ll before':>10} {'ll after':>9}  verdict"
        )
        for summary in summaries:
            before, after = summary["pooled_before"], summary["pooled_after"]
            ece_better = after["ece"] < before["ece"]
            ll_better = after["log_loss"] < before["log_loss"]
            verdict = (
                "helps"
                if ece_better and ll_better
                else ("mixed" if ece_better or ll_better else "no help")
            )
            print(
                f"  {summary['model']:<12} {summary['n_pooled']:>5} "
                f"{before['ece']:>11.4f} {after['ece']:>10.4f} "
                f"{before['log_loss']:>10.4f} {after['log_loss']:>9.4f}  {verdict}"
            )

        if args.save:
            print("\n  saved calibrators (fitted on every season but the last)")
            for summary in summaries:
                calibrator = summary["calibrator"]
                if calibrator is None:
                    continue
                path = write_artifact(
                    f"calibration_{summary['model']}", "v1", calibrator.to_artifact()
                )
                print(f"    {summary['model']}: {artifact_reference(path)}")
    finally:
        db.close()

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
