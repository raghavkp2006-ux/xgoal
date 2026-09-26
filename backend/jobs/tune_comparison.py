#!/usr/bin/env python3
"""Phase 2.4 — walk-forward CV selection of the comparison models' hyper-parameters.

The protocol keeps the test window untouched:

* **selection** happens on seasons strictly *before* the test window, each fold
  training on everything earlier and scoring only that season;
* the test window is then scored once, with the already-selected configuration,
  purely for information.

The command is read-only: it writes no ``model_versions`` row and no artifact.

Usage:
    python -m jobs.tune_comparison
    python -m jobs.tune_comparison --validation-seasons 2018/19,2019/20
"""
import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import get_competition, load_labeled_matches, load_team_names  # noqa: E402
from app.ml.evaluation import PRIMARY_METRIC, walk_forward  # noqa: E402

DEFAULT_VALIDATION_SEASONS = ("2018/19", "2019/20")
DEFAULT_TEST_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25")

LOGISTIC_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"C": 0.03},
    {"C": 0.1},
    {"C": 0.3},
    {"C": 1.0},
)

XGB_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"max_depth": 3, "min_child_weight": 10},
    {"max_depth": 4, "min_child_weight": 10},
    {"max_depth": 4, "min_child_weight": 20},
)
"""Build plan 2.4 fixes subsample and colsample_bytree at 0.8 and bounds
max_depth to 3-4 with min_child_weight >= 10, so only those two knobs vary."""

FIXED_XGB: dict[str, Any] = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 750,
}


def parse_seasons(raw: str | None, fallback: tuple[str, ...]) -> list[str]:
    """Comma-separated season labels, or a default tuple."""
    if not raw:
        return list(fallback)
    return [part.strip() for part in raw.split(",") if part.strip()]


def fold_scores(
    labels: list[str],
    matches: list[Any],
    seasons: list[str],
    *,
    model_name: str,
    **kwargs: Any,
) -> list[tuple[str, float]]:
    """(season, log loss) for one configuration, one fold per validation season."""
    scores: list[tuple[str, float]] = []
    for season in seasons:
        result = walk_forward(
            labels,
            matches,
            [season],
            include_xgboost=model_name == "xgboost",
            include_logistic=model_name == "logistic",
            **kwargs,
        )
        scores.append((season, result.per_model[model_name][PRIMARY_METRIC]))
    return scores


def mean_of(scores: list[tuple[str, float]]) -> float:
    """Mean log loss across folds."""
    return sum(value for _, value in scores) / len(scores)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward CV selection for the comparison models."
    )
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument("--validation-seasons", default=None, help="comma-separated")
    parser.add_argument("--test-seasons", default=None, help="comma-separated")
    parser.add_argument("--window", type=int, default=6, help="rolling feature window")
    args = parser.parse_args()

    validation_seasons = parse_seasons(args.validation_seasons, DEFAULT_VALIDATION_SEASONS)
    test_seasons = parse_seasons(args.test_seasons, DEFAULT_TEST_SEASONS)

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found")
        names = load_team_names(db)
        labels, matches = load_labeled_matches(db, competition.id, names)

        print("=" * 78)
        print(f"  Phase 2.4 CV selection - {competition.code} ({competition.name})")
        print("=" * 78)
        print(f"  {len(matches)} matches")
        print(f"  validation folds : {', '.join(validation_seasons)}")
        print(f"  test window      : {', '.join(test_seasons)}  (scored once, after selection)")
        print("  selection metric : mean log loss on the validation folds only")

        # ---------------------------------------------------------------- #
        # logistic
        # ---------------------------------------------------------------- #
        print("\n" + "=" * 78)
        print("  multinomial logistic regression: regularisation strength C")
        print("=" * 78)
        print(
            f"  {'C':>7} "
            + " ".join(f"{s:>10}" for s in validation_seasons)
            + f" {'mean':>10}"
        )
        print("  " + "-" * (10 + 11 * len(validation_seasons)))
        logistic_table: dict[float, float] = {}
        for candidate in LOGISTIC_CANDIDATES:
            scores = fold_scores(
                labels,
                matches,
                validation_seasons,
                model_name="logistic",
                window=args.window,
                logistic_params=dict(candidate),
            )
            logistic_table[candidate["C"]] = mean_of(scores)
            cells = " ".join(f"{value:>10.4f}" for _, value in scores)
            print(f"  {candidate['C']:>7.3f} {cells} {mean_of(scores):>10.4f}")
        best_c = min(logistic_table, key=lambda c: logistic_table[c])
        print(
            f"\n  selected C = {best_c}  "
            f"(mean validation log loss {logistic_table[best_c]:.4f})"
        )

        # ---------------------------------------------------------------- #
        # xgboost
        # ---------------------------------------------------------------- #
        print("\n" + "=" * 78)
        print("  XGBoost: depth and min_child_weight (subsample/colsample fixed at 0.8)")
        print("=" * 78)
        print(
            f"  {'depth':>6} {'mcw':>5} "
            + " ".join(f"{s:>10}" for s in validation_seasons)
            + f" {'mean':>10}"
        )
        print("  " + "-" * (14 + 11 * len(validation_seasons)))
        xgb_table: dict[tuple[int, int], float] = {}
        for candidate in XGB_CANDIDATES:
            params = {**FIXED_XGB, **candidate}
            scores = fold_scores(
                labels,
                matches,
                validation_seasons,
                model_name="xgboost",
                window=args.window,
                xgb_params=params,
            )
            key = (candidate["max_depth"], candidate["min_child_weight"])
            xgb_table[key] = mean_of(scores)
            cells = " ".join(f"{value:>10.4f}" for _, value in scores)
            print(f"  {key[0]:>6} {key[1]:>5} {cells} {mean_of(scores):>10.4f}")
        best_depth, best_mcw = min(xgb_table, key=lambda k: xgb_table[k])
        print(
            f"\n  selected max_depth = {best_depth}, min_child_weight = {best_mcw}  "
            f"(mean validation log loss {xgb_table[(best_depth, best_mcw)]:.4f})"
        )

        # ---------------------------------------------------------------- #
        # test window, scored once with the selected configuration
        # ---------------------------------------------------------------- #
        print("\n" + "=" * 78)
        print("  test window scored with the SELECTED configuration (information only)")
        print("=" * 78)
        result = walk_forward(
            labels,
            matches,
            test_seasons,
            window=args.window,
            logistic_params={"C": best_c},
            xgb_params={
                **FIXED_XGB,
                "max_depth": best_depth,
                "min_child_weight": best_mcw,
            },
        )
        for line in result.summary_lines():
            print(line)
        print("\n  per-season log loss")
        for line in result.season_lines():
            print(line)
        print("\n  accuracy")
        for line in result.accuracy_lines():
            print(line)
        print("\n  acceptance targets")
        for line in result.target_lines():
            print(line)
    finally:
        db.close()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
