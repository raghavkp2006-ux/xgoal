#!/usr/bin/env python3
"""Half-life (time-decay) sweep — validation-season selection, honestly reported.

Answers one question: can the Dixon-Coles model's gap to the closing odds be
closed by re-tuning the exponential time decay? The protocol is deliberately
split so the test window stays untouched:

1. ``--validation-season`` (2019/20 by default, never one of the test seasons)
   picks the half-life that minimises its own log loss;
2. the whole grid is then reported on the untouched ``--test-seasons`` window,
   including what a test-peeking choice would have picked, labelled as such.

The command is read-only: it never writes a model version or an artifact, so
running it cannot silently change what production serves.

Usage:
    python -m jobs.sweep_half_life
    python -m jobs.sweep_half_life --validation-season 2018/19
    python -m jobs.sweep_half_life --grid 365,540,730 --test-seasons 2024/25
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402


from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import get_competition, load_labeled_matches, load_team_names  # noqa: E402
from app.ml.evaluation import PRIMARY_METRIC, WalkForwardResult, walk_forward  # noqa: E402

DEFAULT_GRID = (180.0, 270.0, 365.0, 450.0, 540.0, 730.0, 1095.0)
DEFAULT_BASELINE = 540.0
DEFAULT_VALIDATION = "2019/20"
DEFAULT_TEST_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25")


def parse_grid(raw: str | None) -> list[float]:
    """Comma-separated half-lives in days, or the default grid."""
    if not raw:
        return list(DEFAULT_GRID)
    values = [float(part) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise SystemExit("--grid needs positive half-lives in days")
    return values


def parse_seasons(raw: str | None, fallback: tuple[str, ...]) -> list[str]:
    """Comma-separated season labels, or a default tuple."""
    if not raw:
        return list(fallback)
    return [part.strip() for part in raw.split(",") if part.strip()]


def validation_lines(result: WalkForwardResult, half_life: float) -> str:
    """One row of the validation-season sweep table."""
    dc = result.per_model["dixon_coles"]
    market = result.per_model.get("market", {})
    return (
        f"  {half_life:>9.0f} {dc['log_loss']:>12.4f} {dc['rps']:>8.4f} "
        f"{dc['ece']:>8.4f} {dc['accuracy'] * 100:>6.1f}% "
        f"{market.get('log_loss', float('nan')):>8.4f} "
        f"{dc['log_loss'] - market.get('log_loss', float('nan')):>+8.4f}"
    )


def test_lines(half_life: float, result: WalkForwardResult, seasons: list[str]) -> list[str]:
    """Pooled and per-season rows for one half-life on the test window."""
    dc = result.per_model["dixon_coles"]
    gap = dc["log_loss"] - result.per_model["market"]["log_loss"]
    per_season = "  ".join(
        f"{season}={result.per_season[season]['dixon_coles'][PRIMARY_METRIC]:.4f}"
        for season in seasons
        if season in result.per_season
    )
    return [
        f"\n  half_life={half_life:.0f}  pooled dc={dc['log_loss']:.4f} "
        f"(gap vs market {gap:+.4f})",
        f"    per season: {per_season}",
        f"    pooled rps {dc['rps']:.4f}  ece {dc['ece']:.4f}  "
        f"accuracy {dc['accuracy'] * 100:.2f}%",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Half-life sweep: pick on a validation season, report the test window as-is."
    )
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument(
        "--validation-season",
        default=DEFAULT_VALIDATION,
        help="season used to choose the half-life (must not be a test season)",
    )
    parser.add_argument("--test-seasons", default=None, help="comma-separated test seasons")
    parser.add_argument(
        "--baseline",
        type=float,
        default=DEFAULT_BASELINE,
        help="half-life treated as the untuned baseline (must be in --grid)",
    )
    parser.add_argument("--grid", default=None, help="comma-separated half-lives in days")
    parser.add_argument("--max-goals", type=int, default=10, help="score matrix size - 1")
    args = parser.parse_args()

    grid = parse_grid(args.grid)
    test_seasons = parse_seasons(args.test_seasons, DEFAULT_TEST_SEASONS)
    if args.validation_season in test_seasons:
        raise SystemExit("--validation-season must not be one of the test seasons")
    if args.baseline not in grid:
        raise SystemExit("--baseline must be one of --grid")

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        names = load_team_names(db)
        labels, matches = load_labeled_matches(db, competition.id, names)

        print("=" * 78)
        print(f"  XGoal half-life sweep - {competition.code} ({competition.name})")
        print("=" * 78)
        print(f"  {len(matches)} matches, seasons {sorted(set(labels))}")

        print(f"\n=== step 1: selection on {args.validation_season} only ===")
        header = (
            f"  {'half_life':>9} {'dc log_loss':>12} {'dc rps':>8} {'dc ece':>8} "
            f"{'dc acc':>7} {'market':>8} {'gap':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        validation: dict[float, float] = {}
        for half_life in grid:
            result = walk_forward(
                labels,
                matches,
                [args.validation_season],
                half_life_days=half_life,
                max_goals=args.max_goals,
                include_xgboost=False,
            )
            validation[half_life] = result.per_model["dixon_coles"][PRIMARY_METRIC]
            print(validation_lines(result, half_life))
        picked = min(validation, key=lambda value: validation[value])
        print(
            f"\n  validation winner: half_life={picked:.0f} "
            f"({args.validation_season} log loss {validation[picked]:.4f} vs "
            f"baseline {args.baseline:.0f} at {validation[args.baseline]:.4f})"
        )

        print(
            f"\n=== step 2: untouched test window ({', '.join(test_seasons)}) "
            "for every half-life ==="
        )
        runs: dict[float, WalkForwardResult] = {}
        for half_life in grid:
            result = walk_forward(
                labels,
                matches,
                test_seasons,
                half_life_days=half_life,
                max_goals=args.max_goals,
                include_xgboost=False,
            )
            runs[half_life] = result
            for line in test_lines(half_life, result, test_seasons):
                print(line)

        oracle = min(
            grid, key=lambda value: runs[value].per_model["dixon_coles"][PRIMARY_METRIC]
        )
        print("\n=== step 3: verdict ===")

        def summarise(label: str, result: WalkForwardResult) -> None:
            dc = result.per_model["dixon_coles"]
            gap = dc["log_loss"] - result.per_model["market"]["log_loss"]
            print(f"  {label:<42} pooled dc {dc['log_loss']:.4f}  gap {gap:+.4f}")

        summarise(f"baseline {args.baseline:.0f} (untuned default)", runs[args.baseline])
        summarise(f"validation pick {picked:.0f} ({args.validation_season})", runs[picked])
        summarise(f"oracle {oracle:.0f} (peeked at test) - NOT ADOPTED", runs[oracle])
        print(
            "\n  the oracle row exists to quantify how much a test-tuned choice would\n"
            "  flatter the numbers; it is not a candidate configuration."
        )

        for label, result in (
            (f"baseline {args.baseline:.0f}", runs[args.baseline]),
            (f"validation pick {picked:.0f}", runs[picked]),
        ):
            print(f"\n  -- targets for {label} --")
            for line in result.target_lines():
                print("  " + line.strip())
            print(f"  -- accuracy for {label} --")
            for line in result.accuracy_lines():
                print("  " + line.strip())
    finally:
        db.close()

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
