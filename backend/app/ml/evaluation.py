"""Walk-forward evaluation harness.

Protocol
--------
* Every match kicking off before the first test season is pre-training data.
* Each test season is then predicted by models refit on *everything* earlier
  (expanding window): Dixon-Coles and XGBoost take one refit per test season,
  while Elo, base rate and uniform replay the stream match by match, and the
  market baseline is read off the stored closing odds.
* All forecasters are pooled and scored with log loss (primary), RPS, Brier and
  ECE — overall and per season.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.ml.baselines import BaseRateBaseline, EloBaseline, UniformBaseline, devig_probs
from app.ml.data import MatchInput, Probs
from app.ml.dixon_coles import DixonColesModel
from app.ml.metrics import evaluate_predictions
from app.ml.xgb_model import XGBoostModel

PRIMARY_METRIC = "log_loss"
MARKET_TOLERANCE = 0.03
ELO_WINS_NEEDED = 4
ACCURACY_BAND = (0.51, 0.54)


@dataclass
class WalkForwardResult:
    """Metrics produced by one walk-forward run."""

    test_seasons: list[str]
    n_test_matches: int
    per_model: dict[str, dict[str, float]]
    per_season: dict[str, dict[str, dict[str, float]]]
    coverage: dict[str, int]
    # Raw out-of-sample forecasts, kept so calibration (Phase 2.5) and the
    # reliability tables can reuse exactly the predictions that were scored.
    per_season_forecasts: dict[str, dict[str, list[tuple[MatchInput, Probs]]]] = field(
        default_factory=dict
    )

    def ranking(self) -> list[tuple[str, float]]:
        """Models ordered by pooled log loss, best first."""
        return sorted(
            ((name, metrics[PRIMARY_METRIC]) for name, metrics in self.per_model.items()),
            key=lambda item: item[1],
        )

    def best(self) -> str:
        """Name of the best model by pooled log loss."""
        return self.ranking()[0][0]

    def to_metrics_payload(self) -> dict[str, Any]:
        """JSONB-ready summary for ``model_versions.eval_metrics``."""
        return {
            "protocol": "walk_forward",
            "primary_metric": PRIMARY_METRIC,
            "test_seasons": list(self.test_seasons),
            "n_test_matches": self.n_test_matches,
            "coverage": dict(self.coverage),
            "pooled": {name: dict(metrics) for name, metrics in self.per_model.items()},
            "per_season": {
                season: {name: dict(metrics) for name, metrics in models.items()}
                for season, models in self.per_season.items()
            },
        }

    def summary_lines(self) -> list[str]:
        """Human-readable ranking table for job output."""
        lines = [
            f"  {'model':<12} {'log_loss':>9} {'rps':>7} {'brier':>7} {'ece':>7} {'n':>6}",
            "  " + "-" * 52,
        ]
        for name, metrics in sorted(
            self.per_model.items(), key=lambda kv: kv[1][PRIMARY_METRIC]
        ):
            lines.append(
                f"  {name:<12} {metrics['log_loss']:>9.4f} {metrics['rps']:>7.4f} "
                f"{metrics['brier']:>7.4f} {metrics['ece']:>7.4f} "
                f"{self.coverage.get(name, 0):>6d}"
            )
        return lines

    def season_lines(self) -> list[str]:
        """Per-season log loss for every model, best model first within a season."""
        lines: list[str] = []
        for season in self.test_seasons:
            models = self.per_season.get(season, {})
            if not models:
                continue
            ranked = sorted(
                ((name, metrics[PRIMARY_METRIC]) for name, metrics in models.items()),
                key=lambda item: item[1],
            )
            joined = "  ".join(f"{name}={value:.4f}" for name, value in ranked)
            lines.append(f"  {season}: {joined}")
        return lines

    def coverage_for(self, season: str, model: str) -> int:
        """Number of scored forecasts for one model in one season."""
        return len(self.per_season_forecasts.get(season, {}).get(model, []))

    def table_lines(self) -> list[str]:
        """Per-season tables: every model with log loss, RPS, Brier, ECE and n."""
        lines: list[str] = []
        header = (
            f"      {'model':<12} {'log_loss':>9} {'rps':>7} {'brier':>7} {'ece':>7} {'n':>5}"
        )
        for season in self.test_seasons:
            models = self.per_season.get(season, {})
            if not models:
                continue
            lines.append(f"  {season}")
            lines.append(header)
            lines.append("      " + "-" * 47)
            for name, metrics in sorted(models.items(), key=lambda kv: kv[1][PRIMARY_METRIC]):
                lines.append(
                    f"      {name:<12} {metrics['log_loss']:>9.4f} {metrics['rps']:>7.4f} "
                    f"{metrics['brier']:>7.4f} {metrics['ece']:>7.4f} "
                    f"{self.coverage_for(season, name):>5d}"
                )
            lines.append("")
        return lines

    def accuracy_lines(self) -> list[str]:
        """Per-season pick accuracy for every model, against the 51-54% band.

        Accuracy is reported rather than optimised: the plan's target is that the
        primary model lands inside the band, so a number above it is not
        automatically better news (it usually means over-confident picks).
        """
        lines: list[str] = []
        for season in self.test_seasons:
            models = self.per_season.get(season, {})
            if not models:
                continue
            ranked = sorted(
                (
                    (name, metrics.get("accuracy", float("nan")))
                    for name, metrics in models.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            joined = "  ".join(f"{name}={value * 100:.1f}%" for name, value in ranked)
            lines.append(f"  {season}: {joined}")
        pooled = self.per_model.get("dixon_coles", {}).get("accuracy")
        if pooled is not None:
            band = f"{ACCURACY_BAND[0] * 100:.0f}-{ACCURACY_BAND[1] * 100:.0f}%"
            lines.append(
                f"  pooled dixon_coles={pooled * 100:.2f}% (plan target band {band})"
            )
        return lines

    def target_lines(
        self,
        *,
        market_tolerance: float = MARKET_TOLERANCE,
        elo_wins_needed: int = ELO_WINS_NEEDED,
    ) -> list[str]:
        """Phase 2 acceptance checks against base rate, Elo and closing odds."""
        seasons = [
            season
            for season in self.test_seasons
            if PRIMARY_METRIC in self.per_season.get(season, {}).get("dixon_coles", {})
        ]
        if not seasons:
            return ["  (no scored seasons)"]

        # With fewer scored seasons than the plan's four, scale the bar down so a
        # short diagnostic run is not reported as a spurious failure.
        elo_wins_needed = min(elo_wins_needed, len(seasons))

        def wins_over(rival: str) -> list[str]:
            return [
                season
                for season in seasons
                if rival in self.per_season[season]
                and self.per_season[season]["dixon_coles"][PRIMARY_METRIC]
                < self.per_season[season][rival][PRIMARY_METRIC]
            ]

        base_wins = wins_over("base_rate")
        elo_wins = wins_over("elo")
        market_seasons = [season for season in seasons if "market" in self.per_season[season]]
        gaps = {
            season: self.per_season[season]["dixon_coles"][PRIMARY_METRIC]
            - self.per_season[season]["market"][PRIMARY_METRIC]
            for season in market_seasons
        }
        worst = max(gaps.values()) if gaps else None
        worst_season = max(gaps, key=lambda season: gaps[season]) if gaps else None

        lines = [
            _verdict(
                f"beats base rate on all {len(seasons)} seasons",
                len(base_wins) == len(seasons),
                f"{len(base_wins)}/{len(seasons)}"
                + (
                    ""
                    if len(base_wins) == len(seasons)
                    else f" (lost {sorted(set(seasons) - set(base_wins))})"
                ),
            ),
            _verdict(
                f"beats Elo on >= {elo_wins_needed} of {len(seasons)} seasons",
                len(elo_wins) >= elo_wins_needed,
                f"{len(elo_wins)}/{len(seasons)}",
            ),
            _verdict(
                f"within {market_tolerance:.2f} log loss of closing odds",
                worst is not None and worst <= market_tolerance,
                "no seasons with closing odds"
                if worst is None
                else (
                    f"worst gap {worst:+.4f} ({worst_season}), "
                    f"mean {sum(gaps.values()) / len(gaps):+.4f}"
                ),
            ),
        ]
        if market_seasons:
            lines.append(
                "  per-season gap vs closing odds: "
                + "  ".join(f"{season}={gaps[season]:+.4f}" for season in market_seasons)
            )

        pooled_accuracy = self.per_model.get("dixon_coles", {}).get("accuracy")
        in_band = [
            season
            for season in seasons
            if ACCURACY_BAND[0]
            <= self.per_season[season]["dixon_coles"].get("accuracy", float("nan"))
            <= ACCURACY_BAND[1]
        ]
        lines.append(
            _verdict(
                f"pick accuracy inside {ACCURACY_BAND[0] * 100:.0f}-{ACCURACY_BAND[1] * 100:.0f}%",
                pooled_accuracy is not None
                and ACCURACY_BAND[0] <= pooled_accuracy <= ACCURACY_BAND[1],
                "no accuracy recorded"
                if pooled_accuracy is None
                else (
                    f"pooled {pooled_accuracy * 100:.2f}%, "
                    f"{len(in_band)}/{len(seasons)} seasons in band"
                ),
            )
        )
        return lines


def _verdict(label: str, passed: bool, detail: str) -> str:
    """One acceptance line: PASS/FAIL, the criterion and the evidence."""
    return f"  [{'PASS' if passed else 'FAIL'}] {label} - {detail}"


def reliability_bins(
    probs: Sequence[Probs], outcomes: Sequence[int], n_bins: int = 5
) -> list[tuple[str, int, float, float]]:
    """Calibration table for the predicted class: (bin, n, predicted, observed)."""
    rows: list[tuple[str, int, float, float]] = []
    edges = [index / n_bins for index in range(n_bins + 1)]
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        picked = [
            (probs[i], outcomes[i])
            for i in range(len(probs))
            if (lower < max(probs[i]) <= upper) or (lower == 0.0 and max(probs[i]) <= upper)
        ]
        if not picked:
            continue
        expected = sum(max(p) for p, _ in picked) / len(picked)
        hits = sum(1 for p, outcome in picked if p.index(max(p)) == outcome) / len(picked)
        rows.append((f"{lower:.1f}-{upper:.1f}", len(picked), expected, hits))
    return rows


def walk_forward(
    labels: Sequence[str],
    matches: Sequence[MatchInput],
    test_seasons: Sequence[str],
    half_life_days: float = 540.0,
    max_goals: int = 10,
    random_seed: int = 42,
    window: int = 6,
    include_xgboost: bool = True,
) -> WalkForwardResult:
    """Run the walk-forward protocol and score every forecaster."""
    if len(labels) != len(matches):
        raise ValueError("labels and matches must be aligned")
    ordered = sorted(zip(labels, matches, strict=True), key=lambda item: item[1].kickoff)
    ordered_labels = [label for label, _ in ordered]
    ordered_matches = [match for _, match in ordered]

    test_set = list(dict.fromkeys(test_seasons))
    wanted = set(test_set)
    first_test = next((i for i, label in enumerate(ordered_labels) if label in wanted), None)
    if first_test is None:
        raise ValueError("no matches belong to the requested test seasons")

    training = ordered_matches[:first_test]
    if not training:
        raise ValueError("no training matches kick off before the first test season")

    elo = EloBaseline().fit(training)
    pooled: dict[str, list[tuple[MatchInput, Probs]]] = defaultdict(list)
    per_season: dict[str, dict[str, list[tuple[MatchInput, Probs]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for season in test_set:
        indices = [
            i for i in range(first_test, len(ordered_matches)) if ordered_labels[i] == season
        ]
        if not indices:
            continue
        history = ordered_matches[: indices[0]]
        season_matches = [ordered_matches[i] for i in indices]

        dc_model = DixonColesModel(half_life_days=half_life_days, max_goals=max_goals).fit(
            history
        )
        booster = (
            XGBoostModel(random_seed=random_seed, window=window).fit(history)
            if include_xgboost
            else None
        )
        base_rate = BaseRateBaseline().fit(history)
        uniform = UniformBaseline().fit(history)

        for match in season_matches:
            forecasts: list[tuple[str, Probs]] = [
                ("dixon_coles", dc_model.predict(match.home, match.away).probs),
                ("elo", elo.predict(match.home, match.away)),
                ("base_rate", base_rate.predict(match.home, match.away)),
                ("uniform", uniform.predict(match.home, match.away)),
            ]
            market = devig_probs(match)
            if market is not None:
                forecasts.append(("market", market))
            if booster is not None:
                forecasts.append(("xgboost", booster.predict_next_probs([match])[0]))
            for name, probs in forecasts:
                pooled[name].append((match, probs))
                per_season[season][name].append((match, probs))
            elo.update(match)

    def metrics_for(items: Sequence[tuple[MatchInput, Probs]]) -> dict[str, float]:
        return evaluate_predictions([p for _, p in items], [m.outcome for m, _ in items])

    return WalkForwardResult(
        test_seasons=test_set,
        n_test_matches=len(ordered_matches) - first_test,
        per_model={name: metrics_for(items) for name, items in pooled.items()},
        per_season={
            season: {name: metrics_for(items) for name, items in models.items()}
            for season, models in per_season.items()
        },
        coverage={name: len(items) for name, items in pooled.items()},
        per_season_forecasts={
            season: {name: list(items) for name, items in models.items()}
            for season, models in per_season.items()
        },
    )
