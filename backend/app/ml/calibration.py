"""Phase 2.5 — probability calibration for the scoreline forecasts.

The walk-forward harness shows Dixon-Coles and XGBoost are both *slightly*
over-confident: the middle confidence bins hit less often than they claim (see
the ECE column). This module fits isotonic regression on a dedicated calibration
split and exposes the before/after view used by ``jobs/calibrate_forecasts.py``.

Method
------
One-vs-rest isotonic regression per outcome: for class *k*, regress the observed
indicator ``1[outcome == k]`` on the predicted ``p_k``, then renormalise the three
calibrated columns so they still sum to one. Isotonic is monotone and
non-parametric, so it fixes over/under-confidence without assuming a shape.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from app.ml.data import FloatArray, Probs

N_OUTCOMES = 3
DEFAULT_BINS = 5


class IsotonicCalibrator:
    """Monotone, one-vs-rest calibration of a (home, draw, away) forecast.

    ``shrink`` mixes the calibrated mapping back towards the raw forecast:
    ``p = (1 - shrink) * p_raw + shrink * p_isotonic``. A calibration split of a
    single season (~380 matches) is small enough that the isotonic curve
    overfits its own sample, so a partial mix is the variance-reduced choice;
    ``shrink=1.0`` is plain isotonic and ``shrink=0.0`` is a no-op.
    """

    def __init__(self, shrink: float = 1.0) -> None:
        if not 0.0 <= shrink <= 1.0:
            raise ValueError("shrink must be between 0.0 and 1.0")
        self.shrink = float(shrink)
        self.models: list[IsotonicRegression] = []
        self.n_train_matches = 0

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has produced a usable mapping."""
        return len(self.models) == N_OUTCOMES

    def fit(self, probs: Sequence[Probs], outcomes: Sequence[int]) -> IsotonicCalibrator:
        """Fit one isotonic curve per outcome on the calibration split."""
        if len(probs) != len(outcomes):
            raise ValueError("forecasts and outcomes must be aligned")
        if not probs:
            raise ValueError("cannot fit calibration without forecasts")
        matrix = np.asarray(probs, dtype=np.float64)
        truth = np.asarray(outcomes, dtype=np.int64)
        if bool(((truth < 0) | (truth >= N_OUTCOMES)).any()):
            raise ValueError(f"outcome indices must be in 0..{N_OUTCOMES - 1}")

        self.models = []
        for outcome in range(N_OUTCOMES):
            indicator = (truth == outcome).astype(np.float64)
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(matrix[:, outcome], indicator)
            self.models.append(model)
        self.n_train_matches = int(matrix.shape[0])
        return self

    def transform(self, probs: Sequence[Probs]) -> list[Probs]:
        """Calibrated probabilities, renormalised to sum to one per fixture."""
        if not self.fitted:
            raise RuntimeError("fit() must be called before transform()")
        matrix = np.asarray(probs, dtype=np.float64)
        iso: FloatArray = np.zeros_like(matrix)
        for outcome, model in enumerate(self.models):
            iso[:, outcome] = model.predict(matrix[:, outcome])
        mixed = (1.0 - self.shrink) * matrix + self.shrink * iso
        mixed = np.clip(mixed, 1e-9, None)
        mixed = mixed / mixed.sum(axis=1, keepdims=True)
        return [(float(row[0]), float(row[1]), float(row[2])) for row in mixed]

    # -- persistence ------------------------------------------------------

    def to_artifact(self) -> dict[str, Any]:
        """JSON-serialisable snapshot: the isotonic knots of each outcome."""
        if not self.fitted:
            raise RuntimeError("fit() must be called before to_artifact()")
        return {
            "model": "isotonic_calibration",
            "n_outcomes": N_OUTCOMES,
            "n_train_matches": self.n_train_matches,
            "shrink": self.shrink,
            "knots": [
                {
                    "p": [float(value) for value in model.X_thresholds_],
                    "y": [float(value) for value in model.y_thresholds_],
                }
                for model in self.models
            ],
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> IsotonicCalibrator:
        """Rebuild a calibrator from :meth:`to_artifact` output."""
        calibrator = cls(shrink=float(artifact.get("shrink", 1.0)))
        for knot in artifact.get("knots") or []:
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(
                np.asarray(knot["p"], dtype=np.float64),
                np.asarray(knot["y"], dtype=np.float64),
            )
            calibrator.models.append(model)
        calibrator.n_train_matches = int(artifact.get("n_train_matches", 0))
        return calibrator


def reliability_lines(
    probs: Sequence[Probs],
    outcomes: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    title: str = "",
) -> list[str]:
    """ASCII reliability diagram: predicted confidence vs observed hit rate.

    Each row is one confidence bin: ``predicted`` is the mean probability the
    model put on its own pick, ``observed`` how often that pick actually won, and
    ``gap`` their difference (positive = over-confident).
    """
    if len(probs) != len(outcomes):
        raise ValueError("forecasts and outcomes must be aligned")
    if not probs:
        return ["  (no forecasts)"]

    confidences = [max(row) for row in probs]
    picks = [row.index(max(row)) for row in probs]
    edges = [index / n_bins for index in range(n_bins + 1)]
    lines = [
        f"  reliability - {title}" if title else "  reliability (predicted class)",
        f"  {'bin':<9} {'n':>5} {'predicted':>10} {'observed':>9} {'gap':>8}",
        "  " + "-" * 45,
    ]
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower < confidence <= upper or (lower == 0.0 and confidence <= upper)
        ]
        if not members:
            continue
        predicted = sum(confidences[index] for index in members) / len(members)
        observed = sum(1 for index in members if picks[index] == outcomes[index]) / len(members)
        label = f"{lower:.1f}-{upper:.1f}"
        lines.append(
            f"  {label:<9} {len(members):>5} {predicted:>10.4f} "
            f"{observed:>9.4f} {predicted - observed:>+8.4f}"
        )
    return lines
