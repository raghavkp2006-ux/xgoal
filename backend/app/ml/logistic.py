"""Multinomial logistic regression: the comparison model's linear baseline.

Phase 2.4 asks for this *first*, because on a few thousand rows with a couple of
dozen weakly-informative features a regularised linear model is usually within a
hair of gradient boosting — and when it is, that is worth knowing before
reaching for the fancier model.

It shares everything with :mod:`app.ml.xgb_model` except the estimator:

* the same :class:`~app.ml.features.FeatureBuilder`, so the feature contract is
  identical and the two models are directly comparable;
* the same ownership rule — the builder lives on the model, so ``fit`` trains on
  the training window and ``predict_next`` keeps feeding the *same* builder,
  advancing feature state match by match exactly like a live system;
* the same JSON artifact shape, so a logistic model round-trips without pickle
  and can be served by the same code path.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.ml.data import FloatArray, MatchInput, Probs
from app.ml.features import FEATURE_NAMES, FeatureBuilder

DEFAULT_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 2000,
    "solver": "lbfgs",
    "tol": 1e-6,
}
"""Regularisation strength is left at sklearn's default here and selected by
walk-forward CV in ``jobs/tune_comparison.py`` rather than guessed."""


class LogisticModel:
    """Multinomial logistic regression over the rolling point-in-time features."""

    def __init__(
        self,
        window: int = 6,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.window = window
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.builder = FeatureBuilder(window=window)
        self.classifier: LogisticRegression | None = None
        self.n_train_matches = 0

    def fit(self, matches: Sequence[MatchInput]) -> LogisticModel:
        """Train on ``matches``, building features in kick-off order."""
        dataset = self.builder.build(matches)
        if len(dataset) == 0:
            raise ValueError("cannot fit without training matches")
        classifier = LogisticRegression(**self.params)
        classifier.fit(dataset.matrix, np.asarray(dataset.outcomes))
        self.classifier = classifier
        self.n_train_matches = len(dataset)
        return self

    def predict_next(self, matches: Sequence[MatchInput]) -> FloatArray:
        """Probabilities for the next matches in the stream (advances history)."""
        if self.classifier is None:
            raise RuntimeError("fit() must be called before predict_next()")
        dataset = self.builder.build(matches)
        if len(dataset) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        probs = np.asarray(self.classifier.predict_proba(dataset.matrix), dtype=np.float64)
        totals = probs.sum(axis=1, keepdims=True)
        return probs / totals

    def predict_next_probs(self, matches: Sequence[MatchInput]) -> list[Probs]:
        """Same as :meth:`predict_next`, as plain (home, draw, away) tuples."""
        matrix = self.predict_next(matches)
        return [(float(row[0]), float(row[1]), float(row[2])) for row in matrix]


    def coefficients(self) -> dict[str, list[float]]:
        """Per-feature coefficients per outcome class (home, draw, away)."""
        if self.classifier is None:
            raise RuntimeError("fit() must be called before coefficients()")
        coef = np.asarray(self.classifier.coef_, dtype=np.float64)
        return {
            name: [float(value) for value in coef[:, column]]
            for column, name in enumerate(FEATURE_NAMES)
        }

    def feature_importance(self, top: int = 10) -> dict[str, float]:
        """Mean absolute coefficient across classes, descending.

        The linear analogue of a split-count importance: it says how much the
        fitted model leans on each feature overall, not which class it favours.
        """
        if self.classifier is None:
            raise RuntimeError("fit() must be called before feature_importance()")
        coef = np.asarray(self.classifier.coef_, dtype=np.float64)
        values = np.abs(coef).mean(axis=0)
        pairs = sorted(zip(FEATURE_NAMES, values), key=lambda kv: kv[1], reverse=True)
        return {name: round(float(value), 5) for name, value in pairs[:top]}

    def hyperparameters(self) -> dict[str, Any]:
        """Settings recorded in ``model_versions.hyperparameters``."""
        return {"window": self.window, **self.params}

    def to_artifact(self) -> dict[str, Any]:
        """JSON-serialisable snapshot: coefficients, not a pickled estimator."""
        if self.classifier is None:
            raise RuntimeError("fit() must be called before to_artifact()")
        return {
            "model": "logistic",
            "hyperparameters": self.hyperparameters(),
            "feature_names": list(FEATURE_NAMES),
            "n_train_matches": self.n_train_matches,
            "feature_importance": self.feature_importance(top=len(FEATURE_NAMES)),
            "classes": [int(c) for c in self.classifier.classes_],
            "coef": [[float(v) for v in row] for row in self.classifier.coef_],
            "intercept": [float(v) for v in self.classifier.intercept_],
            "feature_state": self.builder.export_state(),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> LogisticModel:
        """Rebuild a trained model from :meth:`to_artifact` output.

        The rolling feature state travels with the artefact, so
        ``predict_next`` reproduces the saved model's forecasts instead of
        silently falling back to league priors.
        """
        hyper = dict(artifact.get("hyperparameters") or {})
        window = int(hyper.pop("window", 6))
        model = cls(window=window, params=hyper or None)
        classifier = LogisticRegression(**model.params)
        coef = np.asarray(artifact["coef"], dtype=np.float64)
        classifier.coef_ = coef
        classifier.intercept_ = np.asarray(artifact["intercept"], dtype=np.float64)
        classifier.classes_ = np.asarray(
            artifact.get("classes", [0, 1, 2]), dtype=np.int64
        )
        classifier.n_features_in_ = coef.shape[1]
        model.classifier = classifier
        model.n_train_matches = int(artifact.get("n_train_matches", 0))
        model.builder.import_state(artifact.get("feature_state") or {})
        return model
