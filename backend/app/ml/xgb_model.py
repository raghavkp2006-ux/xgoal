"""XGBoost comparison model: 3-class softmax over point-in-time features.

The feature builder is owned by the model, so ``fit`` trains on the training
window and ``predict_next`` keeps feeding the *same* builder — feature state
advances match by match, exactly like a live system, with no look-ahead.
"""

from __future__ import annotations

import base64
from typing import Any, Sequence

import numpy as np
from xgboost import XGBClassifier

from app.ml.data import FloatArray, MatchInput, Probs
from app.ml.features import FEATURE_NAMES, FeatureBuilder

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}


class XGBoostModel:
    """Gradient-boosted classifier over the rolling features of app.ml.features."""

    def __init__(
        self,
        random_seed: int = 42,
        window: int = 6,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.random_seed = random_seed
        self.window = window
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS if params is None else params)
        self.builder = FeatureBuilder(window=window)
        self.classifier: XGBClassifier | None = None
        self.n_train_matches = 0

    def fit(self, matches: Sequence[MatchInput]) -> XGBoostModel:
        """Train on ``matches``, building features in kick-off order."""
        dataset = self.builder.build(matches)
        if len(dataset) == 0:
            raise ValueError("cannot fit without training matches")
        params = dict(self.params)
        params["random_state"] = self.random_seed
        classifier = XGBClassifier(**params)
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

    def feature_importance(self, top: int = 10) -> dict[str, float]:
        """Split-count importances of the sklearn wrapper, descending."""
        if self.classifier is None:
            raise RuntimeError("fit() must be called before feature_importance()")
        values = np.asarray(self.classifier.feature_importances_, dtype=np.float64)
        pairs = sorted(zip(FEATURE_NAMES, values), key=lambda kv: kv[1], reverse=True)
        return {name: round(float(value), 5) for name, value in pairs[:top]}

    def hyperparameters(self) -> dict[str, Any]:
        """Settings recorded in ``model_versions.hyperparameters``."""
        return {"random_seed": self.random_seed, "window": self.window, **self.params}

    def to_artifact(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (booster embedded as base64 UBJ)."""
        if self.classifier is None:
            raise RuntimeError("fit() must be called before to_artifact()")
        blob = self.classifier.get_booster().save_raw(raw_format="ubj")
        return {
            "model": "xgboost",
            "hyperparameters": self.hyperparameters(),
            "feature_names": list(FEATURE_NAMES),
            "n_train_matches": self.n_train_matches,
            "feature_importance": self.feature_importance(top=len(FEATURE_NAMES)),
            "feature_state": self.builder.export_state(),
            "booster_ubj_base64": base64.b64encode(blob).decode("ascii"),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> XGBoostModel:
        """Rebuild a trained model from :meth:`to_artifact` output.

        The rolling feature state travels with the artifact, so ``predict_next``
        keeps producing the forecasts the saved model produced instead of
        silently falling back to league priors.
        """
        hyper = artifact.get("hyperparameters") or {}
        params = {k: v for k, v in hyper.items() if k not in ("random_seed", "window")}
        model = cls(
            random_seed=int(hyper.get("random_seed", 42)),
            window=int(hyper.get("window", 6)),
            params=params or None,
        )
        classifier = XGBClassifier()
        classifier.load_model(bytearray(base64.b64decode(artifact["booster_ubj_base64"])))
        model.classifier = classifier
        model.n_train_matches = int(artifact.get("n_train_matches", 0))
        model.builder.import_state(artifact.get("feature_state") or {})
        return model
