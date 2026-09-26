"""XGBoost comparison model: 3-class softmax over point-in-time features.

The feature builder is owned by the model, so ``fit`` trains on the training
window and ``predict_next`` keeps feeding the *same* builder — feature state
advances match by match, exactly like a live system, with no look-ahead.
"""

from __future__ import annotations

import base64
import warnings
from typing import Any, Sequence

import numpy as np
from xgboost import XGBClassifier

from app.ml.data import FloatArray, MatchInput, Probs
from app.ml.features import FEATURE_NAMES, FeatureBuilder

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 750,
    "learning_rate": 0.05,
    # Build plan 2.4: "aggressive regularisation — max_depth 3-4,
    # min_child_weight >= 10, subsample 0.8, colsample_bytree 0.8, early stopping
    # on a validation fold". n_estimators is only the ceiling; early stopping
    # chooses the tree count actually shipped.
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}

DEFAULT_EARLY_STOPPING_ROUNDS = 25
"""Rounds without validation improvement before the tree count is frozen."""

MIN_MATCHES_FOR_EARLY_STOPPING = 200
"""Below this the validation split would be too thin to be informative."""

_WRAPPER_KEYS = frozenset(
    {
        "random_seed",
        "window",
        "validation_fraction",
        "early_stopping_rounds",
        "selected_trees",
        "n_estimators_ceiling",
        "validation_matches",
        "validation_log_loss",
    }
)
"""Keys ``hyperparameters()`` adds for provenance that are *not* XGBClassifier
keyword arguments — stripped before rebuilding a classifier from an artefact."""


class XGBoostModel:
    """Gradient-boosted classifier over the rolling features of app.ml.features."""

    def __init__(
        self,
        random_seed: int = 42,
        window: int = 6,
        params: dict[str, Any] | None = None,
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> None:
        self.random_seed = random_seed
        self.window = window
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.builder = FeatureBuilder(window=window)
        self.classifier: XGBClassifier | None = None
        self.n_train_matches = 0
        self.validation_matches = 0
        self.selected_trees: int | None = None
        self.validation_log_loss: float | None = None

    def fit(self, matches: Sequence[MatchInput]) -> XGBoostModel:
        """Train on ``matches``, building features in kick-off order.

        Early stopping runs on a **chronological** validation fold — the last
        ``validation_fraction`` of the training window, which is still strictly
        before anything the model will be asked to predict, so it cannot leak.
        The fold only *selects the tree count*; the shipped model is then refit
        on the whole training window with that count and no early stopping. That
        keeps the stored booster exactly the set of trees used, so reloading an
        artifact reproduces its forecasts bit for bit.
        """
        dataset = self.builder.build(matches)
        if len(dataset) == 0:
            raise ValueError("cannot fit without training matches")

        matrix = np.asarray(dataset.matrix)
        outcomes = np.asarray(dataset.outcomes)
        params = dict(self.params)
        params["random_state"] = self.random_seed
        ceiling = int(params.get("n_estimators", 750))

        split = int(len(dataset) * (1.0 - self.validation_fraction))
        use_early_stopping = (
            self.early_stopping_rounds > 0
            and self.validation_fraction > 0.0
            and len(dataset) >= MIN_MATCHES_FOR_EARLY_STOPPING
            and 0 < split < len(dataset)
        )

        chosen = ceiling
        if use_early_stopping:
            probe_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
            probe_params["early_stopping_rounds"] = self.early_stopping_rounds
            probe = XGBClassifier(**probe_params)
            probe.fit(
                matrix[:split],
                outcomes[:split],
                eval_set=[(matrix[split:], outcomes[split:])],
                verbose=False,
            )
            best = getattr(probe, "best_iteration", None)
            if best is None:
                warnings.warn(
                    f"XGBoost early stopping did not set best_iteration "
                    f"after {self.early_stopping_rounds} patience rounds; "
                    f"using ceiling n_estimators={ceiling}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if best is not None:
                chosen = int(best) + 1
                self.selected_trees = chosen
                probe_probs = np.asarray(
                    probe.predict_proba(matrix[split:]), dtype=np.float64
                )
                self.validation_matches = len(dataset) - split
                self.validation_log_loss = float(
                    -np.mean(
                        np.log(
                            np.clip(
                                probe_probs[np.arange(len(outcomes) - split), outcomes[split:]],
                                1e-15,
                                1.0,
                            )
                        )
                    )
                )

        final_params = dict(params)
        final_params["n_estimators"] = chosen
        classifier = XGBClassifier(**final_params)
        classifier.fit(matrix, outcomes, verbose=False)
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
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        return {
            "random_seed": self.random_seed,
            "window": self.window,
            "validation_fraction": self.validation_fraction,
            "early_stopping_rounds": self.early_stopping_rounds,
            "selected_trees": self.selected_trees,
            "n_estimators_ceiling": int(self.params.get("n_estimators", 750)),
            "validation_matches": self.validation_matches,
            "validation_log_loss": self.validation_log_loss,
            **params,
        }

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

        The rolling feature state travels with the artefact, so ``predict_next``
        keeps producing the forecasts the saved model produced instead of
        silently falling back to league priors.
        """
        hyper = artifact.get("hyperparameters") or {}
        params = {k: v for k, v in hyper.items() if k not in _WRAPPER_KEYS}
        model = cls(
            random_seed=int(hyper.get("random_seed", 42)),
            window=int(hyper.get("window", 6)),
            params=params or None,
            validation_fraction=float(hyper.get("validation_fraction", 0.0)),
            early_stopping_rounds=int(hyper.get("early_stopping_rounds", 0)),
        )
        model.selected_trees = hyper.get("selected_trees")
        classifier = XGBClassifier()
        classifier.load_model(bytearray(base64.b64decode(artifact["booster_ubj_base64"])))
        model.classifier = classifier
        model.n_train_matches = int(artifact.get("n_train_matches", 0))
        model.builder.import_state(artifact.get("feature_state") or {})
        return model
