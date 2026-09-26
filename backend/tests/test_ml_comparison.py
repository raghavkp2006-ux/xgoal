"""Phase 2.4 tests — the comparison models (multinomial logistic, regularised XGBoost).

Two of these test bugs that actually occurred while building this phase:

* ``params`` used to *replace* the model defaults instead of merging over them,
  so a hyper-parameter search silently ran XGBoost at sklearn's
  ``learning_rate=0.3`` and logistic at ``max_iter=100`` (with convergence
  warnings). The CV numbers were wrong until that was fixed.
* the plan fixes the regularisation envelope, so the shipped defaults are
  asserted directly — a future edit that loosens them should fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.evaluation import walk_forward
from app.ml.features import FEATURE_NAMES
from app.ml.logistic import DEFAULT_PARAMS as LOGISTIC_DEFAULTS
from app.ml.logistic import LogisticModel
from app.ml.metrics import log_loss
from app.ml.xgb_model import DEFAULT_EARLY_STOPPING_ROUNDS, XGBoostModel
from app.ml.xgb_model import DEFAULT_PARAMS as XGB_DEFAULTS

# --------------------------------------------------------------------------- #
# multinomial logistic regression
# --------------------------------------------------------------------------- #


def test_logistic_learns_and_round_trips(synthetic_tail):
    """The artifact must reproduce the forecasts of the model that wrote it."""
    _, matches = synthetic_tail
    train, probe = matches[:300], matches[300:330]
    model = LogisticModel().fit(train)
    artifact = model.to_artifact()
    matrix = model.predict_next(probe)
    assert matrix.shape == (len(probe), 3)
    assert matrix.sum(axis=1) == pytest.approx(np.ones(len(probe)))

    restored = LogisticModel.from_artifact(artifact)
    assert restored.predict_next(probe) == pytest.approx(matrix)
    assert restored.n_train_matches == model.n_train_matches


def test_logistic_artifact_carries_the_feature_state(synthetic_tail):
    """Reloading must not silently fall back to the league priors."""
    _, matches = synthetic_tail
    model = LogisticModel().fit(matches[:300])
    restored = LogisticModel.from_artifact(model.to_artifact())
    assert restored.builder.history == model.builder.history
    assert restored.builder.history


def test_logistic_importance_covers_every_feature(synthetic_tail):
    """Coefficients are reported for the shared feature contract."""
    _, matches = synthetic_tail
    model = LogisticModel().fit(matches[:250])
    importance = model.feature_importance(top=len(FEATURE_NAMES))
    assert set(importance) == set(FEATURE_NAMES)
    assert all(value >= 0.0 for value in importance.values())


def test_logistic_coefficients_are_per_class(synthetic_tail):
    """One coefficient per feature per outcome class, aligned to FEATURE_NAMES."""
    _, matches = synthetic_tail
    model = LogisticModel().fit(matches[:250])
    coefficients = model.coefficients()
    assert set(coefficients) == set(FEATURE_NAMES)
    assert all(len(values) == 3 for values in coefficients.values())


def test_logistic_beats_the_uniform_baseline(synthetic_dataset):
    """A linear model on these features must clear the flat 1/3 floor."""
    labels, matches = synthetic_dataset
    train = [m for label, m in zip(labels, matches, strict=True) if label != "2024/25"]
    test = [m for label, m in zip(labels, matches, strict=True) if label == "2024/25"]
    model = LogisticModel().fit(train)
    ours = log_loss(model.predict_next_probs(test), [m.outcome for m in test])
    uniform = log_loss([(1 / 3, 1 / 3, 1 / 3)] * len(test), [m.outcome for m in test])
    assert ours < uniform


def test_logistic_params_merge_over_defaults(synthetic_tail):
    """A partial override must not drop max_iter (the bug that broke the CV)."""
    _, matches = synthetic_tail
    model = LogisticModel(params={"C": 0.3}).fit(matches[:200])
    assert model.params["C"] == 0.3
    assert model.params["max_iter"] == LOGISTIC_DEFAULTS["max_iter"]
    assert "solver" in model.params


# --------------------------------------------------------------------------- #
# XGBoost regularisation envelope (build plan 2.4)
# --------------------------------------------------------------------------- #


def test_xgboost_defaults_sit_inside_the_plans_regularisation_envelope():
    """max_depth 3-4, min_child_weight >= 10, subsample/colsample 0.8."""
    assert 3 <= XGB_DEFAULTS["max_depth"] <= 4
    assert XGB_DEFAULTS["min_child_weight"] >= 10
    assert XGB_DEFAULTS["subsample"] == 0.8
    assert XGB_DEFAULTS["colsample_bytree"] == 0.8
    assert DEFAULT_EARLY_STOPPING_ROUNDS > 0


def test_xgboost_params_merge_over_defaults(synthetic_tail):
    """A CV override must keep learning_rate and reg_lambda from the defaults."""
    _, matches = synthetic_tail
    model = XGBoostModel(params={"max_depth": 3, "min_child_weight": 20}).fit(
        matches[:250]
    )
    assert model.params["max_depth"] == 3
    assert model.params["min_child_weight"] == 20
    assert model.params["learning_rate"] == XGB_DEFAULTS["learning_rate"]
    assert model.params["reg_lambda"] == XGB_DEFAULTS["reg_lambda"]


def test_early_stopping_selects_the_tree_count_and_refits_on_everything(
    synthetic_tail,
):
    """Early stopping picks n_estimators; the final model trains on all rows."""
    _, matches = synthetic_tail
    train = matches[:300]
    model = XGBoostModel(random_seed=7).fit(train)
    assert model.n_train_matches == len(train)
    assert model.validation_matches > 0
    assert model.selected_trees is not None
    assert 1 <= model.selected_trees <= XGB_DEFAULTS["n_estimators"]
    # The shipped booster is exactly the trees selected, so the stored model
    # reproduces its own forecasts after a reload.
    restored = XGBoostModel.from_artifact(model.to_artifact())
    probe = matches[300:330]
    assert restored.predict_next(probe) == pytest.approx(model.predict_next(probe))


def test_early_stopping_can_be_disabled(synthetic_tail):
    """With early stopping off the ceiling is used and no fold is carved."""
    _, matches = synthetic_tail
    model = XGBoostModel(
        random_seed=7, validation_fraction=0.0, early_stopping_rounds=0
    ).fit(matches[:250])
    assert model.selected_trees is None
    assert model.validation_matches == 0


# --------------------------------------------------------------------------- #
# the harness scores both comparison models on identical folds
# --------------------------------------------------------------------------- #


def test_walk_forward_scores_both_comparison_models(synthetic_dataset):
    """Logistic and XGBoost must both appear, over exactly the test matches."""
    labels, matches = synthetic_dataset
    test_seasons = ["2023/24", "2024/25"]
    result = walk_forward(labels, matches, test_seasons)

    expected = {season: sum(1 for label in labels if label == season) for season in test_seasons}
    assert result.n_test_matches == sum(expected.values())
    for model in ("dixon_coles", "elo", "base_rate", "uniform", "xgboost", "logistic"):
        for season, count in expected.items():
            assert result.coverage_for(season, model) == count, f"{model} uncovered for {season}"


def test_walk_forward_accepts_per_model_params(synthetic_dataset):
    """The CV path must be able to drive one model's hyper-parameters."""
    labels, matches = synthetic_dataset
    result = walk_forward(
        labels,
        matches,
        ["2024/25"],
        logistic_params={"C": 0.03},
        xgb_params={"max_depth": 3, "min_child_weight": 20},
    )
    assert "logistic" in result.per_model
    assert "xgboost" in result.per_model


def test_comparison_models_can_be_excluded(synthetic_dataset):
    """Selection folds need to run one family at a time."""
    labels, matches = synthetic_dataset
    result = walk_forward(
        labels,
        matches,
        ["2024/25"],
        include_xgboost=False,
        include_logistic=True,
    )
    assert "logistic" in result.per_model
    assert "xgboost" not in result.per_model
