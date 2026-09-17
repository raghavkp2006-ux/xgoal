"""Tests for the walk-forward evaluation harness."""

import pytest

from app.ml.evaluation import PRIMARY_METRIC, walk_forward


def test_walk_forward_scores_every_forecaster(synthetic_dataset):
    """Every registered forecaster is pooled over the test season."""
    labels, matches = synthetic_dataset
    result = walk_forward(labels, matches, ["2024/25"], include_xgboost=False)
    assert result.test_seasons == ["2024/25"]
    assert result.n_test_matches > 0
    assert {"dixon_coles", "elo", "base_rate", "uniform"} <= set(result.per_model)
    for name in result.per_model:
        assert result.coverage[name] == result.n_test_matches
    assert len(result.per_season["2024/25"]) == len(result.per_model)


def test_walk_forward_payload_is_json_ready(synthetic_dataset):
    """``eval_metrics`` must be a JSONB-safe structure a job can store verbatim."""
    labels, matches = synthetic_dataset
    result = walk_forward(labels, matches, ["2024/25"], include_xgboost=False)
    payload = result.to_metrics_payload()
    assert payload["protocol"] == "walk_forward"
    assert payload["primary_metric"] == PRIMARY_METRIC
    assert payload["test_seasons"] == ["2024/25"]
    assert payload["n_test_matches"] == result.n_test_matches
    assert set(payload["pooled"]) == set(result.per_model)
    assert set(payload["per_season"]) == {"2024/25"}
    for metrics in payload["pooled"].values():
        assert set(metrics) == {"log_loss", "rps", "brier", "ece", "accuracy"}
        assert all(isinstance(value, float) for value in metrics.values())


def test_dixon_coles_beats_the_naive_forecasters(synthetic_dataset):
    """The parametric model must add value over base rate and uniform."""
    labels, matches = synthetic_dataset
    result = walk_forward(labels, matches, ["2023/24", "2024/25"], include_xgboost=False)
    ours = result.per_model["dixon_coles"][PRIMARY_METRIC]
    assert ours < result.per_model["uniform"][PRIMARY_METRIC]
    assert ours < result.per_model["base_rate"][PRIMARY_METRIC]
    assert result.ranking()[0][0] in {"dixon_coles", "elo"}
    assert any("51-54%" in line for line in result.accuracy_lines())
    assert any("pick accuracy inside" in line for line in result.target_lines())
    assert len(result.season_lines()) == 2
    assert result.summary_lines()[0].startswith("  model")


def test_market_baseline_is_absent_without_closing_odds(synthetic_dataset):
    """Synthetic matches carry no prices, so the market forecaster is skipped."""
    labels, matches = synthetic_dataset
    result = walk_forward(labels, matches, ["2024/25"], include_xgboost=False)
    assert "market" not in result.per_model


def test_walk_forward_includes_xgboost_when_asked(synthetic_tail):
    """The comparison model joins the pool behind the same coverage contract."""
    labels, matches = synthetic_tail
    result = walk_forward(labels, matches, ["2024/25"], include_xgboost=True)
    assert "xgboost" in result.per_model
    assert result.coverage["xgboost"] == result.n_test_matches


def test_walk_forward_needs_pretraining_matches(synthetic_dataset):
    """Testing on the very first season leaves nothing to train on."""
    labels, matches = synthetic_dataset
    with pytest.raises(ValueError):
        walk_forward(labels, matches, ["2022/23"], include_xgboost=False)


def test_walk_forward_rejects_misaligned_input(synthetic_dataset):
    """Labels and matches must describe the same fixtures."""
    labels, matches = synthetic_dataset
    with pytest.raises(ValueError):
        walk_forward(labels[:-1], matches, ["2024/25"], include_xgboost=False)


def test_walk_forward_rejects_unknown_seasons(synthetic_dataset):
    """A typo in ``--test-seasons`` must fail loudly, not silently score nothing."""
    labels, matches = synthetic_dataset
    with pytest.raises(ValueError):
        walk_forward(labels, matches, ["1999/00"], include_xgboost=False)
