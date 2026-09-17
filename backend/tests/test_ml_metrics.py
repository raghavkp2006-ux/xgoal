"""Tests for app.ml.metrics — the forecast scoring functions."""

import math

import pytest

from app.ml.metrics import (
    accuracy,
    brier_score,
    evaluate_predictions,
    expected_calibration_error,
    log_loss,
    ranked_probability_score,
)


def test_log_loss_matches_hand_computation():
    """A 50% pick that comes in costs exactly ln(2)."""
    assert log_loss([[0.5, 0.3, 0.2]], [0]) == pytest.approx(math.log(2.0))


def test_uniform_forecast_costs_log_three():
    """The uniform forecaster is the reference line for every real model."""
    probs = [[1 / 3, 1 / 3, 1 / 3] for _ in range(9)]
    outcomes = [index % 3 for index in range(9)]
    assert log_loss(probs, outcomes) == pytest.approx(math.log(3.0))


def test_rps_is_zero_for_a_perfect_certain_forecast():
    """Committing to the realised outcome scores a perfect zero."""
    assert ranked_probability_score([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 2]) == 0.0


def test_rps_respects_the_ordered_outcome_scale():
    """Calling a home win when the away side wins is worse than calling a draw."""
    home_call = ranked_probability_score([[1.0, 0.0, 0.0]], [2])
    draw_call = ranked_probability_score([[0.0, 1.0, 0.0]], [2])
    assert home_call > draw_call > 0.0


def test_brier_score_is_hand_checkable():
    """(0.7 - 1)^2 + 0.2^2 + 0.1^2 = 0.14 for a realised home win."""
    assert brier_score([[0.7, 0.2, 0.1]], [0]) == pytest.approx(0.14)


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    """Six hits out of ten events called at 0.6 is perfectly calibrated."""
    probs = [[0.6, 0.2, 0.2]] * 10
    outcomes = [0] * 6 + [1, 1, 2, 2]
    assert expected_calibration_error(probs, outcomes) == pytest.approx(0.0)


def test_calibration_error_flags_an_overconfident_forecast():
    """Always shouting 0.9 and being right half the time is miscalibrated."""
    probs = [[0.9, 0.05, 0.05]] * 10
    outcomes = [0] * 5 + [2] * 5
    assert expected_calibration_error(probs, outcomes) == pytest.approx(0.4)


def test_accuracy_is_the_share_of_correct_picks():
    """Two right out of three picks is 2/3."""
    probs = [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.5, 0.2, 0.3]]
    assert accuracy(probs, [0, 2, 2]) == pytest.approx(2 / 3)


def test_uniform_forecast_accuracy_is_the_home_win_rate():
    """A flat forecast always picks home, so its accuracy is the home-win share."""
    probs = [[1 / 3, 1 / 3, 1 / 3] for _ in range(4)]
    assert accuracy(probs, [0, 0, 1, 2]) == pytest.approx(0.5)


def test_evaluate_predictions_returns_every_metric():
    metrics = evaluate_predictions([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]], [0, 2])
    assert set(metrics) == {"log_loss", "rps", "brier", "ece", "accuracy"}
    assert all(isinstance(value, float) for value in metrics.values())
    assert metrics["accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("probs", "outcomes"),
    [
        ([], []),
        ([[0.5, 0.5]], [0]),
        ([[0.5, 0.3, 0.2]], [0, 1]),
        ([[0.5, 0.3, 0.2]], [3]),
        ([[0.0, 0.0, 0.0]], [0]),
    ],
)
def test_metrics_reject_malformed_input(probs, outcomes):
    """Empty, ragged, misaligned or impossible input must raise, not guess."""
    with pytest.raises(ValueError):
        log_loss(probs, outcomes)
