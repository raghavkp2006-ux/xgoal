"""Forecast evaluation metrics: log loss (primary), RPS, Brier and ECE.

All metrics take an ``(n, 3)`` matrix of (home, draw, away) probabilities plus
the realised outcome indices and are lower-is-better.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.ml.data import FloatArray, IntArray

_EPS = 1e-15
N_OUTCOMES = 3


def _stack(probs: Sequence[Sequence[float]]) -> FloatArray:
    """Validate and renormalise an (n, 3) probability matrix."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_OUTCOMES:
        raise ValueError(f"expected an (n, {N_OUTCOMES}) probability matrix")
    if arr.shape[0] == 0:
        raise ValueError("cannot score an empty set of predictions")
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=1, keepdims=True)
    if bool((totals <= 0.0).any()):
        raise ValueError("every probability row must sum to a positive number")
    return arr / totals


def _outcomes(outcomes: Sequence[int]) -> IntArray:
    """Validate outcome indices."""
    y = np.asarray(outcomes, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("outcomes must be a flat sequence")
    if bool(((y < 0) | (y >= N_OUTCOMES)).any()):
        raise ValueError(f"outcome indices must be in 0..{N_OUTCOMES - 1}")
    return y


def _check_len(probs: FloatArray, outcomes: IntArray) -> int:
    """Ensure predictions and outcomes line up; return the sample count."""
    if probs.shape[0] != outcomes.shape[0]:
        raise ValueError("predictions and outcomes must have matching lengths")
    return int(probs.shape[0])


def log_loss(probs: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    """Mean negative log-likelihood of the realised outcome (lower is better)."""
    p = _stack(probs)
    y = _outcomes(outcomes)
    _check_len(p, y)
    picked = np.clip(p[np.arange(y.shape[0]), y], _EPS, 1.0)
    return float(-np.mean(np.log(picked)))


def ranked_probability_score(
    probs: Sequence[Sequence[float]], outcomes: Sequence[int]
) -> float:
    """Mean ranked probability score for the ordered H < D < A scale."""
    p = _stack(probs)
    y = _outcomes(outcomes)
    _check_len(p, y)
    cdf_pred = np.cumsum(p, axis=1)[:, : N_OUTCOMES - 1]
    one_hot = np.eye(N_OUTCOMES, dtype=np.float64)[y]
    cdf_true = np.cumsum(one_hot, axis=1)[:, : N_OUTCOMES - 1]
    spread = np.sum((cdf_pred - cdf_true) ** 2, axis=1) / (N_OUTCOMES - 1)
    return float(np.mean(spread))


def brier_score(probs: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    """Mean multiclass Brier score (sum of squared errors over the 3 outcomes)."""
    p = _stack(probs)
    y = _outcomes(outcomes)
    _check_len(p, y)
    one_hot = np.eye(N_OUTCOMES, dtype=np.float64)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: Sequence[Sequence[float]], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    """Confidence-based ECE over the predicted class (predicted prob vs accuracy)."""
    p = _stack(probs)
    y = _outcomes(outcomes)
    _check_len(p, y)
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = (predicted == y).astype(np.float64)
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    total = float(p.shape[0])
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        ece += (n_bin / total) * gap
    return ece


def accuracy(probs: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    """Share of fixtures where the most likely outcome is the one that happened.

    This is the "pick" accuracy a tipster would quote. It is reported, not
    optimised: for a three-way market a well-calibrated model lands in the low
    fifties, and a higher number usually means over-confident, badly calibrated
    picks rather than a better model.
    """
    p = _stack(probs)
    y = _outcomes(outcomes)
    _check_len(p, y)
    return float(np.mean(p.argmax(axis=1) == y))


def evaluate_predictions(
    probs: Sequence[Sequence[float]], outcomes: Sequence[int]
) -> dict[str, float]:
    """Every metric in one dict, ready for ``model_versions.eval_metrics``."""
    return {
        "log_loss": log_loss(probs, outcomes),
        "rps": ranked_probability_score(probs, outcomes),
        "brier": brier_score(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
    }
