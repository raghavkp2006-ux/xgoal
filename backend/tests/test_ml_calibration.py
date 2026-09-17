"""Tests for app.ml.calibration — Phase 2.5 isotonic calibration."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.calibration import IsotonicCalibrator, reliability_lines
from app.ml.data import Probs
from app.ml.metrics import expected_calibration_error, log_loss

TOO_SURE = 0.9


def overconfident_sample(n: int, hit_rate: float, seed: int) -> tuple[list[Probs], list[int]]:
    """Forecasts that shout 0.9 at a class that only wins ``hit_rate`` of the time."""
    rng = np.random.default_rng(seed)
    probs: list[Probs] = []
    outcomes: list[int] = []
    rest = (1.0 - TOO_SURE) / 2
    for _ in range(n):
        outcome = int(rng.integers(0, 3))
        if rng.random() < hit_rate:
            pick = outcome
        else:
            pick = int(rng.choice([candidate for candidate in range(3) if candidate != outcome]))
        row = [rest, rest, rest]
        row[pick] = TOO_SURE
        probs.append((row[0], row[1], row[2]))
        outcomes.append(outcome)
    return probs, outcomes


def test_isotonic_repairs_an_overconfident_forecaster():
    """Fit on the calibration split, then score a held-out split: ECE must drop."""
    calibration_probs, calibration_outcomes = overconfident_sample(600, 0.6, seed=1)
    test_probs, test_outcomes = overconfident_sample(600, 0.6, seed=2)

    calibrator = IsotonicCalibrator().fit(calibration_probs, calibration_outcomes)
    calibrated = calibrator.transform(test_probs)

    before = expected_calibration_error(test_probs, test_outcomes)
    after = expected_calibration_error(calibrated, test_outcomes)
    assert before > 0.2, "the fixture is supposed to start badly calibrated"
    assert after < 0.05
    assert after < before / 4
    assert log_loss(calibrated, test_outcomes) < log_loss(test_probs, test_outcomes)


def test_calibrated_rows_are_still_a_valid_distribution():
    """Renormalising the three isotonic columns keeps probabilities valid."""
    probs, outcomes = overconfident_sample(400, 0.5, seed=3)
    calibrated = IsotonicCalibrator().fit(probs, outcomes).transform(probs)
    assert len(calibrated) == len(probs)
    for row in calibrated:
        assert sum(row) == pytest.approx(1.0)
        assert all(0.0 < value < 1.0 for value in row)


def test_artifact_round_trip_reproduces_the_mapping():
    """Serving reloads the knots, so the reloaded mapping must be identical."""
    probs, outcomes = overconfident_sample(500, 0.55, seed=4)
    calibrator = IsotonicCalibrator(shrink=0.4).fit(probs, outcomes)
    restored = IsotonicCalibrator.from_artifact(calibrator.to_artifact())
    assert restored.fitted
    assert restored.shrink == calibrator.shrink
    assert restored.n_train_matches == calibrator.n_train_matches
    assert restored.transform(probs) == pytest.approx(calibrator.transform(probs))


def test_zero_shrink_leaves_the_forecast_untouched():
    """``shrink=0`` is the control arm: the raw forecast, renormalised only."""
    probs, outcomes = overconfident_sample(200, 0.6, seed=6)
    identity = IsotonicCalibrator(shrink=0.0).fit(probs, outcomes)
    assert identity.transform(probs) == pytest.approx(probs)


def test_shrinkage_interpolates_between_raw_and_isotonic():
    """A partial mix sits between the two ends of the shrinkage range."""
    probs, outcomes = overconfident_sample(400, 0.6, seed=7)
    raw = IsotonicCalibrator(shrink=0.0).fit(probs, outcomes).transform(probs)
    full = IsotonicCalibrator(shrink=1.0).fit(probs, outcomes).transform(probs)
    half = IsotonicCalibrator(shrink=0.5).fit(probs, outcomes).transform(probs)
    for index, row in enumerate(half):
        assert min(raw[index][0], full[index][0]) <= row[0] <= max(raw[index][0], full[index][0])
        assert sum(row) == pytest.approx(1.0)


@pytest.mark.parametrize("shrink", [-0.1, 1.1, 2.0])
def test_shrink_must_be_a_fraction(shrink):
    """Out-of-range shrinkage is a configuration error, not a silent clamp."""
    with pytest.raises(ValueError):
        IsotonicCalibrator(shrink=shrink)


def test_reliability_lines_report_every_occupied_bin():
    """The diagram is reproducible text: predicted, observed and their gap."""
    probs, outcomes = overconfident_sample(300, 0.6, seed=5)
    lines = reliability_lines(probs, outcomes, n_bins=5, title="before")
    assert lines[0].endswith("before")
    body = lines[3:]
    assert body, "expected at least one populated bin"
    for line in body:
        parts = line.split()
        assert len(parts) == 5
        predicted, observed, gap = (float(value) for value in parts[2:])
        assert predicted - observed == pytest.approx(gap, abs=1e-4)
        assert predicted > observed, "the sample is over-confident by construction"
    assert sum(int(line.split()[1]) for line in body) == len(probs)


def test_unfitted_calibrator_refuses_to_transform():
    """Using a calibrator before fitting is an error, not a silent identity map."""
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform([(0.5, 0.3, 0.2)])
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().to_artifact()


def test_fit_validates_input():
    """Empty, misaligned or impossible input must raise."""
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([], [])
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([(0.5, 0.3, 0.2)], [])
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([(0.5, 0.3, 0.2)], [3])
