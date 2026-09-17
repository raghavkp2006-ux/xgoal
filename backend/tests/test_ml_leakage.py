"""The 200-match leakage test: no forecast may see its own or any later result.

Implemented over a fixed 200-match window of the synthetic stream, from three
angles that together cover the whole point-in-time pipeline:

1. **prefix invariance** — the feature row for match *i* is identical whether the
   builder is fed ``matches[:i]`` or the entire stream, so nothing is read ahead;
2. **result tampering** — rewriting the result of match *i* and of every later
   match leaves the row used to forecast match *i* bit-identical;
3. **protocol isolation** — the walk-forward harness refits each test season only
   on matches that kicked off strictly before that season's first fixture.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from app.ml.dixon_coles import DixonColesModel
from app.ml.evaluation import walk_forward
from app.ml.features import FeatureBuilder

WINDOW = 200
HISTORY = 60


def window_slices(matches):
    """(history, 200-match test window) of the synthetic stream."""
    history = list(matches[:HISTORY])
    window = list(matches[HISTORY : HISTORY + WINDOW])
    assert len(window) == WINDOW
    return history, window


def test_feature_rows_are_prefix_invariant_over_200_matches(synthetic_dataset):
    """200/200 rows: the row for a match cannot depend on the full stream."""
    _, matches = synthetic_dataset
    history, window = window_slices(matches)

    full = FeatureBuilder()
    full.build(history)
    streamed = full.build(window)

    checked = 0
    for index in range(WINDOW):
        truncated = FeatureBuilder()
        truncated.build(history)
        row = truncated.build(window[: index + 1]).matrix[-1]
        if not np.array_equal(row, streamed.matrix[index]):
            pytest.fail(f"match {index} of the window saw data from its own future")
        checked += 1

    assert checked == WINDOW
    print(f"\nleakage (prefix invariance): {checked}/{WINDOW} rows unchanged")


def test_result_tampering_cannot_change_the_forecast_features(synthetic_dataset):
    """200/200 rows: own result and every later result are invisible."""
    _, matches = synthetic_dataset
    history, window = window_slices(matches)
    tampered = [
        replace(match, home_goals=9, away_goals=0, home_shots_on_tgt=20, away_shots_on_tgt=0)
        for match in window
    ]

    checked = 0
    for index in range(WINDOW):
        honest = FeatureBuilder()
        honest.build(history)
        honest_row = honest.build(window[index:]).matrix[0]

        rewritten = FeatureBuilder()
        rewritten.build(history)
        rewritten_row = rewritten.build(tampered[index:]).matrix[0]

        if not np.array_equal(honest_row, rewritten_row):
            pytest.fail(f"match {index} of the window used the result of itself or a later match")
        checked += 1

    assert checked == WINDOW
    print(f"\nleakage (result tampering): {checked}/{WINDOW} rows unchanged")


def test_walk_forward_only_trains_on_strictly_earlier_matches(synthetic_six_seasons, monkeypatch):
    """One refit per test season, each fitted strictly before its first kick-off."""
    labels, matches = synthetic_six_seasons
    test_seasons = sorted(set(labels))[1:]  # five scored seasons, one to pre-train on
    seen: list[tuple[int, datetime]] = []
    original_fit = DixonColesModel.fit

    def spy(self, train_matches):
        ordered = sorted(train_matches, key=lambda match: match.kickoff)
        seen.append((len(ordered), ordered[-1].kickoff))
        return original_fit(self, train_matches)

    monkeypatch.setattr(DixonColesModel, "fit", spy)
    result = walk_forward(labels, matches, test_seasons, include_xgboost=False)

    first_kickoff = {
        season: min(
            match.kickoff
            for label, match in zip(labels, matches, strict=True)
            if label == season
        )
        for season in test_seasons
    }
    assert len(seen) == len(test_seasons), "expected exactly one refit per test season"
    for season, (n_train, last_kickoff) in zip(test_seasons, seen, strict=True):
        assert n_train > 0
        assert last_kickoff < first_kickoff[season], f"{season} trained on its own fixtures"
    assert result.n_test_matches == sum(1 for label in labels if label in set(test_seasons))
    print(
        f"\nleakage (protocol): {len(seen)}/{len(test_seasons)} seasons trained strictly "
        f"before their first kick-off (train sizes {[n for n, _ in seen]})"
    )


def test_scored_forecasts_cover_every_test_match(synthetic_dataset):
    """The harness keeps the exact forecasts it scored, one per fixture, XGBoost included."""
    labels, matches = synthetic_dataset
    test_seasons = ["2023/24", "2024/25"]
    result = walk_forward(labels, matches, test_seasons, include_xgboost=True)

    per_season = sum(1 for label in labels if label in set(test_seasons))
    assert result.n_test_matches == per_season
    for season in test_seasons:
        for model in ("dixon_coles", "elo", "base_rate", "uniform", "xgboost"):
            assert result.coverage_for(season, model) == sum(
                1 for label in labels if label == season
            ), f"{model} is not covered for {season}"
