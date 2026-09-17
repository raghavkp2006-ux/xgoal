"""Tests for app.ml.features — the point-in-time rolling feature builder."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.ml.data import MatchInput
from app.ml.features import (
    FEATURE_NAMES,
    PRIOR_PPG,
    PRIOR_REST_DAYS,
    PRIOR_SOT_SHARE,
    FeatureBuilder,
)

KICKOFF = datetime(2024, 8, 16, 19, 0, tzinfo=timezone.utc)


def make_match(
    home,
    away,
    home_goals,
    away_goals,
    hours=0,
    home_sot=None,
    away_sot=None,
    match_id=None,
):
    """A synthetic finished match kicked off ``hours`` after the base time."""
    return MatchInput(
        home=home,
        away=away,
        kickoff=KICKOFF + timedelta(hours=hours),
        home_goals=home_goals,
        away_goals=away_goals,
        match_id=match_id,
        home_shots_on_tgt=home_sot,
        away_shots_on_tgt=away_sot,
    )


def test_first_fixture_uses_league_priors():
    """Nothing is known before kick-off, so every feature is a prior."""
    dataset = FeatureBuilder().build([make_match("A", "B", 2, 1)])
    row = dataset.matrix[0]
    assert row.shape[0] == len(FEATURE_NAMES)
    assert row[0] == PRIOR_PPG
    assert row[5] == PRIOR_REST_DAYS
    assert row[6] == 0.0
    assert row[7] == PRIOR_PPG


def test_features_only_reflect_earlier_matches():
    """The second fixture sees the first one, and nothing after it."""
    builder = FeatureBuilder()
    dataset = builder.build(
        [make_match("A", "B", 2, 1, hours=0), make_match("A", "B", 0, 3, hours=72)]
    )
    home, away = dataset.matrix[1][:7], dataset.matrix[1][7:]
    assert home[6] == 1.0
    assert home[0] == 3.0
    assert home[1] == 2.0
    assert home[2] == 1.0
    assert away[6] == 1.0
    assert away[0] == 0.0
    assert away[1] == 1.0
    assert away[2] == 2.0
    assert home[5] == pytest.approx(3.0)


def test_window_limits_the_rolling_history():
    """With ``window=2`` only the two most recent outings are counted."""
    builder = FeatureBuilder(window=2)
    matches = [make_match("A", "B", 1, 0, hours=index * 24) for index in range(4)]
    dataset = builder.build(matches)
    assert dataset.matrix[-1][6] == 2.0


def test_shot_share_uses_on_target_counts():
    """Shots on target define the documented xG proxy."""
    builder = FeatureBuilder()
    matches = [
        make_match("A", "B", 1, 0, hours=0, home_sot=6, away_sot=2),
        make_match("A", "B", 1, 0, hours=24),
    ]
    dataset = builder.build(matches)
    assert dataset.matrix[1][3] == pytest.approx(6 / 8)
    assert dataset.matrix[1][10] == pytest.approx(2 / 8)


def test_shot_share_falls_back_when_shot_data_is_missing():
    """Rows without shot columns contribute nothing, so the split stays neutral."""
    builder = FeatureBuilder()
    matches = [make_match("A", "B", 1, 0, hours=0), make_match("A", "B", 1, 0, hours=24)]
    dataset = builder.build(matches)
    assert dataset.matrix[1][3] == PRIOR_SOT_SHARE
    assert dataset.matrix[1][10] == PRIOR_SOT_SHARE


def test_dataset_is_sorted_by_kickoff_and_aligned():
    """Rows, outcomes and matches stay aligned even when fed out of order."""
    dataset = FeatureBuilder().build(
        [make_match("A", "B", 0, 2, hours=48), make_match("C", "D", 3, 0, hours=0)]
    )
    assert dataset.matches[0].kickoff < dataset.matches[1].kickoff
    assert dataset.outcomes.tolist() == [0, 2]
    assert len(dataset) == 2


def test_reset_forgets_every_team():
    """A reset builder is indistinguishable from a fresh one."""
    builder = FeatureBuilder()
    builder.build([make_match("A", "B", 2, 1)])
    builder.reset()
    assert builder.history == {}


def test_exported_state_survives_json_and_rebuilds_identically():
    """The XGBoost artifact carries this snapshot, so it must round-trip."""
    builder = FeatureBuilder()
    builder.build(
        [
            make_match("A", "B", 2, 1, hours=0, home_sot=4, away_sot=2),
            make_match("A", "B", 0, 3, hours=72),
        ]
    )
    snapshot = json.loads(json.dumps(builder.export_state()))

    restored = FeatureBuilder()
    restored.import_state(snapshot)
    assert restored.history == builder.history
    upcoming = make_match("A", "B", 1, 1, hours=200)
    assert restored.build([upcoming]).matrix == pytest.approx(builder.build([upcoming]).matrix)
