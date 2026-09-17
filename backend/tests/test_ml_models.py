"""Tests for the Dixon-Coles model, the baselines and the XGBoost comparison model."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from app.ml.baselines import BaseRateBaseline, EloBaseline, UniformBaseline, devig_probs
from app.ml.data import MatchInput
from app.ml.dixon_coles import DixonColesModel
from app.ml.features import FEATURE_NAMES
from app.ml.metrics import log_loss
from app.ml.xgb_model import XGBoostModel

KICKOFF = datetime(2024, 9, 1, 18, 0, tzinfo=timezone.utc)


def test_dixon_coles_learns_the_home_advantage(synthetic_dataset):
    """A league simulated with a home advantage must show one back."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    assert model.fitted
    assert model.home_adv > 0.1
    assert abs(model.rho) <= model.rho_bound
    assert model.n_train_matches == len(matches)
    assert model.train_start == min(match.kickoff for match in matches)
    assert model.train_end == max(match.kickoff for match in matches)


def test_attack_ratings_are_mean_centred(synthetic_dataset):
    """Centring keeps unseen teams at league average instead of offset."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    mean_attack = sum(model.attack.values()) / len(model.attack)
    assert mean_attack == pytest.approx(0.0, abs=1e-9)
    assert all(math.isfinite(value) for value in model.defence.values())


def test_predictions_are_valid_probabilities(synthetic_dataset):
    """Probabilities sum to one and the score matrix matches ``max_goals``."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    prediction = model.predict(matches[0].home, matches[0].away)
    assert sum(prediction.probs) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in prediction.probs)
    assert prediction.exp_home_goals > 0.0
    assert prediction.exp_away_goals > 0.0
    assert prediction.max_goals == model.max_goals
    assert len(prediction.matrix) == model.max_goals + 1


def test_strength_gap_shows_up_in_the_forecast(synthetic_dataset):
    """A strong home side must out-rank both its opponent and its away self."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    strong, weak = "Team 00", "Team 01"
    model.attack[strong], model.attack[weak] = 0.6, -0.6
    model.defence[strong], model.defence[weak] = -0.3, 0.3
    at_home = model.predict(strong, weak)
    away = model.predict(weak, strong)
    assert at_home.p_home > at_home.p_away
    assert at_home.p_home > away.p_home
    assert at_home.exp_home_goals > away.exp_home_goals


def test_unseen_teams_fall_back_to_the_league_average(synthetic_dataset):
    """A promoted side the model has never seen must not break prediction."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    prediction = model.predict("Nonexistent FC", "Also Missing FC")
    assert sum(prediction.probs) == pytest.approx(1.0)
    assert prediction.p_home > prediction.p_away


def test_dixon_coles_beats_the_uniform_baseline(synthetic_dataset):
    """Out-of-sample log loss must beat the flat 1/3 floor."""
    labels, matches = synthetic_dataset
    train = [m for label, m in zip(labels, matches, strict=True) if label != "2024/25"]
    test = [m for label, m in zip(labels, matches, strict=True) if label == "2024/25"]
    model = DixonColesModel().fit(train)
    ours = log_loss([model.predict(m.home, m.away).probs for m in test], [m.outcome for m in test])
    uniform = log_loss(
        [UniformBaseline().predict(m.home, m.away) for m in test], [m.outcome for m in test]
    )
    assert ours < uniform


def test_artifact_round_trip_reproduces_predictions(synthetic_dataset):
    """What is stored in ``ml/artifacts`` must predict like the original."""
    _, matches = synthetic_dataset
    model = DixonColesModel().fit(matches)
    restored = DixonColesModel.from_artifact(model.to_artifact())
    assert restored.predict("Team 00", "Team 05").probs == pytest.approx(
        model.predict("Team 00", "Team 05").probs
    )
    assert restored.home_adv == pytest.approx(model.home_adv)
    assert restored.rho == pytest.approx(model.rho)
    assert restored.n_train_matches == model.n_train_matches


def test_uniform_baseline_is_flat():
    """The uniform forecaster ignores both teams entirely."""
    assert UniformBaseline().fit([]).predict("A", "B") == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_base_rate_matches_the_training_frequencies(synthetic_dataset):
    """The base rate is exactly the observed draw frequency."""
    _, matches = synthetic_dataset
    baseline = BaseRateBaseline().fit(matches)
    draws = sum(1 for match in matches if match.outcome == 1) / len(matches)
    assert baseline.predict("A", "B")[1] == pytest.approx(draws)
    assert baseline.n_train_matches == len(matches)


def test_elo_stays_point_in_time(synthetic_dataset):
    """Predictions use the ratings from before a match; ``update`` moves them."""
    _, matches = synthetic_dataset
    elo = EloBaseline().fit(matches[:200])
    upcoming = matches[200]
    probs = elo.predict(upcoming.home, upcoming.away)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[2]
    before = elo.rating(upcoming.home)
    elo.update(upcoming)
    assert elo.rating(upcoming.home) != before


def test_devig_probs_is_none_without_closing_odds(synthetic_dataset):
    """Rows ingested from fixtures only carry no market prices."""
    _, matches = synthetic_dataset
    assert devig_probs(matches[0]) is None


def test_devig_probs_renormalises_to_one():
    """Legacy rows with an overround are renormalised, not trusted."""
    match = MatchInput(
        home="A",
        away="B",
        kickoff=KICKOFF,
        home_goals=1,
        away_goals=0,
        closing_p_home=0.50,
        closing_p_draw=0.30,
        closing_p_away=0.30,
    )
    probs = devig_probs(match)
    assert probs is not None
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[2]


def test_xgboost_learns_and_round_trips(synthetic_tail):
    """The artifact must reproduce the forecasts of the model that wrote it."""
    _, matches = synthetic_tail
    train, probe = matches[:300], matches[300:330]
    model = XGBoostModel(random_seed=7).fit(train)
    artifact = model.to_artifact()
    matrix = model.predict_next(probe)
    assert matrix.shape == (len(probe), 3)
    assert matrix.sum(axis=1) == pytest.approx(np.ones(len(probe)))

    restored = XGBoostModel.from_artifact(artifact)
    assert restored.predict_next(probe) == pytest.approx(matrix)
    assert restored.n_train_matches == model.n_train_matches
    assert restored.feature_importance(top=3) == model.feature_importance(top=3)


def test_xgboost_artifact_carries_the_feature_state(synthetic_tail):
    """Reloading must not silently fall back to the league priors."""
    _, matches = synthetic_tail
    train = matches[:300]
    model = XGBoostModel(random_seed=7).fit(train)
    restored = XGBoostModel.from_artifact(model.to_artifact())
    assert restored.builder.history == model.builder.history
    assert restored.builder.history


def test_xgboost_importance_covers_every_feature(synthetic_tail):
    """Importances are reported for the shared feature contract."""
    _, matches = synthetic_tail
    model = XGBoostModel().fit(matches[:250])
    importance = model.feature_importance(top=len(FEATURE_NAMES))
    assert set(importance) == set(FEATURE_NAMES)
    assert all(value >= 0.0 for value in importance.values())
