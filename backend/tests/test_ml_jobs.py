"""Tests for the Phase 2 job helpers (all of them database-free)."""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.ml.data import ScorelinePrediction
from app.ml.dataset import to_match_input
from app.ml.store import _rounded_probs, feature_hash
from jobs.simulate_season import current_table, parse_forced, simulate
from jobs.train_model import parse_seasons

KICKOFF = datetime(2024, 9, 1, 18, 0, tzinfo=timezone.utc)
NAMES = {1: "Team 00", 2: "Team 01"}


def fake_match(**overrides):
    """A stand-in ``matches`` row exposing only the columns the adapters read."""
    fields = {
        "id": 10,
        "home_team_id": 1,
        "away_team_id": 2,
        "kickoff_utc": KICKOFF,
        "home_goals": 2,
        "away_goals": 1,
        "home_shots_on_tgt": 5,
        "away_shots_on_tgt": 3,
        "closing_p_home": None,
        "closing_p_draw": None,
        "closing_p_away": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_to_match_input_maps_the_orm_row():
    """Team ids become canonical names and the score survives verbatim."""
    built = to_match_input(fake_match(), NAMES)
    assert built is not None
    assert (built.home, built.away) == ("Team 00", "Team 01")
    assert (built.home_goals, built.away_goals) == (2, 1)
    assert built.match_id == 10
    assert built.kickoff == KICKOFF
    assert built.home_shots_on_tgt == 5
    assert built.closing_p_home is None


def test_to_match_input_skips_unfinished_fixtures():
    """A fixture without a final score is not training data."""
    assert to_match_input(fake_match(home_goals=None), NAMES) is None
    assert to_match_input(fake_match(away_goals=None), NAMES) is None


def test_to_match_input_keeps_closing_odds_as_floats():
    """Numeric columns arrive as Decimal and must be cast for the market model."""
    built = to_match_input(
        fake_match(
            closing_p_home=Decimal("0.5000"),
            closing_p_draw=Decimal("0.2500"),
            closing_p_away=Decimal("0.2500"),
        ),
        NAMES,
    )
    assert built is not None
    assert built.closing_p_home == pytest.approx(0.5)
    assert (built.closing_p_draw, built.closing_p_away) == pytest.approx((0.25, 0.25))


def test_current_table_counts_points_and_goal_difference():
    """Standings are derived from the fixtures, not from an ingested table."""
    played = [
        fake_match(home_team_id=1, away_team_id=2, home_goals=3, away_goals=0),
        fake_match(id=11, home_team_id=2, away_team_id=1, home_goals=1, away_goals=1),
    ]
    table = current_table(NAMES, played)
    assert table["Team 00"] == {"played": 2, "points": 4, "gd": 3, "gf": 4}
    assert table["Team 01"] == {"played": 2, "points": 1, "gd": -3, "gf": 1}


def four_team_table():
    """A fresh, all-zero table for the simulation tests."""
    return {
        team: {"played": 0, "points": 0, "gd": 0, "gf": 0}
        for team in ("Team 00", "Team 01", "Team 02", "Team 03")
    }


def test_simulate_produces_a_complete_distribution(flat_model):
    """Exactly one champion per simulation, so the title shares sum to one."""
    remaining = [("Team 00", "Team 01"), ("Team 02", "Team 03")]
    results = simulate(flat_model, four_team_table(), remaining, 500, 7, {})
    assert results["simulated_matches"] == 2
    assert results["tiebreak"].startswith("points")
    rows = {row["team"]: row for row in results["teams"]}
    assert len(rows) == 4
    assert sum(row["p_title"] for row in rows.values()) == pytest.approx(1.0, abs=1e-4)
    for row in rows.values():
        assert row["p_title"] <= row["p_top4"] <= 1.0
        assert row["p_europe"] >= row["p_top4"]
        assert 0.0 <= row["p_relegation"] <= 1.0
        assert row["points_p10"] <= row["expected_points"] <= row["points_p90"]


def test_simulate_honours_forced_results_without_randomness(flat_model):
    """A fully pinned scenario is deterministic, whatever the seed."""
    table = {
        "Team 00": {"played": 0, "points": 0, "gd": 0, "gf": 0},
        "Team 01": {"played": 0, "points": 0, "gd": 0, "gf": 0},
    }
    remaining = [("Team 00", "Team 01")]
    forced = {"Team 00 vs Team 01": (5, 0)}
    first = simulate(flat_model, table, remaining, 64, 1, forced)
    second = simulate(flat_model, table, remaining, 64, 999, forced)
    assert first == second
    winner = next(row for row in first["teams"] if row["team"] == "Team 00")
    assert winner["p_title"] == 1.0
    assert winner["expected_points"] == 3.0


def test_simulate_is_reproducible_for_a_fixed_seed(flat_model):
    """Same seed and inputs, same probabilities."""
    remaining = [("Team 00", "Team 01"), ("Team 02", "Team 03")]
    first = simulate(flat_model, four_team_table(), remaining, 200, 5, {})
    second = simulate(flat_model, four_team_table(), remaining, 200, 5, {})
    assert first == second
    other = simulate(flat_model, four_team_table(), remaining, 200, 6, {})
    assert other != first


def test_feature_hash_is_order_independent_and_stable():
    """The audit key must not change when a dict is rebuilt in another order."""
    assert feature_hash({"home": "A", "away": "B"}) == feature_hash({"away": "B", "home": "A"})
    assert feature_hash({"home": "A"}) != feature_hash({"home": "B"})
    assert len(feature_hash({"home": "A"})) == 16


def test_rounded_probs_satisfy_the_predictions_check_constraint():
    """``predictions.p_*`` must sum to 1 within 0.001 after rounding."""
    prediction = ScorelinePrediction(
        p_home=0.333333,
        p_draw=0.333333,
        p_away=0.333334,
        exp_home_goals=1.2,
        exp_away_goals=1.1,
        max_goals=10,
        matrix=[[0.0]],
    )
    values = _rounded_probs(prediction)
    assert sum(values) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert [round(value, 5) for value in values] == list(values)


def test_parse_seasons_defaults_to_the_five_season_window():
    """The documented default is the last five La Liga seasons."""
    assert parse_seasons(None) == [
        "2020/21",
        "2021/22",
        "2022/23",
        "2023/24",
        "2024/25",
    ]
    assert parse_seasons("") == parse_seasons(None)
    assert parse_seasons("2023/24, 2024/25") == ["2023/24", "2024/25"]


def test_parse_forced_builds_fixture_keys():
    """``--force`` values become ``{'HOME vs AWAY': (goals, goals)}``."""
    assert parse_forced(["Team 00:Team 01:2-1"]) == {"Team 00 vs Team 01": (2, 1)}
    assert parse_forced([]) == {}


@pytest.mark.parametrize("value", ["Team 00 vs Team 01", "Team 00:Team 01", "A:B:x-y"])
def test_parse_forced_rejects_bad_input(value):
    """A malformed scenario must abort the run instead of being ignored."""
    with pytest.raises(SystemExit):
        parse_forced([value])


def test_parse_forced_round_trips_through_system_exit_messages():
    """A valid scenario never raises, even with surrounding whitespace."""
    assert parse_forced([" Team 00 : Team 01 : 1-0 "]) == {"Team 00 vs Team 01": (1, 0)}
