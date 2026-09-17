"""Shared test fixtures: synthetic match streams for the ML tests (no database)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.ml.data import MatchInput, ScorelinePrediction

TEAMS: tuple[str, ...] = tuple(f"Team {index:02d}" for index in range(12))
DEFAULT_SEASONS: tuple[str, ...] = ("2022/23", "2023/24", "2024/25")
MATCHES_PER_SEASON = 180


def build_schedule(
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    matches_per_season: int = MATCHES_PER_SEASON,
    seed: int = 11,
) -> tuple[list[str], list[MatchInput]]:
    """Synthesise matches from a latent Poisson model with a real home advantage.

    Returns ``(season labels, matches)`` aligned and already in kick-off order.
    """
    rng = np.random.default_rng(seed)
    attack = {team: float(rng.normal(0.0, 0.25)) for team in TEAMS}
    defence = {team: float(rng.normal(0.0, 0.25)) for team in TEAMS}

    labels: list[str] = []
    matches: list[MatchInput] = []
    kickoff = datetime(2022, 8, 12, 19, 0, tzinfo=timezone.utc)
    pairings = matches_per_season // (len(TEAMS) // 2)

    for label in seasons:
        for _ in range(pairings):
            order = rng.permutation(len(TEAMS))
            for position in range(0, len(TEAMS), 2):
                home = TEAMS[int(order[position])]
                away = TEAMS[int(order[position + 1])]
                lam = math.exp(0.25 + attack[home] - defence[away])
                mu = math.exp(attack[away] - defence[home])
                matches.append(
                    MatchInput(
                        home=home,
                        away=away,
                        kickoff=kickoff,
                        home_goals=int(rng.poisson(lam)),
                        away_goals=int(rng.poisson(mu)),
                        match_id=len(matches) + 1,
                    )
                )
                labels.append(label)
                kickoff += timedelta(hours=6)
        kickoff = kickoff.replace(year=kickoff.year + 1, month=8, day=12)

    return labels, matches


@pytest.fixture(scope="session")
def synthetic_dataset() -> tuple[list[str], list[MatchInput]]:
    """Three seasons of synthetic La Liga-shaped results."""
    return build_schedule()


@pytest.fixture(scope="session")
def synthetic_tail() -> tuple[list[str], list[MatchInput]]:
    """The last two synthetic seasons only: a small, fast dataset."""
    labels, matches = build_schedule()
    return labels[MATCHES_PER_SEASON:], matches[MATCHES_PER_SEASON:]


class FlatModel:
    """Stand-in Dixon-Coles model: the same 2x2 score grid for every fixture."""

    def predict(self, home: str, away: str) -> ScorelinePrediction:
        """The same four scorelines, all equally likely."""
        return ScorelinePrediction(
            p_home=0.25,
            p_draw=0.5,
            p_away=0.25,
            exp_home_goals=0.5,
            exp_away_goals=0.5,
            max_goals=1,
            matrix=[[0.25, 0.25], [0.25, 0.25]],
        )


@pytest.fixture(scope="session")
def flat_model() -> FlatModel:
    """A model with no team information, used to test the simulation maths."""
    return FlatModel()


SIX_SEASON_LABELS: tuple[str, ...] = (
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
)


@pytest.fixture(scope="session")
def synthetic_six_seasons() -> tuple[list[str], list[MatchInput]]:
    """Six synthetic seasons: one to pre-train on, then a five-season test window."""
    return build_schedule(seasons=SIX_SEASON_LABELS, seed=13)
