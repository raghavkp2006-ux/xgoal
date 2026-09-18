"""DB-agnostic data structures shared by the prediction models.

The ML package deliberately knows nothing about SQLAlchemy: jobs convert ORM
rows into :class:`MatchInput` objects and feed them to the models, which keeps
every model unit-testable without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np
from numpy.typing import NDArray

TeamKey = Union[int, str]
"""Team identifier: DB id (``int``) or canonical name (``str``)."""

Probs = tuple[float, float, float]
"""Outcome probabilities, ordered as (home, draw, away)."""

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

OUTCOME_HOME = 0
OUTCOME_DRAW = 1
OUTCOME_AWAY = 2


def outcome_index(home_goals: int, away_goals: int) -> int:
    """Map a final scoreline onto the outcome index (0=home, 1=draw, 2=away)."""
    if home_goals > away_goals:
        return OUTCOME_HOME
    if home_goals < away_goals:
        return OUTCOME_AWAY
    return OUTCOME_DRAW


@dataclass(frozen=True)
class MatchInput:
    """A completed match, normalised for model training and evaluation."""

    home: TeamKey
    away: TeamKey
    kickoff: datetime
    home_goals: int
    away_goals: int
    match_id: int | None = None
    home_shots_on_tgt: int | None = None
    away_shots_on_tgt: int | None = None
    closing_p_home: float | None = None
    closing_p_draw: float | None = None
    closing_p_away: float | None = None

    @property
    def outcome(self) -> int:
        """Outcome index of the final scoreline."""
        return outcome_index(self.home_goals, self.away_goals)

    @property
    def goal_diff(self) -> int:
        """Home goals minus away goals."""
        return self.home_goals - self.away_goals


@dataclass(frozen=True)
class PointInTimeMatch:
    """One completed fixture as the point-in-time feature builder sees it.

    Everything the Phase 2.2 features need travels with the row — including the
    competition tier and season start year that the promotion flag and division
    indicator depend on — so the builder can be handed a plain list (a filtered
    view) instead of a database session.
    """

    match_id: int
    competition_id: int
    competition_tier: int
    season_id: int
    season_start_year: int
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    home_shots_on_tgt: int | None = None
    away_shots_on_tgt: int | None = None

    @property
    def outcome(self) -> int:
        """Outcome index of the final scoreline."""
        return outcome_index(self.home_goals, self.away_goals)

    def points(self, team_id: int) -> int:
        """League points ``team_id`` collected from this fixture (3/1/0)."""
        if self.home_goals == self.away_goals:
            return 1
        winner = self.home_team_id if self.home_goals > self.away_goals else self.away_team_id
        return 3 if winner == team_id else 0

    def goals_for(self, team_id: int) -> int:
        """Goals ``team_id`` scored in this fixture."""
        return self.home_goals if team_id == self.home_team_id else self.away_goals

    def goals_against(self, team_id: int) -> int:
        """Goals ``team_id`` conceded in this fixture."""
        return self.away_goals if team_id == self.home_team_id else self.home_goals

    def shots_on_target_for(self, team_id: int) -> int | None:
        """Shots on target ``team_id`` took, or ``None`` when unavailable."""
        return self.home_shots_on_tgt if team_id == self.home_team_id else self.away_shots_on_tgt

    def shots_on_target_against(self, team_id: int) -> int | None:
        """Shots on target ``team_id`` faced, or ``None`` when unavailable."""
        return self.away_shots_on_tgt if team_id == self.home_team_id else self.home_shots_on_tgt

    def involves(self, team_id: int) -> bool:
        """Whether either side of this fixture is ``team_id``."""
        return team_id in (self.home_team_id, self.away_team_id)

    def to_match_input(self) -> MatchInput:
        """The Phase 2.1 Elo / Phase 2.3 goals models' view of the same fixture."""
        return MatchInput(
            home=self.home_team_id,
            away=self.away_team_id,
            kickoff=self.kickoff_utc,
            home_goals=self.home_goals,
            away_goals=self.away_goals,
            match_id=self.match_id,
            home_shots_on_tgt=self.home_shots_on_tgt,
            away_shots_on_tgt=self.away_shots_on_tgt,
        )


@dataclass(frozen=True)
class MatchIdentity:
    """The fixture being forecast: who plays whom, where, when and in which division.

    Deliberately carries **no score**. A forecaster always knows the fixture it is
    pricing; it never knows the result, so the identity is the one row the leakage
    guard is allowed to read through the full table rather than the filtered view.
    """

    match_id: int
    competition_id: int
    competition_tier: int
    season_id: int
    season_start_year: int
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    matchday: int | None = None


@dataclass(frozen=True)
class MatrixSummary:
    """A score matrix reduced to the numbers the product and the simulator need."""

    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    max_goals: int


def summarise_score_matrix(matrix: FloatArray) -> MatrixSummary:
    """Sum a score matrix into W/D/L probabilities and expected goals.

    The single place where a scoreline distribution becomes an outcome
    distribution. Phase 2.7 (serving) and Phase 5 (the season simulator) both
    consume this, so they cannot drift apart. Rows are home goals, columns away
    goals; the matrix is normalised first, so callers may pass unnormalised
    Poisson products.
    """
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("score matrix must sum to a positive number")
    m = matrix / total
    size = m.shape[0]
    goals = np.arange(size, dtype=np.float64)
    return MatrixSummary(
        p_home=float(np.tril(m, -1).sum()),
        p_draw=float(np.trace(m)),
        p_away=float(np.triu(m, 1).sum()),
        exp_home_goals=float((m.sum(axis=1) * goals).sum()),
        exp_away_goals=float((m.sum(axis=0) * goals).sum()),
        max_goals=size - 1,
    )


@dataclass(frozen=True)
class ScorelinePrediction:
    """Outcome probabilities plus the full scoreline distribution behind them."""

    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    max_goals: int
    matrix: list[list[float]]

    @property
    def probs(self) -> Probs:
        """(p_home, p_draw, p_away) — the shape the evaluation metrics expect."""
        return (self.p_home, self.p_draw, self.p_away)

    @classmethod
    def from_matrix(cls, matrix: FloatArray) -> ScorelinePrediction:
        """Aggregate a square score matrix (rows=home goals, columns=away goals).

        Delegates the statistics to :func:`summarise_score_matrix` so there is
        exactly one implementation of the W/D/L and expected-goals summation.
        """
        summary = summarise_score_matrix(matrix)
        normalised = matrix / float(matrix.sum())
        cells = [[round(float(v), 6) for v in row] for row in normalised]
        return cls(
            p_home=summary.p_home,
            p_draw=summary.p_draw,
            p_away=summary.p_away,
            exp_home_goals=summary.exp_home_goals,
            exp_away_goals=summary.exp_away_goals,
            max_goals=summary.max_goals,
            matrix=cells,
        )
