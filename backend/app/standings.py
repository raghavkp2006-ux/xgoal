"""La Liga standings and head-to-head tie-breaking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Protocol

import numpy as np
from numpy.typing import NDArray

class SeasonMatch(Protocol):
    """The match fields needed by the pure standings ranker."""

    season_id: int
    home_team_id: Hashable
    away_team_id: Hashable
    home_goals: int | None
    away_goals: int | None


@dataclass(frozen=True)
class TeamRow:
    """One club's reconstructed season totals and final position."""

    position: int
    team_id: Hashable
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    points: int

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class _Totals:
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0


def rank_standings(matches: list[SeasonMatch], season_id: int) -> list[TeamRow]:
    """Rebuild and rank one season from its complete fixture list.

    Include scheduled but unfinished fixtures in ``matches`` so the ranker can
    tell whether every head-to-head fixture for a tied group has been played.
    """
    selected = [match for match in matches if match.season_id == season_id]
    teams = {
        team_id
        for match in selected
        for team_id in (match.home_team_id, match.away_team_id)
    }
    totals = {team_id: _Totals() for team_id in teams}
    pair_fixtures: Counter[frozenset[Hashable]] = Counter()
    pair_results: Counter[frozenset[Hashable]] = Counter()
    head_to_head: dict[tuple[Hashable, Hashable], list[int]] = {}

    for match in selected:
        home_id, away_id = match.home_team_id, match.away_team_id
        pair_fixtures[frozenset((home_id, away_id))] += 1
        if (match.home_goals is None) != (match.away_goals is None):
            raise ValueError("a fixture must have either both goals or neither")
        if match.home_goals is None or match.away_goals is None:
            continue

        home_goals, away_goals = match.home_goals, match.away_goals
        pair_results[frozenset((home_id, away_id))] += 1
        home, away = totals[home_id], totals[away_id]
        home.played += 1
        away.played += 1
        home.goals_for += home_goals
        home.goals_against += away_goals
        away.goals_for += away_goals
        away.goals_against += home_goals
        if home_goals > away_goals:
            home.won += 1
            away.lost += 1
            home.points += 3
            head_to_head.setdefault((home_id, away_id), [0, 0])[0] += 3
        elif home_goals < away_goals:
            away.won += 1
            home.lost += 1
            away.points += 3
            head_to_head.setdefault((away_id, home_id), [0, 0])[0] += 3
        else:
            home.drawn += 1
            away.drawn += 1
            home.points += 1
            away.points += 1
            head_to_head.setdefault((home_id, away_id), [0, 0])[0] += 1
            head_to_head.setdefault((away_id, home_id), [0, 0])[0] += 1

        head_to_head.setdefault((home_id, away_id), [0, 0])[1] += (
            home_goals - away_goals
        )
        head_to_head.setdefault((away_id, home_id), [0, 0])[1] += (
            away_goals - home_goals
        )

    by_points: dict[int, list[Hashable]] = {}
    for team_id, row in totals.items():
        by_points.setdefault(row.points, []).append(team_id)

    ordered: list[Hashable] = []
    for points in sorted(by_points, reverse=True):
        tied = by_points[points]
        if len(tied) == 1:
            ordered.extend(tied)
            continue

        pair_keys = [
            frozenset((left, right))
            for index, left in enumerate(tied)
            for right in tied[index + 1 :]
        ]
        # La Liga plays two legs per pairing; do not use partial mini-table
        # figures when either head-to-head fixture is still outstanding.
        complete_head_to_head = all(
            pair_fixtures[pair] == 2 and pair_results[pair] == 2
            for pair in pair_keys
        )
        if complete_head_to_head:
            def tie_key(team_id: Hashable) -> tuple[int, int, int, int, str]:
                h2h_points = sum(
                    head_to_head.get((team_id, opponent), [0, 0])[0]
                    for opponent in tied
                    if opponent != team_id
                )
                h2h_goal_difference = sum(
                    head_to_head.get((team_id, opponent), [0, 0])[1]
                    for opponent in tied
                    if opponent != team_id
                )
                row = totals[team_id]
                return (
                    -h2h_points,
                    -h2h_goal_difference,
                    -(row.goals_for - row.goals_against),
                    -row.goals_for,
                    str(team_id),
                )
        else:
            def tie_key(team_id: Hashable) -> tuple[int, int, int, int, str]:
                row = totals[team_id]
                return (
                    0,
                    0,
                    -(row.goals_for - row.goals_against),
                    -row.goals_for,
                    str(team_id),
                )

        # Criteria/order follow laliga-buildplan-v2.md, Appendix B.
        ordered.extend(sorted(tied, key=tie_key))

    return [
        TeamRow(
            position=position,
            team_id=team_id,
            played=totals[team_id].played,
            won=totals[team_id].won,
            drawn=totals[team_id].drawn,
            lost=totals[team_id].lost,
            goals_for=totals[team_id].goals_for,
            goals_against=totals[team_id].goals_against,
            points=totals[team_id].points,
        )
        for position, team_id in enumerate(ordered, start=1)
    ]


def rank_standings_many(
    matches: list[SeasonMatch],
    season_id: int,
    team_ids: list[Hashable],
    home_goals: NDArray[np.int64],
    away_goals: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Rank many fully completed season score matrices with one vectorized pass.

    ``positions`` and ``points`` are shaped ``(runs, teams)``. This uses the
    same M14/Appendix B rule order as :func:`rank_standings`, without Python
    loops over simulated runs.
    """
    selected = [match for match in matches if match.season_id == season_id]
    n_runs, n_fixtures = home_goals.shape
    n_teams = len(team_ids)
    if away_goals.shape != (n_runs, n_fixtures):
        raise ValueError("home and away goal matrices must have the same shape")
    if n_fixtures != len(selected):
        raise ValueError("goal matrix fixture count does not match the season schedule")
    if n_fixtures != n_teams * (n_teams - 1):
        raise ValueError("batch ranking requires a complete double round robin")
    if len(set(team_ids)) != n_teams:
        raise ValueError("team_ids must be unique")

    index = {team_id: team_index for team_index, team_id in enumerate(team_ids)}
    home_index = np.asarray(
        [index[match.home_team_id] for match in selected], dtype=np.int64
    )
    away_index = np.asarray(
        [index[match.away_team_id] for match in selected], dtype=np.int64
    )
    pair_counts: Counter[frozenset[Hashable]] = Counter(
        frozenset((match.home_team_id, match.away_team_id))
        for match in selected
    )
    if len(pair_counts) != n_teams * (n_teams - 1) // 2 or any(
        count != 2 for count in pair_counts.values()
    ):
        raise ValueError("batch ranking schedule is not a double round robin")

    home_points = np.where(
        home_goals > away_goals,
        3,
        np.where(home_goals == away_goals, 1, 0),
    )
    away_points = np.where(
        away_goals > home_goals,
        3,
        np.where(home_goals == away_goals, 1, 0),
    )
    goal_difference = np.zeros((n_runs, n_teams), dtype=np.int64)
    goals_for = np.zeros((n_runs, n_teams), dtype=np.int64)
    points = np.zeros((n_runs, n_teams), dtype=np.int64)
    h2h_points = np.zeros((n_runs, n_teams, n_teams), dtype=np.int64)
    h2h_goal_difference = np.zeros_like(h2h_points)
    run_indices = np.arange(n_runs, dtype=np.int64)[:, None]
    home_indices = home_index[None, :]
    away_indices = away_index[None, :]

    np.add.at(points, (run_indices, home_indices), home_points)
    np.add.at(points, (run_indices, away_indices), away_points)
    np.add.at(
        goal_difference,
        (run_indices, home_indices),
        home_goals - away_goals,
    )
    np.add.at(
        goal_difference,
        (run_indices, away_indices),
        away_goals - home_goals,
    )
    np.add.at(goals_for, (run_indices, home_indices), home_goals)
    np.add.at(goals_for, (run_indices, away_indices), away_goals)
    np.add.at(
        h2h_points,
        (run_indices, home_indices, away_indices),
        home_points,
    )
    np.add.at(
        h2h_points,
        (run_indices, away_indices, home_indices),
        away_points,
    )
    np.add.at(
        h2h_goal_difference,
        (run_indices, home_indices, away_indices),
        home_goals - away_goals,
    )
    np.add.at(
        h2h_goal_difference,
        (run_indices, away_indices, home_indices),
        away_goals - home_goals,
    )

    tied_on_points = points[:, :, None] == points[:, None, :]
    mini_table_points = np.sum(h2h_points * tied_on_points, axis=2)
    mini_table_goal_difference = np.sum(
        h2h_goal_difference * tied_on_points, axis=2
    )
    sorted_team_ids = sorted(range(n_teams), key=lambda index: str(team_ids[index]))
    stable_ranks = np.empty(n_teams, dtype=np.int64)
    stable_ranks[sorted_team_ids] = np.arange(n_teams, dtype=np.int64)
    stable_team_order = np.broadcast_to(-stable_ranks, (n_runs, n_teams))
    order = np.lexsort(
        (
            stable_team_order,
            goals_for,
            goal_difference,
            mini_table_goal_difference,
            mini_table_points,
            points,
        ),
        axis=1,
    )
    positions = np.empty_like(order)
    positions[run_indices, order] = np.arange(n_teams, 0, -1, dtype=np.int64)
    return positions, points
