"""Uncertainty-aware, vectorized season simulation primitives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from app.ml.data import MatchInput
from app.ml.dixon_coles import DixonColesModel
from app.standings import rank_standings, rank_standings_many

FloatMatrix = NDArray[np.float64]
IntMatrix = NDArray[np.int64]
TOP4_SLOTS = 4
TOP6_SLOTS = 6
RELEGATION_SLOTS = 3


@dataclass(slots=True)
class Fixture:
    """One scheduled fixture; scores are ``None`` until sampled."""

    season_id: int
    home_team_id: Hashable
    away_team_id: Hashable
    home_goals: int | None
    away_goals: int | None

    def key(self, names: Mapping[Hashable, str]) -> str:
        return f"{names[self.home_team_id]} vs {names[self.away_team_id]}"


def derive_missing_round_robin_fixtures(
    team_ids: Sequence[Hashable],
    played_fixtures: Sequence[Fixture],
    *,
    season_id: int,
) -> list[tuple[Hashable, Hashable]]:
    """Derive absent directed pairings for a 20-club double round robin."""
    if len(team_ids) != 20 or len(set(team_ids)) != 20:
        raise ValueError("La Liga round-robin reconstruction requires 20 unique teams")
    possible = {
        (home, away)
        for home in team_ids
        for away in team_ids
        if home != away
    }
    played: set[tuple[Hashable, Hashable]] = set()
    for fixture in played_fixtures:
        if fixture.season_id != season_id:
            continue
        pair = (fixture.home_team_id, fixture.away_team_id)
        if pair not in possible:
            raise ValueError(f"played fixture contains a team outside the season: {pair}")
        if pair in played:
            raise ValueError(f"duplicate played home/away pairing: {pair}")
        if fixture.home_goals is None or fixture.away_goals is None:
            raise ValueError("only scored fixtures belong in the played-pair set")
        played.add(pair)

    remaining = sorted(possible - played, key=lambda pair: (str(pair[0]), str(pair[1])))
    complete = played | set(remaining)
    home_counts = Counter(home for home, _ in complete)
    away_counts = Counter(away for _, away in complete)
    if len(complete) != 380 or any(home_counts[t] != 19 for t in team_ids):
        raise AssertionError("derived schedule does not have 380 fixtures and 19 home games")
    if any(away_counts[t] != 19 for t in team_ids):
        raise AssertionError("derived schedule does not give every team 19 away games")
    return remaining


def _parameter_vector(
    model: DixonColesModel,
    team_names: Sequence[str],
    rng: np.random.Generator,
    *,
    sample_unseen: bool = True,
) -> FloatMatrix:
    attacks = np.asarray([model.attack.get(team, 0.0) for team in team_names])
    defences = np.asarray([model.defence.get(team, 0.0) for team in team_names])
    known_attacks = np.asarray(list(model.attack.values()), dtype=np.float64)
    known_defences = np.asarray(list(model.defence.values()), dtype=np.float64)
    attack_sd = float(np.std(known_attacks)) if known_attacks.size else 0.0
    defence_sd = float(np.std(known_defences)) if known_defences.size else 0.0
    for index, team in enumerate(team_names):
        if sample_unseen and team not in model.attack:
            attacks[index] = rng.normal(0.0, attack_sd)
        if sample_unseen and team not in model.defence:
            defences[index] = rng.normal(0.0, defence_sd)
    return np.concatenate(
        (attacks, defences, np.asarray([model.home_adv, model.rho], dtype=np.float64))
    )


def bootstrap_parameter_ensemble(
    training_matches: Sequence[MatchInput],
    team_names: Sequence[str],
    *,
    base_model: DixonColesModel,
    n_bootstrap: int,
    seed: int,
) -> FloatMatrix:
    """Fit match-resampled models whose parameters are sampled per simulation."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if not training_matches:
        raise ValueError("parameter bootstrap requires training matches")
    rng = np.random.default_rng(seed)
    vectors = [_parameter_vector(base_model, team_names, rng, sample_unseen=False)]
    size = len(training_matches)
    for replicate in range(n_bootstrap):
        fit: DixonColesModel | None = None
        for attempt in range(3):
            indices = rng.integers(0, size, size=size)
            sample = [training_matches[int(index)] for index in indices]
            candidate = DixonColesModel(
                xi=base_model.xi,
                max_goals=base_model.max_goals,
                rho_bound=base_model.rho_bound,
                l2=base_model.l2,
                seed=base_model.seed,
            )
            try:
                fit = candidate.fit(sample)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
        if fit is None:
            raise RuntimeError(f"bootstrap fit {replicate + 1} failed without a model")
        vectors.append(_parameter_vector(fit, team_names, rng))
    ensemble = np.asarray(vectors, dtype=np.float64)
    if not np.isfinite(ensemble).all():
        raise ValueError("bootstrap parameter ensemble contains non-finite values")
    return ensemble


def _tau_factors(
    home_goals: IntMatrix,
    away_goals: IntMatrix,
    lam: FloatMatrix,
    mu: FloatMatrix,
    rho: NDArray[np.float64],
) -> FloatMatrix:
    factors = np.ones(home_goals.shape, dtype=np.float64)
    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)
    factors[zero_zero] = (1.0 - lam * mu * rho[:, None])[zero_zero]
    factors[zero_one] = (1.0 + lam * rho[:, None])[zero_one]
    factors[one_zero] = (1.0 + mu * rho[:, None])[one_zero]
    factors[one_one] = np.broadcast_to(
        1.0 - rho[:, None], home_goals.shape
    )[one_one]
    return factors


def _draw_dixon_coles_goals(
    lam: FloatMatrix,
    mu: FloatMatrix,
    rho: NDArray[np.float64],
    rng: np.random.Generator,
) -> IntMatrix:
    """Draw and rejection-correct all sampled scorelines in vectorized batches."""
    if lam.shape != mu.shape or lam.ndim != 2 or rho.shape != (lam.shape[0],):
        raise ValueError("expected lambdas shaped (runs, fixtures) and one rho per run")
    tau_00 = 1.0 - lam * mu * rho[:, None]
    tau_01 = 1.0 + lam * rho[:, None]
    tau_10 = 1.0 + mu * rho[:, None]
    tau_11 = np.broadcast_to(1.0 - rho[:, None], lam.shape)
    minimum_tau = np.minimum.reduce((tau_00, tau_01, tau_10, tau_11))
    if not np.isfinite(minimum_tau).all() or np.any(minimum_tau <= 0.0):
        raise ValueError("sampled Dixon-Coles parameters give a non-positive tau factor")
    envelope = np.maximum.reduce(
        (
            np.ones(lam.shape, dtype=np.float64),
            tau_00,
            tau_01,
            tau_10,
            tau_11,
        )
    )

    goals = rng.poisson(lam=np.stack((lam, mu), axis=-1)).astype(np.int64)
    factors = _tau_factors(goals[:, :, 0], goals[:, :, 1], lam, mu, rho)
    accepted = rng.random(lam.shape) < factors / envelope
    while not accepted.all():
        rejected_runs, rejected_fixtures = np.nonzero(~accepted)
        retry_lam = lam[rejected_runs, rejected_fixtures]
        retry_mu = mu[rejected_runs, rejected_fixtures]
        retry = rng.poisson(
            lam=np.column_stack((retry_lam, retry_mu))
        ).astype(np.int64)
        goals[rejected_runs, rejected_fixtures] = retry
        retry_factors = _tau_factors(
            retry[:, 0, None],
            retry[:, 1, None],
            retry_lam[:, None],
            retry_mu[:, None],
            rho[rejected_runs],
        )[:, 0]
        accepted[rejected_runs, rejected_fixtures] = (
            rng.random(len(rejected_runs))
            < retry_factors / envelope[rejected_runs, rejected_fixtures]
        )
    return goals


def simulate_season(
    *,
    season_id: int,
    team_ids: Sequence[Hashable],
    team_names: Mapping[Any, str],
    fixtures: list[Fixture],
    parameter_ensemble: FloatMatrix,
    n_simulations: int,
    seed: int,
    forced_results: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Simulate every unscored fixture and rank each completed season."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    if len(set(team_ids)) != len(team_ids) or not team_ids:
        raise ValueError("team_ids must be non-empty and unique")
    if parameter_ensemble.ndim != 2 or parameter_ensemble.shape[1] != 2 * len(
        team_ids
    ) + 2:
        raise ValueError("parameter ensemble does not match the team set")
    if any(fixture.season_id != season_id for fixture in fixtures):
        raise ValueError("all fixtures must belong to the simulated season")
    if len(fixtures) != len(
        {(fixture.home_team_id, fixture.away_team_id) for fixture in fixtures}
    ):
        raise ValueError("duplicate directed pairing in the season schedule")
    if len(fixtures) != len(team_ids) * (len(team_ids) - 1):
        raise ValueError("fixture list is not a complete directed round robin")
    current_standings = rank_standings(fixtures, season_id)
    current_by_team = {row.team_id: row for row in current_standings}

    ordered_teams = list(team_ids)
    team_index = {team_id: index for index, team_id in enumerate(ordered_teams)}
    fixture_home = np.asarray(
        [team_index[fixture.home_team_id] for fixture in fixtures], dtype=np.int64
    )
    fixture_away = np.asarray(
        [team_index[fixture.away_team_id] for fixture in fixtures], dtype=np.int64
    )
    simulated_indices = [
        index
        for index, fixture in enumerate(fixtures)
        if fixture.home_goals is None and fixture.away_goals is None
    ]
    played_indices = [
        index
        for index, fixture in enumerate(fixtures)
        if fixture.home_goals is not None and fixture.away_goals is not None
    ]
    if len(simulated_indices) + len(played_indices) != len(fixtures):
        raise ValueError("each fixture must have two scores or neither")

    forced = dict(forced_results or {})
    fixture_keys = {fixture.key(team_names) for fixture in fixtures}
    unknown_forced = sorted(set(forced) - fixture_keys)
    if unknown_forced:
        raise ValueError(f"forced result fixture(s) not in the season: {unknown_forced}")
    forced_fixture_indices = {
        index: forced[fixtures[index].key(team_names)]
        for index in simulated_indices
        if fixtures[index].key(team_names) in forced
    }

    rng = np.random.default_rng(seed)
    sampled_parameters = parameter_ensemble[
        rng.integers(0, len(parameter_ensemble), size=n_simulations)
    ]
    n_teams = len(ordered_teams)
    attack = sampled_parameters[:, :n_teams]
    defence = sampled_parameters[:, n_teams : 2 * n_teams]
    home_advantage = sampled_parameters[:, -2]
    rho = sampled_parameters[:, -1]

    simulated_home = fixture_home[simulated_indices]
    simulated_away = fixture_away[simulated_indices]
    if simulated_indices:
        lam = np.exp(
            attack[:, simulated_home]
            - defence[:, simulated_away]
            + home_advantage[:, None]
        )
        mu = np.exp(attack[:, simulated_away] - defence[:, simulated_home])
        if not np.isfinite(lam).all() or not np.isfinite(mu).all():
            raise ValueError("sampled strengths produced non-finite scoring rates")
        sampled_goals = _draw_dixon_coles_goals(lam, mu, rho, rng)
        for fixture_column, fixture_index in enumerate(simulated_indices):
            forced_score = forced_fixture_indices.get(fixture_index)
            if forced_score is not None:
                sampled_goals[:, fixture_column, 0] = forced_score[0]
                sampled_goals[:, fixture_column, 1] = forced_score[1]
    else:
        sampled_goals = np.empty((n_simulations, 0, 2), dtype=np.int64)

    all_home_goals = np.empty((n_simulations, len(fixtures)), dtype=np.int64)
    all_away_goals = np.empty_like(all_home_goals)
    for index in played_indices:
        fixture = fixtures[index]
        if fixture.home_goals is None or fixture.away_goals is None:
            raise AssertionError("played fixture has missing goals")
        all_home_goals[:, index] = fixture.home_goals
        all_away_goals[:, index] = fixture.away_goals
    if simulated_indices:
        all_home_goals[:, simulated_indices] = sampled_goals[:, :, 0]
        all_away_goals[:, simulated_indices] = sampled_goals[:, :, 1]

    finish_positions, final_points = rank_standings_many(
        fixtures,
        season_id,
        ordered_teams,
        all_home_goals,
        all_away_goals,
    )

    draws_per_simulation = np.count_nonzero(
        all_home_goals == all_away_goals, axis=1
    )
    points_per_simulation = final_points.sum(axis=1, dtype=np.int64)
    required_points = (
        3 * (len(fixtures) - draws_per_simulation) + 2 * draws_per_simulation
    )
    if not np.array_equal(points_per_simulation, required_points):
        raise AssertionError("a simulated season violated exact points conservation")

    position_counts = np.stack(
        [
            np.count_nonzero(finish_positions == position, axis=0)
            for position in range(1, n_teams + 1)
        ],
        axis=0,
    )
    points_interval = np.quantile(final_points, [0.025, 0.975], axis=0)
    rows: list[dict[str, Any]] = []
    for team_position, team_id in enumerate(ordered_teams):
        distribution = {
            str(position + 1): round(
                int(position_counts[position, team_position]) / n_simulations, 6
            )
            for position in range(n_teams)
        }
        rows.append(
            {
                "team_id": team_id,
                "team": team_names[team_id],
                "p_champion": distribution["1"],
                "p_top4": round(
                    float(np.mean(finish_positions[:, team_position] <= TOP4_SLOTS)), 6
                ),
                "p_top6": round(
                    float(np.mean(finish_positions[:, team_position] <= TOP6_SLOTS)), 6
                ),
                "p_relegation": round(
                    float(
                        np.mean(
                            finish_positions[:, team_position]
                            > n_teams - RELEGATION_SLOTS
                        )
                    ),
                    6,
                ),
                "finish_position_distribution": distribution,
                "expected_final_points": round(
                    float(final_points[:, team_position].mean()), 3
                ),
                "expected_final_points_95ci": {
                    "lower": round(float(points_interval[0, team_position]), 3),
                    "upper": round(float(points_interval[1, team_position]), 3),
                    "level": 0.95,
                },
                "current_points": current_by_team[team_id].points,
                "current_goal_difference": current_by_team[team_id].goal_difference,
                "played": current_by_team[team_id].played,
            }
        )
    rows.sort(key=lambda row: (-float(row["p_champion"]), str(row["team"])))

    return {
        "team_ids": ordered_teams,
        "teams": rows,
        "simulated_matches": len(simulated_indices),
        "completed_matches": len(played_indices),
        "finish_position_counts": position_counts.T.tolist(),
        "raw_simulations": {
            "finish_positions": finish_positions.tolist(),
            "final_points": final_points.tolist(),
            "draws": draws_per_simulation.tolist(),
        },
        "parameter_uncertainty": {
            "method": "nonparametric match bootstrap; one fitted parameter vector sampled per run",
            "available_parameter_vectors": int(len(parameter_ensemble)),
        },
    }
