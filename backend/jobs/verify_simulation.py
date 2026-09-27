#!/usr/bin/env python3
"""Independently verify one stored simulation_runs payload.

This script intentionally does not import any simulator or standings code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal  # noqa: E402
from app.models import SimulationRun  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the three named checks from stored raw simulations."
    )
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--season-id", type=int, default=None)
    args = parser.parse_args()

    with SessionLocal() as db:
        query = db.query(SimulationRun)
        if args.run_id is not None:
            query = query.filter(SimulationRun.id == args.run_id)
        if args.season_id is not None:
            query = query.filter(SimulationRun.season_id == args.season_id)
        run = query.order_by(SimulationRun.run_at.desc(), SimulationRun.id.desc()).first()
        if run is None:
            raise SystemExit("no matching simulation_runs row exists")
        run_id = run.id
        results = run.results
        n_simulations = run.n_simulations

    raw = results["raw_simulations"]
    positions = np.asarray(raw["finish_positions"], dtype=np.int64)
    points = np.asarray(raw["final_points"], dtype=np.int64)
    draws = np.asarray(raw["draws"], dtype=np.int64)
    if positions.shape != points.shape or positions.shape[0] != n_simulations:
        raise AssertionError("raw finish-position and points arrays have inconsistent shapes")
    if draws.shape != (n_simulations,):
        raise AssertionError("raw draw-count array does not match n_simulations")
    n_teams = positions.shape[1]
    team_index = {
        team_id: index for index, team_id in enumerate(results["team_ids"])
    }
    match_count = int(results["completed_matches"]) + int(results["simulated_matches"])

    team_sums = np.count_nonzero(
        (positions[:, :, None] == np.arange(1, n_teams + 1)[None, None, :]),
        axis=2,
    ).sum(axis=0) / n_simulations
    position_sums = np.count_nonzero(
        positions[:, :, None] == np.arange(1, n_teams + 1)[None, None, :],
        axis=1,
    ).sum(axis=0) / n_simulations
    if not np.allclose(team_sums, 1.0, atol=0.001, rtol=0.0):
        raise AssertionError("a team's finish-position probabilities do not sum to one")
    if not np.allclose(position_sums, 1.0, atol=0.001, rtol=0.0):
        raise AssertionError("finish-position probabilities across teams do not sum to one")

    for team_row in results["teams"]:
        raw_team_index = team_index[team_row["team_id"]]
        raw_distribution = {
            str(position): float(
                np.count_nonzero(
                    positions[:, raw_team_index] == position
                )
                / n_simulations
            )
            for position in range(1, n_teams + 1)
        }
        stored = team_row["finish_position_distribution"]
        if any(
            abs(raw_distribution[position] - float(stored[position])) > 0.000001
            for position in raw_distribution
        ):
            raise AssertionError(
                f"stored finish-position distribution is inconsistent for "
                f"{team_row['team']}"
            )

    points_by_run = points.sum(axis=1, dtype=np.int64)
    expected_points_by_run = 3 * (match_count - draws) + 2 * draws
    point_mismatches = int(
        np.count_nonzero(points_by_run != expected_points_by_run)
    )
    if match_count != 380:
        raise AssertionError(f"expected 380 season fixtures, found {match_count}")
    if point_mismatches:
        raise AssertionError(
            f"{point_mismatches} of {n_simulations} runs failed exact point conservation"
        )

    print(f"simulation_run_id={run_id}")
    print(f"simulations={n_simulations}; teams={n_teams}; fixtures={match_count}")
    print(
        "team_finish_distribution_sums="
        + ",".join(f"{value:.6f}" for value in team_sums)
    )
    print(
        "position_probability_sums="
        + ",".join(f"{value:.6f}" for value in position_sums)
    )
    print(
        f"points_conservation_runs={n_simulations}/{n_simulations}; "
        f"mismatches={point_mismatches}; "
        f"observed_total_points_range={points_by_run.min()}..{points_by_run.max()}; "
        f"expected_total_points_range={expected_points_by_run.min()}.."
        f"{expected_points_by_run.max()}; "
        f"draw_count_range={draws.min()}..{draws.max()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
