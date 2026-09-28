#!/usr/bin/env python3
"""Run the five-season matchday-20 simulation gate on completed DB seasons."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import get_competition, load_completed_matches, load_team_names  # noqa: E402
from app.ml.dixon_coles import DixonColesModel  # noqa: E402
from app.ml.simulation import (  # noqa: E402
    Fixture,
    bootstrap_parameter_ensemble,
    simulate_season,
)
from app.models import Match, Season  # noqa: E402
from app.standings import rank_standings  # noqa: E402

BACKTEST_SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24", "2024/25")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Back-test the matchday-20 title-probability gate."
    )
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--parameter-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sims < 1 or args.parameter_bootstrap < 2:
        raise SystemExit("sims must be positive and parameter-bootstrap at least 2")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    db = SessionLocal()
    passed = 0
    try:
        competition = get_competition(db, "SP1")
        if competition is None:
            raise SystemExit("SP1 competition was not found")
        names = load_team_names(db)
        seasons = {
            season.label: season
            for season in db.query(Season)
            .filter_by(competition_id=competition.id)
            .all()
        }
        summary: list[tuple[str, str, str, str, int, int, int, float]] = []
        for season_number, label in enumerate(BACKTEST_SEASONS):
            season = seasons[label]
            matches = (
                db.query(Match)
                .filter_by(season_id=season.id)
                .order_by(Match.kickoff_utc, Match.id)
                .all()
            )
            if len(matches) != 380 or any(
                match.home_goals is None or match.away_goals is None
                for match in matches
            ):
                raise ValueError(f"{label} is not a completed 380-match season")
            team_ids = sorted(
                {
                    team_id
                    for match in matches
                    for team_id in (match.home_team_id, match.away_team_id)
                },
                key=lambda team_id: names[team_id],
            )
            if len(team_ids) != 20:
                raise ValueError(f"{label} has {len(team_ids)} clubs, expected 20")

            first_200 = matches[:200]
            first_200_counts = Counter(
                team_id
                for match in first_200
                for team_id in (match.home_team_id, match.away_team_id)
            )
            personal_twentieth = {}
            for team_id in team_ids:
                team_matches = [
                    match
                    for match in matches
                    if team_id in (match.home_team_id, match.away_team_id)
                ]
                personal_twentieth[team_id] = team_matches[19].kickoff_utc
            cutoff = max(personal_twentieth.values())
            snapshot_matches = [
                match for match in matches if match.kickoff_utc <= cutoff
            ]
            snapshot_counts = Counter(
                team_id
                for match in snapshot_matches
                for team_id in (match.home_team_id, match.away_team_id)
            )
            if any(snapshot_counts[team_id] < 20 for team_id in team_ids):
                raise AssertionError(
                    f"{label}: common as-of cut missed a club's twentieth fixture"
                )
            exact_first_200 = all(first_200_counts[team_id] == 20 for team_id in team_ids)

            print(f"\n[{label}] first 200: {len(first_200)} fixtures; "
                  f"all clubs exactly 20 appearances={exact_first_200}")
            print(
                "first_200_per_team="
                + "; ".join(
                    f"{names[team_id]}={first_200_counts[team_id]}"
                    for team_id in team_ids
                )
            )
            print(
                f"cutoff=latest personal 20th fixture ({cutoff.isoformat()}); "
                f"fixtures included={len(snapshot_matches)}"
            )
            print(
                "adjusted_snapshot_per_team="
                + "; ".join(
                    f"{names[team_id]}={snapshot_counts[team_id]}"
                    for team_id in team_ids
                )
            )

            training_cut = cutoff + timedelta(microseconds=1)
            training = load_completed_matches(
                db, competition.id, names, before=training_cut
            )
            model = DixonColesModel(half_life_days=540.0).fit(training)
            model_team_names = [names[team_id] for team_id in team_ids]
            ensemble = bootstrap_parameter_ensemble(
                training,
                model_team_names,
                base_model=model,
                n_bootstrap=args.parameter_bootstrap,
                seed=args.seed + season_number + 1,
            )

            snapshot_ids = {match.id for match in snapshot_matches}
            fixtures = [
                Fixture(
                    season.id,
                    match.home_team_id,
                    match.away_team_id,
                    match.home_goals if match.id in snapshot_ids else None,
                    match.away_goals if match.id in snapshot_ids else None,
                )
                for match in matches
            ]
            simulation_started = perf_counter()
            results = simulate_season(
                season_id=season.id,
                team_ids=team_ids,
                team_names=names,
                fixtures=fixtures,
                parameter_ensemble=ensemble,
                n_simulations=args.sims,
                seed=args.seed + season_number,
            )
            simulation_seconds = perf_counter() - simulation_started

            actual_champion = rank_standings(matches, season.id)[0].team_id
            title_rows = results["teams"][:2]
            title_order = [row["team_id"] for row in title_rows]
            title_summary = ", ".join(
                f"{names[row['team_id']]}={row['p_champion']:.3f}"
                for row in title_rows
            )
            champion_pass = actual_champion in title_order
            passed += int(champion_pass)
            print(
                f"actual_champion={names[actual_champion]}; "
                f"top_two_by_title_probability={title_summary}; "
                f"result={'PASS' if champion_pass else 'FAIL'}; "
                f"sim_seconds={simulation_seconds:.3f}; "
                f"training_matches={len(training)}"
            )
            summary.append(
                (
                    label,
                    names[actual_champion],
                    title_summary,
                    "PASS" if champion_pass else "FAIL",
                    len(training),
                    len(snapshot_matches),
                    len(matches) - len(snapshot_matches),
                    simulation_seconds,
                )
            )
            del results, ensemble, fixtures

        print("\nBACKTEST SUMMARY")
        print(
            "season | actual champion | top two by title probability | gate | "
            "train n | as-of matches | remaining | simulation seconds"
        )
        for row in summary:
            print(" | ".join(map(str, row)))
        print(f"gate_result={passed}/5; required=at least 4/5")
        print(f"overall_gate={'PASS' if passed >= 4 else 'FAIL'}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
