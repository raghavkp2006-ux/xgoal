#!/usr/bin/env python3
"""Monte Carlo season simulation — the job that fills ``simulation_runs``.

Fits Dixon-Coles at the current information cut, estimates parameter uncertainty
with match bootstrap resampling, vectorizes the scoreline draws, ranks every run
with La Liga head-to-head tiebreakers and appends a raw-verifiable result row.

Usage:
    python -m jobs.simulate_season
    python -m jobs.simulate_season --season 2024/25 --sims 20000
    python -m jobs.simulate_season --as-of-matchday 30 --seed 7
    python -m jobs.simulate_season --parameter-bootstrap 100
"""
import argparse
import os
import sys
from datetime import datetime, time, timezone
from time import perf_counter
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import (  # noqa: E402
    get_competition,
    load_completed_matches,
    load_team_names,
)
from app.ml.dixon_coles import DixonColesModel  # noqa: E402
from app.ml.simulation import (  # noqa: E402
    Fixture,
    bootstrap_parameter_ensemble,
    derive_missing_round_robin_fixtures,
    simulate_season as simulate_vectorized,
)
from app.ml.store import write_simulation_run  # noqa: E402
from app.models import Match, ModelVersion, Season  # noqa: E402

MODEL_NAME = "dixon_coles"


def load_season(db: Session, competition_id: int, label: str | None) -> Season:
    """A season by label, or the season flagged ``is_current``."""
    query = db.query(Season).filter(Season.competition_id == competition_id)
    row = (
        query.filter(Season.label == label).one_or_none()
        if label
        else query.filter(Season.is_current.is_(True)).one_or_none()
    )
    if row is None:
        wanted = label or "the current season"
        raise SystemExit(f"season {wanted} not found for this competition")
    return row


def season_fixtures(
    db: Session, season_id: int
) -> tuple[list[Match], list[Match]]:
    """(played, unplayed) fixtures of a season, chronological."""
    played = (
        db.query(Match)
        .filter(
            Match.season_id == season_id,
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
        )
        .order_by(Match.kickoff_utc)
        .all()
    )
    unplayed = (
        db.query(Match)
        .filter(Match.season_id == season_id, Match.home_goals.is_(None))
        .order_by(Match.kickoff_utc, Match.id)
        .all()
    )
    return played, unplayed


def current_table(names: dict[int, str], played: list[Match]) -> dict[str, dict[str, int]]:
    """Points, goal difference and goals scored from the fixtures played so far."""
    table: dict[str, dict[str, int]] = {}

    def add(name: str) -> dict[str, int]:
        return table.setdefault(name, {"played": 0, "points": 0, "gd": 0, "gf": 0})

    for match in played:
        home = add(names[match.home_team_id])
        away = add(names[match.away_team_id])
        home["played"] += 1
        away["played"] += 1
        home["gf"] += match.home_goals
        away["gf"] += match.away_goals
        home["gd"] += match.home_goals - match.away_goals
        away["gd"] += match.away_goals - match.home_goals
        if match.home_goals > match.away_goals:
            home["points"] += 3
        elif match.home_goals < match.away_goals:
            away["points"] += 3
        else:
            home["points"] += 1
            away["points"] += 1
    return table


def parse_forced(items: list[str]) -> dict[str, tuple[int, int]]:
    """``--force`` values as ``{"Home vs Away": (goals, goals)}``."""
    forced: dict[str, tuple[int, int]] = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 3 or "-" not in parts[2]:
            raise SystemExit(f"cannot parse --force {item!r} (expected 'HOME:AWAY:2-1')")
        try:
            home_goals, away_goals = (int(value) for value in parts[2].split("-", 1))
        except ValueError as exc:
            raise SystemExit(f"cannot parse --force {item!r} (score must be '2-1')") from exc
        forced[f"{parts[0].strip()} vs {parts[1].strip()}"] = (home_goals, away_goals)
    return forced


def print_table(rows: list[dict[str, Any]]) -> None:
    """Position, team and outcome probabilities of the simulated table."""
    header = (
        f"  {'#':>2}  {'team':<28} {'played':>6} {'pts':>4} {'xPts':>6} "
        f"{'title':>7} {'top4':>7} {'top6':>7} {'down':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for position, row in enumerate(rows, start=1):
        print(
            f"  {position:>2}  {row['team'][:28]:<28} {row['played']:>6} "
            f"{row['current_points']:>4} {row['expected_final_points']:>6.1f} "
            f"{row['p_champion']:>7.3f} {row['p_top4']:>7.3f} "
            f"{row['p_top6']:>7.3f} {row['p_relegation']:>7.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo the rest of a season.")
    parser.add_argument("--comp", default="SP1", help="competition code (default SP1)")
    parser.add_argument("--season", default=None, help="season label (default: current season)")
    parser.add_argument(
        "--as-of-matchday",
        type=int,
        default=None,
        help="simulate only fixtures after this matchday (default: every unplayed one)",
    )
    parser.add_argument("--sims", type=int, default=10000, help="number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--parameter-bootstrap",
        type=int,
        default=100,
        help="match-resampled model fits used to estimate parameter uncertainty",
    )
    parser.add_argument(
        "--force",
        action="append",
        default=[],
        metavar="HOME:AWAY:2-1",
        help="pin one fixture's scoreline (repeatable)",
    )
    args = parser.parse_args()
    if args.sims < 1:
        raise SystemExit("--sims must be at least 1")
    if args.parameter_bootstrap < 2:
        raise SystemExit("--parameter-bootstrap must be at least 2")

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        season = load_season(db, competition.id, args.season)
        names = load_team_names(db)
        played, unplayed = season_fixtures(db, season.id)
        derived_calendar = not unplayed and season.is_current
        if not unplayed and not derived_calendar:
            raise SystemExit(f"season {season.label} has no unplayed fixtures to simulate")

        if derived_calendar:
            if args.as_of_matchday is not None:
                raise SystemExit(
                    "matchday filtering is unavailable for the derived current-season "
                    "schedule; the stored as_of_matchday will be 0 (unknown)"
                )
            team_ids = sorted(
                {
                    team_id
                    for match in played
                    for team_id in (match.home_team_id, match.away_team_id)
                },
                key=lambda team_id: names[team_id],
            )
            if len(team_ids) != 20:
                raise SystemExit(
                    f"cannot derive La Liga fixtures: expected 20 teams, found {len(team_ids)}"
                )
            played_fixtures = [
                Fixture(
                    season.id,
                    match.home_team_id,
                    match.away_team_id,
                    match.home_goals,
                    match.away_goals,
                )
                for match in played
            ]
            remaining_pairs = derive_missing_round_robin_fixtures(
                team_ids, played_fixtures, season_id=season.id
            )
            remaining_fixtures = [
                Fixture(season.id, home_id, away_id, None, None)
                for home_id, away_id in remaining_pairs
            ]
            fixtures = played_fixtures + remaining_fixtures
            as_of = 0
            cut = datetime.combine(
                season.start_date, time.min, tzinfo=timezone.utc
            )
        else:
            matchdays = [match.matchday for match in played if match.matchday is not None]
            as_of = (
                args.as_of_matchday
                if args.as_of_matchday is not None
                else max(matchdays, default=0)
            )
            remaining_matches = [
                match
                for match in unplayed
                if match.matchday is None or match.matchday > as_of
            ]
            if not remaining_matches:
                raise SystemExit(f"no unplayed fixtures after matchday {as_of}")
            cut = min(match.kickoff_utc for match in remaining_matches)
            team_ids = sorted(
                {
                    team_id
                    for match in played + remaining_matches
                    for team_id in (match.home_team_id, match.away_team_id)
                },
                key=lambda team_id: names[team_id],
            )
            fixtures = [
                Fixture(
                    season.id,
                    match.home_team_id,
                    match.away_team_id,
                    match.home_goals,
                    match.away_goals,
                )
                for match in played
            ] + [
                Fixture(season.id, match.home_team_id, match.away_team_id, None, None)
                for match in remaining_matches
            ]

        version_row = (
            db.query(ModelVersion)
            .filter(ModelVersion.name == MODEL_NAME, ModelVersion.is_production.is_(True))
            .one_or_none()
        )
        if version_row is None:
            raise SystemExit("no production model version - run jobs.train_model first")

        history = load_completed_matches(db, competition.id, names, before=cut)
        model = DixonColesModel().fit(history)
        source = f"fit on {len(history)} matches before {cut:%Y-%m-%d}"
        model_team_names = [names[team_id] for team_id in team_ids]
        bootstrap_started = perf_counter()
        parameter_ensemble = bootstrap_parameter_ensemble(
            history,
            model_team_names,
            base_model=model,
            n_bootstrap=args.parameter_bootstrap,
            seed=args.seed + 1,
        )
        bootstrap_seconds = perf_counter() - bootstrap_started
        forced = parse_forced(args.force)

        print("=" * 78)
        print(f"  XGoal season simulation - {competition.code} ({competition.name})")
        print("=" * 78)
        print(
            f"  season={season.label}  as_of_matchday={as_of}  "
            f"sims={args.sims}  seed={args.seed}"
        )
        print(
            f"  fixtures played={len(played)} simulated={len(fixtures) - len(played)}  "
            f"model={source}"
        )
        print(
            f"  parameter bootstrap={args.parameter_bootstrap} fits in "
            f"{bootstrap_seconds:.3f}s"
        )
        if forced:
            print(f"  forced results: {forced}")

        simulation_started = perf_counter()
        results = simulate_vectorized(
            season_id=season.id,
            team_ids=team_ids,
            team_names=names,
            fixtures=fixtures,
            parameter_ensemble=parameter_ensemble,
            n_simulations=args.sims,
            seed=args.seed,
            forced_results=forced,
        )
        simulation_seconds = perf_counter() - simulation_started
        results["runtime_seconds"] = round(simulation_seconds, 6)
        results["context"] = {
            "competition": competition.code,
            "season": season.label,
            "as_of_matchday": as_of,
            "model_source": source,
            "derived_calendar": derived_calendar,
            "fixture_calendar_note": (
                "The 311 remaining fixtures are derived from the missing directed "
                "round-robin pairs; their real matchdays and kickoff dates are unknown."
                if derived_calendar
                else None
            ),
            "parameter_bootstrap_fits": args.parameter_bootstrap,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        run = write_simulation_run(
            db,
            season_id=season.id,
            model_version_id=version_row.id,
            n_simulations=args.sims,
            random_seed=args.seed,
            as_of_matchday=as_of,
            results=results,
            forced_results=forced or None,
        )
        print()
        print_table(results["teams"])  # type: ignore[arg-type]
        print(f"\n  vectorized simulation and ranking: {simulation_seconds:.3f}s")
    finally:
        db.close()

    print(
        f"\n  simulation_run id={run.id} stored ({run.n_simulations} sims, "
        f"seed {run.random_seed}, as_of_matchday {run.as_of_matchday})"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
