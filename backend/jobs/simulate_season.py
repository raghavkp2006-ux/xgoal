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
from datetime import datetime, time, timedelta, timezone
from time import perf_counter
from typing import Any, Hashable, Mapping, cast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.data import MatchInput  # noqa: E402
from app.ml.dataset import (  # noqa: E402
    get_competition,
    load_completed_matches,
    load_team_names,
    to_match_input,
)
from app.ml.dixon_coles import DixonColesModel  # noqa: E402
from app.ml.simulation import (  # noqa: E402
    Fixture,
    bootstrap_parameter_ensemble,
    derive_missing_round_robin_fixtures,
)
from app.ml.simulation import (
    simulate_season as simulate_vectorized,
)
from app.ml.store import read_artifact, write_simulation_run  # noqa: E402
from app.models import Match, ModelVersion, Season  # noqa: E402
from app.standings import rank_standings  # noqa: E402
from jobs.load_calendar import (  # noqa: E402
    DEFAULT_CSV,
    load_aliases,
    read_calendar,
    resolve_team_id,
)

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
) -> list[Match]:
    """All fixtures of a season, chronological."""
    return (
        db.query(Match)
        .filter(Match.season_id == season_id)
        .order_by(Match.kickoff_utc, Match.id)
        .all()
    )


def production_training_matches(
    db: Session,
    competition_id: int,
    names: dict[int, str],
    model: DixonColesModel,
    expected_count: int,
) -> list[MatchInput]:
    """Load the exact date-bounded training window recorded in the model artifact."""
    if model.train_start is None or model.train_end is None:
        raise ValueError("production model artifact must include its training dates")
    rows = (
        db.query(Match)
        .filter(
            Match.competition_id == competition_id,
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
            Match.kickoff_utc >= model.train_start,
            Match.kickoff_utc <= model.train_end,
        )
        .order_by(Match.kickoff_utc)
        .all()
    )
    history = [item for row in rows if (item := to_match_input(row, names)) is not None]
    if len(history) != expected_count:
        raise ValueError(
            "production model training history does not match its recorded size: "
            f"expected {expected_count}, found {len(history)}"
        )
    return history


def infer_latest_played_matchday(
    completed_pairs: set[tuple[int, int]],
    canonical_team_ids: dict[str, int],
    *,
    calendar_path=DEFAULT_CSV,
) -> int:
    """Infer the latest completed round when legacy played rows have no matchday."""
    aliases = load_aliases()
    rows = read_calendar(calendar_path)
    completed_matchdays = []
    for row in rows:
        home_id = resolve_team_id(row.home, aliases, canonical_team_ids)
        away_id = resolve_team_id(row.away, aliases, canonical_team_ids)
        if (home_id, away_id) in completed_pairs:
            completed_matchdays.append(row.matchday)
    if not completed_matchdays:
        raise ValueError(
            "cannot infer current matchday: no played season fixtures match the "
            f"official calendar at {calendar_path}"
        )
    return max(completed_matchdays)


def historical_cutoff(
    matches: list[Match],
    as_of_matchday: int,
    fixtures_per_matchday: int,
) -> tuple[datetime, str, int | None]:
    """Get an as-of boundary, falling back to a labeled chronological proxy."""
    matchday_kickoffs = [
        match.kickoff_utc
        for match in matches
        if match.matchday is not None and match.matchday <= as_of_matchday
    ]
    if matchday_kickoffs:
        return (
            max(matchday_kickoffs) + timedelta(microseconds=1),
            "stored_matchday",
            None,
        )

    target_fixtures = as_of_matchday * fixtures_per_matchday
    ordered = sorted(matches, key=lambda match: (match.kickoff_utc, match.id))
    if target_fixtures < 1 or target_fixtures > len(ordered):
        raise ValueError(
            f"cannot derive a chronological proxy for matchday {as_of_matchday}: "
            f"need {target_fixtures} fixtures, found {len(ordered)}"
        )
    cutoff = ordered[target_fixtures - 1].kickoff_utc + timedelta(microseconds=1)
    return cutoff, "chronological_fixture_count_proxy", target_fixtures


def current_table(names: dict[int, str], played: list[Match]) -> dict[str, dict[str, int]]:
    """Points, goal difference and goals scored from the fixtures played so far."""
    table: dict[str, dict[str, int]] = {}

    def add(name: str) -> dict[str, int]:
        return table.setdefault(name, {"played": 0, "points": 0, "gd": 0, "gf": 0})

    for match in played:
        if match.home_goals is None or match.away_goals is None:
            raise ValueError(f"played match {match.id} is missing a final score")
        home_goals = match.home_goals
        away_goals = match.away_goals
        home = add(names[match.home_team_id])
        away = add(names[match.away_team_id])
        home["played"] += 1
        away["played"] += 1
        home["gf"] += home_goals
        away["gf"] += away_goals
        home["gd"] += home_goals - away_goals
        away["gd"] += away_goals - home_goals
        if home_goals > away_goals:
            home["points"] += 3
        elif home_goals < away_goals:
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
        "--backtest",
        action="store_true",
        help="fit only on data available at the selected historical matchday; do not store",
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
    if args.backtest and (args.season is None or args.as_of_matchday is None):
        raise SystemExit("--backtest requires both --season and --as-of-matchday")
    if args.as_of_matchday is not None and not args.backtest:
        raise SystemExit("--as-of-matchday is only supported with --backtest")

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        season = load_season(db, competition.id, args.season)
        names = load_team_names(db)
        all_matches = season_fixtures(db, season.id)
        played = [
            match
            for match in all_matches
            if match.home_goals is not None and match.away_goals is not None
        ]
        unplayed = [
            match
            for match in all_matches
            if match.home_goals is None and match.away_goals is None
        ]
        derived_calendar = not unplayed and season.is_current
        if not unplayed and not derived_calendar and not args.backtest:
            raise SystemExit(f"season {season.label} has no unplayed fixtures to simulate")

        actual_champion_id: Hashable | None = None
        backtest_cutoff_method: str | None = None
        backtest_proxy_target: int | None = None
        if args.backtest:
            team_ids = sorted(
                {
                    team_id
                    for match in all_matches
                    for team_id in (match.home_team_id, match.away_team_id)
                },
                key=lambda team_id: names[team_id],
            )
            try:
                cut, backtest_cutoff_method, backtest_proxy_target = historical_cutoff(
                    all_matches,
                    args.as_of_matchday,
                    fixtures_per_matchday=len(team_ids) // 2,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            played = [
                match
                for match in all_matches
                if match.home_goals is not None
                and match.away_goals is not None
                and match.kickoff_utc < cut
            ]
            remaining_matches = [
                match
                for match in all_matches
                if match.kickoff_utc >= cut
                or match.home_goals is None
                or match.away_goals is None
            ]
            if not remaining_matches:
                raise SystemExit(
                    f"no fixtures remain after matchday {args.as_of_matchday}"
                )
            as_of = args.as_of_matchday
            actual_fixtures = [
                Fixture(
                    season.id,
                    match.home_team_id,
                    match.away_team_id,
                    match.home_goals,
                    match.away_goals,
                )
                for match in all_matches
            ]
            actual_table = rank_standings(actual_fixtures, season.id)
            if len(actual_table) != len(team_ids):
                raise SystemExit(
                    f"cannot backtest season {season.label}: expected "
                    f"{len(team_ids)} final table rows, found {len(actual_table)}"
                )
            actual_champion_id = actual_table[0].team_id
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
                Fixture(
                    season.id,
                    match.home_team_id,
                    match.away_team_id,
                    None,
                    None,
                )
                for match in remaining_matches
            ]
        elif derived_calendar:
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
            if matchdays:
                as_of = max(matchdays)
            else:
                canonical_team_ids = {name: team_id for team_id, name in names.items()}
                completed_pairs = {
                    (match.home_team_id, match.away_team_id) for match in played
                }
                as_of = infer_latest_played_matchday(
                    completed_pairs, canonical_team_ids
                )
            remaining_matches = unplayed
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

        version_row: ModelVersion | None = None
        if args.backtest:
            history = load_completed_matches(db, competition.id, names, before=cut)
            model = DixonColesModel().fit(history)
            bootstrap_history = history
            source = f"historical fit on {len(history)} matches before {cut:%Y-%m-%d}"
        else:
            version_row = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.name == MODEL_NAME,
                    ModelVersion.is_production.is_(True),
                )
                .one_or_none()
            )
            if version_row is None:
                raise SystemExit("no production model version - run jobs.train_model first")
            model = DixonColesModel.from_artifact(read_artifact(version_row.artifact_path))
            bootstrap_history = production_training_matches(
                db,
                competition.id,
                names,
                model,
                version_row.n_train_matches,
            )
            source = f"production artifact {version_row.version}"
        model_team_names = [names[team_id] for team_id in team_ids]
        bootstrap_started = perf_counter()
        parameter_ensemble = bootstrap_parameter_ensemble(
            bootstrap_history,
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
            f"{'' if version_row is None else f' (model_version_id={version_row.id})'}"
        )
        print(
            f"  parameter bootstrap={args.parameter_bootstrap} fits in "
            f"{bootstrap_seconds:.3f}s"
        )
        print(
            f"  parameter ensemble={len(parameter_ensemble)} vectors "
            f"(base + {args.parameter_bootstrap} match-resampled refits; "
            f"resampling {len(bootstrap_history)} training matches)"
        )
        if args.backtest:
            if backtest_cutoff_method == "chronological_fixture_count_proxy":
                print(
                    "  backtest cutoff=APPROXIMATE chronological fixture-count proxy "
                    f"target={backtest_proxy_target} matches; "
                    "this is not an exact matchday reconstruction"
                )
            else:
                print("  backtest cutoff=stored matchday labels")
        if forced:
            print(f"  forced results: {forced}")

        simulation_started = perf_counter()
        team_name_map = cast(Mapping[Hashable, str], names)
        results = simulate_vectorized(
            season_id=season.id,
            team_ids=team_ids,
            team_names=team_name_map,
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
            "model_version": None if version_row is None else version_row.version,
            "backtest_cutoff_method": backtest_cutoff_method,
            "backtest_proxy_target_fixtures": backtest_proxy_target,
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
        if not args.backtest:
            assert version_row is not None
            persisted_results = {
                key: value
                for key, value in results.items()
                if key not in {"raw_simulations", "finish_position_counts"}
            }
            run = write_simulation_run(
                db,
                season_id=season.id,
                model_version_id=version_row.id,
                n_simulations=args.sims,
                random_seed=args.seed,
                as_of_matchday=as_of,
                results=persisted_results,
                forced_results=forced or None,
            )
        print()
        print_table(results["teams"])  # type: ignore[arg-type]
        print(f"\n  vectorized simulation and ranking: {simulation_seconds:.3f}s")
        if actual_champion_id is not None:
            team_results = results["teams"]
            assert isinstance(team_results, list)
            champion_result = next(
                row for row in team_results if row["team_id"] == actual_champion_id
            )
            champion_rank = next(
                rank
                for rank, row in enumerate(team_results, start=1)
                if row["team_id"] == actual_champion_id
            )
            print(
                f"  BACKTEST champion={champion_result['team']} "
                f"title_probability={champion_result['p_champion']:.6f} "
                f"title_probability_rank={champion_rank}"
            )
    finally:
        db.close()

    if not args.backtest:
        print(
            f"\n  simulation_run id={run.id} stored ({run.n_simulations} sims, "
            f"seed {run.random_seed}, as_of_matchday {run.as_of_matchday})"
        )
    else:
        print("\n  backtest results not stored")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
