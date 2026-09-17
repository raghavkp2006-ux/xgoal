#!/usr/bin/env python3
"""Monte Carlo season simulation — the job that fills ``simulation_runs``.

Loads the production Dixon-Coles artifact, replays a season's remaining fixtures
thousands of times and stores the resulting title / top-four / European-place /
relegation probabilities together with a points distribution per team.

Ties are broken on points, then goal difference, then goals scored. La Liga
breaks ties on head-to-head results first, so the simulated table is an
approximation of the official rule (which the CSV feed does not expose).

Forced results (``--force "Real Madrid:FC Barcelona:2-1"``) are recorded in
``simulation_runs.forced_results`` and used instead of sampled scorelines, which
is how a "what if" scenario is published.

Usage:
    python -m jobs.simulate_season
    python -m jobs.simulate_season --season 2024/25 --sims 20000
    python -m jobs.simulate_season --as-of-matchday 30 --seed 7
    python -m jobs.simulate_season --fit --force "Real Madrid:FC Barcelona:2-1"
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sqlalchemy.orm import Session  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.ml.dataset import get_competition, load_completed_matches, load_team_names  # noqa: E402
from app.ml.dixon_coles import DixonColesModel  # noqa: E402
from app.ml.store import read_artifact, write_simulation_run  # noqa: E402
from app.models import Match, ModelVersion, Season  # noqa: E402

MODEL_NAME = "dixon_coles"
TOP4_SLOTS = 4
EUROPE_SLOTS = 6
RELEGATION_SLOTS = 3


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
        f"{'title':>7} {'top4':>7} {'euro':>7} {'down':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for position, row in enumerate(rows, start=1):
        print(
            f"  {position:>2}  {row['team'][:28]:<28} {row['played']:>6} "
            f"{row['current_points']:>4} {row['expected_points']:>6.1f} "
            f"{row['p_title']:>7.3f} {row['p_top4']:>7.3f} "
            f"{row['p_europe']:>7.3f} {row['p_relegation']:>7.3f}"
        )


def simulate(
    model: DixonColesModel,
    table: dict[str, dict[str, int]],
    remaining: list[tuple[str, str]],
    n_sims: int,
    seed: int,
    forced: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Play ``remaining`` ``n_sims`` times and summarise the final table."""
    teams = sorted(table)
    index = {team: position for position, team in enumerate(teams)}
    n_teams = len(teams)
    columns = np.arange(n_sims)

    points = np.tile(
        np.array([table[t]["points"] for t in teams], dtype=np.int64)[:, None],
        (1, n_sims),
    )
    goal_diff = np.tile(
        np.array([table[t]["gd"] for t in teams], dtype=np.int64)[:, None], (1, n_sims)
    )
    goals_for = np.tile(
        np.array([table[t]["gf"] for t in teams], dtype=np.int64)[:, None], (1, n_sims)
    )

    rng = np.random.default_rng(seed)
    for home, away in remaining:
        forced_score = forced.get(f"{home} vs {away}")
        if forced_score is not None:
            home_goals = np.full(n_sims, forced_score[0], dtype=np.int64)
            away_goals = np.full(n_sims, forced_score[1], dtype=np.int64)
        else:
            matrix = np.asarray(model.predict(home, away).matrix, dtype=np.float64)
            flat = matrix.ravel()
            flat = flat / flat.sum()
            cdf = np.cumsum(flat)
            cdf[-1] = 1.0
            drawn = rng.random(n_sims)
            cells = np.searchsorted(cdf, drawn, side="right")
            home_goals = cells // matrix.shape[0]
            away_goals = cells % matrix.shape[1]

        home_index = index[home]
        away_index = index[away]
        home_win = home_goals > away_goals
        away_win = home_goals < away_goals
        points[home_index] += np.where(home_win, 3, np.where(away_win, 0, 1))
        points[away_index] += np.where(away_win, 3, np.where(home_win, 0, 1))
        goal_diff[home_index] += home_goals - away_goals
        goal_diff[away_index] += away_goals - home_goals
        goals_for[home_index] += home_goals
        goals_for[away_index] += away_goals

    # lexsort uses the last key as the primary one: points, then gd, then goals.
    order = np.lexsort((goals_for, goal_diff, points), axis=0)
    ranks = np.empty_like(order)
    ranks[order, columns] = np.arange(n_teams)[:, None]
    from_top = n_teams - 1 - ranks

    rows: list[dict[str, Any]] = []
    for position, team in enumerate(teams):
        rows.append(
            {
                "team": team,
                "p_title": round(float((from_top[position] == 0).mean()), 5),
                "p_top4": round(float((from_top[position] < TOP4_SLOTS).mean()), 5),
                "p_europe": round(float((from_top[position] < EUROPE_SLOTS).mean()), 5),
                "p_relegation": round(
                    float((from_top[position] >= n_teams - RELEGATION_SLOTS).mean()), 5
                ),
                "expected_points": round(float(points[position].mean()), 2),
                "points_p10": float(np.percentile(points[position], 10)),
                "points_p90": float(np.percentile(points[position], 90)),
                "current_points": int(table[team]["points"]),
                "current_goal_diff": int(table[team]["gd"]),
                "played": int(table[team]["played"]),
            }
        )
    rows.sort(key=lambda row: (-row["p_title"], -row["expected_points"]))
    return {
        "tiebreak": "points, goal difference, goals scored",
        "simulated_matches": len(remaining),
        "teams": rows,
    }


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
        "--fit",
        action="store_true",
        help="refit instead of loading the production artifact",
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

    db = SessionLocal()
    try:
        competition = get_competition(db, args.comp)
        if competition is None:
            raise SystemExit(f"competition {args.comp} not found - run ingestion first")
        season = load_season(db, competition.id, args.season)
        names = load_team_names(db)
        played, unplayed = season_fixtures(db, season.id)
        if not unplayed:
            raise SystemExit(f"season {season.label} has no unplayed fixtures to simulate")

        matchdays = [match.matchday for match in played if match.matchday is not None]
        as_of = (
            args.as_of_matchday
            if args.as_of_matchday is not None
            else max(matchdays, default=0)
        )
        remaining_matches = [
            match for match in unplayed if match.matchday is None or match.matchday > as_of
        ]
        if not remaining_matches:
            raise SystemExit(f"no unplayed fixtures after matchday {as_of}")

        version_row = (
            db.query(ModelVersion)
            .filter(ModelVersion.name == MODEL_NAME, ModelVersion.is_production.is_(True))
            .one_or_none()
        )
        if version_row is None:
            raise SystemExit("no production model version - run jobs.train_model first")

        cut = min(match.kickoff_utc for match in remaining_matches)
        if args.fit:
            history = load_completed_matches(db, competition.id, names, before=cut)
            model = DixonColesModel().fit(history)
            source = f"refit on {len(history)} matches before {cut:%Y-%m-%d}"
        else:
            model = DixonColesModel.from_artifact(read_artifact(version_row.artifact_path))
            source = f"{version_row.name} {version_row.version} ({version_row.artifact_path})"

        table = current_table(names, played)
        remaining = [(names[m.home_team_id], names[m.away_team_id]) for m in remaining_matches]
        forced = parse_forced(args.force)
        unknown = sorted(set(forced) - {f"{home} vs {away}" for home, away in remaining})
        if unknown:
            raise SystemExit(f"--force fixture(s) not in the remaining set: {unknown}")

        print("=" * 78)
        print(f"  XGoal season simulation - {competition.code} ({competition.name})")
        print("=" * 78)
        print(
            f"  season={season.label}  as_of_matchday={as_of}  "
            f"sims={args.sims}  seed={args.seed}"
        )
        print(f"  fixtures played={len(played)} simulated={len(remaining)}  model={source}")
        if forced:
            print(f"  forced results: {forced}")

        results = simulate(model, table, remaining, args.sims, args.seed, forced)
        results["context"] = {
            "competition": competition.code,
            "season": season.label,
            "as_of_matchday": as_of,
            "model_source": source,
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
        print_table(results["teams"])
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
