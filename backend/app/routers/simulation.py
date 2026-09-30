"""Simulation endpoints — cached baseline reads and interactive what-if runs.

GET  /api/v1/simulation?season=...   → latest precomputed simulation
POST /api/v1/simulation/whatif       → 2,000-sim what-if with forced results
"""

from __future__ import annotations

import hashlib
from datetime import datetime, time, timezone
from time import perf_counter
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Match, MatchStatus, Season, SimulationRun
from app.rate_limiting import limiter


class WhatIfValidationError(ValueError):
    """A requested fixture cannot be forced in the current season."""


class ProblemRoute(APIRoute):
    """Render request validation errors as RFC 7807 for this router."""

    def get_route_handler(self) -> Any:
        handler = super().get_route_handler()

        async def problem_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except (RequestValidationError, WhatIfValidationError) as exc:
                if isinstance(exc, RequestValidationError):
                    detail = "; ".join(
                        f"{'.'.join(map(str, issue['loc']))}: {issue['msg']}"
                        for issue in exc.errors()
                    )
                else:
                    detail = str(exc)
                return JSONResponse(
                    status_code=422,
                    media_type="application/problem+json",
                    content={
                        "type": "about:blank",
                        "title": "Unprocessable Content",
                        "status": 422,
                        "detail": detail,
                    },
                )

        return problem_handler


router = APIRouter(
    prefix="/api/v1/simulation", tags=["simulation"], route_class=ProblemRoute
)

# ---------------------------------------------------------------------------
# GET — read the latest cached baseline run
# ---------------------------------------------------------------------------


@router.get("")
def get_simulation(
    season: str | None = Query(
        None,
        description="Season label (e.g. '2025/26'). Defaults to the current season.",
    ),
    if_none_match: str | None = Header(None),
    db: Session = Depends(get_db),
) -> Response:
    """Return the most recent baseline simulation run for a season.

    This reads from the ``simulation_runs`` table — it never re-runs a
    simulation.  Baseline runs have ``forced_results IS NULL``.
    """
    season_filter = Season.label == season if season else Season.is_current.is_(True)
    statement = (
        select(
            SimulationRun.id,
            SimulationRun.season_id,
            SimulationRun.model_version_id,
            SimulationRun.n_simulations,
            SimulationRun.random_seed,
            SimulationRun.as_of_matchday,
            SimulationRun.run_at,
            SimulationRun.results,
        )
        .join(Season, SimulationRun.season_id == Season.id)
        .where(
            season_filter,
            or_(
                SimulationRun.forced_results.is_(None),
                text("forced_results = 'null'::jsonb"),
            ),
        )
        .order_by(SimulationRun.run_at.desc())
        .limit(1)
    )
    run = db.execute(statement).one_or_none()
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No baseline simulation run found for season "
                f"{season or 'current'}. Run `python -m jobs.simulate_season` first."
            ),
        )

    results: dict[str, Any] = dict(run.results)
    # Strip bulky raw arrays the frontend doesn't need
    results.pop("raw_simulations", None)
    results.pop("finish_position_counts", None)

    payload = {
        "id": run.id,
        "season_id": run.season_id,
        "model_version_id": run.model_version_id,
        "n_simulations": run.n_simulations,
        "random_seed": run.random_seed,
        "as_of_matchday": run.as_of_matchday,
        "run_at": run.run_at.isoformat() if run.run_at else None,
        **results,
    }
    response = JSONResponse(content=payload)
    etag = f'"{hashlib.sha256(response.body).hexdigest()}"'
    headers = {
        "ETag": etag,
        "Cache-Control": "public, max-age=300, stale-while-revalidate=3600",
    }
    if if_none_match:
        candidates = (candidate.strip() for candidate in if_none_match.split(","))
        if any(candidate in {"*", etag, f"W/{etag}"} for candidate in candidates):
            return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return response


# ---------------------------------------------------------------------------
# POST — interactive what-if
# ---------------------------------------------------------------------------

WHATIF_SIMS = 2_000
WHATIF_BOOTSTRAP = 20  # much smaller than the nightly 100
WHATIF_SEED = 42
Goal = Annotated[int, Field(strict=True, ge=0, le=20)]


class ForcedResult(BaseModel):
    """A single forced match result for what-if mode."""

    match_id: Annotated[int, Field(strict=True, gt=0)]
    home_goals: Goal
    away_goals: Goal


class WhatIfRequest(BaseModel):
    """Request body for the what-if simulation endpoint."""

    forced_results: list[ForcedResult] = Field(..., max_length=50)


def validate_forced_results(
    requested: list[ForcedResult], season_id: int, matches: list[Match]
) -> dict[int, tuple[int, int]]:
    """Validate the entire request before any expensive simulation work."""
    by_id = {match.id: match for match in matches}
    forced: dict[int, tuple[int, int]] = {}
    for result in requested:
        if result.match_id in forced:
            raise WhatIfValidationError(f"Duplicate match_id {result.match_id}")
        match = by_id.get(result.match_id)
        if match is None or match.season_id != season_id:
            raise WhatIfValidationError(
                f"match_id {result.match_id} is not in the current season"
            )
        if match.status != MatchStatus.NS:
            raise WhatIfValidationError(f"match_id {result.match_id} is not NS")
        forced[result.match_id] = (result.home_goals, result.away_goals)
    return forced


DELTA_FIELDS = (
    "p_champion", "p_top4", "p_top6", "p_relegation", "expected_final_points"
)


def paired_team_deltas(
    forced_teams: list[dict[str, Any]], unforced_teams: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Subtract runs drawn from the same parameters and random stream."""
    unforced_by_id = {row["team_id"]: row for row in unforced_teams}
    if len(unforced_by_id) != len(unforced_teams) or len(forced_teams) != len(unforced_teams):
        raise ValueError("Paired simulations have different team sets")
    deltas = []
    for row in forced_teams:
        original = unforced_by_id.get(row["team_id"])
        if original is None:
            raise ValueError(f"Unforced simulation has no team_id {row['team_id']}")
        deltas.append({
            "team_id": row["team_id"],
            "team": row["team"],
            **{field: round(row[field] - original[field], 6) for field in DELTA_FIELDS},
        })
    return deltas


@router.post("/whatif")
@limiter.limit(settings.rate_limit_expensive, override_defaults=False)
def run_whatif(
    request: Request,
    response: Response,
    body: WhatIfRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run a reduced (2,000-sim) Monte Carlo with forced match results.

    The result is transient — it is **not** persisted to the database.
    A paired unforced run supplies the baseline for team deltas.
    """
    # Lazy imports — these pull in numpy/scipy which are heavy at import time
    from app.ml.dataset import (  # noqa: E402
        load_completed_matches,
        load_team_names,
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

    season_row = _resolve_season(db, None)
    all_matches = (
        db.query(Match)
        .filter(Match.season_id == season_row.id)
        .order_by(Match.kickoff_utc, Match.id)
        .all()
    )
    forced_by_id = validate_forced_results(body.forced_results, season_row.id, all_matches)
    cached_run: SimulationRun | None = (
        db.query(SimulationRun)
        .filter(
            SimulationRun.season_id == season_row.id,
            or_(
                SimulationRun.forced_results.is_(None),
                text("forced_results = 'null'::jsonb"),
            ),
        )
        .order_by(SimulationRun.run_at.desc())
        .first()
    )
    if cached_run is None:
        raise HTTPException(status_code=404, detail="No cached baseline simulation run")

    # Resolve competition for training data
    from app.models import Competition  # noqa: E402

    comp_row = (
        db.query(Competition)
        .join(Season, Season.competition_id == Competition.id)
        .filter(Season.id == season_row.id)
        .first()
    )
    if comp_row is None:
        raise HTTPException(status_code=404, detail="Competition not found")

    names = load_team_names(db)

    # Load season matches
    played_matches = [match for match in all_matches if match.status == MatchStatus.FT]
    unplayed_matches = [match for match in all_matches if match.status == MatchStatus.NS]

    team_ids = sorted(
        {
            tid
            for m in all_matches
            for tid in (m.home_team_id, m.away_team_id)
        },
        key=lambda tid: names.get(tid, str(tid)),
    )
    if len(team_ids) != 20:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 20 teams, found {len(team_ids)}",
        )

    played_fixtures = [
        Fixture(season_row.id, m.home_team_id, m.away_team_id, m.home_goals, m.away_goals)
        for m in played_matches
    ]
    remaining_fixtures = [
        Fixture(season_row.id, m.home_team_id, m.away_team_id, None, None)
        for m in unplayed_matches
    ]
    if not remaining_fixtures:
        remaining_pairs = derive_missing_round_robin_fixtures(
            team_ids, played_fixtures, season_id=season_row.id
        )
        remaining_fixtures = [
            Fixture(season_row.id, h, a, None, None) for h, a in remaining_pairs
        ]
    fixtures = played_fixtures + remaining_fixtures

    # Build forced results mapping
    forced = {
        f"{names[match.home_team_id]} vs {names[match.away_team_id]}": forced_by_id[match.id]
        for match in unplayed_matches
        if match.id in forced_by_id
    }

    # Fit model and bootstrap (small ensemble for speed)
    cut = datetime.combine(
        (
            season_row.start_date
            if hasattr(season_row, "start_date")
            else datetime.now(timezone.utc).date()
        ),
        time.min,
        tzinfo=timezone.utc,
    )
    history = load_completed_matches(db, comp_row.id, names, before=cut)
    model = DixonColesModel().fit(history)

    team_name_list = [names[tid] for tid in team_ids]
    seed = WHATIF_SEED
    ensemble = bootstrap_parameter_ensemble(
        history,
        team_name_list,
        base_model=model,
        n_bootstrap=WHATIF_BOOTSTRAP,
        seed=seed + 1,
    )

    t0 = perf_counter()
    unforced_results = simulate_vectorized(
        season_id=season_row.id,
        team_ids=team_ids,
        team_names=names,
        fixtures=fixtures,
        parameter_ensemble=ensemble,
        n_simulations=WHATIF_SIMS,
        seed=seed,
    )
    results = simulate_vectorized(
        season_id=season_row.id,
        team_ids=team_ids,
        team_names=names,
        fixtures=fixtures,
        parameter_ensemble=ensemble,
        n_simulations=WHATIF_SIMS,
        seed=seed,
        forced_results=forced,
    )
    elapsed = perf_counter() - t0

    # Strip raw arrays
    results.pop("raw_simulations", None)
    results.pop("finish_position_counts", None)

    deltas = paired_team_deltas(
        cast(list[dict[str, Any]], results["teams"]),
        cast(list[dict[str, Any]], unforced_results["teams"]),
    )

    now = datetime.now(timezone.utc).isoformat()
    context = dict(cached_run.results.get("context", {}))
    context.update(generated_at=now, parameter_bootstrap_fits=WHATIF_BOOTSTRAP)

    return {
        "id": None,
        "season_id": season_row.id,
        "model_version_id": cached_run.model_version_id,
        "n_simulations": WHATIF_SIMS,
        "random_seed": seed,
        "as_of_matchday": cached_run.as_of_matchday,
        "run_at": now,
        "forced_results": [item.model_dump() for item in body.forced_results],
        "runtime_seconds": round(elapsed, 3),
        "note": "What-if runs use 2,000 simulations. Session only.",
        **results,
        "context": context,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_season(db: Session, label: str | None) -> Season:
    """Find a season by label, or the one flagged ``is_current``."""
    if label:
        row = db.query(Season).filter(Season.label == label).first()
    else:
        row = db.query(Season).filter(Season.is_current.is_(True)).first()
    if row is None:
        detail = f"Season '{label}'" if label else "Current season"
        raise HTTPException(status_code=404, detail=f"{detail} not found.")
    return row
