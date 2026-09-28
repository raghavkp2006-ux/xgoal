"""Simulation endpoints — cached baseline reads and interactive what-if runs.

GET  /api/v1/simulation?season=...   → latest precomputed simulation
POST /api/v1/simulation/whatif       → 2,000-sim what-if with forced results
"""

from __future__ import annotations

import hashlib
from datetime import datetime, time, timezone
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Match, Season, SimulationRun

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])

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


class ForcedResult(BaseModel):
    """A single forced match result for what-if mode."""

    home_team: str = Field(..., description="Home team canonical name")
    away_team: str = Field(..., description="Away team canonical name")
    home_goals: int = Field(..., ge=0, le=20)
    away_goals: int = Field(..., ge=0, le=20)


class WhatIfRequest(BaseModel):
    """Request body for the what-if simulation endpoint."""

    forced_results: list[ForcedResult] = Field(
        ..., min_length=1, max_length=20
    )
    season: str | None = Field(
        None,
        description="Season label. Defaults to the current season.",
    )


@router.post("/whatif")
def run_whatif(
    body: WhatIfRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run a reduced (2,000-sim) Monte Carlo with forced match results.

    The result is transient — it is **not** persisted to the database.
    A baseline comparison is included so the frontend can compute deltas.
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

    season_row = _resolve_season(db, body.season)
    competition = (
        db.query(Match)
        .filter(Match.season_id == season_row.id)
        .first()
    )
    if competition is None:
        raise HTTPException(
            status_code=404,
            detail="No matches found for this season.",
        )

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
    played_matches = (
        db.query(Match)
        .filter(
            Match.season_id == season_row.id,
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
        )
        .order_by(Match.kickoff_utc)
        .all()
    )

    team_ids = sorted(
        {
            tid
            for m in played_matches
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
    remaining_pairs = derive_missing_round_robin_fixtures(
        team_ids, played_fixtures, season_id=season_row.id
    )
    remaining_fixtures = [
        Fixture(season_row.id, h, a, None, None) for h, a in remaining_pairs
    ]
    fixtures = played_fixtures + remaining_fixtures

    # Build forced results mapping
    forced: dict[str, tuple[int, int]] = {}
    for fr in body.forced_results:
        key = f"{fr.home_team} vs {fr.away_team}"
        forced[key] = (fr.home_goals, fr.away_goals)

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
    seed = 42
    ensemble = bootstrap_parameter_ensemble(
        history,
        team_name_list,
        base_model=model,
        n_bootstrap=WHATIF_BOOTSTRAP,
        seed=seed + 1,
    )

    t0 = perf_counter()
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

    # Load baseline for delta computation
    baseline_run: SimulationRun | None = (
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

    baseline_teams = None
    if baseline_run and baseline_run.results:
        baseline_results = dict(baseline_run.results)
        baseline_teams = baseline_results.get("teams")

    return {
        "n_simulations": WHATIF_SIMS,
        "random_seed": seed,
        "forced_results": {
            f"{fr.home_team} vs {fr.away_team}": [fr.home_goals, fr.away_goals]
            for fr in body.forced_results
        },
        "runtime_seconds": round(elapsed, 3),
        "note": (
            f"What-if result from {WHATIF_SIMS} simulations (not 10,000). "
            "Session-only — not persisted."
        ),
        **results,
        "baseline_teams": baseline_teams,
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
