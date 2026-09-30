"""FastAPI application entry point for xgoal backend."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import anyio
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings
from app.database import SessionLocal, get_engine
from app.models import DataFreshness
from app.rate_limiting import check_rate_limit, limiter
from app.routers import (
    competitions,
    matches,
    model_versions,
    predictions,
    simulation,
    standings,
    teams,
)


def rate_limit_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RateLimitExceeded):
        raise exc
    response = _rate_limit_exceeded_handler(request, exc)
    return JSONResponse(
        status_code=429,
        media_type="application/problem+json",
        headers={
            name: value for name, value in response.headers.items()
            if name not in {"content-type", "content-length"}
        },
        content={
            "type": "about:blank",
            "title": "Too Many Requests",
            "status": 429,
            "detail": f"Rate limit exceeded: {exc.detail}",
        },
    )


def _warm_database_pool() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    await anyio.to_thread.run_sync(_warm_database_pool)
    yield


app = FastAPI(
    title="xgoal — La Liga Analytics",
    version="0.1.0",
    description="La Liga analytics platform with Dixon-Coles model and season simulator",
    lifespan=lifespan,
    dependencies=[Depends(check_rate_limit)],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(competitions.router)
app.include_router(matches.router)
app.include_router(model_versions.router)
app.include_router(predictions.router)
app.include_router(standings.router)
app.include_router(teams.router)
app.include_router(simulation.router)


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint. Returns DB connectivity and data freshness."""
    db_ok = False
    sources = []
    try:
        db = SessionLocal()
        db_ok = True
        freshness = db.query(DataFreshness).all()
        for f in freshness:
            age_seconds = None
            if f.last_success_at:
                age_seconds = int(
                    (datetime.now(timezone.utc) - f.last_success_at).total_seconds()
                )
            sources.append(
                {
                    "name": f.source,
                    "age_seconds": age_seconds,
                    "last_error": f.last_error,
                }
            )
        db.close()
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "sources": sources,
    }


limiter.exempt(health)  # type: ignore[no-untyped-call]  # slowapi's exempt lacks annotations.
