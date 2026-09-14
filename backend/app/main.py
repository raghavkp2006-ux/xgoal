"""FastAPI application entry point for xgoal backend."""

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.database import SessionLocal
from app.models import DataFreshness
from app.routers import teams

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="xgoal — La Liga Analytics",
    version="0.1.0",
    description="La Liga analytics platform with Dixon-Coles model and season simulator",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(teams.router)


@app.get("/health")
def health():
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