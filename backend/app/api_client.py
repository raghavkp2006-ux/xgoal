"""API-Football v3 client with automatic request logging and rate limiting."""
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import ApiRequestLog


class ApiFootballClient:
    """Thin client for api-football.com v3 with request logging."""

    BASE_URL: str = settings.api_football_base_url
    KEY: str = settings.api_football_key or ""

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=self.BASE_URL, timeout=30.0)

    # ── low-level request ─────────────────────────────────────────────

    def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        db: Optional[Session] = None,
    ) -> dict:
        """Make a request, log to DB, and return the JSON response."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        t0 = time.monotonic()
        try:
            resp = self._client.get(
                endpoint,
                params=params,
                headers={"x-apisports-key": self.KEY},
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            data = resp.json()

            # Parse quota from headers
            quota_remaining = resp.headers.get("x-ratelimit-requests-remaining")
            if quota_remaining is not None:
                try:
                    quota_remaining = int(quota_remaining)
                except ValueError:
                    quota_remaining = None

            # Log the request
            log_entry = ApiRequestLog(
                requested_at=datetime.now(timezone.utc),
                endpoint=endpoint,
                params=params,
                status_code=resp.status_code,
                quota_remaining=quota_remaining,
                duration_ms=elapsed_ms,
            )
            db.add(log_entry)
            db.commit()

            if not data.get("get", ""):
                data["get"] = endpoint
            return data
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log_entry = ApiRequestLog(
                requested_at=datetime.now(timezone.utc),
                endpoint=endpoint,
                params=params,
                status_code=None,
                quota_remaining=None,
                duration_ms=elapsed_ms,
            )
            db.add(log_entry)
            db.commit()
            raise
        finally:
            if close_db:
                db.close()

    # ── convenience helpers ───────────────────────────────────────────

    def get_status(self) -> dict:
        return self._request("/status")

    def get_leagues(self, league_id: int) -> dict:
        return self._request("/leagues", {"id": league_id})

    def get_teams(self, league_id: int, season: int) -> dict:
        return self._request("/teams", {"league": league_id, "season": season})

    def get_standings(self, league_id: int, season: int) -> dict:
        return self._request("/standings", {"league": league_id, "season": season})

    def get_fixtures(self, league_id: int, season: int) -> dict:
        return self._request("/fixtures", {"league": league_id, "season": season})

    def get_fixture_events(self, fixture_id: int) -> dict:
        return self._request("/fixtures/events", {"fixture": fixture_id})

    def get_players(self, league_id: int, season: int) -> dict:
        return self._request("/players", {"league": league_id, "season": season})

    def close(self) -> None:
        self._client.close()