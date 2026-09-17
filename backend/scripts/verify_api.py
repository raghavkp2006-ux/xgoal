#!/usr/bin/env python3
"""Verify the API-Football key and connectivity (Phase 1.2 acceptance check).

Hits the two endpoints the plan calls out — ``/status`` (account, daily quota,
subscription end) and ``/leagues?id=140`` (La Liga seasons) — so a dead key, an
exhausted free-tier quota or an expired subscription is obvious *before* an
ingest run starts and burns requests.

Usage:
    python scripts/verify_api.py
    python scripts/verify_api.py --league 141
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402

TIMEOUT = 20.0


def fetch(client: httpx.Client, endpoint: str, params: dict[str, object] | None = None) -> dict:
    """GET one endpoint and report its HTTP status, API errors and JSON body."""
    response = client.get(endpoint, params=params or {})
    print(f"  {endpoint} -> HTTP {response.status_code}")
    payload = response.json()
    errors = payload.get("errors")
    if errors:
        print(f"    API errors: {errors}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the API-Football key and endpoints.")
    parser.add_argument("--league", type=int, default=140, help="league id (140 = La Liga)")
    args = parser.parse_args()

    if not settings.api_football_key:
        raise SystemExit("API_FOOTBALL_KEY is not set - copy .env.example to .env")

    headers = {"x-apisports-key": settings.api_football_key}
    print(f"checking {settings.api_football_base_url}")
    with httpx.Client(
        base_url=settings.api_football_base_url, headers=headers, timeout=TIMEOUT
    ) as client:
        status = fetch(client, "/status").get("response") or {}
        account = status.get("account") or {}
        quota = status.get("requests") or {}
        subscription = status.get("subscription") or {}
        name = f"{account.get('firstname', '?')} {account.get('lastname', '')}".strip()
        print(f"    account: {name}")
        print(f"    quota today: {quota.get('current')}/{quota.get('limit_day')}")
        print(f"    subscription ends: {subscription.get('end')}")

        leagues = fetch(client, "/leagues", {"id": args.league}).get("response") or []
        for entry in leagues:
            league = entry.get("league") or {}
            country = entry.get("country") or {}
            seasons = [season.get("season") for season in (league.get("seasons") or [])]
            current = [
                season.get("season")
                for season in (league.get("seasons") or [])
                if season.get("current")
            ]
            print(f"    {league.get('name')} ({country.get('name')})")
            print(f"    seasons: {seasons}")
            print(f"    current: {current}")

    if not leagues:
        print("  ! no league returned - the key may be invalid or the request throttled")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
