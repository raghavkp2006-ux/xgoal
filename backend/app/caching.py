"""Shared HTTP validators and cache policies for Postgres read responses."""

import hashlib

from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LONG_CACHE = "public, max-age=3600"
SHORT_CACHE = "public, max-age=60, stale-while-revalidate=300"
SIMULATION_CACHE = "public, max-age=300, stale-while-revalidate=3600"


def cache_response(response: Response, if_none_match: str | None, cache_control: str) -> Response:
    """Hash the serialized body and implement weak/comma-separated GET validators."""
    if response.status_code >= 400 or cache_control == "no-store":
        response.headers["Cache-Control"] = "no-store"
        if "etag" in response.headers:
            del response.headers["etag"]
        return response
    if not 200 <= response.status_code < 300:
        return response
    etag = f'"{hashlib.sha256(response.body).hexdigest()}"'
    response.headers.update({"ETag": etag, "Cache-Control": cache_control})
    if if_none_match and any(
        candidate.strip() in {"*", etag, f"W/{etag}"} for candidate in if_none_match.split(",")
    ):
        headers = {
            key: value
            for key, value in response.headers.items()
            if key not in {"content-length", "content-type"}
        }
        return Response(status_code=304, headers=headers)
    return response


def read_cache_policy(path: str) -> str | None:
    if path == "/health":
        return "no-store"
    parts = path.strip("/").split("/")
    if len(parts) < 3 or parts[:2] != ["api", "v1"]:
        return None
    if parts[2] in {"competitions", "teams", "model-versions"}:
        return LONG_CACHE
    if parts[2] in {"standings", "matches", "predictions"}:
        return SHORT_CACHE
    if parts[2] == "simulation":
        return SIMULATION_CACHE
    return None


class ReadCacheMiddleware:
    """Apply one policy after serialization, including handled GET errors."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        policy = read_cache_policy(scope.get("path", ""))
        if scope["type"] != "http" or scope.get("method") != "GET" or policy is None:
            await self.app(scope, receive, send)
            return
        start: Message = {}
        body = bytearray()

        async def cache_send(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    response = Response(content=bytes(body), status_code=start["status"])
                    response.raw_headers = list(start["headers"])
                    response = cache_response(
                        response, Headers(scope=scope).get("if-none-match"), policy
                    )
                    await response(scope, receive, send)
            else:
                await send(message)

        await self.app(scope, receive, cache_send)
