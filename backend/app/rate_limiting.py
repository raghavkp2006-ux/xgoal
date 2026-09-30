"""Shared API limiter and coverage for FastAPI's grouped included routers."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

limiter = Limiter(
    key_func=get_remote_address, default_limits=["60/minute"], headers_enabled=True
)


def check_rate_limit(request: Request) -> None:
    """Cover included routes when SlowAPI middleware cannot resolve their handler."""
    if not getattr(request.state, "_rate_limiting_complete", False):
        limiter._check_request_limit(request, request.scope["endpoint"], False)
        request.state._rate_limiting_complete = True
