"""Deployment settings checks that do not access a database or external service."""

import pytest
from pydantic import ValidationError
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import Settings


def test_single_cors_origin_keeps_localhost():
    settings = Settings(_env_file=None, cors_origin="https://a.example")
    assert settings.cors_origins == ["https://a.example", "http://localhost:3000"]


def test_comma_separated_cors_origins():
    settings = Settings(_env_file=None, cors_origin="https://a.example,https://b.example")
    assert settings.cors_origins == [
        "https://a.example", "https://b.example", "http://localhost:3000"
    ]


def test_cors_whitespace_empty_entries_and_duplicates():
    settings = Settings(
        _env_file=None,
        cors_origin=" , https://a.example , , https://b.example, http://localhost:3000, ",
    )
    assert settings.cors_origins == [
        "https://a.example", "https://b.example", "http://localhost:3000"
    ]


@pytest.mark.parametrize("origins", ["*", " * ", "https://a.example, *, https://b.example"])
def test_wildcard_cors_origin_is_rejected(origins):
    with pytest.raises(ValidationError, match="CORS_ORIGIN must list explicit origins"):
        Settings(_env_file=None, cors_origin=origins)


def test_rate_limit_defaults(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_DEFAULT", raising=False)
    monkeypatch.delenv("RATE_LIMIT_EXPENSIVE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.rate_limit_default == "60/minute"
    assert settings.rate_limit_expensive == "10/minute"


def test_rate_limits_are_configurable_from_environment(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "120/minute")
    monkeypatch.setenv("RATE_LIMIT_EXPENSIVE", "20/minute")
    settings = Settings(_env_file=None)
    assert settings.rate_limit_default == "120/minute"
    assert settings.rate_limit_expensive == "20/minute"


def test_rate_key_does_not_trust_a_raw_forwarded_header():
    request = Request({
        "type": "http",
        "client": ("127.0.0.1", 50000),
        "headers": [(b"x-forwarded-for", b"1.1.1.1")],
    })
    assert get_remote_address(request) == "127.0.0.1"
