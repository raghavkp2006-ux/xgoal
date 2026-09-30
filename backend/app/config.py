from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str = "postgresql://postgres:postgres@localhost:5432/xgoal"
    cors_origin: str = "http://localhost:3000"
    api_football_key: Optional[str] = None
    api_football_base_url: str = "https://v3.football.api-sports.io"
    log_level: str = "INFO"
    rate_limit_default: str = "60/minute"
    rate_limit_expensive: str = "10/minute"

    @field_validator("cors_origin")
    @classmethod
    def validate_cors_origin(cls, value: str) -> str:
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if "*" in origins:
            raise ValueError("CORS_ORIGIN must list explicit origins; '*' is not allowed")
        return ",".join(origins)

    @property
    def cors_origins(self) -> list[str]:
        origins = [origin for origin in self.cors_origin.split(",") if origin]
        return list(dict.fromkeys([*origins, "http://localhost:3000"]))

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
