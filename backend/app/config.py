from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str = "postgresql://postgres:postgres@localhost:5432/xgoal"
    cors_origin: str = "http://localhost:3000"
    api_football_key: Optional[str] = None
    api_football_base_url: str = "https://v3.football.api-sports.io"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()