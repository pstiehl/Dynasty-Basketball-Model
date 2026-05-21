"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///dynasty_bball.db"

    # HTTP defaults
    request_timeout_seconds: int = 30
    user_agent: str = (
        "DynastyBasketballModel/0.1 "
        "(open-source dynasty NBA aggregator; https://github.com/pstiehl/Dynasty-Basketball-Model)"
    )


settings = Settings()
