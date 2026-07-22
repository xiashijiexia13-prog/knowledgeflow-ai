"""Typed application configuration loaded from the environment."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Configuration values shared across the application."""

    app_name: str = "KnowledgeFlow AI"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: Path = Path("logs")
    log_max_bytes: int = 1_048_576
    log_backup_count: int = 3
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    data_dir: Path = Path("data")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def load_settings() -> AppSettings:
    """Load and validate configuration from defaults, .env, and the environment."""

    return AppSettings()
