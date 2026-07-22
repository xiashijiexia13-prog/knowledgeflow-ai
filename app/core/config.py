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
    ollama_model: str = "qwen3:4b"
    ollama_timeout_seconds: float = 120.0
    ollama_temperature: float = 0.1
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_batch_size: int = 16
    chunk_size: int = 500
    chunk_overlap: int = 80
    vector_store_dir: Path = Path("data/vector_store")
    chroma_collection: str = "knowledgeflow"
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.86
    max_context_chars: int = 8_000
    max_upload_bytes: int = 10 * 1024 * 1024
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
