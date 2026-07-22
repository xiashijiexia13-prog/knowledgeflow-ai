"""Typed application configuration loaded from the environment."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Configuration values shared across the application."""

    app_name: str = "KnowledgeFlow AI"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: Path = Path("logs")
    log_max_bytes: int = Field(default=1_048_576, gt=0)
    log_backup_count: int = Field(default=3, ge=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_batch_size: int = Field(default=16, gt=0)
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)
    vector_store_dir: Path = Path("data/vector_store")
    chroma_collection: str = "knowledgeflow"
    retrieval_top_k: int = Field(default=4, gt=0)
    retrieval_min_score: float = Field(default=0.86, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=8_000, gt=0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    data_dir: Path = Path("data")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "AppSettings":
        """Reject chunk settings that could make the splitter stop progressing."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def load_settings() -> AppSettings:
    """Load and validate configuration from defaults, .env, and the environment."""

    return AppSettings()
