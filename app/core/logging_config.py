"""Central logging configuration for the application."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.exceptions import ConfigurationError


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_NAME = "knowledgeflow.log"


def configure_logging(
    log_level: str,
    log_dir: Path | None = None,
    max_bytes: int = 1_048_576,
    backup_count: int = 3,
) -> None:
    """Configure consistent console and optional rotating file logging."""

    normalized_level = log_level.upper()
    numeric_level = getattr(logging, normalized_level, None)

    if not isinstance(numeric_level, int):
        raise ConfigurationError(f"Unsupported log level: {log_level}")

    if max_bytes <= 0:
        raise ConfigurationError("max_bytes must be greater than zero")

    if backup_count < 0:
        raise ConfigurationError("backup_count cannot be negative")

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                filename=log_dir / LOG_FILE_NAME,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,
            )
        )

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
