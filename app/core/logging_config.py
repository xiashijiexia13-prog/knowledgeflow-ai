"""Central logging configuration for the application."""

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_level: str) -> None:
    """Configure consistent console logging for the current process."""

    normalized_level = log_level.upper()
    numeric_level = getattr(logging, normalized_level, None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )
