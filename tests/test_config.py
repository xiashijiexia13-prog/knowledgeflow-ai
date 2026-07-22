"""Tests for early validation of environment-driven application settings."""

from pydantic import ValidationError
import pytest

from app.core.config import AppSettings


def test_accepts_valid_defaults() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.chunk_size == 500
    assert settings.retrieval_min_score == 0.86


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_batch_size", 0),
        ("ollama_temperature", 2.1),
        ("retrieval_min_score", 1.1),
        ("max_upload_bytes", 0),
    ],
)
def test_rejects_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **{field: value})


def test_rejects_overlap_not_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        AppSettings(_env_file=None, chunk_size=100, chunk_overlap=100)
