"""Tests for manual whitespace cleaning and overlapping chunk creation."""

import pytest

from app.models import DocumentPage
from app.rag.text_splitter import clean_text, split_pages


def make_page(text: str) -> DocumentPage:
    return DocumentPage(
        document_id="doc-1",
        document_name="handbook.md",
        file_path="handbook.md",
        file_type="md",
        page_number=3,
        text=text,
    )


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  first\tline\r\n\r\n\r\n second  line ") == (
        "first line\n\nsecond line"
    )


def test_split_pages_preserves_metadata_and_overlap() -> None:
    chunks = split_pages([make_page("A" * 600)], chunk_size=200, chunk_overlap=40)

    assert len(chunks) == 4
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert chunks[1].start_char == chunks[0].end_char - 40
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_reject_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        split_pages([], chunk_size=100, chunk_overlap=100)
