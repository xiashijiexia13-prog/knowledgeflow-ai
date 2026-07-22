"""Tests for TXT, Markdown, and PDF source extraction."""

from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import DocumentLoadError
from app.services.document_loader import load_document


def test_load_utf8_text_with_stable_document_id(tmp_path: Path) -> None:
    document = tmp_path / "handbook.txt"
    document.write_text("年假需要直属主管审批。", encoding="utf-8")

    first = load_document(document)
    second = load_document(document)

    assert first[0].text == "年假需要直属主管审批。"
    assert first[0].document_id == second[0].document_id
    assert first[0].page_number is None


def test_load_pdf_preserves_one_based_page_numbers(tmp_path: Path) -> None:
    document = tmp_path / "manual.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "First page")
    pdf.new_page().insert_text((72, 72), "Second page")
    pdf.save(document)
    pdf.close()

    pages = load_document(document)

    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].document_id == pages[1].document_id


def test_reject_unsupported_document_type(tmp_path: Path) -> None:
    document = tmp_path / "notes.docx"
    document.write_bytes(b"not a supported document")

    with pytest.raises(DocumentLoadError, match="Unsupported document type"):
        load_document(document)
