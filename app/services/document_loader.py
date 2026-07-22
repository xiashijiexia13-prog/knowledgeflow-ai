"""Load supported files into a common page-oriented document model."""

from hashlib import sha256
from pathlib import Path
from typing import cast

import pymupdf

from app.core.exceptions import DocumentLoadError
from app.models.document import DocumentPage, SupportedFileType


SUPPORTED_SUFFIXES: dict[str, SupportedFileType] = {
    ".txt": "txt",
    ".md": "md",
    ".pdf": "pdf",
}
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def load_document(file_path: Path) -> list[DocumentPage]:
    """Validate and extract a TXT, Markdown, or PDF file."""

    path = file_path.resolve()
    if not path.is_file():
        raise DocumentLoadError(f"Document does not exist: {path}")

    file_type = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if file_type is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentLoadError(
            f"Unsupported document type '{path.suffix}'. Supported: {supported}"
        )

    document_id = _calculate_file_hash(path)
    if file_type == "pdf":
        pages = _load_pdf(path, document_id)
    else:
        pages = [_load_text_file(path, document_id, file_type)]

    if not pages:
        raise DocumentLoadError(f"Document contains no extractable text: {path.name}")
    return pages


def _calculate_file_hash(path: Path) -> str:
    """Return a stable content-based identifier without loading the whole file."""

    digest = sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DocumentLoadError(f"Cannot read document: {path}") from error
    return digest.hexdigest()


def _load_text_file(
    path: Path,
    document_id: str,
    file_type: SupportedFileType,
) -> DocumentPage:
    """Decode a text document with common Chinese-compatible encodings."""

    for encoding in TEXT_ENCODINGS:
        try:
            text = path.read_text(encoding=encoding).strip()
            break
        except UnicodeDecodeError:
            continue
        except OSError as error:
            raise DocumentLoadError(f"Cannot read document: {path}") from error
    else:
        raise DocumentLoadError(f"Cannot decode text document: {path.name}")

    if not text:
        raise DocumentLoadError(f"Document contains no text: {path.name}")

    return _create_page(path, document_id, file_type, text)


def _load_pdf(path: Path, document_id: str) -> list[DocumentPage]:
    """Extract non-empty PDF pages while preserving one-based page numbers."""

    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise DocumentLoadError(f"Password-protected PDF is not supported: {path.name}")
            return [
                _create_page(path, document_id, "pdf", text, page.number + 1)
                for page in pdf
                if (text := page.get_text().strip())
            ]
    except DocumentLoadError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise DocumentLoadError(f"Cannot parse PDF document: {path.name}") from error


def _create_page(
    path: Path,
    document_id: str,
    file_type: str,
    text: str,
    page_number: int | None = None,
) -> DocumentPage:
    """Build one validated page with consistent source metadata."""

    return DocumentPage(
        document_id=document_id,
        document_name=path.name,
        file_path=path,
        file_type=cast(SupportedFileType, file_type),
        page_number=page_number,
        text=text,
    )
