"""Manual text cleaning and character-based chunking."""

from hashlib import sha256
import re

from app.models import DocumentPage, TextChunk


PARAGRAPH_BREAK = re.compile(r"\n{3,}")
HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")
PREFERRED_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "; ", "；")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful paragraph boundaries."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n")
    )
    return PARAGRAPH_BREAK.sub("\n\n", normalized).strip()


def split_pages(
    pages: list[DocumentPage],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[TextChunk]:
    """Clean and split pages into overlapping chunks with inherited metadata."""

    _validate_chunk_settings(chunk_size, chunk_overlap)
    chunks: list[TextChunk] = []

    for page in pages:
        text = clean_text(page.text)
        if not text:
            continue

        page_chunks = _split_text(text, chunk_size, chunk_overlap)
        for index, (start, end, content) in enumerate(page_chunks):
            identity = f"{page.document_id}:{page.page_number}:{index}:{content}"
            chunks.append(
                TextChunk(
                    chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
                    document_id=page.document_id,
                    document_name=page.document_name,
                    file_path=page.file_path,
                    file_type=page.file_type,
                    page_number=page.page_number,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    text=content,
                )
            )

    return chunks


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int, str]]:
    """Return character ranges and text, preferring natural sentence boundaries."""

    pieces: list[tuple[int, int, str]] = []
    start = 0

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = _choose_boundary(text, start, hard_end, chunk_size)
        content = text[start:end].strip()
        if content:
            pieces.append((start, end, content))
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return pieces


def _choose_boundary(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    if hard_end >= len(text):
        return len(text)

    earliest = start + max(chunk_size // 2, 1)
    window = text[earliest:hard_end]
    best_end = -1

    for separator in PREFERRED_SEPARATORS:
        position = window.rfind(separator)
        if position >= 0:
            best_end = max(best_end, earliest + position + len(separator))

    return best_end if best_end > start else hard_end
