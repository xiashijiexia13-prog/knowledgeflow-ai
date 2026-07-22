"""Structured document data passed through the RAG pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SupportedFileType = Literal["txt", "md", "pdf"]


class DocumentPage(BaseModel):
    """Text and source metadata extracted from one logical document page."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    file_path: Path
    file_type: SupportedFileType
    page_number: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)


class TextChunk(BaseModel):
    """A searchable text segment that retains its original source metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    file_path: Path
    file_type: SupportedFileType
    page_number: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)


class SearchResult(BaseModel):
    """One retrieved chunk and its cosine similarity score."""

    model_config = ConfigDict(frozen=True)

    chunk: TextChunk
    score: float = Field(ge=-1.0, le=1.0)


class SourceReference(BaseModel):
    """Source information safe to expose with a generated answer."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_name: str
    page_number: int | None
    chunk_id: str
    score: float
    excerpt: str


class RAGAnswer(BaseModel):
    """Grounded answer and the exact retrieved sources used to produce it."""

    model_config = ConfigDict(frozen=True)

    answer: str
    answered: bool
    sources: list[SourceReference]
