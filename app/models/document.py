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
