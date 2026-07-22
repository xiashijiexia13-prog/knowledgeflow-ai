"""Validated HTTP request and response bodies."""

from pydantic import BaseModel, Field

from app.models import RAGAnswer, StoredDocument


class DocumentResponse(BaseModel):
    """Public metadata for one managed document."""

    document_id: str
    document_name: str
    file_type: str
    page_count: int
    size_bytes: int

    @classmethod
    def from_document(cls, document: StoredDocument) -> "DocumentResponse":
        return cls(**document.model_dump(exclude={"file_path"}))


class BuildRequest(BaseModel):
    """Knowledge-base rebuild options."""

    reset: bool = True


class BuildResponse(BaseModel):
    """Indexing counts returned after a successful build."""

    documents: int
    pages: int
    chunks: int
    stored_chunks: int


class ChatRequest(BaseModel):
    """One user question for the RAG pipeline."""

    question: str = Field(min_length=1, max_length=2_000)


class DeleteResponse(BaseModel):
    """Confirmation that one document was removed."""

    document_id: str
    message: str


ChatResponse = RAGAnswer
