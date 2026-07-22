"""FastAPI route handlers."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi import HTTPException, Request
from pydantic import BaseModel

from app.api.schemas import (
    BuildRequest,
    BuildResponse,
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentResponse,
)
from app.core.exceptions import DocumentLoadError
from app.core.container import ApplicationContainer


router = APIRouter()


class HealthResponse(BaseModel):
    """Current application and dependency health."""

    status: str
    ollama_ready: bool
    stored_chunks: int


def get_container(request: Request) -> ApplicationContainer:
    """Read the application-scoped dependency container."""

    return request.app.state.container


ContainerDependency = Annotated[ApplicationContainer, Depends(get_container)]


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(container: ContainerDependency) -> HealthResponse:
    """Report API availability and important local dependency state."""

    ollama_ready = container.ollama.health_check()
    return HealthResponse(
        status="healthy" if ollama_ready else "degraded",
        ollama_ready=ollama_ready,
        stored_chunks=container.vector_store.count,
    )


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_document(
    container: ContainerDependency,
    file: UploadFile = File(...),
) -> DocumentResponse:
    """Validate and persist one PDF, TXT, or Markdown document."""

    content = await file.read(container.settings.max_upload_bytes + 1)
    document = container.document_manager.save(file.filename or "", content)
    return DocumentResponse.from_document(document)


@router.get("/documents", response_model=list[DocumentResponse], tags=["documents"])
def list_documents(container: ContainerDependency) -> list[DocumentResponse]:
    """List documents currently managed by the application."""

    return [
        DocumentResponse.from_document(document)
        for document in container.document_manager.list_documents()
    ]


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    tags=["documents"],
)
def delete_document(
    document_id: str,
    container: ContainerDependency,
) -> DeleteResponse:
    """Remove a source file and its vectors."""

    deleted = container.document_manager.delete(document_id)
    return DeleteResponse(document_id=deleted.document_id, message="Document deleted")


@router.post(
    "/knowledge-base/build",
    response_model=BuildResponse,
    tags=["knowledge-base"],
)
def build_knowledge_base(
    request: BuildRequest,
    container: ContainerDependency,
) -> BuildResponse:
    """Build vectors for every currently managed source document."""

    paths = container.document_manager.document_paths()
    if not paths:
        raise DocumentLoadError("No documents are available to index")
    summary = container.knowledge_base.index_files(paths, reset=request.reset)
    return BuildResponse(**summary.__dict__)


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest, container: ContainerDependency) -> ChatResponse:
    """Run the complete retrieval-augmented question-answering pipeline."""

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be blank")
    return container.rag_pipeline.ask(request.question)
