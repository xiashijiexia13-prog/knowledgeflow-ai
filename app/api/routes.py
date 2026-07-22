"""FastAPI route handlers."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

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
