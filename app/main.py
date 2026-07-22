"""FastAPI application factory and lifecycle configuration."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import load_settings
from app.core.container import ApplicationContainer
from app.core.logging_config import configure_logging


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    """Create an API app with optionally injected services for testing."""

    settings = container.settings if container else load_settings()
    configure_logging(
        settings.log_level,
        settings.log_dir,
        settings.log_max_bytes,
        settings.log_backup_count,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_container = container or ApplicationContainer.create(settings)
        app.state.container = active_container
        yield
        active_container.close()

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Retrieval-augmented local knowledge-base assistant",
        lifespan=lifespan,
    )
    application.include_router(router, prefix="/api/v1")
    return application


app = create_app()
