"""Integration tests for FastAPI request validation and route coordination."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.core.config import AppSettings
from app.main import create_app
from app.models import RAGAnswer
from app.rag.vector_store import ChromaVectorStore
from app.services.document_manager import DocumentManager
from app.services.knowledge_base import IndexingSummary


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = AppSettings(
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vectors",
        log_dir=tmp_path / "logs",
        max_upload_bytes=1_024,
    )
    vector_store = ChromaVectorStore(settings.vector_store_dir, "api_tests")
    manager = DocumentManager(tmp_path / "raw", vector_store, 1_024)
    container = SimpleNamespace(
        settings=settings,
        ollama=SimpleNamespace(health_check=lambda: True),
        vector_store=vector_store,
        document_manager=manager,
        knowledge_base=SimpleNamespace(
            index_files=lambda paths, reset: IndexingSummary(
                documents=len(paths),
                pages=len(paths),
                chunks=len(paths),
                stored_chunks=len(paths),
            )
        ),
        rag_pipeline=SimpleNamespace(
            ask=lambda question: RAGAnswer(
                answer=f"grounded: {question}",
                answered=True,
                sources=[],
            )
        ),
        close=lambda: None,
    )

    with TestClient(create_app(container)) as test_client:
        yield test_client


def test_health_and_openapi(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {
        "status": "healthy",
        "ollama_ready": True,
        "stored_chunks": 0,
    }
    assert "/api/v1/chat" in client.get("/openapi.json").json()["paths"]


def test_document_lifecycle_and_build(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("handbook.txt", "annual leave policy", "text/plain")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]

    duplicate = client.post(
        "/api/v1/documents/upload",
        files={"file": ("copy.txt", "annual leave policy", "text/plain")},
    )
    assert duplicate.status_code == 409
    assert len(client.get("/api/v1/documents").json()) == 1

    build = client.post("/api/v1/knowledge-base/build", json={"reset": True})
    assert build.status_code == 200
    assert build.json()["documents"] == 1

    assert client.delete(f"/api/v1/documents/{document_id}").status_code == 200
    assert client.delete(f"/api/v1/documents/{document_id}").status_code == 404


def test_chat_validation_and_response(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"question": "leave process"})
    assert response.status_code == 200
    assert response.json()["answer"] == "grounded: leave process"

    assert client.post("/api/v1/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/v1/chat", json={"question": "   "}).status_code == 422


def test_rejects_unsupported_and_oversized_uploads(client: TestClient) -> None:
    unsupported = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.docx", b"content", "application/octet-stream")},
    )
    oversized = client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.txt", b"x" * 1_025, "text/plain")},
    )

    assert unsupported.status_code == 400
    assert oversized.status_code == 400
