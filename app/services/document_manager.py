"""Secure local persistence and lifecycle operations for uploaded documents."""

from hashlib import sha256
import logging
from pathlib import Path
import re

from app.core.exceptions import (
    DocumentLoadError,
    DocumentNotFoundError,
    DuplicateDocumentError,
)
from app.models import DocumentPage, StoredDocument
from app.rag.vector_store import ChromaVectorStore
from app.services.document_loader import SUPPORTED_SUFFIXES, load_document


logger = logging.getLogger(__name__)
SAFE_STEM = re.compile(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+")


class DocumentManager:
    """Validate, save, list, and delete files in the local raw-data directory."""

    def __init__(
        self,
        raw_directory: Path,
        vector_store: ChromaVectorStore,
        max_upload_bytes: int,
    ):
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be greater than zero")
        self.raw_directory = raw_directory
        self.vector_store = vector_store
        self.max_upload_bytes = max_upload_bytes
        raw_directory.mkdir(parents=True, exist_ok=True)

    def save(self, original_filename: str, content: bytes) -> StoredDocument:
        """Validate and atomically keep one uploaded document on local disk."""

        if not original_filename.strip():
            raise DocumentLoadError("Uploaded filename cannot be empty")
        if not content:
            raise DocumentLoadError("Uploaded document cannot be empty")
        if len(content) > self.max_upload_bytes:
            raise DocumentLoadError(
                f"Uploaded document exceeds {self.max_upload_bytes} bytes"
            )

        original_name = Path(original_filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise DocumentLoadError("Only PDF, TXT, and Markdown files are supported")

        document_id = sha256(content).hexdigest()
        existing = {document.document_id: document for document in self.list_documents()}
        if document_id in existing:
            raise DuplicateDocumentError(
                f"Document already exists: {existing[document_id].document_name}"
            )

        safe_stem = SAFE_STEM.sub("_", Path(original_name).stem).strip("._")[:80]
        safe_name = f"{safe_stem or 'document'}{suffix}"
        destination = self.raw_directory / safe_name
        if destination.exists():
            destination = self.raw_directory / f"{safe_stem}_{document_id[:12]}{suffix}"

        temporary = destination.with_suffix(f"{destination.suffix}.uploading")
        try:
            temporary.write_bytes(content)
            temporary.replace(destination)
            pages = load_document(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return _to_stored_document(destination, pages)

    def list_documents(self) -> list[StoredDocument]:
        """Return valid managed files, skipping damaged files with a warning."""

        documents: list[StoredDocument] = []
        for path in sorted(self.raw_directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                pages = load_document(path)
                documents.append(_to_stored_document(path, pages))
            except DocumentLoadError:
                logger.warning("Skipping unreadable managed document: %s", path.name)
        return documents

    def document_paths(self) -> list[Path]:
        """Return the current valid source paths for knowledge-base rebuilding."""

        return [document.file_path for document in self.list_documents()]

    def delete(self, document_id: str) -> StoredDocument:
        """Delete one source file and all vectors derived from it."""

        document = self._find(document_id)
        document.file_path.unlink()
        self.vector_store.delete_document(document_id)
        return document

    def replace(
        self,
        document_id: str,
        original_filename: str,
        content: bytes,
    ) -> StoredDocument:
        """Validate a replacement before removing the old file and vectors."""

        existing = self._find(document_id)
        replacement = self.save(original_filename, content)
        try:
            existing.file_path.unlink()
            self.vector_store.delete_document(document_id)
        except Exception:
            replacement.file_path.unlink(missing_ok=True)
            raise
        return replacement

    def _find(self, document_id: str) -> StoredDocument:
        for document in self.list_documents():
            if document.document_id == document_id:
                return document
        raise DocumentNotFoundError(f"Document not found: {document_id}")


def _to_stored_document(path: Path, pages: list[DocumentPage]) -> StoredDocument:
    first_page = pages[0]
    return StoredDocument(
        document_id=first_page.document_id,
        document_name=first_page.document_name,
        file_path=path,
        file_type=first_page.file_type,
        page_count=len(pages),
        size_bytes=path.stat().st_size,
    )
