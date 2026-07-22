"""Persistent ChromaDB storage for text chunks and their vectors."""

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from app.models import SearchResult, TextChunk
from app.models.document import SupportedFileType


class ChromaVectorStore:
    """Store, query, delete, and rebuild a local cosine-similarity collection."""

    def __init__(self, persist_directory: Path, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._get_or_create_collection()

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        """Insert or replace chunks and return the resulting collection size."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return self.count
        if any(not vector for vector in embeddings):
            raise ValueError("embedding vectors cannot be empty")

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_to_metadata(chunk) for chunk in chunks],
        )
        return self.count

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks above a cosine similarity threshold."""

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        if self.count == 0:
            return []

        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = (response["documents"] or [[]])[0]
        metadatas = (response["metadatas"] or [[]])[0]
        distances = (response["distances"] or [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            if score >= min_score and document is not None and metadata is not None:
                results.append(
                    SearchResult(
                        chunk=_metadata_to_chunk(chunk_id, document, metadata),
                        score=score,
                    )
                )
        return results

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk belonging to one source document."""

        self._collection.delete(where={"document_id": document_id})

    def reset(self) -> None:
        """Delete and recreate the complete collection."""

        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _chunk_to_metadata(chunk: TextChunk) -> dict[str, str | int]:
    return {
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "file_path": str(chunk.file_path),
        "file_type": chunk.file_type,
        "page_number": chunk.page_number or -1,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def _metadata_to_chunk(
    chunk_id: str,
    document: str,
    metadata: dict[str, Any],
) -> TextChunk:
    page_number = int(metadata["page_number"])
    return TextChunk(
        chunk_id=chunk_id,
        document_id=str(metadata["document_id"]),
        document_name=str(metadata["document_name"]),
        file_path=Path(str(metadata["file_path"])),
        file_type=cast(SupportedFileType, str(metadata["file_type"])),
        page_number=None if page_number < 1 else page_number,
        chunk_index=int(metadata["chunk_index"]),
        start_char=int(metadata["start_char"]),
        end_char=int(metadata["end_char"]),
        text=document,
    )
