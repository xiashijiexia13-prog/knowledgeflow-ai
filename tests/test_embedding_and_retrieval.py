"""Tests for embedding prefixes and persistent vector retrieval."""

from pathlib import Path

import numpy as np

from app.models import TextChunk
from app.rag.embeddings import SentenceTransformerEmbeddingService
from app.rag.vector_store import ChromaVectorStore


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.last_texts = texts
        return np.array([[1.0, 0.0] for _ in texts])


def make_chunk(chunk_id: str, document_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.txt",
        file_path=f"{document_id}.txt",
        file_type="txt",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
    )


def test_e5_query_and_passage_prefixes() -> None:
    service = SentenceTransformerEmbeddingService("fake-model")
    fake_model = FakeSentenceTransformer()
    service._model = fake_model

    assert service.embed_query("leave") == [1.0, 0.0]
    assert fake_model.last_texts == ["query: leave"]
    service.embed_documents(["policy"])
    assert fake_model.last_texts == ["passage: policy"]
    assert service.dimension == 2


def test_chroma_persists_and_filters_by_similarity(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "vectors", "test_collection")
    chunks = [
        make_chunk("leave", "doc-leave", "annual leave"),
        make_chunk("backup", "doc-backup", "database backup"),
    ]
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    result = store.query([1.0, 0.0], top_k=2, min_score=0.5)
    reopened = ChromaVectorStore(tmp_path / "vectors", "test_collection")

    assert [item.chunk.chunk_id for item in result] == ["leave"]
    assert result[0].score > 0.99
    assert reopened.count == 2

    reopened.delete_document("doc-leave")
    assert reopened.count == 1
