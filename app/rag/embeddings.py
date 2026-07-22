"""Embedding interfaces and a sentence-transformers implementation."""

import logging
from typing import Protocol

from sentence_transformers import SentenceTransformer

from app.core.exceptions import EmbeddingError


logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Minimal contract required by indexing and retrieval components."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document passages into normalized vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Convert one search query into a normalized vector."""


class SentenceTransformerEmbeddingService:
    """Generate multilingual E5 embeddings with lazy model loading."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        if batch_size <= 0:
            raise EmbeddingError("embedding batch_size must be greater than zero")
        self.model_name = model_name
        self.device = None if device == "auto" else device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        """Return the number of numeric coordinates in each vector."""

        dimension = self._get_model().get_embedding_dimension()
        if dimension is None:
            raise EmbeddingError("Embedding model did not report its vector dimension")
        return dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed non-empty document passages using the E5 passage prefix."""

        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Document texts cannot contain empty values")
        return self._encode([f"passage: {text.strip()}" for text in texts])

    def embed_query(self, query: str) -> list[float]:
        """Embed one non-empty question using the E5 query prefix."""

        if not query.strip():
            raise EmbeddingError("Query cannot be empty")
        return self._encode([f"query: {query.strip()}"])[0]

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as error:
                raise EmbeddingError(
                    f"Cannot load embedding model: {self.model_name}"
                ) from error
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._get_model().encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError("Embedding generation failed") from error
        return vectors.tolist()
