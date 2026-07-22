"""Semantic retrieval without language-model generation."""

from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import ChromaVectorStore
from app.models import SearchResult


class SemanticRetriever:
    """Embed a question and retrieve the most similar stored chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        top_k: int = 4,
        min_score: float = 0.45,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def search(self, question: str) -> list[SearchResult]:
        """Return relevant source chunks ordered from highest score to lowest."""

        if not question.strip():
            raise ValueError("Question cannot be empty")
        query_embedding = self.embedding_provider.embed_query(question)
        return self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
            min_score=self.min_score,
        )
