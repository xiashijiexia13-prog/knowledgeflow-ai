"""Construct and hold the application's long-lived service objects."""

from dataclasses import dataclass

from app.core.config import AppSettings
from app.rag.embeddings import SentenceTransformerEmbeddingService
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import SemanticRetriever
from app.rag.vector_store import ChromaVectorStore
from app.services.document_manager import DocumentManager
from app.services.knowledge_base import KnowledgeBaseService
from app.services.ollama_client import OllamaClient


@dataclass
class ApplicationContainer:
    """Explicit dependency container shared by API request handlers."""

    settings: AppSettings
    embeddings: SentenceTransformerEmbeddingService
    vector_store: ChromaVectorStore
    document_manager: DocumentManager
    knowledge_base: KnowledgeBaseService
    retriever: SemanticRetriever
    ollama: OllamaClient
    rag_pipeline: RAGPipeline

    @classmethod
    def create(cls, settings: AppSettings) -> "ApplicationContainer":
        """Build the production dependency graph from validated settings."""

        embeddings = SentenceTransformerEmbeddingService(
            settings.embedding_model,
            settings.embedding_device,
            settings.embedding_batch_size,
        )
        vector_store = ChromaVectorStore(
            settings.vector_store_dir,
            settings.chroma_collection,
        )
        document_manager = DocumentManager(
            settings.data_dir / "raw",
            vector_store,
            settings.max_upload_bytes,
        )
        knowledge_base = KnowledgeBaseService(
            embeddings,
            vector_store,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        retriever = SemanticRetriever(
            embeddings,
            vector_store,
            settings.retrieval_top_k,
            settings.retrieval_min_score,
        )
        ollama = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
            settings.ollama_temperature,
        )
        rag_pipeline = RAGPipeline(
            retriever,
            ollama,
            settings.max_context_chars,
        )
        return cls(
            settings=settings,
            embeddings=embeddings,
            vector_store=vector_store,
            document_manager=document_manager,
            knowledge_base=knowledge_base,
            retriever=retriever,
            ollama=ollama,
            rag_pipeline=rag_pipeline,
        )

    def close(self) -> None:
        """Release resources owned by container services."""

        self.ollama.close()
