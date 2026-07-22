"""Build the vector knowledge base from source documents."""

from dataclasses import dataclass
from pathlib import Path

from app.rag.embeddings import EmbeddingProvider
from app.rag.text_splitter import split_pages
from app.rag.vector_store import ChromaVectorStore
from app.services.document_loader import load_document


@dataclass(frozen=True)
class IndexingSummary:
    """Counts produced by one indexing operation."""

    documents: int
    pages: int
    chunks: int
    stored_chunks: int


class KnowledgeBaseService:
    """Coordinate document loading, chunking, embedding, and persistence."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_files(
        self,
        file_paths: list[Path],
        reset: bool = False,
    ) -> IndexingSummary:
        """Index supported files and optionally rebuild the collection first."""

        if not file_paths:
            raise ValueError("At least one document path is required")
        if reset:
            self.vector_store.reset()

        all_pages = []
        for file_path in file_paths:
            all_pages.extend(load_document(file_path))

        chunks = split_pages(all_pages, self.chunk_size, self.chunk_overlap)
        embeddings = self.embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        stored_chunks = self.vector_store.upsert(chunks, embeddings)

        return IndexingSummary(
            documents=len({page.document_id for page in all_pages}),
            pages=len(all_pages),
            chunks=len(chunks),
            stored_chunks=stored_chunks,
        )
