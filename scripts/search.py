"""Build or query the knowledge base without calling an LLM."""

import argparse
from pathlib import Path

from app.core.config import load_settings
from app.core.logging_config import configure_logging
from app.rag.embeddings import SentenceTransformerEmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.vector_store import ChromaVectorStore
from app.services.knowledge_base import KnowledgeBaseService


def main() -> None:
    """Run the retrieval-only command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for")
    parser.add_argument(
        "--documents",
        nargs="*",
        type=Path,
        default=[],
        help="Optional documents to index before searching",
    )
    parser.add_argument("--reset", action="store_true", help="Rebuild the collection")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(
        settings.log_level,
        settings.log_dir,
        settings.log_max_bytes,
        settings.log_backup_count,
    )
    embeddings = SentenceTransformerEmbeddingService(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
    )
    store = ChromaVectorStore(settings.vector_store_dir, settings.chroma_collection)

    if args.documents:
        summary = KnowledgeBaseService(
            embeddings,
            store,
            settings.chunk_size,
            settings.chunk_overlap,
        ).index_files(args.documents, reset=args.reset)
        print(f"Indexed {summary.documents} documents and {summary.chunks} chunks.")

    results = SemanticRetriever(
        embeddings,
        store,
        settings.retrieval_top_k,
        settings.retrieval_min_score,
    ).search(args.question)
    if not results:
        print("No relevant content found in the knowledge base.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        page = f", page {chunk.page_number}" if chunk.page_number else ""
        print(f"[{rank}] score={result.score:.4f} source={chunk.document_name}{page}")
        print(chunk.text)


if __name__ == "__main__":
    main()
