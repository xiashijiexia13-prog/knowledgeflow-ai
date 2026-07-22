"""Evaluate retrieval and optional generation on a small labelled dataset."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from app.core.config import load_settings
from app.rag.embeddings import SentenceTransformerEmbeddingService
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import SemanticRetriever
from app.rag.vector_store import ChromaVectorStore
from app.services.knowledge_base import KnowledgeBaseService
from app.services.ollama_client import OllamaClient


def evaluate(
    dataset_path: Path,
    documents: list[Path],
    top_k: int,
    min_score: float,
    with_generation: bool,
) -> dict[str, Any]:
    """Run labelled retrieval metrics in an isolated temporary vector store."""

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    settings = load_settings()
    embeddings = SentenceTransformerEmbeddingService(
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
    )

    store = ChromaVectorStore(None, "evaluation")
    KnowledgeBaseService(
        embeddings,
        store,
        settings.chunk_size,
        settings.chunk_overlap,
    ).index_files(documents, reset=True)
    retriever = SemanticRetriever(embeddings, store, top_k, min_score)
    ollama = (
        OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
            settings.ollama_temperature,
        )
        if with_generation
        else None
    )
    pipeline = (
        RAGPipeline(retriever, ollama, settings.max_context_chars) if ollama else None
    )

    answerable = [case for case in cases if case["should_answer"]]
    unanswerable = [case for case in cases if not case["should_answer"]]
    retrieval_hits = 0
    top_one_hits = 0
    refusal_hits = 0
    required_term_hits = 0
    generation_cases = 0
    latencies: list[float] = []
    details: list[dict[str, Any]] = []

    for case in cases:
        started = perf_counter()
        results = retriever.search(case["question"])
        latency_ms = (perf_counter() - started) * 1_000
        latencies.append(latency_ms)
        source_names = [result.chunk.document_name for result in results]

        if case["should_answer"]:
            hit = case["expected_document"] in source_names
            retrieval_hits += int(hit)
            top_one_hits += int(
                bool(source_names) and source_names[0] == case["expected_document"]
            )
        else:
            refusal_hits += int(not results)

        generated_answer = None
        if pipeline:
            answer = pipeline.ask(case["question"])
            generated_answer = answer.answer
            if case["should_answer"]:
                generation_cases += 1
                terms = case.get("required_terms", [])
                required_term_hits += int(
                    all(term.lower() in answer.answer.lower() for term in terms)
                )

        details.append(
            {
                "question": case["question"],
                "expected_document": case["expected_document"],
                "retrieved_documents": source_names,
                "top_score": round(results[0].score, 4) if results else None,
                "retrieval_ms": round(latency_ms, 2),
                "generated_answer": generated_answer,
            }
        )

    if ollama:
        ollama.close()

    metrics: dict[str, Any] = {
        "total_questions": len(cases),
        "answerable_questions": len(answerable),
        f"recall_at_{top_k}": round(retrieval_hits / len(answerable), 4),
        "top_1_source_accuracy": round(top_one_hits / len(answerable), 4),
        "unanswerable_refusal_accuracy": round(
            refusal_hits / len(unanswerable), 4
        ),
        "average_retrieval_ms": round(sum(latencies) / len(latencies), 2),
    }
    if generation_cases:
        metrics["required_term_accuracy"] = round(
            required_term_hits / generation_cases, 4
        )
    return {"settings": {"top_k": top_k, "min_score": min_score}, "metrics": metrics, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset.json"))
    parser.add_argument("--documents", type=Path, nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.86)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    documents = args.documents or sorted(Path("data/examples").glob("*.md"))
    report = evaluate(
        args.dataset,
        documents,
        args.top_k,
        args.min_score,
        args.with_generation,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
