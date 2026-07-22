"""End-to-end retrieval-augmented question-answering pipeline."""

from app.models import RAGAnswer, SearchResult, SourceReference
from app.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from app.rag.retriever import SemanticRetriever
from app.services.ollama_client import LanguageModelProvider


NO_ANSWER_MESSAGE = "无法从现有知识库中找到答案。"


class RAGPipeline:
    """Retrieve evidence, build a guarded prompt, and generate a cited answer."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        language_model: LanguageModelProvider,
        max_context_chars: int = 8_000,
    ):
        self.retriever = retriever
        self.language_model = language_model
        self.max_context_chars = max_context_chars

    def ask(self, question: str) -> RAGAnswer:
        """Answer one question or reject it when retrieval finds no evidence."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        results = self.retriever.search(question)
        if not results:
            return RAGAnswer(answer=NO_ANSWER_MESSAGE, answered=False, sources=[])

        user_prompt, included_results = build_user_prompt(
            question,
            results,
            self.max_context_chars,
        )
        answer = self.language_model.generate(SYSTEM_PROMPT, user_prompt)
        return RAGAnswer(
            answer=answer,
            answered=True,
            sources=[_to_source(result) for result in included_results],
        )


def _to_source(result: SearchResult) -> SourceReference:
    chunk = result.chunk
    return SourceReference(
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page_number=chunk.page_number,
        chunk_id=chunk.chunk_id,
        score=result.score,
        excerpt=chunk.text[:300],
    )
