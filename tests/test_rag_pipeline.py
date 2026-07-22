"""Tests for grounded generation, citations, and no-answer behavior."""

from app.models import SearchResult, TextChunk
from app.rag.pipeline import NO_ANSWER_MESSAGE, RAGPipeline


class FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, question: str) -> list[SearchResult]:
        return self.results


class FakeLanguageModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        assert "只能根据" in system_prompt
        assert "<knowledge_base>" in user_prompt
        return "向直属主管提交申请。[来源1]"


def make_result() -> SearchResult:
    chunk = TextChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_name="handbook.txt",
        file_path="handbook.txt",
        file_type="txt",
        page_number=2,
        chunk_index=0,
        start_char=0,
        end_char=12,
        text="年假需要直属主管审批。",
    )
    return SearchResult(chunk=chunk, score=0.91)


def test_answer_contains_structured_source() -> None:
    model = FakeLanguageModel()
    answer = RAGPipeline(FakeRetriever([make_result()]), model).ask("如何请年假？")

    assert answer.answered is True
    assert answer.sources[0].document_name == "handbook.txt"
    assert answer.sources[0].page_number == 2
    assert model.calls == 1


def test_no_retrieval_result_refuses_without_calling_model() -> None:
    model = FakeLanguageModel()
    answer = RAGPipeline(FakeRetriever([]), model).ask("不存在的问题")

    assert answer.answered is False
    assert answer.answer == NO_ANSWER_MESSAGE
    assert answer.sources == []
    assert model.calls == 0
