"""Small HTTP client for the local Ollama chat API."""

from typing import Protocol

import httpx

from app.core.exceptions import LLMServiceError


class LanguageModelProvider(Protocol):
    """Generation contract required by the RAG pipeline."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one answer from system and user instructions."""


class OllamaClient:
    """Call one local Ollama model with explicit timeouts and error mapping."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.1,
        transport: httpx.BaseTransport | None = None,
    ):
        if not model.strip():
            raise LLMServiceError("Ollama model name cannot be empty")
        if timeout_seconds <= 0:
            raise LLMServiceError("Ollama timeout must be greater than zero")
        if not 0.0 <= temperature <= 2.0:
            raise LLMServiceError("Ollama temperature must be between 0 and 2")

        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    def health_check(self) -> bool:
        """Return whether Ollama responds and the configured model is installed."""

        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {str(model.get("name", "")) for model in models}
            return self.model in names
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a non-streaming response from Ollama's chat endpoint."""

        if not user_prompt.strip():
            raise LLMServiceError("User prompt cannot be empty")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            raw_content = response.json()["message"]["content"].strip()
            content = _strip_thinking(raw_content)
        except httpx.TimeoutException as error:
            raise LLMServiceError("Ollama request timed out") from error
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise LLMServiceError("Ollama generation request failed") from error

        if not content:
            raise LLMServiceError("Ollama returned an empty answer")
        return content

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""

        self._client.close()


def _strip_thinking(content: str) -> str:
    """Return only the final answer when a Qwen model emits think tags."""

    if "</think>" in content:
        final_answer = content.rsplit("</think>", maxsplit=1)[1].strip()
        if final_answer:
            return final_answer
    return content
