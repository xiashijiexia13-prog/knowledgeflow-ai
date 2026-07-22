"""Application-specific exception types."""


class KnowledgeFlowError(Exception):
    """Base exception for expected errors raised by the application."""


class ConfigurationError(KnowledgeFlowError):
    """Raised when application configuration contains an invalid value."""


class DocumentLoadError(KnowledgeFlowError):
    """Raised when a source document cannot be validated or parsed."""


class EmbeddingError(KnowledgeFlowError):
    """Raised when text cannot be converted into embedding vectors."""


class LLMServiceError(KnowledgeFlowError):
    """Raised when the configured language-model service cannot respond."""


class DuplicateDocumentError(KnowledgeFlowError):
    """Raised when uploaded content already exists in the document store."""


class DocumentNotFoundError(KnowledgeFlowError):
    """Raised when a requested document identifier does not exist."""
