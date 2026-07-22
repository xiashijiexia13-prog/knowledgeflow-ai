"""Application-specific exception types."""


class KnowledgeFlowError(Exception):
    """Base exception for expected errors raised by the application."""


class ConfigurationError(KnowledgeFlowError):
    """Raised when application configuration contains an invalid value."""
