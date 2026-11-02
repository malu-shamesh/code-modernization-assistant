"""Typed errors raised by the model-client layer."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every error raised by an ``LLMClient`` implementation."""


class LLMRequestError(LLMError):
    """The provider rejected or failed a request.

    ``status_code`` is the HTTP status when one is known. ``retry_after_s`` carries a
    server-provided back-off hint (``Retry-After``) when present.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class LLMRetryExhaustedError(LLMRequestError):
    """A transient failure persisted through every allowed attempt."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.attempts = attempts


class LLMBlockedError(LLMError):
    """The prompt or the response was blocked by the provider (safety, recitation ...)."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class LLMEmptyResponseError(LLMError):
    """The provider answered successfully but produced no usable text."""


class ScriptExhaustedError(LLMError):
    """A ``ScriptedLLM`` built from a list of responses was called more times than scripted."""
