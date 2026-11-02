"""Retry with exponential backoff for transient provider failures.

Only failures that are likely to disappear on their own are retried: HTTP 408/429/5xx,
connection errors and timeouts, and the matching ``google.api_core`` / ``google.genai``
errors. Authentication and request-shape errors (400/401/403 ...) fail immediately.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from .errors import LLMError, LLMRequestError, LLMRetryExhaustedError

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    from google.api_core import exceptions as gax
except ImportError:
    gax = None  # type: ignore[assignment]

try:
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None  # type: ignore[assignment]

try:
    from google.auth import exceptions as auth_exceptions
except ImportError:
    auth_exceptions = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_STATUS_CODES = frozenset({408, 429})


def _is_transient_status(code: int | None) -> bool:
    return code is not None and (code in TRANSIENT_STATUS_CODES or 500 <= code <= 599)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff: ``base_delay_s * 2**(attempt-1)`` capped at ``max_delay_s``.

    With ``jitter`` the wait is drawn uniformly from the upper half of that interval. ``sleep``
    is injectable so tests do not wait.
    """

    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("delays must be non-negative")

    def delay_for(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Seconds to wait after failed attempt number ``attempt`` (1-based)."""
        exponent = min(max(attempt - 1, 0), 62)  # keeps huge attempt counts finite
        delay = min(self.max_delay_s, self.base_delay_s * 2**exponent)
        if self.jitter:
            delay = random.uniform(delay / 2, delay)
        if retry_after_s is not None:
            delay = min(self.max_delay_s, max(delay, retry_after_s))
        return delay


def status_code_of(exc: BaseException) -> int | None:
    """HTTP status carried by a provider exception, if any."""
    if isinstance(exc, LLMRequestError):
        return exc.status_code
    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        return exc.code if isinstance(exc.code, int) else None
    if gax is not None and isinstance(exc, gax.GoogleAPICallError):
        return int(exc.code) if exc.code is not None else None
    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def is_transient(exc: BaseException) -> bool:
    """True when retrying ``exc`` could plausibly succeed."""
    if isinstance(exc, LLMError) and not isinstance(exc, LLMRequestError):
        return False
    if _is_transient_status(status_code_of(exc)):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if httpx is not None and isinstance(
        exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    ):
        return True
    if gax is not None and isinstance(
        exc, (gax.ServiceUnavailable, gax.ResourceExhausted, gax.DeadlineExceeded)
    ):
        return True
    if auth_exceptions is not None and isinstance(exc, auth_exceptions.GoogleAuthError):
        return isinstance(exc, auth_exceptions.TransportError) or bool(exc.retryable)
    return False


def _is_provider_error(exc: BaseException) -> bool:
    """Errors that originate from the provider or the network, as opposed to local bugs."""
    candidates = (
        (httpx, "HTTPError"),
        (gax, "GoogleAPICallError"),
        (genai_errors, "APIError"),
        (auth_exceptions, "GoogleAuthError"),
    )
    for module, name in candidates:
        if module is not None and isinstance(exc, getattr(module, name)):
            return True
    return isinstance(exc, (ConnectionError, TimeoutError))


def call_with_retry(
    fn: Callable[[], T], policy: RetryPolicy | None = None, *, describe: str = "LLM call"
) -> T:
    """Run ``fn`` until it succeeds, fails permanently, or ``policy.max_attempts`` is spent.

    Non-transient provider errors are raised as ``LLMRequestError`` (original chained), local
    programming errors propagate untouched, and exhausted retries raise
    ``LLMRetryExhaustedError``.
    """
    policy = policy or RetryPolicy()
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc):
                if isinstance(exc, LLMError) or not _is_provider_error(exc):
                    raise
                raise LLMRequestError(
                    f"{describe} failed: {exc}", status_code=status_code_of(exc)
                ) from exc
            if attempt == policy.max_attempts:
                raise LLMRetryExhaustedError(
                    f"{describe} failed after {attempt} attempts: {exc}",
                    attempts=attempt,
                    status_code=status_code_of(exc),
                ) from exc
            retry_after = exc.retry_after_s if isinstance(exc, LLMRequestError) else None
            delay = policy.delay_for(attempt, retry_after)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                describe,
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            policy.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
