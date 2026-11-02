"""Request limits and the concurrency gate, configured from ``MODERNIZER_*`` variables.

Size limits guard memory and the per-request LLM bill. Only the ``Content-Length`` header is
checked before the body is read; a chunked upload without that header is only measured after
parsing. On Cloud Run the front end already caps request size, so this is a second line.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class LimitExceeded(Exception):
    """A request is larger than the configured limits (HTTP 413). The message is client-safe."""


def _positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


@dataclass(frozen=True)
class Limits:
    max_files: int = 25
    max_file_chars: int = 150_000
    max_total_chars: int = 400_000
    max_concurrent: int = 2  # in-flight /v1/modernize requests per instance
    retry_after_s: int = 5  # value of the Retry-After header on a 429

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Limits:
        source = os.environ if env is None else env
        d = cls()
        return cls(
            max_files=_positive_int(source, "MODERNIZER_MAX_FILES", d.max_files),
            max_file_chars=_positive_int(source, "MODERNIZER_MAX_FILE_CHARS", d.max_file_chars),
            max_total_chars=_positive_int(source, "MODERNIZER_MAX_TOTAL_CHARS", d.max_total_chars),
            max_concurrent=_positive_int(source, "MODERNIZER_MAX_CONCURRENT", d.max_concurrent),
            retry_after_s=_positive_int(source, "MODERNIZER_RETRY_AFTER_S", d.retry_after_s),
        )

    @property
    def max_body_bytes(self) -> int:
        """Largest accepted ``Content-Length``.

        JSON can spend up to six bytes per character (unicode escapes), plus per-file framing,
        so this is deliberately looser than the character limits: those stay the authority,
        this only stops absurdly large uploads before they are read.
        """
        return self.max_total_chars * 6 + self.max_files * 1024 + 64 * 1024

    def check_sources(self, sources: Sequence[tuple[str, str]]) -> None:
        """Raise ``LimitExceeded`` for a batch of ``(path, source)`` pairs that is too large."""
        if len(sources) > self.max_files:
            raise LimitExceeded(f"too many files: {len(sources)} > {self.max_files}")
        total = 0
        for path, source in sources:
            if len(source) > self.max_file_chars:
                raise LimitExceeded(
                    f"file {path!r} is too large: {len(source)} > {self.max_file_chars} characters"
                )
            total += len(source)
        if total > self.max_total_chars:
            raise LimitExceeded(
                f"request is too large: {total} > {self.max_total_chars} characters"
            )


class ConcurrencyGate:
    """Non-blocking admission control: a request either gets a slot at once or is refused."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self._slots = threading.BoundedSemaphore(limit)

    def try_acquire(self) -> bool:
        return self._slots.acquire(blocking=False)

    def release(self) -> None:
        self._slots.release()

    def acquire_slot(self) -> Slot | None:
        """Like ``try_acquire`` but returns a handle whose ``release`` may safely be repeated."""
        return Slot(self) if self.try_acquire() else None


class Slot:
    """One held place in a ``ConcurrencyGate``; releasing twice gives the place back only once.

    Both the worker thread and the request handler release the slot, so that it is returned
    whichever of them is left holding it (see ``service.app``).
    """

    def __init__(self, gate: ConcurrencyGate) -> None:
        self._gate = gate
        self._lock = threading.Lock()
        self._held = True

    def release(self) -> None:
        with self._lock:
            if not self._held:
                return
            self._held = False
        self._gate.release()
