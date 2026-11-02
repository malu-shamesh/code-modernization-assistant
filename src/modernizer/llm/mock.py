"""Deterministic offline model client for tests and smoke runs."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..models import LLMResponse, TokenUsage
from .errors import ScriptExhaustedError


def estimate_tokens(text: str) -> int:
    """Rough token count used by the mock: one token per four characters, rounded up."""
    return math.ceil(len(text) / 4)


class ScriptedLLM:
    """``LLMClient`` that replays canned answers.

    ``responses`` is either a sequence consumed one entry per call (an exhausted script raises
    ``ScriptExhaustedError``) or a callable mapping the prompt to the answer. Safe to share
    across threads; ``calls`` records every request in arrival order.
    """

    def __init__(
        self, responses: Sequence[str] | Callable[[str], str], name: str = "mock-model"
    ) -> None:
        if isinstance(responses, str):
            raise TypeError("responses must be a sequence of strings or a callable, not a str")
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next = 0
        self._script: list[str] | None
        self._fn: Callable[[str], str] | None
        if callable(responses):
            self._script = None
            self._fn = responses
        else:
            self._script = list(responses)
            self._fn = None

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        text = ""
        with self._lock:
            self.calls.append({"prompt": prompt, "system": system, "tags": dict(tags or {})})
            if self._script is not None:
                if self._next >= len(self._script):
                    raise ScriptExhaustedError(
                        f"ScriptedLLM {self.name!r} received call #{self._next + 1} "
                        f"but only {len(self._script)} responses were scripted"
                    )
                text = self._script[self._next]
                self._next += 1
        if self._fn is not None:
            text = self._fn(prompt)
        usage = TokenUsage(
            input_tokens=estimate_tokens((system or "") + prompt),
            output_tokens=estimate_tokens(text),
        )
        return LLMResponse(
            text=text, usage=usage, model=self.name, latency_s=time.perf_counter() - started
        )
