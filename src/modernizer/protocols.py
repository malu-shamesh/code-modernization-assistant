"""Interfaces between subpackages. Implementations live in their own subpackages so that each
one can be developed, swapped and mocked independently."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .models import LLMCallRecord, LLMResponse, ModuleResult


@runtime_checkable
class LLMClient(Protocol):
    """Synchronous text-in/text-out model client.

    ``tags`` carries bookkeeping such as ``{"stage": "convert", "module": "pkg/a.py"}``. Raw
    provider clients ignore it; the metering wrapper uses it for metric attributes and records.
    """

    name: str  # model id, e.g. "gemini-2.5-pro"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into L2-normalised float32 vectors of shape (n, dim)."""

    dim: int

    def embed(
        self, texts: Sequence[str], *, task: Literal["document", "query"] = "document"
    ) -> np.ndarray: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Nearest-neighbour index over document ids. Similarity: higher is better."""

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...


@runtime_checkable
class ResultSink(Protocol):
    """Where per-module results and per-call records are persisted (JSONL, BigQuery ...)."""

    def write_module(self, run_id: str, result: ModuleResult) -> None: ...

    def write_llm_call(self, record: LLMCallRecord) -> None: ...

    def close(self) -> None: ...
