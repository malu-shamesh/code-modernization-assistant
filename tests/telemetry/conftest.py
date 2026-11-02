from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from modernizer.models import LLMCallRecord, LLMResponse, TokenUsage
from modernizer.telemetry import Telemetry, setup_telemetry


class FakeLLM:
    """Returns a fixed usage per call; optionally raises."""

    def __init__(
        self,
        name: str = "fake-model",
        usage: TokenUsage | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.usage = usage or TokenUsage(100, 50)
        self.error = error
        self.calls = 0
        self.last_call: dict[str, object] = {}
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        with self._lock:
            self.calls += 1
            self.last_call = {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "tags": tags,
            }
        if self.error is not None:
            raise self.error
        return LLMResponse(text="ok", usage=self.usage, model=self.name)


class ListSink:
    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []
        self._lock = threading.Lock()

    def write_module(self, run_id: str, result: object) -> None:
        raise NotImplementedError

    def write_llm_call(self, record: LLMCallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def close(self) -> None:
        pass


def fake_cost(model: str, usage: TokenUsage) -> float:
    """Exactly representable in binary floating point, so sums are order independent."""
    return usage.input_tokens * 0.0001220703125 + usage.output_tokens * 0.000244140625


@pytest.fixture
def telemetry() -> Iterator[Telemetry]:
    tel = setup_telemetry("none")
    yield tel
    tel.shutdown()


@pytest.fixture
def span_exporter(telemetry: Telemetry) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    telemetry.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter
