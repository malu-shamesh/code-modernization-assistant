from __future__ import annotations

import threading
from datetime import datetime

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from modernizer.models import Tier, TokenUsage
from modernizer.protocols import LLMClient
from modernizer.telemetry import (
    MeteredLLM,
    Telemetry,
    UsageLedger,
    call_counts,
    cost_total,
    token_totals,
)

from .conftest import FakeLLM, ListSink, fake_cost


def make(
    telemetry: Telemetry,
    inner: FakeLLM,
    tier: Tier = Tier.OPEN,
    ledger: UsageLedger | None = None,
    **kwargs: object,
) -> tuple[MeteredLLM, UsageLedger]:
    ledger = ledger or UsageLedger()
    llm = MeteredLLM(
        inner,
        tier=tier,
        meter=telemetry.llm_meter,
        ledger=ledger,
        tracer=telemetry.tracer("test"),
        cost_fn=fake_cost,
        **kwargs,  # type: ignore[arg-type]
    )
    return llm, ledger


def test_implements_protocol_and_passes_through(telemetry: Telemetry) -> None:
    llm, _ = make(telemetry, FakeLLM("m1"))
    assert isinstance(llm, LLMClient)
    assert llm.name == "m1"
    response = llm.generate("hi", tags={"stage": "convert", "module": "a.py"})
    assert response.text == "ok"
    assert response.usage == TokenUsage(100, 50)


def test_arguments_are_forwarded_unchanged(telemetry: Telemetry) -> None:
    inner = FakeLLM("m1")
    llm, _ = make(telemetry, inner)
    tags = {"stage": "convert", "module": "a.py"}
    llm.generate("the prompt", system="sys", temperature=0.7, max_output_tokens=123, tags=tags)
    assert inner.last_call == {
        "prompt": "the prompt",
        "system": "sys",
        "temperature": 0.7,
        "max_output_tokens": 123,
        "tags": tags,
    }


def test_latency_recorded_for_ok_and_error_calls(telemetry: Telemetry) -> None:
    ok_llm, _ = make(telemetry, FakeLLM("m1"))
    bad_llm, _ = make(telemetry, FakeLLM("m2", error=ValueError("x")))
    ok_llm.generate("p", tags={"stage": "convert"})
    with pytest.raises(ValueError):
        bad_llm.generate("p", tags={"stage": "convert"})
    data = telemetry.metric_reader.get_metrics_data()
    assert data is not None
    (latency,) = [
        m
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "modernizer.llm.latency"
    ]
    counts = {p.attributes["gen_ai.request.model"]: p.count for p in latency.data.data_points}
    assert counts == {"m1": 1, "m2": 1}


def test_metrics_match_ledger(telemetry: Telemetry) -> None:
    inner = FakeLLM("m1", usage=TokenUsage(120, 30))
    llm, ledger = make(telemetry, inner, tier=Tier.FRONTIER)
    for i in range(5):
        llm.generate("p", tags={"stage": "convert", "module": f"m{i % 2}.py"})
    llm.generate("p", tags={"stage": "repair", "module": "m0.py"})

    reader = telemetry.metric_reader
    tokens = token_totals(reader)
    assert tokens == {("m1", "frontier", "input"): 720, ("m1", "frontier", "output"): 180}

    per_model, total_cost = ledger.totals()
    assert per_model["m1"] == TokenUsage(720, 180)
    assert tokens[("m1", "frontier", "input")] == per_model["m1"].input_tokens
    assert tokens[("m1", "frontier", "output")] == per_model["m1"].output_tokens
    assert cost_total(reader) == pytest.approx(total_cost)
    assert total_cost == pytest.approx(6 * fake_cost("m1", TokenUsage(120, 30)))
    assert call_counts(reader) == {("m1", "frontier", "ok"): 6}
    assert ledger.usage_by_model("m0.py") == {"m1": TokenUsage(480, 120)}
    assert ledger.cost("m1.py") == pytest.approx(2 * fake_cost("m1", TokenUsage(120, 30)))


def test_repeated_collection_does_not_double_count(telemetry: Telemetry) -> None:
    llm, _ = make(telemetry, FakeLLM("m1"))
    llm.generate("p")
    first = token_totals(telemetry.metric_reader)
    assert token_totals(telemetry.metric_reader) == first
    llm.generate("p")
    second = token_totals(telemetry.metric_reader)
    assert second[("m1", "open", "input")] == 2 * first[("m1", "open", "input")]


def test_missing_tags_default_to_unknown(telemetry: Telemetry) -> None:
    llm, ledger = make(telemetry, FakeLLM())
    llm.generate("p")
    assert ledger.modules() == ["unknown"]


def test_concurrent_calls_produce_exact_sums(telemetry: Telemetry) -> None:
    n_threads, per_thread = 16, 50
    sink = ListSink()
    inner = FakeLLM("m1", usage=TokenUsage(7, 3))
    llm, ledger = make(telemetry, inner, sink=sink, run_id="r1")
    barrier = threading.Barrier(n_threads)

    def work(idx: int) -> None:
        barrier.wait()
        for _ in range(per_thread):
            llm.generate("p", tags={"stage": "convert", "module": f"mod{idx % 4}.py"})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    calls = n_threads * per_thread
    assert inner.calls == calls
    tokens = token_totals(telemetry.metric_reader)
    assert tokens == {("m1", "open", "input"): 7 * calls, ("m1", "open", "output"): 3 * calls}
    per_model, total_cost = ledger.totals()
    assert per_model == {"m1": TokenUsage(7 * calls, 3 * calls)}
    expected_cost = calls * fake_cost("m1", TokenUsage(7, 3))
    assert total_cost == expected_cost
    assert cost_total(telemetry.metric_reader) == expected_cost
    assert call_counts(telemetry.metric_reader) == {("m1", "open", "ok"): calls}
    assert len(sink.records) == calls
    assert sum(ledger.usage_by_model(m)["m1"].total for m in ledger.modules()) == 10 * calls


def test_error_path_records_error_call_and_reraises(
    telemetry: Telemetry, span_exporter: InMemorySpanExporter
) -> None:
    sink = ListSink()
    llm, ledger = make(telemetry, FakeLLM("m1", error=TimeoutError("boom")), sink=sink)
    with pytest.raises(TimeoutError, match="boom"):
        llm.generate("p", tags={"stage": "convert", "module": "a.py"})

    reader = telemetry.metric_reader
    assert call_counts(reader) == {("m1", "open", "error"): 1}
    assert token_totals(reader) == {}
    assert cost_total(reader) == 0.0
    assert ledger.totals() == ({}, 0.0)
    assert sink.records == []

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["error.type"] == "TimeoutError"
    assert not span.status.is_ok


def test_span_attributes(telemetry: Telemetry, span_exporter: InMemorySpanExporter) -> None:
    llm, _ = make(telemetry, FakeLLM("m1", usage=TokenUsage(11, 22)), tier=Tier.FRONTIER)
    llm.generate(
        "p", temperature=0.2, max_output_tokens=99, tags={"stage": "repair", "module": "x.py"}
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "llm.generate"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.request.model"] == "m1"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 22
    assert attrs["modernizer.tier"] == "frontier"
    assert attrs["modernizer.stage"] == "repair"
    assert attrs["modernizer.module"] == "x.py"
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["gen_ai.request.max_tokens"] == 99
    assert attrs["modernizer.cost_usd"] == fake_cost("m1", TokenUsage(11, 22))


def test_sink_receives_call_record(telemetry: Telemetry) -> None:
    sink = ListSink()
    llm, _ = make(telemetry, FakeLLM("m1", usage=TokenUsage(10, 5)), sink=sink, run_id="run-7")
    llm.generate("p", tags={"stage": "test_gen", "module": "pkg/a.py"})

    (record,) = sink.records
    assert record.run_id == "run-7"
    assert record.module == "pkg/a.py"
    assert record.stage == "test_gen"
    assert record.model == "m1"
    assert record.tier == "open"
    assert (record.input_tokens, record.output_tokens) == (10, 5)
    assert record.cost_usd == fake_cost("m1", TokenUsage(10, 5))
    assert record.latency_s >= 0.0
    parsed = datetime.fromisoformat(record.timestamp)
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def test_sink_failure_does_not_lose_response(telemetry: Telemetry) -> None:
    class BrokenSink(ListSink):
        def write_llm_call(self, record: object) -> None:
            raise OSError("disk full")

    llm, ledger = make(telemetry, FakeLLM("m1"), sink=BrokenSink())
    assert llm.generate("p").text == "ok"
    assert ledger.totals()[0]["m1"] == TokenUsage(100, 50)


def test_cost_fn_failure_records_zero_cost(telemetry: Telemetry) -> None:
    def broken(model: str, usage: TokenUsage) -> float:
        raise KeyError(model)

    llm = MeteredLLM(
        FakeLLM("m1"),
        tier=Tier.OPEN,
        meter=telemetry.llm_meter,
        ledger=UsageLedger(),
        cost_fn=broken,
    )
    llm.generate("p")
    assert cost_total(telemetry.metric_reader) == 0.0
    assert token_totals(telemetry.metric_reader)[("m1", "open", "input")] == 100


def test_works_without_tracer(telemetry: Telemetry) -> None:
    llm = MeteredLLM(
        FakeLLM(),
        tier=Tier.OPEN,
        meter=telemetry.llm_meter,
        ledger=UsageLedger(),
        cost_fn=fake_cost,
    )
    assert llm.generate("p").text == "ok"
