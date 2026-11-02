from __future__ import annotations

import threading

import pytest
from opentelemetry.sdk.metrics import Counter, MeterProvider
from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader

from modernizer.models import Tier, TokenUsage
from modernizer.telemetry import (
    LLMMeter,
    Telemetry,
    UsageLedger,
    call_counts,
    cost_total,
    module_outcome_counts,
    token_totals,
)


def test_ledger_aggregates_by_module_model_and_tier() -> None:
    ledger = UsageLedger()
    ledger.record("a.py", "cheap", Tier.OPEN, "convert", TokenUsage(10, 5), 0.5)
    ledger.record("a.py", "cheap", Tier.OPEN, "repair", TokenUsage(2, 1), 0.25)
    ledger.record("a.py", "big", Tier.FRONTIER, "convert", TokenUsage(100, 50), 2.0)
    ledger.record("b.py", "big", Tier.FRONTIER, "convert", TokenUsage(1, 1), 0.5)

    assert ledger.usage_by_model("a.py") == {"cheap": TokenUsage(12, 6), "big": TokenUsage(100, 50)}
    assert ledger.cost("a.py") == 2.75
    assert ledger.cost("b.py") == 0.5
    per_model, total = ledger.totals()
    assert per_model == {"cheap": TokenUsage(12, 6), "big": TokenUsage(101, 51)}
    assert total == 3.25
    assert ledger.by_tier() == {"open": TokenUsage(12, 6), "frontier": TokenUsage(101, 51)}
    assert ledger.cost_by_tier() == {"open": 0.75, "frontier": 2.5}


def test_ledger_unknown_module_is_empty_and_not_created() -> None:
    ledger = UsageLedger()
    assert ledger.usage_by_model("missing") == {}
    assert ledger.cost("missing") == 0.0
    assert ledger.modules() == []
    assert ledger.totals() == ({}, 0.0)


def test_ledger_returns_copies() -> None:
    ledger = UsageLedger()
    ledger.record("a.py", "m", Tier.OPEN, "convert", TokenUsage(1, 1), 0.0)
    ledger.usage_by_model("a.py")["m"].input_tokens = 999
    ledger.by_tier()["open"].input_tokens = 999
    assert ledger.usage_by_model("a.py")["m"] == TokenUsage(1, 1)
    assert ledger.by_tier()["open"] == TokenUsage(1, 1)


def test_ledger_thread_safety() -> None:
    ledger = UsageLedger()
    n_threads, per_thread = 12, 200

    def work() -> None:
        for _ in range(per_thread):
            ledger.record("a.py", "m", Tier.OPEN, "convert", TokenUsage(3, 2), 0.25)

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    n = n_threads * per_thread
    assert ledger.usage_by_model("a.py") == {"m": TokenUsage(3 * n, 2 * n)}
    assert ledger.cost("a.py") == 0.25 * n


def test_query_on_empty_reader(telemetry: Telemetry) -> None:
    reader = telemetry.metric_reader
    assert token_totals(reader) == {}
    assert cost_total(reader) == 0.0
    assert module_outcome_counts(reader) == {}
    assert call_counts(reader) == {}


def test_record_call_and_cumulative_reads_across_collections(telemetry: Telemetry) -> None:
    meter = telemetry.llm_meter
    reader = telemetry.metric_reader
    meter.record_call("m", Tier.OPEN, "convert", TokenUsage(10, 4), 0.5, 1.2)
    assert token_totals(reader) == {("m", "open", "input"): 10, ("m", "open", "output"): 4}

    meter.record_call("m", "open", "repair", TokenUsage(5, 1), 0.25, 0.3)
    meter.record_call("big", Tier.FRONTIER, "convert", TokenUsage(7, 7), 1.0, 2.0)
    assert token_totals(reader) == {
        ("m", "open", "input"): 15,
        ("m", "open", "output"): 5,
        ("big", "frontier", "input"): 7,
        ("big", "frontier", "output"): 7,
    }
    assert cost_total(reader) == 1.75
    assert cost_total(reader) == 1.75


def test_zero_usage_adds_no_token_series(telemetry: Telemetry) -> None:
    telemetry.llm_meter.record_call("m", Tier.OPEN, "convert", TokenUsage(0, 3), 0.0, 0.1)
    assert token_totals(telemetry.metric_reader) == {("m", "open", "output"): 3}
    assert cost_total(telemetry.metric_reader) == 0.0


def test_module_outcomes(telemetry: Telemetry) -> None:
    meter = telemetry.llm_meter
    meter.record_module_outcome(Tier.OPEN, True, False)
    meter.record_module_outcome(Tier.OPEN, True, False)
    meter.record_module_outcome(Tier.FRONTIER, True, True)
    meter.record_module_outcome("frontier", False, True)
    assert module_outcome_counts(telemetry.metric_reader) == {
        ("open", "true", "false"): 2,
        ("frontier", "true", "true"): 1,
        ("frontier", "false", "true"): 1,
    }


def test_instrument_names_units_and_attributes(telemetry: Telemetry) -> None:
    telemetry.llm_meter.record_call("m", Tier.OPEN, "convert", TokenUsage(1, 1), 0.5, 0.1)
    telemetry.llm_meter.record_module_outcome(Tier.OPEN, True, False)
    data = telemetry.metric_reader.get_metrics_data()
    assert data is not None
    metrics = {
        m.name: m for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
    }
    assert set(metrics) == {
        "modernizer.llm.tokens",
        "modernizer.llm.cost",
        "modernizer.llm.calls",
        "modernizer.llm.latency",
        "modernizer.module.outcomes",
    }
    assert metrics["modernizer.llm.tokens"].unit == "{token}"
    assert metrics["modernizer.llm.cost"].unit == "USD"
    assert metrics["modernizer.llm.latency"].unit == "s"
    token_attrs = {
        frozenset(p.attributes.items()) for p in metrics["modernizer.llm.tokens"].data.data_points
    }
    assert (
        frozenset(
            {
                "gen_ai.request.model": "m",
                "gen_ai.token.type": "input",
                "modernizer.tier": "open",
                "modernizer.stage": "convert",
            }.items()
        )
        in token_attrs
    )
    (latency_point,) = metrics["modernizer.llm.latency"].data.data_points
    assert latency_point.count == 1
    assert latency_point.explicit_bounds[0] == 0.25


def test_llm_meter_accepts_meter_or_provider() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    LLMMeter(provider).record_call("a", Tier.OPEN, "s", TokenUsage(1, 0), 0.0, 0.0)
    LLMMeter(provider.get_meter("other")).record_call(
        "a", Tier.OPEN, "s", TokenUsage(2, 0), 0.0, 0.0
    )
    assert token_totals(reader) == {("a", "open", "input"): 3}
    provider.shutdown()


def test_delta_reader_is_rejected() -> None:
    reader = InMemoryMetricReader(preferred_temporality={Counter: AggregationTemporality.DELTA})
    provider = MeterProvider(metric_readers=[reader])
    LLMMeter(provider).record_call("a", Tier.OPEN, "s", TokenUsage(1, 0), 0.0, 0.0)
    with pytest.raises(ValueError, match="cumulative"):
        token_totals(reader)
    provider.shutdown()
