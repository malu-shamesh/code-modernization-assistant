from __future__ import annotations

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

import modernizer.telemetry.setup as tel_setup
from modernizer.models import Tier, TokenUsage
from modernizer.telemetry import LLMMeter, setup_telemetry, token_totals


class _Sink:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


def test_none_exporter_builds_isolated_providers() -> None:
    tel = setup_telemetry("none", service_name="svc")
    try:
        assert isinstance(tel.meter_provider, MeterProvider)
        assert isinstance(tel.tracer_provider, TracerProvider)
        assert isinstance(tel.metric_reader, InMemoryMetricReader)
        assert isinstance(tel.llm_meter, LLMMeter)
        assert tel.tracer_provider.resource.attributes["service.name"] == "svc"
        assert tel.meter("x") is not None
        assert tel.tracer("x") is not None
        # Globals are never replaced.
        assert not isinstance(trace.get_tracer_provider(), TracerProvider)
        assert not isinstance(metrics.get_meter_provider(), MeterProvider)
        tel.flush()
    finally:
        tel.shutdown()


def test_many_instances_do_not_conflict() -> None:
    instances = [setup_telemetry("none") for _ in range(5)]
    for i, tel in enumerate(instances):
        tel.llm_meter.record_call("m", Tier.OPEN, "s", TokenUsage(i + 1, 0), 0.0, 0.0)
    for i, tel in enumerate(instances):
        assert token_totals(tel.metric_reader) == {("m", "open", "input"): i + 1}
    for tel in instances:
        tel.shutdown()


def test_shutdown_is_idempotent() -> None:
    tel = setup_telemetry("none")
    tel.shutdown()
    tel.shutdown()


def test_console_exporter_follows_current_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # capsys swaps sys.stdout after import, so a stream bound at import time would miss it.
    tel = setup_telemetry("console", export_interval_ms=60000)
    tel.llm_meter.record_call("m-console", Tier.OPEN, "convert", TokenUsage(3, 4), 0.5, 0.1)
    with tel.tracer("t").start_as_current_span("unit-span"):
        pass
    assert token_totals(tel.metric_reader)[("m-console", "open", "input")] == 3
    tel.shutdown()
    out = capsys.readouterr().out
    assert "modernizer.llm.tokens" in out
    assert "unit-span" in out


def test_shutdown_still_stops_tracer_when_meter_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tel = setup_telemetry("none")
    stopped: list[str] = []

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("exporter down")

    monkeypatch.setattr(tel.meter_provider, "shutdown", boom)
    monkeypatch.setattr(tel.tracer_provider, "shutdown", lambda: stopped.append("tracer"))
    with pytest.raises(RuntimeError, match="exporter down"):
        tel.shutdown()
    assert stopped == ["tracer"]


def test_unknown_exporter_rejected() -> None:
    with pytest.raises(ValueError, match="unknown exporter"):
        setup_telemetry("statsd")


def test_exporter_name_is_case_insensitive() -> None:
    setup_telemetry("NONE").shutdown()


def test_otlp_uses_signal_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str | None] = {}

    class StubMetricExporter(tel_setup.ConsoleMetricExporter):
        def __init__(self, endpoint: str | None = None) -> None:
            super().__init__(out=_Sink())
            seen["metrics"] = endpoint

    class StubSpanExporter(tel_setup.ConsoleSpanExporter):
        def __init__(self, endpoint: str | None = None) -> None:
            super().__init__(out=_Sink())
            seen["traces"] = endpoint

    monkeypatch.setattr(tel_setup, "OTLPMetricExporter", StubMetricExporter)
    monkeypatch.setattr(tel_setup, "OTLPSpanExporter", StubSpanExporter)
    setup_telemetry("otlp", otlp_endpoint="http://collector:4318/").shutdown()
    assert seen == {
        "metrics": "http://collector:4318/v1/metrics",
        "traces": "http://collector:4318/v1/traces",
    }


def test_otlp_missing_dependency_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tel_setup, "OTLPMetricExporter", None)
    with pytest.raises(ImportError, match="pip install opentelemetry-exporter-otlp-proto-http"):
        setup_telemetry("otlp")


def test_gcp_missing_dependency_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tel_setup, "CloudMonitoringMetricsExporter", None)
    with pytest.raises(ImportError, match="opentelemetry-exporter-gcp-monitoring"):
        setup_telemetry("gcp", project_id="p")

    class Dummy:
        def __init__(self, project_id: str | None = None) -> None:
            pass

    monkeypatch.setattr(tel_setup, "CloudMonitoringMetricsExporter", Dummy)
    monkeypatch.setattr(tel_setup, "CloudTraceSpanExporter", None)
    with pytest.raises(ImportError, match="opentelemetry-exporter-gcp-trace"):
        setup_telemetry("gcp", project_id="p")


def test_gcp_exporter_selection_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class StubMonitoring(tel_setup.ConsoleMetricExporter):
        def __init__(self, project_id: str | None = None) -> None:
            super().__init__()
            created["monitoring"] = project_id

    class StubTrace(tel_setup.ConsoleSpanExporter):
        def __init__(self, project_id: str | None = None) -> None:
            super().__init__(out=_Sink())
            created["trace"] = project_id

    monkeypatch.setattr(tel_setup, "CloudMonitoringMetricsExporter", StubMonitoring)
    monkeypatch.setattr(tel_setup, "CloudTraceSpanExporter", StubTrace)
    tel = setup_telemetry("gcp", project_id="my-project", export_interval_ms=60000)
    try:
        assert created == {"monitoring": "my-project", "trace": "my-project"}
        tel.llm_meter.record_call("m", Tier.OPEN, "s", TokenUsage(1, 1), 0.0, 0.0)
        assert token_totals(tel.metric_reader)[("m", "open", "input")] == 1
    finally:
        tel.shutdown()
