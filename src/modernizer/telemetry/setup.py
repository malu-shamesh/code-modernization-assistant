"""Build tracer/meter providers without touching the OpenTelemetry globals."""

from __future__ import annotations

import sys
import threading

from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    InMemoryMetricReader,
    MetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.trace import Tracer

from .meter import LATENCY_BUCKETS_S, METRIC_LATENCY, LLMMeter

try:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:
    OTLPMetricExporter = None
    OTLPSpanExporter = None

try:
    from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
except ImportError:
    CloudMonitoringMetricsExporter = None

try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
except ImportError:
    CloudTraceSpanExporter = None

EXPORTERS = ("none", "console", "otlp", "gcp")


class Telemetry:
    """Providers plus a cumulative in-memory reader, so totals can always be read back."""

    def __init__(
        self,
        meter_provider: MeterProvider,
        tracer_provider: TracerProvider,
        metric_reader: InMemoryMetricReader,
    ) -> None:
        self.meter_provider = meter_provider
        self.tracer_provider = tracer_provider
        self.metric_reader = metric_reader
        self.llm_meter = LLMMeter(meter_provider)
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    def meter(self, name: str) -> Meter:
        return self.meter_provider.get_meter(name)

    def tracer(self, name: str) -> Tracer:
        return self.tracer_provider.get_tracer(name)

    def flush(self) -> None:
        self.meter_provider.force_flush()
        self.tracer_provider.force_flush()

    def shutdown(self) -> None:
        """Flush exporters and stop background threads. Safe to call more than once."""
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
        try:
            self.meter_provider.shutdown()
        finally:
            self.tracer_provider.shutdown()


def _require(obj: object, exporter: str, package: str) -> None:
    if obj is None:
        raise ImportError(
            f"exporter {exporter!r} needs {package}; install it with: pip install {package}"
        )


def _signal_url(base: str | None, signal: str) -> str | None:
    return f"{base.rstrip('/')}/v1/{signal}" if base else None


def _build_exporters(
    exporter: str, otlp_endpoint: str | None, project_id: str | None
) -> tuple[MetricExporter | None, SpanExporter | None]:
    if exporter == "console":
        # The exporters' default stream is bound to sys.stdout at import time, which would
        # ignore later redirection (pytest capture, notebooks, contextlib.redirect_stdout).
        return ConsoleMetricExporter(out=sys.stdout), ConsoleSpanExporter(out=sys.stdout)
    if exporter == "otlp":
        _require(OTLPMetricExporter, exporter, "opentelemetry-exporter-otlp-proto-http")
        return (
            OTLPMetricExporter(endpoint=_signal_url(otlp_endpoint, "metrics")),
            OTLPSpanExporter(endpoint=_signal_url(otlp_endpoint, "traces")),
        )
    if exporter == "gcp":
        _require(CloudMonitoringMetricsExporter, exporter, "opentelemetry-exporter-gcp-monitoring")
        _require(CloudTraceSpanExporter, exporter, "opentelemetry-exporter-gcp-trace")
        return (
            CloudMonitoringMetricsExporter(project_id=project_id),
            CloudTraceSpanExporter(project_id=project_id),
        )
    return None, None


def setup_telemetry(
    exporter: str = "none",
    *,
    service_name: str = "code-modernization-assistant",
    export_interval_ms: int = 15000,
    otlp_endpoint: str | None = None,
    project_id: str | None = None,
) -> Telemetry:
    """Create an isolated telemetry stack.

    ``exporter`` is one of "none", "console", "otlp" (``otlp_endpoint`` is the collector base
    URL, e.g. http://localhost:4318) or "gcp" (Cloud Monitoring and Cloud Trace). Providers
    are returned on the ``Telemetry`` object and never installed as OpenTelemetry globals.
    Call ``Telemetry.shutdown`` when done; providers do not register exit hooks.
    """
    exporter = exporter.lower()
    if exporter not in EXPORTERS:
        raise ValueError(f"unknown exporter {exporter!r}; expected one of {EXPORTERS}")

    # Exporters are built before any provider so a configuration error leaks no threads.
    metric_exporter, span_exporter = _build_exporters(exporter, otlp_endpoint, project_id)

    resource = Resource.create({"service.name": service_name})
    in_memory = InMemoryMetricReader()
    readers: list[MetricReader] = [in_memory]
    if metric_exporter is not None:
        readers.append(
            PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=export_interval_ms
            )
        )
    meter_provider = MeterProvider(
        metric_readers=readers,
        resource=resource,
        views=[
            View(
                instrument_name=METRIC_LATENCY,
                aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS_S),
            )
        ],
        shutdown_on_exit=False,
    )
    tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
    if span_exporter is not None:
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    return Telemetry(meter_provider, tracer_provider, in_memory)
