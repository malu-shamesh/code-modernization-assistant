"""LLM metering: OpenTelemetry instruments and the client wrapper that feeds them."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.metrics import Meter, MeterProvider
from opentelemetry.trace import Tracer

from ..llm.pricing import cost_usd as _default_cost_fn
from ..models import LLMCallRecord, LLMResponse, Tier, TokenUsage
from ..protocols import LLMClient, ResultSink
from .ledger import UsageLedger, tier_value

logger = logging.getLogger(__name__)

METRIC_TOKENS = "modernizer.llm.tokens"
METRIC_COST = "modernizer.llm.cost"
METRIC_CALLS = "modernizer.llm.calls"
METRIC_LATENCY = "modernizer.llm.latency"
METRIC_MODULE_OUTCOMES = "modernizer.module.outcomes"

ATTR_MODEL = "gen_ai.request.model"
ATTR_TOKEN_TYPE = "gen_ai.token.type"
ATTR_TIER = "modernizer.tier"
ATTR_STAGE = "modernizer.stage"
ATTR_MODULE = "modernizer.module"
ATTR_OUTCOME = "modernizer.outcome"
ATTR_SUCCESS = "modernizer.success"
ATTR_ESCALATED = "modernizer.escalated"

LATENCY_BUCKETS_S = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)

UNKNOWN = "unknown"


def _flag(value: bool) -> str:
    return "true" if value else "false"


class LLMMeter:
    """Owns the modernizer instruments. Safe to share between threads."""

    def __init__(self, meter: Meter | MeterProvider) -> None:
        if isinstance(meter, MeterProvider):
            meter = meter.get_meter("modernizer.llm")
        self._tokens = meter.create_counter(
            METRIC_TOKENS, unit="{token}", description="LLM tokens consumed"
        )
        self._cost = meter.create_counter(METRIC_COST, unit="USD", description="LLM spend")
        self._calls = meter.create_counter(METRIC_CALLS, unit="{call}", description="LLM calls")
        self._latency = meter.create_histogram(
            METRIC_LATENCY, unit="s", description="LLM call latency"
        )
        self._module_outcomes = meter.create_counter(
            METRIC_MODULE_OUTCOMES, unit="{module}", description="Converted modules by outcome"
        )

    def record_call(
        self,
        model: str,
        tier: Tier | str,
        stage: str,
        usage: TokenUsage,
        cost_usd: float,
        latency_s: float,
        outcome: str = "ok",
    ) -> None:
        base = {ATTR_MODEL: model, ATTR_TIER: tier_value(tier), ATTR_STAGE: stage}
        if usage.input_tokens > 0:
            self._tokens.add(usage.input_tokens, {**base, ATTR_TOKEN_TYPE: "input"})
        if usage.output_tokens > 0:
            self._tokens.add(usage.output_tokens, {**base, ATTR_TOKEN_TYPE: "output"})
        if cost_usd > 0:
            self._cost.add(cost_usd, base)
        self._calls.add(1, {**base, ATTR_OUTCOME: outcome})
        self._latency.record(max(latency_s, 0.0), base)

    def record_module_outcome(self, tier: Tier | str, success: bool, escalated: bool) -> None:
        """``tier`` is the tier that produced the final result."""
        self._module_outcomes.add(
            1,
            {
                ATTR_TIER: tier_value(tier),
                ATTR_SUCCESS: _flag(success),
                ATTR_ESCALATED: _flag(escalated),
            },
        )


class MeteredLLM:
    """Wraps an ``LLMClient`` and records every call as a span, metrics and ledger entries."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        tier: Tier,
        meter: LLMMeter,
        ledger: UsageLedger,
        tracer: Tracer | None = None,
        run_id: str = "",
        sink: ResultSink | None = None,
        cost_fn: Callable[[str, TokenUsage], float] | None = None,
    ) -> None:
        cost_fn = cost_fn or _default_cost_fn
        self.name = inner.name
        self._inner = inner
        self._tier = tier
        self._meter = meter
        self._ledger = ledger
        self._tracer: Tracer = tracer or trace.NoOpTracer()
        self._run_id = run_id
        self._sink = sink
        self._cost_fn = cost_fn

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        stage = (tags or {}).get("stage", UNKNOWN)
        module = (tags or {}).get("module", UNKNOWN)
        with self._tracer.start_as_current_span("llm.generate") as span:
            span.set_attributes(
                {
                    "gen_ai.operation.name": "chat",
                    ATTR_MODEL: self.name,
                    "gen_ai.request.temperature": temperature,
                    "gen_ai.request.max_tokens": max_output_tokens,
                    ATTR_TIER: self._tier.value,
                    ATTR_STAGE: stage,
                    ATTR_MODULE: module,
                }
            )
            started = time.perf_counter()
            try:
                response = self._inner.generate(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    tags=tags,
                )
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                self._meter.record_call(
                    self.name,
                    self._tier,
                    stage,
                    TokenUsage(),
                    0.0,
                    time.perf_counter() - started,
                    outcome="error",
                )
                raise
            latency = time.perf_counter() - started

            usage = response.usage
            cost = self._cost(usage)
            span.set_attributes(
                {
                    "gen_ai.usage.input_tokens": usage.input_tokens,
                    "gen_ai.usage.output_tokens": usage.output_tokens,
                    "gen_ai.response.model": response.model or self.name,
                    "modernizer.cost_usd": cost,
                }
            )
            self._meter.record_call(self.name, self._tier, stage, usage, cost, latency)
            self._ledger.record(module, self.name, self._tier, stage, usage, cost)
            self._write_record(module, stage, usage, cost, latency)
            return response

    def _cost(self, usage: TokenUsage) -> float:
        # A pricing gap must not discard a response that was already paid for.
        try:
            return float(self._cost_fn(self.name, usage))
        except Exception:
            logger.warning("cost lookup failed for model %r; recording 0", self.name, exc_info=True)
            return 0.0

    def _write_record(
        self, module: str, stage: str, usage: TokenUsage, cost: float, latency: float
    ) -> None:
        if self._sink is None:
            return
        record = LLMCallRecord(
            run_id=self._run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=module,
            stage=stage,
            model=self.name,
            tier=self._tier.value,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            latency_s=latency,
        )
        try:
            self._sink.write_llm_call(record)
        except Exception:
            logger.warning("sink rejected llm call record", exc_info=True)
