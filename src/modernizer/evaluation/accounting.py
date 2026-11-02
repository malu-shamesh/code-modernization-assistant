"""Token and cost accounting for one policy run, with independent cross-checks.

The OpenTelemetry in-memory reader is the source of truth for every total in a report. Because
a reported saving is only as trustworthy as the meter behind it, the totals are verified against
three other records of the same calls before they are used:

* the pipeline's ``UsageLedger`` (per tier and per model),
* the per-module results returned by the pipeline,
* a call-level record of every ``LLMCallRecord`` the metered clients emit (``CallCollector``),
  which also carries the stage that the ledger drops.

Any disagreement raises ``EvaluationConsistencyError``.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader, Sum

from ..models import LLMCallRecord, ModuleResult
from ..telemetry import UsageLedger, cost_total, token_totals
from ..telemetry.meter import (
    ATTR_MODEL,
    ATTR_STAGE,
    ATTR_TIER,
    ATTR_TOKEN_TYPE,
    METRIC_COST,
    METRIC_TOKENS,
)
from .results import TokenCounts

TESTGEN_STAGE = "test_gen"
_COST_REL_TOL = 1e-9
_COST_ABS_TOL = 1e-12


class EvaluationConsistencyError(AssertionError):
    """Two independent records of the same LLM calls disagree; the numbers cannot be trusted."""


class CallCollector:
    """Thread-safe ``ResultSink`` that keeps every LLM call record in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[LLMCallRecord] = []

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        return None

    def write_llm_call(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)

    def close(self) -> None:
        return None

    @property
    def records(self) -> list[LLMCallRecord]:
        with self._lock:
            return list(self._records)


@dataclass(frozen=True)
class StageMeter:
    """Meter totals split by stage. Keys: (model, tier, stage, direction) and (model, tier, stage)."""

    tokens: dict[tuple[str, str, str, str], int]
    cost: dict[tuple[str, str, str], float]


def read_stage_meter(reader: InMemoryMetricReader) -> StageMeter:
    """Stage-aware view of the token and cost counters.

    ``telemetry.query`` aggregates over stage; this reads the same cumulative data points and
    keeps the ``modernizer.stage`` attribute.
    """
    tokens: dict[tuple[str, str, str, str], int] = defaultdict(int)
    cost: dict[tuple[str, str, str], float] = defaultdict(float)
    data = reader.get_metrics_data()
    if data is not None:
        for resource_metrics in data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name not in (METRIC_TOKENS, METRIC_COST):
                        continue
                    if not isinstance(metric.data, Sum):
                        continue
                    if metric.data.aggregation_temporality != AggregationTemporality.CUMULATIVE:
                        raise ValueError(f"metric {metric.name!r} is not cumulative")
                    for point in metric.data.data_points:
                        attrs = dict(point.attributes or {})
                        base = (
                            str(attrs.get(ATTR_MODEL, "")),
                            str(attrs.get(ATTR_TIER, "")),
                            str(attrs.get(ATTR_STAGE, "")),
                        )
                        if metric.name == METRIC_TOKENS:
                            direction = str(attrs.get(ATTR_TOKEN_TYPE, ""))
                            tokens[(*base, direction)] += int(point.value)
                        else:
                            cost[base] += float(point.value)
    return StageMeter(dict(tokens), dict(cost))


@dataclass(frozen=True)
class RunAccounting:
    """Verified totals of one policy run, all taken from the OpenTelemetry meter."""

    tokens: TokenCounts
    tokens_by_tier: dict[str, TokenCounts]
    tokens_by_stage: dict[str, TokenCounts]
    tokens_excluding_testgen: TokenCounts
    cost_usd_total: float
    cost_usd_by_tier: dict[str, float]
    cost_usd_by_stage: dict[str, float]
    cost_usd_excluding_testgen: float


def _fail(message: str) -> EvaluationConsistencyError:
    return EvaluationConsistencyError(message)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_COST_REL_TOL, abs_tol=_COST_ABS_TOL)


def _split(
    stage_meter: StageMeter, key_index: int
) -> tuple[dict[str, TokenCounts], dict[str, float]]:
    """Group the stage meter by tier (index 1) or stage (index 2)."""
    tok: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key, value in stage_meter.tokens.items():
        slot = tok[key[key_index]]
        if key[3] == "input":
            slot[0] += value
        elif key[3] == "output":
            slot[1] += value
    cost: dict[str, float] = defaultdict(float)
    for key, value in stage_meter.cost.items():
        cost[key[key_index]] += value
    return {k: TokenCounts(v[0], v[1]) for k, v in tok.items()}, dict(cost)


def _sum_counts(counts: Iterable[TokenCounts]) -> TokenCounts:
    items = list(counts)
    return TokenCounts(sum(c.input_tokens for c in items), sum(c.output_tokens for c in items))


def collect_and_verify(
    reader: InMemoryMetricReader,
    ledger: UsageLedger,
    collector: CallCollector,
    results: Sequence[ModuleResult],
) -> RunAccounting:
    """Read the meter and check it against the ledger, the call records and the module results."""
    # Totals through the public query helpers, stage split through our own reader.
    meter_models = token_totals(reader)
    meter_cost = cost_total(reader)
    stage_meter = read_stage_meter(reader)
    tokens_by_tier, cost_by_tier = _split(stage_meter, 1)
    tokens_by_stage, cost_by_stage = _split(stage_meter, 2)

    overall_in = sum(v for (_, _, d), v in meter_models.items() if d == "input")
    overall_out = sum(v for (_, _, d), v in meter_models.items() if d == "output")
    overall = TokenCounts(overall_in, overall_out)

    if _sum_counts(tokens_by_stage.values()) != overall:
        raise _fail("per-stage token split does not add up to the metered total")
    if not _close(sum(cost_by_stage.values()), meter_cost):
        raise _fail("per-stage cost split does not add up to the metered total")

    # 1. ledger, per tier and per model
    ledger_tiers = ledger.by_tier()
    for tier in sorted(set(ledger_tiers) | set(tokens_by_tier)):
        metered = tokens_by_tier.get(tier, TokenCounts())
        booked = ledger_tiers.get(tier)
        expected = (
            TokenCounts(booked.input_tokens, booked.output_tokens) if booked else TokenCounts()
        )
        if metered != expected:
            raise _fail(f"tier {tier!r}: meter {metered} != ledger {expected}")
    ledger_cost_tiers = ledger.cost_by_tier()
    for tier in sorted(set(ledger_cost_tiers) | set(cost_by_tier)):
        if not _close(cost_by_tier.get(tier, 0.0), ledger_cost_tiers.get(tier, 0.0)):
            raise _fail(f"tier {tier!r}: metered cost differs from ledger cost")
    ledger_models, ledger_cost = ledger.totals()
    for model in sorted({m for m, _, _ in meter_models} | set(ledger_models)):
        usage = ledger_models.get(model)
        for direction, booked_value in (
            ("input", usage.input_tokens if usage else 0),
            ("output", usage.output_tokens if usage else 0),
        ):
            metered_value = sum(
                v for (m, _, d), v in meter_models.items() if m == model and d == direction
            )
            if metered_value != booked_value:
                raise _fail(f"model {model!r} {direction} tokens: meter != ledger")
    if not _close(meter_cost, ledger_cost):
        raise _fail(f"total cost: meter {meter_cost!r} != ledger {ledger_cost!r}")

    # 2. the per-module results (what the report rows are built from)
    module_tokens = _sum_counts(
        TokenCounts(r.total_usage.input_tokens, r.total_usage.output_tokens) for r in results
    )
    if module_tokens != overall:
        raise _fail("sum of per-module token usage differs from the metered total")

    # 3. call records, which also carry the stage
    records = collector.records
    recorded_stage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    recorded_cost: dict[str, float] = defaultdict(float)
    for rec in records:
        recorded_stage[rec.stage][0] += rec.input_tokens
        recorded_stage[rec.stage][1] += rec.output_tokens
        recorded_cost[rec.stage] += rec.cost_usd
    for stage in sorted(set(recorded_stage) | set(tokens_by_stage)):
        recorded = TokenCounts(*recorded_stage.get(stage, [0, 0]))
        if recorded != tokens_by_stage.get(stage, TokenCounts()):
            raise _fail(f"stage {stage!r}: call records disagree with the meter")
        if not _close(recorded_cost.get(stage, 0.0), cost_by_stage.get(stage, 0.0)):
            raise _fail(f"stage {stage!r}: call-record cost disagrees with the meter")

    testgen_tokens = tokens_by_stage.get(TESTGEN_STAGE, TokenCounts())
    return RunAccounting(
        tokens=overall,
        tokens_by_tier=tokens_by_tier,
        tokens_by_stage=tokens_by_stage,
        tokens_excluding_testgen=TokenCounts(
            overall.input_tokens - testgen_tokens.input_tokens,
            overall.output_tokens - testgen_tokens.output_tokens,
        ),
        cost_usd_total=meter_cost,
        cost_usd_by_tier=cost_by_tier,
        cost_usd_by_stage=cost_by_stage,
        cost_usd_excluding_testgen=math.fsum(
            v for stage, v in cost_by_stage.items() if stage != TESTGEN_STAGE
        ),
    )


def per_module_stage_totals(
    collector: CallCollector,
) -> dict[str, tuple[int, float]]:
    """Per module: (tokens, cost) excluding test generation, from the call records."""
    out: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for rec in collector.records:
        if rec.stage == TESTGEN_STAGE:
            continue
        out[rec.module][0] += rec.input_tokens + rec.output_tokens
        out[rec.module][1] += rec.cost_usd
    return {module: (int(v[0]), v[1]) for module, v in out.items()}
