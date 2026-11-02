"""Read cumulative totals back out of an ``InMemoryMetricReader``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader, Sum

from .meter import (
    ATTR_ESCALATED,
    ATTR_MODEL,
    ATTR_OUTCOME,
    ATTR_SUCCESS,
    ATTR_TIER,
    ATTR_TOKEN_TYPE,
    METRIC_CALLS,
    METRIC_COST,
    METRIC_MODULE_OUTCOMES,
    METRIC_TOKENS,
)


def _sum_points(
    reader: InMemoryMetricReader, metric_name: str
) -> Iterator[tuple[dict[str, Any], float]]:
    """Yield (attributes, cumulative value) for every data point of the named Sum metric.

    ``get_metrics_data`` collects afresh on each call, and a cumulative reader reports the
    running total each time, so repeated calls never double count. A delta reader would, so it
    is rejected.
    """
    data = reader.get_metrics_data()
    if data is None:
        return
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != metric_name or not isinstance(metric.data, Sum):
                    continue
                if metric.data.aggregation_temporality != AggregationTemporality.CUMULATIVE:
                    raise ValueError(
                        f"metric {metric_name!r} is not cumulative; use a cumulative reader"
                    )
                for point in metric.data.data_points:
                    yield dict(point.attributes or {}), point.value


def _grouped(
    reader: InMemoryMetricReader, metric_name: str, k1: str, k2: str, k3: str
) -> dict[tuple[str, str, str], int]:
    out: dict[tuple[str, str, str], int] = defaultdict(int)
    for attrs, value in _sum_points(reader, metric_name):
        out[(str(attrs.get(k1, "")), str(attrs.get(k2, "")), str(attrs.get(k3, "")))] += int(value)
    return dict(out)


def token_totals(reader: InMemoryMetricReader) -> dict[tuple[str, str, str], int]:
    """(model, tier, direction) -> tokens, where direction is "input" or "output"."""
    return _grouped(reader, METRIC_TOKENS, ATTR_MODEL, ATTR_TIER, ATTR_TOKEN_TYPE)


def cost_total(reader: InMemoryMetricReader) -> float:
    return float(sum(value for _, value in _sum_points(reader, METRIC_COST)))


def module_outcome_counts(reader: InMemoryMetricReader) -> dict[tuple[str, str, str], int]:
    """(final tier, success, escalated) -> number of modules; flags are "true"/"false"."""
    return _grouped(reader, METRIC_MODULE_OUTCOMES, ATTR_TIER, ATTR_SUCCESS, ATTR_ESCALATED)


def call_counts(reader: InMemoryMetricReader) -> dict[tuple[str, str, str], int]:
    """(model, tier, outcome) -> number of LLM calls; outcome is "ok" or "error"."""
    return _grouped(reader, METRIC_CALLS, ATTR_MODEL, ATTR_TIER, ATTR_OUTCOME)


__all__ = ["call_counts", "cost_total", "module_outcome_counts", "token_totals"]
