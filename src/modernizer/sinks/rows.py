"""Flat, JSON-serialisable rows shared by every sink and by the BigQuery table schemas."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ..models import LLMCallRecord, ModuleResult, TestRunResult


def _finite(value: float | None) -> float | None:
    """BigQuery's JSON ingestion and strict JSON both reject NaN/Infinity."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _text(value: str | None) -> str | None:
    """Replaces lone surrogates (e.g. undecodable file names) that no UTF-8 sink can encode."""
    if value is None:
        return None
    return value.encode("utf-8", "replace").decode("utf-8")


def _tier_value(tier: Any) -> str | None:
    if tier is None:
        return None
    return str(getattr(tier, "value", tier))


def _effective_test(result: ModuleResult) -> TestRunResult | None:
    if result.test is not None:
        return result.test
    # Only the last attempt: an earlier attempt's tests describe different code.
    return result.attempts[-1].test if result.attempts else None


def module_row(
    run_id: str,
    result: ModuleResult,
    *,
    include_source: bool = False,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build the ``module_results`` row for one converted module.

    Source code is left out unless ``include_source`` is set. Test columns are ``None`` when no
    tests were run for the module.
    """
    ts = timestamp or datetime.now(timezone.utc)
    usage = result.total_usage
    last = result.attempts[-1] if result.attempts else None
    test = _effective_test(result)
    routing = result.routing

    row: dict[str, Any] = {
        "run_id": _text(run_id),
        "timestamp": ts.isoformat(),
        "path": _text(result.path),
        "success": bool(result.success),
        "routed_tier": _tier_value(routing.tier),
        "final_tier": _tier_value(result.final_tier),
        "escalated": bool(result.escalated),
        "difficulty": _finite(routing.difficulty),
        "routing_policy": routing.policy,
        "routing_model": routing.model,
        "final_model": last.model if last else None,
        "n_attempts": len(result.attempts),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": _finite(result.cost_usd),
        "duration_s": _finite(result.duration_s),
        "tests_total": test.total if test else None,
        "tests_passed": test.passed if test else None,
        "tests_failed": test.failed if test else None,
        "tests_errors": test.errors if test else None,
        "tests_timed_out": test.timed_out if test else None,
        "validation_issue_codes": [i.code for i in last.validation.issues] if last else [],
        "error": _text(result.error),
    }
    if include_source:
        row["converted_source"] = _text(result.converted_source)
    return row


def call_row(record: LLMCallRecord) -> dict[str, Any]:
    """Build the ``llm_calls`` row for one metered model call."""
    row = asdict(record)
    for key, value in row.items():
        if isinstance(value, str):
            row[key] = _text(value)
    row["cost_usd"] = _finite(row["cost_usd"])
    row["latency_s"] = _finite(row["latency_s"])
    return row
