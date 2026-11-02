from __future__ import annotations

import pytest

from modernizer.models import (
    Attempt,
    LLMCallRecord,
    ModuleResult,
    RoutingDecision,
    TestRunResult,
    Tier,
    TokenUsage,
    ValidationIssue,
    ValidationReport,
)


def make_result(
    path: str = "pkg/a.py",
    *,
    success: bool = True,
    escalated: bool = False,
    with_test: bool = True,
) -> ModuleResult:
    test = (
        TestRunResult(passed=3, failed=1, errors=0, total=4, duration_s=0.5) if with_test else None
    )
    attempts = [
        Attempt(
            index=0,
            tier=Tier.OPEN,
            model="open-model",
            stage="convert",
            usage=TokenUsage(100, 50),
            validation=ValidationReport(
                ok=False, issues=[ValidationIssue("syntax_error", "bad", 3)]
            ),
        ),
        Attempt(
            index=1,
            tier=Tier.FRONTIER,
            model="frontier-model",
            stage="repair",
            usage=TokenUsage(200, 80),
            validation=ValidationReport(
                ok=False,
                issues=[
                    ValidationIssue("legacy_construct_remaining", "print"),
                    ValidationIssue("api_changed", "sig"),
                ],
            ),
            test=test,
        ),
    ]
    return ModuleResult(
        path=path,
        success=success,
        converted_source="print('hi')\n",
        generated_tests="def test_x(): pass\n",
        routing=RoutingDecision(
            tier=Tier.OPEN, model="open-model", difficulty=0.25, threshold=0.5, policy="complexity"
        ),
        final_tier=Tier.FRONTIER,
        escalated=escalated,
        attempts=attempts,
        test=test,
        usage_by_model={
            "open-model": TokenUsage(100, 50),
            "frontier-model": TokenUsage(200, 80),
        },
        cost_usd=0.0123,
        duration_s=4.5,
        error=None,
    )


def make_call(i: int = 0) -> LLMCallRecord:
    return LLMCallRecord(
        run_id="run-1",
        timestamp="2026-01-01T00:00:00+00:00",
        module=f"pkg/m{i}.py",
        stage="convert",
        model="open-model",
        tier="open",
        input_tokens=10 + i,
        output_tokens=5,
        cost_usd=0.001,
        latency_s=0.2,
    )


@pytest.fixture
def result() -> ModuleResult:
    return make_result()
