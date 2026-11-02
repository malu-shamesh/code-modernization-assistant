from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pytest

from modernizer.conversion import SYSTEM_PROMPT
from modernizer.llm import LLMRequestError, ModelSet
from modernizer.models import LLMResponse, Tier, TokenUsage
from modernizer.pipeline import Pipeline, PipelineConfig
from modernizer.routing import Router
from modernizer.sinks import JsonlSink
from modernizer.telemetry import cost_total, setup_telemetry, token_totals
from modernizer.testing import TEST_SYSTEM_PROMPT

LEGACY = """      SUBROUTINE SCALE(N, A, X)
      INTEGER N
      REAL A, X(*)
      INTEGER I
      DO 10 I = 1, N
         X(I) = A * X(I)
   10 CONTINUE
      RETURN
      END
"""

CLAMP_LEGACY = """      SUBROUTINE CLAMP(X, LO, HI)
      REAL X, LO, HI
      IF (X - LO) 10, 20, 20
   10 X = LO
      GO TO 40
   20 IF (X - HI) 40, 40, 30
   30 X = HI
   40 RETURN
      END
"""

GOOD = """def scale(n, a, x):
    for i in range(1, n + 1):
        x[i - 1] = a * x[i - 1]
"""

CLAMP_GOOD = """def clamp(x, lo, hi):
    if x < lo:
        x = lo
    elif x > hi:
        x = hi
    return x
"""

STILL_FORTRAN = LEGACY

WRONG_BEHAVIOUR = """def scale(n, a, x):
    for i in range(1, n + 1):
        x[i - 1] = a + x[i - 1]
"""

TESTS = """import numpy as np
import target_module


def test_scales_in_place():
    x = np.array([1.0, 2.0, 3.0])
    target_module.scale(3, 2.0, x)
    assert x.tolist() == [2.0, 4.0, 6.0]


def test_zero_trip_loop_leaves_array_alone():
    x = np.array([1.0])
    target_module.scale(0, 2.0, x)
    assert x[0] == 1.0
"""


def fence(code: str) -> str:
    return f"```python\n{code}```"


class FakeLLM:
    """Answers conversion prompts from a queue and test prompts with a fixed suite."""

    def __init__(self, name: str, conversions: list[str | Exception], tests: str = TESTS) -> None:
        self.name = name
        self.conversions = list(conversions)
        self.tests = tests
        self.conversion_calls = 0
        self.test_calls = 0
        self.stages: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        if system == TEST_SYSTEM_PROMPT:
            self.test_calls += 1
            text = fence(self.tests)
        else:
            assert system == SYSTEM_PROMPT
            self.conversion_calls += 1
            self.stages.append((tags or {}).get("stage", ""))
            item = self.conversions.pop(0)
            if isinstance(item, Exception):
                raise item
            text = fence(item)
        return LLMResponse(
            text=text, usage=TokenUsage(len(prompt) // 4, len(text) // 4), model=self.name
        )


def cost_fn(model: str, usage: TokenUsage) -> float:
    rate = 10.0 if model == "big" else 1.0
    return rate * usage.total / 1_000_000


@pytest.fixture
def telemetry():
    tel = setup_telemetry("none")
    yield tel
    tel.shutdown()


def build(
    telemetry,
    open_llm: FakeLLM,
    big_llm: FakeLLM,
    *,
    policy: str = "always_open",
    sink=None,
    **config,
) -> Pipeline:
    return Pipeline(
        ModelSet(frontier=big_llm, open=open_llm),
        Router("big", "small", policy=policy),
        None,
        telemetry=telemetry,
        config=PipelineConfig(**config),
        sink=sink,
        run_id="test-run",
    )


def patch_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("modernizer.telemetry.meter._default_cost_fn", cost_fn)


def test_open_model_succeeds_without_escalation(telemetry, monkeypatch):
    patch_cost(monkeypatch)
    small = FakeLLM("small", [GOOD])
    big = FakeLLM("big", [])
    result = build(telemetry, small, big).process("pkg/scale.f", LEGACY)

    assert result.success
    assert result.final_tier == Tier.OPEN
    assert not result.escalated
    assert result.routing.tier == Tier.OPEN
    assert [a.stage for a in result.attempts] == ["convert"]
    assert result.test is not None and result.test.all_passed
    assert big.conversion_calls == 0
    assert big.test_calls == 1, "tests come from the frontier model by default"
    assert set(result.usage_by_model) == {"small", "big"}
    assert result.cost_usd > 0


def test_escalates_to_frontier_after_open_tier_is_exhausted(telemetry, monkeypatch):
    patch_cost(monkeypatch)
    small = FakeLLM("small", [STILL_FORTRAN, STILL_FORTRAN, STILL_FORTRAN])
    big = FakeLLM("big", [GOOD])
    result = build(telemetry, small, big, max_repairs=2).process("scale.f", LEGACY)

    assert result.success
    assert result.escalated
    assert result.final_tier == Tier.FRONTIER
    assert small.conversion_calls == 3
    assert big.conversion_calls == 1
    assert small.stages == ["convert", "repair", "repair"]
    assert big.stages == ["repair"], "the escalated model sees the failed attempt"
    assert [a.tier for a in result.attempts] == [Tier.OPEN] * 3 + [Tier.FRONTIER]
    assert not result.attempts[0].validation.ok


def test_repair_round_fixes_failing_tests(telemetry):
    small = FakeLLM("small", [WRONG_BEHAVIOUR, GOOD])
    big = FakeLLM("big", [])
    result = build(telemetry, small, big).process("scale.f", LEGACY)

    assert result.success
    assert not result.escalated
    assert [a.stage for a in result.attempts] == ["convert", "repair"]
    assert result.attempts[0].validation.ok
    assert result.attempts[0].test is not None and not result.attempts[0].test.all_passed
    assert big.test_calls == 1, "tests are generated once and reused"


def test_gives_up_when_nothing_passes(telemetry):
    small = FakeLLM("small", [WRONG_BEHAVIOUR] * 2)
    big = FakeLLM("big", [WRONG_BEHAVIOUR] * 2)
    result = build(telemetry, small, big, max_repairs=1).process("scale.f", LEGACY)

    assert not result.success
    assert result.escalated
    assert result.error and "tests failed" in result.error
    assert len(result.attempts) == 4


def test_frontier_policy_never_escalates(telemetry):
    small = FakeLLM("small", [])
    big = FakeLLM("big", [GOOD])
    result = build(telemetry, small, big, policy="always_frontier").process("scale.f", LEGACY)

    assert result.success
    assert result.routing.tier == Tier.FRONTIER
    assert small.conversion_calls == 0


def test_provider_error_on_open_tier_escalates(telemetry):
    small = FakeLLM("small", [LLMRequestError("boom", status_code=500)])
    big = FakeLLM("big", [GOOD])
    result = build(telemetry, small, big).process("scale.f", LEGACY)

    assert result.success and result.escalated
    assert result.final_tier == Tier.FRONTIER


def test_validation_only_mode_skips_tests(telemetry):
    small = FakeLLM("small", [GOOD])
    big = FakeLLM("big", [])
    result = build(telemetry, small, big, run_tests=False).process("scale.f", LEGACY)

    assert result.success
    assert result.test is None
    assert big.test_calls == 0


def test_unusable_generated_tests_stop_the_module(telemetry):
    broken_tests = "import target_module\n\ndef test_x(:\n    pass\n"
    small = FakeLLM("small", [GOOD])
    big = FakeLLM("big", [], tests=broken_tests)
    result = build(telemetry, small, big).process("scale.f", LEGACY)

    assert not result.success
    assert result.error and "test generation failed" in result.error
    assert not result.escalated


def test_run_keeps_order_and_meters_tokens(telemetry, monkeypatch, tmp_path):
    patch_cost(monkeypatch)
    small = FakeLLM("small", [GOOD, GOOD])
    big = FakeLLM("big", [])
    sink = JsonlSink(tmp_path)
    pipeline = build(telemetry, small, big, sink=sink, max_workers=2)
    results = pipeline.run({"a.py": LEGACY, "b.py": LEGACY})
    sink.close()

    assert [r.path for r in results] == ["a.py", "b.py"]
    assert all(r.success for r in results)

    ledger_usage, ledger_cost = pipeline.ledger.totals()
    metered = token_totals(telemetry.metric_reader)
    for model, usage in ledger_usage.items():
        tier = "frontier" if model == "big" else "open"
        assert metered[(model, tier, "input")] == usage.input_tokens
        assert metered[(model, tier, "output")] == usage.output_tokens
    assert math.isclose(cost_total(telemetry.metric_reader), ledger_cost)

    rows = [json.loads(line) for line in (tmp_path / "modules.jsonl").read_text().splitlines()]
    assert {r["path"] for r in rows} == {"a.py", "b.py"}
    calls = (tmp_path / "llm_calls.jsonl").read_text().splitlines()
    assert len(calls) == small.conversion_calls + big.test_calls


def test_unexpected_errors_become_failed_results(telemetry, monkeypatch):
    small = FakeLLM("small", [GOOD])
    big = FakeLLM("big", [])
    pipeline = build(telemetry, small, big)

    def explode(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("modernizer.pipeline.convert_module", explode)
    result = pipeline.process("scale.f", LEGACY)
    assert not result.success
    assert result.error == "unexpected error (RuntimeError), see the server log"
    assert "kaboom" not in result.error


class RoutingFake(FakeLLM):
    """Answers by module: the fake just echoes a correct port for whichever unit it is shown."""

    def generate(self, prompt, *, system=None, **kwargs):
        if system == TEST_SYSTEM_PROMPT:
            self.test_calls += 1
            return LLMResponse(text=fence(TESTS), usage=TokenUsage(10, 10), model=self.name)
        self.conversion_calls += 1
        code = CLAMP_GOOD if "CLAMP" in prompt else GOOD
        return LLMResponse(text=fence(code), usage=TokenUsage(10, 10), model=self.name)


def test_batch_calibration_sends_the_easier_module_to_the_open_model(telemetry):
    small = RoutingFake("small", [])
    big = RoutingFake("big", [])
    pipeline = Pipeline(
        ModelSet(frontier=big, open=small),
        Router("big", "small", policy="complexity"),
        None,
        telemetry=telemetry,
        config=PipelineConfig(run_tests=False, target_open_fraction=0.5),
        run_id="calibrated",
    )
    results = pipeline.run({"scale.f": LEGACY, "clamp.f": CLAMP_LEGACY})

    assert [r.path for r in results] == ["scale.f", "clamp.f"]
    assert all(r.success for r in results)
    assert pipeline.planned_open_share == 0.5
    by_path = {r.path: r for r in results}
    assert by_path["scale.f"].routing.tier == Tier.OPEN
    assert by_path["clamp.f"].routing.tier == Tier.FRONTIER
    assert by_path["scale.f"].routing.difficulty < by_path["clamp.f"].routing.difficulty
