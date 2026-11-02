from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from modernizer.evaluation import EvaluationConsistencyError, TokenCounts
from modernizer.evaluation.accounting import (
    CallCollector,
    collect_and_verify,
    per_module_stage_totals,
    read_stage_meter,
)
from modernizer.llm import cost_usd
from modernizer.models import LLMResponse, ModuleResult, RoutingDecision, Tier, TokenUsage
from modernizer.telemetry import MeteredLLM, UsageLedger, setup_telemetry

FRONTIER = "gemini-2.5-pro"
OPEN = "gemini-2.5-flash"


class StubLLM:
    def __init__(self, name: str, usage: TokenUsage) -> None:
        self.name = name
        self.usage = usage

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        return LLMResponse(text="x", usage=self.usage, model=self.name)


class World:
    """Two metered clients wired to a fresh telemetry stack, ledger and call collector."""

    def __init__(self) -> None:
        self.telemetry = setup_telemetry("none")
        self.ledger = UsageLedger()
        self.collector = CallCollector()
        self.clients = {
            Tier.FRONTIER: self._client(FRONTIER, Tier.FRONTIER, TokenUsage(1000, 400)),
            Tier.OPEN: self._client(OPEN, Tier.OPEN, TokenUsage(300, 100)),
        }

    def _client(self, name: str, tier: Tier, usage: TokenUsage) -> MeteredLLM:
        return MeteredLLM(
            StubLLM(name, usage),
            tier=tier,
            meter=self.telemetry.llm_meter,
            ledger=self.ledger,
            sink=self.collector,
        )

    def call(self, tier: Tier, stage: str, module: str) -> None:
        self.clients[tier].generate("p", tags={"stage": stage, "module": module})

    def results(self) -> list[ModuleResult]:
        decision = RoutingDecision(Tier.OPEN, OPEN, 0.1, 0.5, "complexity")
        return [
            ModuleResult(
                path=path,
                success=True,
                converted_source="",
                generated_tests="",
                routing=decision,
                final_tier=Tier.OPEN,
                escalated=False,
                usage_by_model=self.ledger.usage_by_model(path),
            )
            for path in self.ledger.modules()
        ]

    def verify(self):
        return collect_and_verify(
            self.telemetry.metric_reader, self.ledger, self.collector, self.results()
        )


@pytest.fixture
def world():
    w = World()
    w.call(Tier.OPEN, "convert", "a.py")
    w.call(Tier.OPEN, "repair", "a.py")
    w.call(Tier.FRONTIER, "test_gen", "a.py")
    w.call(Tier.FRONTIER, "repair", "a.py")
    w.call(Tier.OPEN, "convert", "b.py")
    w.call(Tier.FRONTIER, "test_gen", "b.py")
    yield w
    w.telemetry.shutdown()


def test_totals_split_by_tier_stage_and_testgen(world):
    acct = world.verify()

    assert acct.tokens == TokenCounts(3 * 300 + 3 * 1000, 3 * 100 + 3 * 400)
    assert acct.tokens_by_tier["open"] == TokenCounts(900, 300)
    assert acct.tokens_by_tier["frontier"] == TokenCounts(3000, 1200)
    assert acct.tokens_by_stage["test_gen"] == TokenCounts(2000, 800)
    assert acct.tokens_by_stage["convert"] == TokenCounts(600, 200)
    assert acct.tokens_excluding_testgen == TokenCounts(
        acct.tokens.input_tokens - 2000, acct.tokens.output_tokens - 800
    )

    frontier_call = cost_usd(FRONTIER, TokenUsage(1000, 400))
    open_call = cost_usd(OPEN, TokenUsage(300, 100))
    assert frontier_call > 0 and open_call > 0
    assert acct.cost_usd_total == pytest.approx(3 * frontier_call + 3 * open_call)
    assert acct.cost_usd_by_stage["test_gen"] == pytest.approx(2 * frontier_call)
    assert acct.cost_usd_excluding_testgen == pytest.approx(acct.cost_usd_total - 2 * frontier_call)
    assert math.isclose(sum(acct.cost_usd_by_tier.values()), acct.cost_usd_total)


def test_stage_meter_keeps_the_stage_attribute(world):
    meter = read_stage_meter(world.telemetry.metric_reader)
    assert meter.tokens[(FRONTIER, "frontier", "test_gen", "input")] == 2000
    assert meter.tokens[(OPEN, "open", "convert", "output")] == 200
    assert (OPEN, "open", "repair") in meter.cost


def test_per_module_totals_exclude_testgen(world):
    totals = per_module_stage_totals(world.collector)
    assert totals["a.py"][0] == 400 + 400 + 1400
    assert totals["b.py"][0] == 400


def test_an_empty_run_verifies_to_zero():
    w = World()
    try:
        acct = w.verify()
    finally:
        w.telemetry.shutdown()
    assert acct.tokens == TokenCounts()
    assert acct.cost_usd_total == 0.0


def test_ledger_drift_is_detected(world):
    world.ledger.record("a.py", OPEN, Tier.OPEN, "convert", TokenUsage(1, 0), 0.0)
    with pytest.raises(EvaluationConsistencyError, match="tier"):
        world.verify()


def test_ledger_cost_drift_is_detected(world):
    world.ledger.record("a.py", OPEN, Tier.OPEN, "convert", TokenUsage(0, 0), 0.5)
    with pytest.raises(EvaluationConsistencyError, match="cost"):
        world.verify()


def test_a_missing_call_record_is_detected(world):
    world.collector._records.pop()
    with pytest.raises(EvaluationConsistencyError, match="stage"):
        world.verify()


def test_module_results_that_disagree_are_detected(world):
    results = world.results()
    results[0].usage_by_model[OPEN] = TokenUsage(1, 1)
    with pytest.raises(EvaluationConsistencyError, match="per-module"):
        collect_and_verify(world.telemetry.metric_reader, world.ledger, world.collector, results)


def test_consistency_error_is_an_assertion_error():
    assert issubclass(EvaluationConsistencyError, AssertionError)
