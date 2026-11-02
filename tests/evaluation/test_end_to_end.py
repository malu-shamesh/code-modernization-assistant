"""Dry run of the whole harness over four real benchmark modules with the oracle model.

Real pytest subprocesses are used (generated tests and held-out tests), so this is the slow
test of the package. The oracle replays the answer key: nothing here says anything about how
well a language model converts code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

from modernizer.evaluation import (
    EvaluationReport,
    PolicySpec,
    default_policies,
    load_corpus,
    oracle_model_set,
    run_evaluation,
)
from modernizer.factory import build_retriever
from modernizer.llm import ModelConfig, build_model_set
from modernizer.models import TokenUsage
from modernizer.parsing import analyze
from modernizer.pipeline import PipelineConfig
from modernizer.routing import Router, batch_open_share
from modernizer.settings import Settings

from .helpers import BENCHMARK_DIR

# Two easy, one medium, one hard module. None of them contains a construct that forces the
# complexity router onto the frontier tier, so a threshold of 1.0 routes all of them open.
CASE_IDS = [
    "a01_horner_poly",
    "a03_factorial_fibonacci",
    "a08_taylor_series",
    "a10_calendar_loop_traps",
]
HARD_ID = "a10_calendar_loop_traps"
# Contains an alternate RETURN, so the feature rule sends it to the frontier whatever its score.
FORCED_ID = "a06_root_finding"
OPEN_THRESHOLD = 1.0
PER_MILLION = 1_000_000


def fake_cost(model: str, usage: TokenUsage) -> float:
    rate = 10.0 if model == "oracle-frontier" else 1.0
    return rate * usage.total / PER_MILLION


@dataclass
class DryRun:
    report: EvaluationReport
    progress: list[str]


@pytest.fixture(scope="module")
def cases():
    return load_corpus(BENCHMARK_DIR, ids=CASE_IDS)


@pytest.fixture(scope="module")
def dry_run(cases) -> DryRun:
    settings = Settings()
    progress: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("modernizer.telemetry.meter._default_cost_fn", fake_cost)
        report = run_evaluation(
            cases,
            default_policies(OPEN_THRESHOLD),
            settings=settings,
            models=oracle_model_set(cases, fail_difficulties=frozenset({"hard"})),
            retriever=build_retriever(settings),
            run_tests=True,
            workers=4,
            dry_run=True,
            progress=progress.append,
        )
    return DryRun(report, progress)


def test_report_is_flagged_as_a_dry_run(dry_run):
    report = dry_run.report
    assert report.dry_run is True
    assert report.corpus["n"] == 4
    assert report.corpus["by_difficulty"] == {"easy": 2, "medium": 1, "hard": 1}
    assert report.corpus["ids"] == sorted(CASE_IDS)
    assert report.settings["frontier_model"] == "oracle-frontier"
    assert report.settings["open_model"] == "oracle-open"
    assert report.settings["threshold"] == Settings().threshold
    assert report.baseline == "always_frontier"
    # One line per module when its conversion finishes and one when its held-out score is in.
    assert len(dry_run.progress) == 24
    assert sum("converted" in line for line in dry_run.progress) == 12
    assert sum("held-out" in line for line in dry_run.progress) == 12
    assert "DRY RUN - NOT MODEL RESULTS" in report.to_markdown()


def test_frontier_baseline_passes_every_held_out_test(dry_run):
    result = dry_run.report.policy("always_frontier")
    assert (result.n, result.generated_pass, result.reference_pass) == (4, 4, 4)
    assert result.reference_test_fraction == 1.0
    assert result.routed_open == 0 and result.escalated == 0
    assert set(result.tokens_by_tier) == {"frontier"}


def test_open_only_policy_fails_the_hard_module_without_escalating(dry_run):
    result = dry_run.report.policy("always_open")
    assert (result.generated_pass, result.reference_pass) == (3, 3)
    assert result.routed_open == 4 and result.escalated == 0
    hard = next(r for r in result.rows if r.id == HARD_ID)
    assert hard.routed_tier == hard.final_tier == "open"
    assert not hard.generated_success and hard.reference_success is False
    # The Fortran source is not importable as Python, so no held-out test can pass.
    assert hard.reference_passed < (hard.reference_total or 1)
    assert hard.reference_fraction == 0.0
    assert hard.error and "validation failed" in hard.error
    assert result.reference_test_fraction == pytest.approx(3 / 4)


def test_hard_module_escalates_under_the_complexity_policy(dry_run):
    result = dry_run.report.policy("complexity")
    assert result.threshold_used == OPEN_THRESHOLD
    assert all(r.routed_tier == "open" for r in result.rows), "threshold 1.0 routes all open"

    hard = next(r for r in result.rows if r.id == HARD_ID)
    assert hard.escalated
    assert hard.final_tier == "frontier"
    assert hard.generated_success and hard.reference_success
    assert hard.attempts == 4, "three open attempts (convert and two repairs), then the frontier"

    easy = [r for r in result.rows if r.difficulty != "hard"]
    assert all(r.final_tier == "open" and not r.escalated for r in easy)
    assert (result.generated_pass, result.reference_pass) == (4, 4)
    assert (result.routed_open, result.escalated, result.final_open_successes) == (4, 1, 3)
    assert result.final_open_reference_successes == 3
    # A fixed threshold: no target, no planned share, nothing forced; escalation lowers the
    # final open share below the routed one.
    assert result.target_open_fraction is None and result.planned_open_share is None
    assert result.forced_frontier == 0
    assert (result.final_open, result.open_fraction, result.final_open_share) == (3, 1.0, 0.75)


def test_meter_totals_agree_with_rows_and_tiers(dry_run):
    for result in dry_run.report.policies:
        assert result.tokens.total > 0
        assert sum(r.input_tokens + r.output_tokens for r in result.rows) == result.tokens.total
        assert sum(t.total for t in result.tokens_by_tier.values()) == result.tokens.total
        assert sum(t.total for t in result.tokens_by_stage.values()) == result.tokens.total
        assert sum(r.tokens_excluding_testgen for r in result.rows) == (
            result.tokens_excluding_testgen.total
        )
        assert math.isclose(sum(r.cost_usd for r in result.rows), result.cost_usd_total)
        assert math.isclose(sum(result.cost_usd_by_tier.values()), result.cost_usd_total)


def test_test_generation_is_split_out(dry_run):
    for result in dry_run.report.policies:
        testgen = result.tokens_by_stage["test_gen"]
        assert testgen.total > 0
        assert result.tokens_excluding_testgen.total == result.tokens.total - testgen.total
        assert result.cost_usd_excluding_testgen == pytest.approx(
            result.cost_usd_total - result.cost_usd_by_stage["test_gen"]
        )
        assert 0 < result.cost_usd_excluding_testgen < result.cost_usd_total


def test_cost_comes_from_the_metered_calls(dry_run):
    baseline = dry_run.report.policy("always_frontier")
    expected = fake_cost(
        "oracle-frontier", TokenUsage(baseline.tokens.input_tokens, baseline.tokens.output_tokens)
    )
    assert baseline.cost_usd_total == pytest.approx(expected)

    complexity = dry_run.report.policy("complexity")
    open_tokens = complexity.tokens_by_tier["open"].total
    frontier_tokens = complexity.tokens_by_tier["frontier"].total
    assert complexity.cost_usd_total == pytest.approx(
        (open_tokens * 1.0 + frontier_tokens * 10.0) / PER_MILLION
    )


def test_comparisons_use_paired_discordant_counts(dry_run):
    by_policy = {c.policy: c for c in dry_run.report.comparisons}
    assert set(by_policy) == {"always_open", "complexity"}

    open_only = by_policy["always_open"].reference
    assert (open_only.only_baseline_passes, open_only.only_policy_passes) == (1, 0)
    assert open_only.delta == pytest.approx(-0.25)

    complexity = by_policy["complexity"]
    assert complexity.reference.discordant == 0
    assert complexity.generated.discordant == 0
    assert complexity.token_reduction_pct is not None
    assert complexity.cost_reduction_pct is not None
    assert "Evidence is limited" in complexity.verdict
    assert complexity.verdict.startswith("DRY RUN")


def test_report_round_trips_through_json(dry_run):
    report = dry_run.report
    text = report.to_json()
    data = json.loads(text)
    assert data["dry_run"] is True
    assert data == report.to_dict()
    assert EvaluationReport.from_json(text) == report
    assert [p["spec"]["name"] for p in data["policies"]] == [
        "always_frontier",
        "always_open",
        "complexity",
    ]


def test_validation_only_run_skips_every_test_and_says_so(cases):
    settings = Settings()
    report = run_evaluation(
        cases[:2],
        [PolicySpec("always_open", "always_open", None, False)],
        settings=settings,
        models=oracle_model_set(cases),
        retriever=build_retriever(settings),
        run_tests=False,
        dry_run=True,
    )
    result = report.policies[0]
    assert result.reference_pass is None and result.reference_test_fraction is None
    assert not result.generated_tests_run
    assert all(r.reference_success is None for r in result.rows)
    assert result.tokens_by_stage.get("test_gen") is None
    markdown = report.to_markdown()
    assert "not run" in markdown
    assert "static validation" in markdown
    assert EvaluationReport.from_json(report.to_json()) == report


def test_canned_models_are_refused_unless_flagged_as_a_dry_run(cases):
    settings = Settings()
    with pytest.raises(ValueError, match="dry_run"):
        run_evaluation(
            cases[:1],
            default_policies(0.5),
            settings=settings,
            models=oracle_model_set(cases),
            retriever=None,
            dry_run=False,
        )
    mock_settings = Settings(models=ModelConfig(provider="mock"))
    with pytest.raises(ValueError, match="dry_run"):
        run_evaluation(
            cases[:1],
            default_policies(0.5),
            settings=mock_settings,
            models=build_model_set(mock_settings.models),
            retriever=None,
            dry_run=False,
        )


def test_invalid_inputs_are_rejected(cases):
    models = oracle_model_set(cases)
    common = {"settings": Settings(), "models": models, "retriever": None, "dry_run": True}
    with pytest.raises(ValueError, match="no benchmark cases"):
        run_evaluation([], default_policies(0.5), **common)
    with pytest.raises(ValueError, match="no policies"):
        run_evaluation(cases, [], **common)
    with pytest.raises(ValueError, match="unique"):
        run_evaluation(
            cases, [PolicySpec("x", "always_open"), PolicySpec("x", "complexity")], **common
        )
    with pytest.raises(ValueError, match="unique"):
        run_evaluation([cases[0], cases[0]], default_policies(0.5), **common)


def test_an_oracle_that_cannot_match_a_prompt_invalidates_the_dry_run(cases):
    # A tiny chunk size splits the module, so the prompts no longer hold the whole legacy source.
    # The pipeline would absorb the oracle's error into a failed module, which must not pass
    # as a model failure.
    settings = Settings(pipeline=PipelineConfig(max_chunk_chars=300))
    with pytest.raises(RuntimeError, match="dry run is invalid"):
        run_evaluation(
            cases[:1],
            [PolicySpec("always_frontier", "always_frontier", None, False)],
            settings=settings,
            models=oracle_model_set(cases),
            retriever=None,
            workers=1,
            dry_run=True,
        )


# ---------------------------------------------------------------------------------------------
# Batch-calibrated routing (target open share). No tests are run, so these are fast.
# ---------------------------------------------------------------------------------------------

BATCH_IDS = [*CASE_IDS, FORCED_ID]


@dataclass
class TargetRun:
    report: EvaluationReport
    router: Router
    parsed: list


@pytest.fixture(scope="module")
def target_run() -> TargetRun:
    cases = load_corpus(BENCHMARK_DIR, ids=BATCH_IDS)
    # The settings carry a batch target of their own; an arm that sets none must ignore it.
    settings = Settings(pipeline=PipelineConfig(target_open_fraction=0.25))
    policies = [
        PolicySpec("fixed", "complexity", OPEN_THRESHOLD, True),
        PolicySpec("half", "complexity", OPEN_THRESHOLD, True, target_open_fraction=0.5),
        PolicySpec("everything", "complexity", OPEN_THRESHOLD, True, target_open_fraction=1.0),
        PolicySpec("always_open", "always_open", None, False),
    ]
    models = oracle_model_set(cases, fail_difficulties=frozenset({"hard"}))
    report = run_evaluation(
        cases,
        policies,
        settings=settings,
        models=models,
        retriever=None,
        run_tests=False,
        workers=2,
        dry_run=True,
    )
    parsed = [analyze(f"{c.id}.f", c.legacy_source) for c in cases]
    router = Router(models.frontier.name, models.open.name, threshold=settings.threshold)
    return TargetRun(report, router, parsed)


def test_target_share_yields_the_planned_share_the_router_computes(target_run):
    result = target_run.report.policy("half")
    expected = target_run.router.calibrated_for_batch(target_run.parsed, 0.5)

    assert result.target_open_fraction == 0.5
    assert result.planned_open_share == batch_open_share(expected, target_run.parsed)
    assert result.threshold_used == expected.threshold
    # The router is deterministic, so what was planned is what was routed.
    assert result.routed_open / result.n == result.planned_open_share
    assert 0.0 < result.planned_open_share < 1.0
    # The open set is the easiest eligible modules: equal scores are routed alike, so every
    # open score lies at or below the threshold and every unforced frontier score above it.
    open_rows = [r for r in result.rows if r.routed_tier == "open"]
    frontier_rows = [
        r for r in result.rows if r.routed_tier == "frontier" and not r.forced_frontier
    ]
    assert open_rows and frontier_rows
    assert max(r.difficulty_score for r in open_rows) <= expected.threshold
    assert min(r.difficulty_score for r in frontier_rows) > expected.threshold


def test_modules_forced_to_the_frontier_are_counted(target_run):
    for name in ("half", "everything"):
        result = target_run.report.policy(name)
        forced = [r for r in result.rows if r.forced_frontier]
        assert [r.id for r in forced] == [FORCED_ID]
        assert forced[0].routed_tier == "frontier"
        assert result.forced_frontier == 1
    # The baselines ignore the feature rule, so nothing is forced there.
    assert target_run.report.policy("always_open").forced_frontier == 0


def test_a_target_that_cannot_be_reached_is_planned_honestly(target_run):
    result = target_run.report.policy("everything")
    expected = target_run.router.calibrated_for_batch(target_run.parsed, 1.0)
    # Four of five modules are eligible; the forced one stays on the frontier.
    assert result.planned_open_share == batch_open_share(expected, target_run.parsed)
    assert result.planned_open_share == pytest.approx(4 / 5)
    assert result.routed_open == 4


def test_escalation_lowers_the_final_open_share_below_the_routed_share(target_run):
    result = target_run.report.policy("everything")
    hard = next(r for r in result.rows if r.id == HARD_ID)
    assert hard.routed_tier == "open" and hard.escalated and hard.final_tier == "frontier"
    assert (result.routed_open, result.final_open, result.escalated) == (4, 3, 1)
    assert result.final_open_share < result.open_fraction


def test_an_arm_without_a_target_keeps_its_fixed_threshold(target_run):
    for name in ("fixed", "always_open"):
        result = target_run.report.policy(name)
        assert result.target_open_fraction is None
        assert result.planned_open_share is None
    fixed = target_run.report.policy("fixed")
    assert fixed.threshold_used == OPEN_THRESHOLD
    assert fixed.routed_open == 4, "everything but the forced module is routed open"
    assert fixed.forced_frontier == 1


def test_target_fields_survive_the_json_round_trip_and_reach_the_markdown(target_run):
    report = target_run.report
    restored = EvaluationReport.from_json(report.to_json())
    assert restored == report
    half = restored.policy("half")
    assert half.spec.target_open_fraction == 0.5
    assert half.planned_open_share == report.policy("half").planned_open_share
    markdown = report.to_markdown()
    assert "## Routing" in markdown
    assert "policy parameter" in markdown and "not a result" in markdown
    assert "calibrated threshold" in markdown
