"""Run routing policies over the benchmark corpus and score them against held-out tests.

Two success metrics are kept apart on purpose:

* ``generated_success``: the pipeline's own verdict. It is judged by tests that an LLM wrote
  from the legacy source, which is what exists in production but is a weak, possibly
  circular check.
* ``reference_success``: the converted module run against the corpus's hand-written,
  held-out tests, which no model saw. This is the stricter and independent metric.

Every policy gets a fresh telemetry stack, so its OpenTelemetry in-memory reader holds exactly
that policy's calls.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from ..factory import build_pipeline
from ..llm import ModelSet, ScriptedLLM
from ..models import ModuleResult, ParsedModule, TestRunResult, Tier
from ..parsing import analyze
from ..retrieval import Retriever
from ..routing import Router
from ..settings import Settings
from ..telemetry import setup_telemetry
from ..testing import run_test_file
from .accounting import (
    CallCollector,
    EvaluationConsistencyError,
    collect_and_verify,
    per_module_stage_totals,
)
from .compare import compare_results
from .corpus import BenchmarkCase, corpus_summary
from .oracle import OracleLLM
from .policies import PolicySpec, default_policies
from .results import Comparison, EvaluationReport, ModuleRow, PolicyResult

__all__ = [
    "EvaluationConsistencyError",
    "PolicySpec",
    "default_policies",
    "run_evaluation",
]

Progress = Callable[[str], None]


@dataclass(frozen=True)
class _CaseOutcome:
    result: ModuleResult
    reference: TestRunResult | None


def _is_synthetic(models: ModelSet) -> bool:
    """True for clients that replay canned answers instead of calling a model."""
    return any(isinstance(m, OracleLLM | ScriptedLLM) for m in (models.frontier, models.open))


def _check_oracle_lookups(models: ModelSet, policy_name: str) -> None:
    """The pipeline turns any exception into a failed module, which would make a harness
    problem (such as a chunked prompt the oracle cannot match) look like a model failure."""
    for client in (models.frontier, models.open):
        if isinstance(client, OracleLLM) and client.lookup_failures:
            raise RuntimeError(
                f"policy {policy_name!r}: {client.name} could not answer "
                f"{client.lookup_failures} request(s); the dry run is invalid (a module larger "
                "than max_chunk_chars cannot be matched to its corpus case)"
            )


def _score_reference(case: BenchmarkCase, result: ModuleResult, timeout_s: float) -> TestRunResult:
    """Run the held-out tests against whatever module the pipeline ended with."""
    if not result.converted_source.strip():
        return TestRunResult(collection_error=True, output="no converted source was produced")
    return run_test_file(result.converted_source, case.reference_tests, timeout_s=timeout_s)


def _validate_inputs(
    cases: Sequence[BenchmarkCase],
    policies: Sequence[PolicySpec],
    settings: Settings,
    models: ModelSet,
    dry_run: bool,
) -> None:
    if not cases:
        raise ValueError("no benchmark cases to evaluate")
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("benchmark case ids must be unique")
    if not policies:
        raise ValueError("no policies to evaluate")
    if len({p.name for p in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    if not dry_run and (_is_synthetic(models) or settings.models.provider == "mock"):
        raise ValueError(
            "the models replay canned answers (oracle or mock): pass dry_run=True, and do not "
            "present the output as model results"
        )


class _ProgressCollector(CallCollector):
    """``CallCollector`` that also reports each finished module as the pipeline writes it."""

    def __init__(self, on_module: Callable[[ModuleResult], None]) -> None:
        super().__init__()
        self._on_module = on_module

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        self._on_module(result)


def _source_path(case: BenchmarkCase) -> str:
    return f"{case.id}.f"


def _forced_to_frontier(router: Router, module: ParsedModule) -> bool:
    """Whether the feature rule of ``router`` sends ``module`` to the frontier tier."""
    return (
        router.policy == "complexity"
        and router.force_frontier_on_features
        and any(hit.feature_id in router.force_frontier_features for hit in module.hits)
    )


def _run_policy(
    spec: PolicySpec,
    cases: Sequence[BenchmarkCase],
    *,
    settings: Settings,
    models: ModelSet,
    retriever: Retriever | None,
    run_tests: bool,
    workers: int,
    progress: Progress | None,
) -> PolicyResult:
    threshold = spec.threshold if spec.threshold is not None else settings.threshold
    # The target comes from the spec alone: an arm without one keeps its fixed threshold even
    # when the settings carry a batch target (MODERNIZER_OPEN_FRACTION).
    target = spec.target_open_fraction if spec.policy == "complexity" else None
    pipeline_cfg = replace(
        settings.pipeline,
        escalate=spec.escalate,
        run_tests=run_tests,
        max_workers=workers,
        target_open_fraction=target,
    )
    policy_settings = replace(
        settings, policy=spec.policy, threshold=threshold, pipeline=pipeline_cfg
    )
    sources = {_source_path(case): case.legacy_source for case in cases}
    telemetry = setup_telemetry("none")
    emit_lock = threading.Lock()
    converted = 0

    def emit(message: str) -> None:
        if progress is not None:
            with emit_lock:
                progress(f"[{spec.name}] {message}")

    def on_module(result: ModuleResult) -> None:
        nonlocal converted
        with emit_lock:
            converted += 1
            index = converted
        tiers = result.routing.tier.value
        if result.final_tier is not result.routing.tier:
            tiers += f"->{result.final_tier.value}"
        emit(
            f"converted {index}/{len(cases)} {result.path}: {tiers}, "
            f"generated {'pass' if result.success else 'FAIL'}"
        )

    collector = _ProgressCollector(on_module)

    try:
        # retriever=None makes build_pipeline construct the default one from the settings.
        pipeline = build_pipeline(
            policy_settings,
            models=models,
            retriever=retriever,
            telemetry=telemetry,
            sink=collector,
        )
        # Pipeline.run analyses the whole batch first and, with a target share, calibrates the
        # complexity threshold on exactly these modules before any model is called.
        results = pipeline.run(sources)
        _check_oracle_lookups(models, spec.name)
        router = pipeline.router  # the calibrated router when a target share was set
        planned = pipeline.planned_open_share
        forced = [
            _forced_to_frontier(router, analyze(_source_path(case), case.legacy_source))
            for case in cases
        ]

        scored = 0

        def score(case: BenchmarkCase, result: ModuleResult) -> TestRunResult | None:
            nonlocal scored
            reference = (
                _score_reference(case, result, pipeline_cfg.test_timeout_s) if run_tests else None
            )
            with emit_lock:
                scored += 1
                index = scored
            held_out = (
                "not run" if reference is None else "pass" if reference.all_passed else "FAIL"
            )
            emit(f"scored {index}/{len(cases)} {case.id}: held-out {held_out}")
            return reference

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evaluate") as pool:
            references = list(pool.map(score, cases, results))
        outcomes = [_CaseOutcome(r, ref) for r, ref in zip(results, references, strict=True)]

        accounting = collect_and_verify(
            telemetry.metric_reader, pipeline.ledger, collector, results
        )
        excluding_testgen = per_module_stage_totals(collector)
        final_threshold = router.threshold if spec.policy == "complexity" else None
    finally:
        telemetry.shutdown()

    rows = [
        _row(case, outcome, excluding_testgen.get(_source_path(case), (0, 0.0)), is_forced)
        for case, outcome, is_forced in zip(cases, outcomes, forced, strict=True)
    ]
    return PolicyResult.from_rows(
        spec,
        rows,
        tokens=accounting.tokens,
        tokens_excluding_testgen=accounting.tokens_excluding_testgen,
        cost_usd_total=accounting.cost_usd_total,
        cost_usd_excluding_testgen=accounting.cost_usd_excluding_testgen,
        tokens_by_tier=accounting.tokens_by_tier,
        tokens_by_stage=accounting.tokens_by_stage,
        cost_usd_by_tier=accounting.cost_usd_by_tier,
        cost_usd_by_stage=accounting.cost_usd_by_stage,
        threshold_used=final_threshold,
        generated_tests_run=run_tests,
        target_open_fraction=target,
        planned_open_share=planned,
    )


def _row(
    case: BenchmarkCase,
    outcome: _CaseOutcome,
    excluding: tuple[int, float],
    forced_frontier: bool,
) -> ModuleRow:
    result, reference = outcome.result, outcome.reference
    usage = result.total_usage
    return ModuleRow(
        id=case.id,
        difficulty=case.difficulty,
        difficulty_score=result.routing.difficulty,
        routed_tier=result.routing.tier.value,
        final_tier=result.final_tier.value,
        escalated=result.escalated,
        generated_success=result.success,
        reference_success=None if reference is None else reference.all_passed,
        reference_passed=None if reference is None else reference.passed,
        reference_total=None if reference is None else reference.total,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        tokens_excluding_testgen=excluding[0],
        cost_usd=result.cost_usd,
        cost_usd_excluding_testgen=excluding[1],
        attempts=len(result.attempts),
        error=result.error,
        forced_frontier=forced_frontier,
    )


def _settings_snapshot(
    settings: Settings, models: ModelSet, retriever: Retriever | None, workers: int
) -> dict[str, Any]:
    cfg = settings.pipeline
    return {
        "provider": settings.models.provider,
        "frontier_model": models.frontier.name,
        "open_model": models.open.name,
        "frontier_client": type(models.frontier).__name__,
        "open_client": type(models.open).__name__,
        "threshold": settings.threshold,
        "retrieval": settings.retrieval,
        "retriever_supplied": retriever is not None,
        "max_repairs": cfg.max_repairs,
        "testgen_tier": Tier(cfg.testgen_tier).value,
        "min_tests": cfg.min_tests,
        "test_timeout_s": cfg.test_timeout_s,
        "retrieval_k": cfg.retrieval_k,
        "max_chunk_chars": cfg.max_chunk_chars,
        "workers": workers,
    }


def choose_baseline(results: Sequence[PolicyResult]) -> PolicyResult:
    """The always_frontier arm if present, else the first policy."""
    for result in results:
        if result.spec.policy == "always_frontier":
            return result
    return results[0]


def run_evaluation(
    cases: Sequence[BenchmarkCase],
    policies: Sequence[PolicySpec],
    *,
    settings: Settings,
    models: ModelSet,
    retriever: Retriever | None,
    run_tests: bool = True,
    workers: int | None = None,
    dry_run: bool = False,
    progress: Progress | None = None,
) -> EvaluationReport:
    """Evaluate every policy on every case and return a report that is hard to over-read.

    ``retriever=None`` lets the pipeline factory build the default retriever from ``settings``.
    ``run_tests=False`` skips both the pipeline's generated tests and the held-out scoring; the
    pass columns then only reflect static validation and the report says so. Policies run one
    after another (each with a fresh telemetry stack); cases within a policy run concurrently.

    Each case is submitted to the pipeline as ``<id>.f`` and a policy with a
    ``target_open_fraction`` has its threshold calibrated by the pipeline on exactly this batch
    of cases, so the planned open share is in-sample by construction. The report records the
    target, the planned share and the realised shares side by side.

    ``dry_run`` must be True when the models replay canned answers (``OracleLLM``, the mock
    provider); such a report is flagged and rendered as plumbing evidence, not model results.
    The unusual direction is refused: canned models with ``dry_run=False`` raise ``ValueError``.
    """
    _validate_inputs(cases, policies, settings, models, dry_run)
    pool_size = max(1, workers if workers is not None else settings.pipeline.max_workers)

    results = [
        _run_policy(
            spec,
            cases,
            settings=settings,
            models=models,
            retriever=retriever,
            run_tests=run_tests,
            workers=pool_size,
            progress=progress,
        )
        for spec in policies
    ]
    baseline = choose_baseline(results)
    comparisons: list[Comparison] = [
        compare_results(baseline, result, dry_run=dry_run)
        for result in results
        if result is not baseline
    ]
    corpus = {**corpus_summary(cases), "ids": [c.id for c in cases]}
    return EvaluationReport(
        dry_run=dry_run,
        policies=results,
        comparisons=comparisons,
        corpus=corpus,
        settings=_settings_snapshot(settings, models, retriever, pool_size),
        baseline=baseline.spec.name,
        run_tests=run_tests,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
