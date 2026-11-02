"""End-to-end conversion of modules: parse, retrieve, route, convert, validate, test, repair.

One module goes through:

1. tree-sitter analysis (``parsing.analyze``) which yields legacy-construct hits and metrics,
2. retrieval of migration notes from the vector index,
3. a routing decision between the open and the frontier model,
4. conversion, then AST validation, then LLM-written pytest tests run in a subprocess,
5. up to ``max_repairs`` repair rounds on the same tier, then escalation to the frontier model
   with the failing attempt and its feedback.

The tests are generated once per module (after the first candidate passes validation) and reused
for every later attempt, so different tiers are judged by the same tests.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .conversion import ConversionError, convert_module
from .llm import LLMError, ModelSet
from .models import (
    Attempt,
    ContextSnippet,
    ModuleResult,
    ParsedModule,
    RoutingDecision,
    TestRunResult,
    Tier,
    TokenUsage,
    ValidationIssue,
    ValidationReport,
)
from .parsing import analyze, validate_conversion
from .protocols import ResultSink
from .retrieval import Retriever
from .routing import Router, batch_open_share
from .sinks import NullSink
from .telemetry import MeteredLLM, Telemetry, UsageLedger
from .testing import TestGenerationError, generate_tests, run_tests

logger = logging.getLogger(__name__)

_CANDIDATE_TRACEBACK = re.compile(r"(?<!test_)target_module\.py")
_MAX_FEEDBACK_CHARS = 4000


@dataclass(frozen=True)
class PipelineConfig:
    max_repairs: int = 2  # repair rounds per tier after the first conversion
    escalate: bool = True  # open -> frontier after the open tier is exhausted
    run_tests: bool = True
    testgen_tier: Tier = Tier.FRONTIER
    min_tests: int = 8
    test_timeout_s: float = 60.0
    retrieval_k: int = 6
    max_chunk_chars: int = 12000
    max_workers: int = 4
    target_open_fraction: float | None = (
        None  # complexity policy: share of a batch for the open model
    )


class _Candidate:
    """Mutable per-module state shared by the attempts of one run."""

    def __init__(self) -> None:
        self.source = ""
        self.tests: str | None = None
        self.tests_regenerated = False
        self.attempts: list[Attempt] = []
        self.last_test: TestRunResult | None = None
        self.last_validation = ValidationReport(ok=False)
        self.error: str | None = None
        self.fatal = False  # the tests are unusable, no further attempt can be judged


class Pipeline:
    def __init__(
        self,
        models: ModelSet,
        router: Router,
        retriever: Retriever | None,
        *,
        telemetry: Telemetry,
        config: PipelineConfig | None = None,
        sink: ResultSink | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.router = router
        self.retriever = retriever
        self.telemetry = telemetry
        self.sink: ResultSink = sink or NullSink()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.ledger = UsageLedger()
        self.planned_open_share: float | None = None
        tracer = telemetry.tracer("modernizer.pipeline")
        self._tracer = tracer
        self._clients: dict[Tier, MeteredLLM] = {
            tier: MeteredLLM(
                models.for_tier(tier),
                tier=tier,
                meter=telemetry.llm_meter,
                ledger=self.ledger,
                tracer=tracer,
                run_id=self.run_id,
                sink=self.sink,
            )
            for tier in Tier
        }

    def run(self, sources: Mapping[str, str]) -> list[ModuleResult]:
        """Convert every module concurrently; results keep the input order.

        With ``config.target_open_fraction`` set and the complexity policy, the routing threshold
        is calibrated on exactly this batch before any model is called.
        """
        items = list(sources.items())
        parsed = [analyze(path, source) for path, source in items]
        self._calibrate(parsed)
        workers = max(1, min(self.config.max_workers, len(items) or 1))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="modernize") as pool:
            futures = [
                pool.submit(self.process, path, source, parsed=module)
                for (path, source), module in zip(items, parsed, strict=True)
            ]
            return [f.result() for f in futures]

    def _calibrate(self, parsed: list[ParsedModule]) -> None:
        target = self.config.target_open_fraction
        if target is None or self.router.policy != "complexity" or not parsed:
            return
        self.router = self.router.calibrated_for_batch(parsed, target)
        self.planned_open_share = batch_open_share(self.router, parsed)
        logger.info(
            "calibrated threshold %.3f for target open share %.0f%% (planned %.0f%%)",
            self.router.threshold,
            target * 100,
            self.planned_open_share * 100,
        )

    def process(
        self, path: str, source: str, *, parsed: ParsedModule | None = None
    ) -> ModuleResult:
        started = time.perf_counter()
        with self._tracer.start_as_current_span("pipeline.module") as span:
            span.set_attribute("modernizer.module", path)
            try:
                result = self._convert(path, source, parsed, started)
            except Exception as exc:
                logger.exception("module %s failed unexpectedly", path)
                result = self._crash_result(path, source, exc, started)
            span.set_attribute("modernizer.success", result.success)
            span.set_attribute("modernizer.escalated", result.escalated)
        self.telemetry.llm_meter.record_module_outcome(
            result.final_tier, result.success, result.escalated
        )
        try:
            self.sink.write_module(self.run_id, result)
        except Exception:
            logger.exception("sink failed to store the result of %s", path)
        return result

    def _convert(
        self, path: str, source: str, parsed: ParsedModule | None, started: float
    ) -> ModuleResult:
        cfg = self.config
        parsed = parsed or analyze(path, source)
        context = self._retrieve(parsed)
        routing = self.router.route(parsed)
        state = _Candidate()
        tags = {"module": path}

        current: RoutingDecision | None = routing
        escalated = False
        final_tier = routing.tier
        success = False
        previous: str | None = None
        feedback: str | None = None

        while current is not None and not success and not state.fatal:
            final_tier = current.tier
            for _ in range(1 + cfg.max_repairs):
                stage = "repair" if previous is not None else "convert"
                outcome = self._attempt(
                    parsed, context, current, state, tags, stage, previous, feedback
                )
                if outcome is None:
                    break
                success, previous, feedback = outcome
                if success or state.fatal:
                    break
            can_escalate = cfg.escalate and not success and not state.fatal
            nxt = self.router.escalate(current) if can_escalate else None
            if nxt is not None:
                escalated = True
            current = nxt

        usage_by_model = self.ledger.usage_by_model(path)
        return ModuleResult(
            path=path,
            success=success,
            converted_source=state.source,
            generated_tests=state.tests or "",
            routing=routing,
            final_tier=final_tier,
            escalated=escalated,
            attempts=state.attempts,
            test=state.last_test,
            usage_by_model=usage_by_model,
            cost_usd=self.ledger.cost(path),
            duration_s=time.perf_counter() - started,
            error=None if success else state.error or _failure_summary(state),
        )

    def _retrieve(self, parsed: ParsedModule) -> list[ContextSnippet]:
        if self.retriever is None:
            return []
        try:
            return self.retriever.retrieve(parsed, k=self.config.retrieval_k)
        except Exception:
            logger.exception("retrieval failed for %s, converting without context", parsed.path)
            return []

    def _attempt(
        self,
        parsed: ParsedModule,
        context: list[ContextSnippet],
        decision: RoutingDecision,
        state: _Candidate,
        tags: dict[str, str],
        stage: str,
        previous: str | None,
        feedback: str | None,
    ) -> tuple[bool, str | None, str | None] | None:
        """One conversion attempt. Returns (success, previous, feedback) or None to stop the tier."""
        cfg = self.config
        index = len(state.attempts)
        state.error = None
        try:
            output = convert_module(
                parsed,
                context,
                self._clients[decision.tier],
                feedback=feedback,
                previous=previous,
                max_chunk_chars=cfg.max_chunk_chars,
                tags=tags,
            )
        except ConversionError as exc:
            report = ValidationReport(
                ok=False, issues=[ValidationIssue("no_code", "the reply contained no usable code")]
            )
            state.attempts.append(
                Attempt(
                    index,
                    decision.tier,
                    decision.model,
                    stage,
                    exc.usage or TokenUsage(),
                    report,
                )
            )
            state.last_validation = report
            return False, previous, "Your previous reply contained no usable code. " + _ONE_BLOCK
        except LLMError as exc:
            state.error = (
                f"model call failed on the {decision.tier.value} tier ({type(exc).__name__})"
            )
            logger.warning("%s: %s: %r", parsed.path, state.error, exc)
            return None

        state.source = output.source
        report = validate_conversion(parsed, output.source)
        state.last_validation = report
        attempt = Attempt(index, decision.tier, decision.model, stage, output.usage, report)
        state.attempts.append(attempt)
        if not report.ok:
            return False, output.source, _validation_feedback(report)
        if not cfg.run_tests:
            return True, output.source, None

        test_result = self._test(parsed, output.source, state, tags)
        attempt.test = test_result
        state.last_test = test_result
        if test_result is None:
            return None
        if test_result.all_passed:
            return True, output.source, None
        return False, output.source, _test_feedback(test_result)

    def _test(
        self, parsed: ParsedModule, candidate: str, state: _Candidate, tags: dict[str, str]
    ) -> TestRunResult | None:
        cfg = self.config
        if state.tests is None and not self._write_tests(parsed, candidate, state, tags):
            return None
        result = run_tests(candidate, state.tests or "", timeout_s=cfg.test_timeout_s)
        if result.collection_error and not _candidate_to_blame(result.output):
            if state.tests_regenerated:
                state.error = "the generated tests could not be collected"
                state.fatal = True
                return result
            state.tests_regenerated = True
            if not self._write_tests(parsed, candidate, state, tags):
                return None
            result = run_tests(candidate, state.tests or "", timeout_s=cfg.test_timeout_s)
        return result

    def _write_tests(
        self, parsed: ParsedModule, candidate: str, state: _Candidate, tags: dict[str, str]
    ) -> bool:
        client = self._clients[self.config.testgen_tier]
        last_error = ""
        for _ in range(2):
            try:
                generated = generate_tests(
                    parsed, candidate, client, min_tests=self.config.min_tests, tags=tags
                )
            except TestGenerationError as exc:
                last_error = type(exc).__name__
                continue
            except LLMError as exc:
                last_error = type(exc).__name__
                break
            state.tests = generated.source
            return True
        state.error = f"test generation failed: {last_error}"
        state.fatal = True
        return False

    def _crash_result(self, path: str, source: str, exc: Exception, started: float) -> ModuleResult:
        try:
            routing = self.router.route(analyze(path, source))
        except Exception:
            routing = RoutingDecision(
                tier=Tier.FRONTIER,
                model=self._clients[Tier.FRONTIER].name,
                difficulty=1.0,
                threshold=None,
                policy="unknown",
                reasons=["analysis failed"],
            )
        return ModuleResult(
            path=path,
            success=False,
            converted_source="",
            generated_tests="",
            routing=routing,
            final_tier=routing.tier,
            escalated=False,
            usage_by_model=self.ledger.usage_by_model(path),
            cost_usd=self.ledger.cost(path),
            duration_s=time.perf_counter() - started,
            error=f"unexpected error ({type(exc).__name__}), see the server log",
        )


_ONE_BLOCK = "Reply with the complete converted module in one python fenced code block."


def _candidate_to_blame(output: str) -> bool:
    """True when a collection error is raised from the converted module, not from the tests."""
    return bool(_CANDIDATE_TRACEBACK.search(output))


def _validation_feedback(report: ValidationReport) -> str:
    return "The converted module failed static validation:\n" + report.feedback()


def _test_feedback(result: TestRunResult) -> str:
    if result.timed_out:
        head = "The tests timed out, the converted code probably loops forever or blocks."
    elif result.collection_error:
        head = "The tests could not be collected, the converted module probably fails on import."
    else:
        head = (
            f"{result.failed} failed, {result.errors} errors, {result.passed} passed out of "
            f"{result.total} tests."
        )
    tail = result.output[-_MAX_FEEDBACK_CHARS:]
    return (
        f"{head}\nThe tests were machine-written from the legacy code; when a test contradicts "
        "the legacy behaviour, the legacy code is the authority.\n\npytest output:\n" + tail
    )


def _failure_summary(state: _Candidate) -> str:
    last = state.attempts[-1] if state.attempts else None
    if last is None:
        return "no attempt produced a candidate"
    if not last.validation.ok:
        codes = ", ".join(sorted({i.code for i in last.validation.issues}))
        return f"validation failed: {codes}"
    if last.test is not None:
        return f"tests failed: {last.test.passed}/{last.test.total} passed"
    return "conversion did not succeed"
