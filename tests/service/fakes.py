"""Offline fakes for the service tests: Fortran fixtures, scripted LLMs, a sink and a harness.

The fake models answer like real ones would: a conversion reply is a fenced python block that has
to pass ``modernizer.parsing.validate_conversion`` for the Fortran module in the prompt, and a
test-generation reply is a fenced pytest file importing ``target_module``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from modernizer.conversion import SYSTEM_PROMPT
from modernizer.llm import ModelSet
from modernizer.models import LLMCallRecord, LLMResponse, ModuleResult, TokenUsage
from modernizer.pipeline import Pipeline, PipelineConfig
from modernizer.routing import Router
from modernizer.telemetry import Telemetry, setup_telemetry
from modernizer.testing import TEST_SYSTEM_PROMPT

# --- Fortran 77 fixtures (fixed form). Difficulty scores differ, so routing tests can rely on
# an order: SUMSQ < HALF < ACCUM < GCD (GCD has GO TO and lands above the default threshold).

LEGACY = """\
C     SUM OF THE FIRST N SQUARES
      INTEGER FUNCTION ISUMSQ(N)
      INTEGER N, I
      ISUMSQ = 0
      DO 10 I = 1, N
         ISUMSQ = ISUMSQ + I*I
   10 CONTINUE
      RETURN
      END
"""

GOOD = """\
def isumsq(n):
    result = 0
    for i in range(1, n + 1):
        result += i * i
    return result
"""

TESTS = """\
import pytest

import target_module


@pytest.mark.parametrize(
    ("n", "expected"), [(0, 0), (1, 1), (2, 5), (3, 14), (4, 30), (-5, 0)]
)
def test_isumsq(n, expected):
    assert target_module.isumsq(n) == expected


def test_isumsq_returns_an_int():
    assert isinstance(target_module.isumsq(5), int)
"""
TEST_COUNT = 7

HALF_LEGACY = """\
C     INTEGER HALF, TRUNCATING TOWARD ZERO
      INTEGER FUNCTION IHALF(N)
      INTEGER N
      IHALF = N/2
      RETURN
      END
"""

HALF_GOOD = """\
from f77rt import idiv


def ihalf(n):
    return idiv(n, 2)
"""

HALF_TESTS = """\
import pytest

import target_module


@pytest.mark.parametrize(("n", "expected"), [(7, 3), (-7, -3), (0, 0), (1, 0), (-1, 0)])
def test_ihalf_truncates_toward_zero(n, expected):
    assert target_module.ihalf(n) == expected
"""
HALF_TEST_COUNT = 5

ACCUM_LEGACY = """\
C     RUNNING TOTAL KEPT IN COMMON
      SUBROUTINE ACCUM(K, TOTAL)
      INTEGER K, TOTAL, ISUM
      COMMON /RUN/ ISUM
      ISUM = ISUM + K
      TOTAL = ISUM
      RETURN
      END
"""

ACCUM_GOOD = """\
class _Run:
    isum = 0


_run = _Run()


def accum(k, total):
    _run.isum += k
    return _run.isum
"""

GCD_LEGACY = """\
C     GREATEST COMMON DIVISOR, ALWAYS >= 0
      INTEGER FUNCTION IGCD(M, N)
      INTEGER M, N, A, B, T
      A = IABS(M)
      B = IABS(N)
   10 IF (B .EQ. 0) GO TO 20
      T = MOD(A, B)
      A = B
      B = T
      GO TO 10
   20 IGCD = A
      RETURN
      END
"""

GCD_GOOD = """\
def igcd(m, n):
    a = abs(m)
    b = abs(n)
    while b != 0:
        a, b = b, a % b
    return a
"""

# Fortran modules and their valid Python ports, keyed by the unit name that appears in the prompt.
PORTS: dict[str, str] = {
    "ISUMSQ": GOOD,
    "IHALF": HALF_GOOD,
    "ACCUM": ACCUM_GOOD,
    "IGCD": GCD_GOOD,
}

# An EQUIVALENCE is on the default forced-to-frontier list, whatever the difficulty score.
EQUIV_LEGACY = """\
C     OVERLAID STORAGE
      SUBROUTINE OVERLY(X, Y)
      REAL X, Y, A(2), B
      EQUIVALENCE (A(1), B)
      B = X
      Y = A(1)
      RETURN
      END
"""

WAIT_S = 30.0  # upper bound for every wait, so a broken test fails instead of hanging


def fence(code: str) -> str:
    return f"```python\n{code}```"


def port_for(prompt: str) -> str:
    """The valid Python port of whichever fixture module the conversion prompt contains."""
    for unit, code in PORTS.items():
        if unit in prompt:
            return code
    raise AssertionError("the conversion prompt contains no known fixture module")


class FakeLLM:
    """Answers conversion prompts from a script and test prompts with a fixed suite.

    ``conversions`` is a list of replies consumed in order (the last one repeats) or, for several
    modules converted concurrently, a function of the prompt. ``error`` makes every conversion
    call raise it, like a failing model endpoint.
    """

    def __init__(
        self,
        name: str,
        conversions: list[str] | Callable[[str], str],
        tests: str | Callable[[str], str] = TESTS,
        *,
        entered: threading.Event | None = None,
        hold: threading.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.conversions = conversions
        self.tests = tests
        self.entered = entered
        self.hold = hold
        self.error = error
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def _next_conversion(self, prompt: str) -> str:
        if callable(self.conversions):
            return self.conversions(prompt)
        with self._lock:
            if len(self.conversions) > 1:
                return self.conversions.pop(0)
            return self.conversions[0]

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
            tests = self.tests(prompt) if callable(self.tests) else self.tests
            text = fence(tests)
        else:
            assert system == SYSTEM_PROMPT
            if self.entered is not None:
                self.entered.set()
            if self.hold is not None:
                assert self.hold.wait(WAIT_S), "the test never released the slow model"
            if self.error is not None:
                raise self.error
            with self._lock:
                self.prompts.append(prompt)
            text = fence(self._next_conversion(prompt))
        return LLMResponse(
            text=text, usage=TokenUsage(len(prompt) // 4, len(text) // 4), model=self.name
        )


class RecordingSink:
    def __init__(self) -> None:
        self.run_ids: list[str] = []
        self.flushes = 0
        self.closed = False

    def write_module(self, run_id: str, result: ModuleResult) -> None:
        self.run_ids.append(run_id)

    def write_llm_call(self, record: LLMCallRecord) -> None:
        return None

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


@dataclass
class Harness:
    """Builds a fresh ``Pipeline`` over scripted fake models for every call, like production."""

    open_script: list[str] | Callable[[str], str] = field(default_factory=lambda: [GOOD])
    frontier_script: list[str] | Callable[[str], str] = field(default_factory=lambda: [GOOD])
    tests: str | Callable[[str], str] = TESTS
    policy: str = "always_open"
    router_options: dict[str, Any] = field(default_factory=dict)
    config: PipelineConfig = field(default_factory=PipelineConfig)
    build_error: Exception | None = None
    run_error: Exception | None = None
    flush_error: Exception | None = None
    open_error: Exception | None = None
    frontier_error: Exception | None = None

    def __post_init__(self) -> None:
        self.entered = threading.Event()
        self.hold = threading.Event()  # set means the open model answers at once
        self.hold.set()
        self.sink = RecordingSink()
        self.pipelines: list[Pipeline] = []
        self.flushes = 0
        self.telemetry: Telemetry = setup_telemetry("none")
        real_flush = self.telemetry.flush

        def counting_flush() -> None:
            self.flushes += 1
            if self.flush_error is not None:
                raise self.flush_error
            real_flush()

        self.telemetry.flush = counting_flush  # type: ignore[method-assign]

    def __call__(self) -> Pipeline:
        if self.build_error is not None:
            raise self.build_error
        small = FakeLLM(
            "small",
            self.open_script,
            self.tests,
            entered=self.entered,
            hold=self.hold,
            error=self.open_error,
        )
        big = FakeLLM("big", self.frontier_script, self.tests, error=self.frontier_error)
        pipeline = Pipeline(
            ModelSet(frontier=big, open=small),
            Router("big", "small", policy=self.policy, **self.router_options),  # type: ignore[arg-type]
            None,
            telemetry=self.telemetry,
            config=self.config,
            sink=self.sink,
        )
        if self.run_error is not None:
            error = self.run_error

            def failing_run(sources: Mapping[str, str]) -> list[ModuleResult]:
                raise error

            pipeline.run = failing_run  # type: ignore[method-assign]
        self.pipelines.append(pipeline)
        return pipeline

    def close(self) -> None:
        self.telemetry.shutdown()
