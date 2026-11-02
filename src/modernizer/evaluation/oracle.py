"""A fake model that answers from the benchmark answer key. DRY RUNS ONLY.

WARNING: ``OracleLLM`` does not convert anything. It reads the hand-written reference
port and the held-out reference tests out of the corpus and replays them, so every
number produced with it is a statement about the harness plumbing (parsing, routing, metering,
scoring, rendering) and says NOTHING about any language model. Reports built from it must be
marked ``dry_run`` and must never be quoted as results. ``run_evaluation`` refuses to run an
oracle with ``dry_run=False``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Collection, Mapping, Sequence

from ..conversion import SYSTEM_PROMPT
from ..fileio import read_source
from ..llm import ModelSet
from ..models import LLMResponse, TokenUsage
from ..testing import TEST_SYSTEM_PROMPT
from .corpus import BenchmarkCase

_FENCE = "`" * 3

# Section headers of conversion.prompts.build_conversion_prompt that precede the numbered source
# listing (the second one is used in repair mode), and the header line of the listing in
# testing.generator.build_test_prompt.
_CONVERSION_HEADERS = ("## Fortran 77 source to convert", "## Fortran 77 source (the authority)")
_TEST_HEADER = "FORTRAN 77 SOURCE (fixed form)"
_FORTRAN_FENCE = f"{_FENCE}fortran\n"
_FENCE_CLOSE = f"\n{_FENCE}"
# Conversion gutter: '{n:>4} | text' (the trailing space and text vanish on blank lines).
_CONVERSION_GUTTER = re.compile(r"^ *(\d+) \|(?: (.*))?$")
# Test gutter: '{n:>4}| text'.
_TEST_GUTTER = re.compile(r"^ *(\d+)\| ?(.*)$")

# A normalised listing: every line right-stripped, trailing blank lines dropped.
_Key = tuple[str, ...]


class OracleLookupError(LookupError):
    """The prompt does not contain the Fortran source of any corpus case.

    Deliberately not an ``LLMError``: it signals a harness problem (for example a module large
    enough to be chunked), not a model failure, and must not be absorbed by escalation.
    """


def _fenced(code: str) -> str:
    # The corpus may be checked out with CRLF line endings; the reply is always LF.
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    return f"{_FENCE}python\n{text.strip(chr(10))}\n{_FENCE}"


def _normalised(lines: Sequence[str]) -> _Key:
    stripped = [line.rstrip() for line in lines]
    while stripped and not stripped[-1]:
        stripped.pop()
    return tuple(stripped)


def _listing(prompt: str, marker: str, gutter: re.Pattern[str]) -> _Key:
    """The Fortran listing that follows ``marker`` in ``prompt``, with its gutter removed.

    The last occurrence of the marker wins, because repair prompts quote other material before
    the source section. Raises ``OracleLookupError`` when the listing is absent or its line
    numbers are not 1..n (a slice of a larger file keeps the original numbers).
    """
    start = prompt.rfind(marker)
    if start < 0:
        raise OracleLookupError("the prompt has no Fortran source section (prelude prompt?)")
    opening = prompt.find(_FORTRAN_FENCE, start)
    if opening < 0:
        raise OracleLookupError("the Fortran source section has no fortran code block")
    body_start = opening + len(_FORTRAN_FENCE)
    closing = prompt.find(_FENCE_CLOSE, body_start)
    if closing < 0:
        raise OracleLookupError("the Fortran code block is not closed")
    texts: list[str] = []
    for expected, raw in enumerate(prompt[body_start:closing].split("\n"), start=1):
        match = gutter.match(raw)
        if match is None or int(match.group(1)) != expected:
            raise OracleLookupError(
                "the Fortran listing is not a whole file numbered from 1 (chunked module?)"
            )
        texts.append(match.group(2) or "")
    return _normalised(texts)


def _index(
    cases: Sequence[BenchmarkCase], split: Callable[[str], list[str]]
) -> dict[_Key, BenchmarkCase]:
    index: dict[_Key, BenchmarkCase] = {}
    for case in cases:
        key = _normalised(split(case.legacy_source))
        if key in index:
            raise ValueError(
                f"benchmark cases {index[key].id} and {case.id} have the same Fortran source, "
                "so the oracle cannot tell their prompts apart"
            )
        index[key] = case
    return index


class OracleLLM:
    """``LLMClient`` that returns the corpus answer key. See the module warning: dry runs only.

    Conversion prompts (``system == conversion.SYSTEM_PROMPT``) are answered with the case's
    reference port (fenced as python), test prompts (``system == testing.TEST_SYSTEM_PROMPT``)
    with its held-out reference tests. The case is found by extracting the Fortran listing from
    the prompt (the line-number gutter is stripped) and comparing it, line by line with trailing
    blanks ignored, to each case's ``legacy.f``. Prompts that do not carry a whole file (chunk
    or shared-state prompts) raise ``OracleLookupError``. For every difficulty in
    ``fail_difficulties`` the conversion answer is the Fortran source itself fenced as python,
    which does not compile and so drives the repair and escalation paths. Token counts are
    estimated as ``len // 4``; the cost comes from the pricing table, where an unknown model
    (such as "oracle-open") costs 0.0. Thread-safe.
    """

    def __init__(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        name: str = "oracle",
        fail_difficulties: Collection[str] = frozenset(),
    ) -> None:
        if not cases:
            raise ValueError("the oracle needs at least one benchmark case")
        self.name = name
        self.fail_difficulties = frozenset(fail_difficulties)
        # The conversion prompt splits the source on "\n" and the test prompt uses
        # str.splitlines(), so every case is indexed under both normalisations.
        self._by_conversion_key = _index(cases, lambda text: text.split("\n"))
        self._by_test_key = _index(cases, str.splitlines)
        self._lock = threading.Lock()
        self._calls = 0
        self._lookup_failures = 0

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._calls

    @property
    def lookup_failures(self) -> int:
        """Requests the oracle could not answer. The pipeline absorbs the exception into a
        failed module, so the runner checks this counter to keep it from passing as a model
        failure."""
        with self._lock:
            return self._lookup_failures

    def _unanswerable(self, message: str) -> OracleLookupError:
        with self._lock:
            self._lookup_failures += 1
        return OracleLookupError(message)

    def _lookup(
        self,
        prompt: str,
        marker: str,
        gutter: re.Pattern[str],
        index: Mapping[_Key, BenchmarkCase],
    ) -> BenchmarkCase:
        try:
            key = _listing(prompt, marker, gutter)
        except OracleLookupError as exc:
            raise self._unanswerable(str(exc)) from exc
        case = index.get(key)
        if case is None:
            raise self._unanswerable("the Fortran source in the prompt matches no corpus case")
        return case

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        with self._lock:
            self._calls += 1
        if system == TEST_SYSTEM_PROMPT:
            case = self._lookup(prompt, _TEST_HEADER, _TEST_GUTTER, self._by_test_key)
            text = _fenced(read_source(case.reference_tests))
        elif system == SYSTEM_PROMPT:
            marker = max(_CONVERSION_HEADERS, key=prompt.rfind)
            case = self._lookup(prompt, marker, _CONVERSION_GUTTER, self._by_conversion_key)
            source = (
                case.legacy_source
                if case.difficulty in self.fail_difficulties
                else case.reference_source
            )
            text = _fenced(source)
        else:
            raise self._unanswerable("unrecognised system prompt: not a conversion or test request")
        usage = TokenUsage(
            input_tokens=len((system or "") + prompt) // 4, output_tokens=len(text) // 4
        )
        return LLMResponse(text=text, usage=usage, model=self.name)


def oracle_model_set(
    cases: Sequence[BenchmarkCase],
    fail_difficulties: Collection[str] = frozenset({"hard"}),
) -> ModelSet:
    """Oracle pair for dry runs: "oracle-frontier" never fails, "oracle-open" fails the given
    difficulties (default: hard) so escalation is exercised. Plumbing only, not model results."""
    return ModelSet(
        frontier=OracleLLM(cases, name="oracle-frontier"),
        open=OracleLLM(cases, name="oracle-open", fail_difficulties=fail_difficulties),
    )
