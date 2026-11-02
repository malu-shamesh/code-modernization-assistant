from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from modernizer.models import (
    ApiParam,
    ApiSymbol,
    CodeUnit,
    FeatureHit,
    LLMResponse,
    ModuleMetrics,
    ParsedModule,
    TokenUsage,
)


@dataclass
class FakeLLM:
    """Replays canned replies (the last one repeats) and records every call."""

    replies: list[str]
    name: str = "fake-model"
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(10, 5))
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "tags": dict(tags or {}),
            }
        )
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return LLMResponse(text=text, usage=self.usage, model=self.name)


def fenced(code: str) -> str:
    return f"```python\n{code}\n```"


AXPY = """\
      SUBROUTINE AXPY(N, A, X, Y, IERR)
C     Y = A*X + Y
      INTEGER N, IERR
      REAL A, X(N), Y(N)
      IERR = 0
      DO 10 I = 1, N
         Y(I) = A*X(I) + Y(I)
   10 CONTINUE
      RETURN
      END"""

BLOCK_INIT = """\
      BLOCK DATA INIT
      COMMON /CNT/ NCALLS
      DATA NCALLS /0/
      END"""

BUMP = """\
      SUBROUTINE BUMP(N)
      INTEGER N
      COMMON /CNT/ NCALLS
      NCALLS = NCALLS + N
      END"""

TOTAL = """\
      INTEGER FUNCTION TOTAL()
      COMMON /CNT/ NCALLS
      TOTAL = NCALLS
      END"""

AXPY_API = ApiSymbol(
    "axpy",
    "subroutine",
    (
        ApiParam("n", "integer"),
        ApiParam("a", "real"),
        ApiParam("x", "real", rank=1),
        ApiParam("y", "real", rank=1, intent="inout"),
        ApiParam("ierr", "integer", intent="out"),
    ),
    returns=("ierr",),
)
BUMP_API = ApiSymbol("bump", "subroutine", (ApiParam("n", "integer"),))
TOTAL_API = ApiSymbol("total", "function", result_type="integer")


def make_module(
    parts: list[tuple[str, str, str]],
    hits: list[FeatureHit] | None = None,
    api: tuple[ApiSymbol, ...] = (),
    path: str = "lib/demo.f",
) -> ParsedModule:
    """Build a module from ``(name, kind, text)`` units separated by one blank line."""
    units: list[CodeUnit] = []
    line = 1
    texts: list[str] = []
    for name, kind, text in parts:
        count = text.count("\n") + 1
        units.append(CodeUnit(name, kind, text, line, line + count - 1, defines=(name.lower(),)))
        texts.append(text)
        line += count + 1
    source = "\n\n".join(texts) + "\n"
    return ParsedModule(
        path=path,
        source=source,
        units=units,
        hits=list(hits or []),
        metrics=ModuleMetrics(loc=source.count("\n")),
        api=api,
    )


def axpy_module() -> ParsedModule:
    hits = [
        FeatureHit("do_labeled", 6, 8, "DO 10 I = 1, N"),
        FeatureHit("implicit_typing", 6, 6, "I"),
        FeatureHit("arg_output", 5, 5, "IERR = 0"),
    ]
    return make_module([("axpy", "subroutine", AXPY)], hits, (AXPY_API,))


def state_module() -> ParsedModule:
    """Four units; BLOCK DATA at lines 1-4, BUMP 6-10, TOTAL 12-15, AXPY 17-25."""
    hits = [
        FeatureHit("block_data", 1, 1, "BLOCK DATA INIT"),
        FeatureHit("common_block", 2, 2, "COMMON /CNT/ NCALLS"),
        FeatureHit("data_statement", 3, 3, "DATA NCALLS /0/"),
        FeatureHit("common_block", 8, 8, "COMMON /CNT/ NCALLS"),
        FeatureHit("common_block", 13, 13, "COMMON /CNT/ NCALLS"),
        FeatureHit("do_labeled", 22, 24, "DO 10 I = 1, N"),
    ]
    return make_module(
        [
            ("init", "block_data", BLOCK_INIT),
            ("bump", "subroutine", BUMP),
            ("total", "function", TOTAL),
            ("axpy", "subroutine", AXPY),
        ],
        hits,
        (BUMP_API, TOTAL_API, AXPY_API),
    )
