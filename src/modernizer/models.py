"""Shared data types. Every subpackage depends on this module and on ``protocols``; neither
imports from any other ``modernizer`` subpackage (except ``features``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    """Which class of model handles a module."""

    FRONTIER = "frontier"
    OPEN = "open"


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureHit:
    """One occurrence of a legacy construct (see ``modernizer.features``)."""

    feature_id: str
    line: int  # 1-based
    end_line: int
    snippet: str


@dataclass(frozen=True)
class ApiParam:
    """One dummy argument of a Fortran program unit, as inferred from its declarations/body."""

    name: str  # lower case
    type: str  # "integer" | "real" | "double" | "complex" | "logical" | "character" | "external" | "unknown"
    rank: int = 0  # 0 = scalar
    intent: str = (
        "in"  # "in" | "out" | "inout": scalars by assignment analysis, arrays by in-place writes
    )


@dataclass(frozen=True)
class ApiSymbol:
    """A program unit whose Python signature and return shape must follow the calling convention."""

    name: str  # lower case unit name
    kind: str  # "subroutine" | "function" | "program" | "block_data"
    params: tuple[ApiParam, ...] = ()
    result_type: str | None = None  # functions only, same vocabulary as ApiParam.type
    returns: tuple[str, ...] = ()  # names of scalar dummies written by the unit, in parameter order
    alt_returns: int = 0  # number of "*" dummy arguments (alternate RETURN targets)


@dataclass
class CodeUnit:
    """A Fortran program unit (subroutine, function, program, block data)."""

    name: str
    kind: str  # "subroutine" | "function" | "program" | "block_data" | "statements"
    source: str
    start_line: int
    end_line: int
    defines: tuple[str, ...] = ()
    references: tuple[str, ...] = ()  # free names referenced inside the unit
    hits: list[FeatureHit] = field(default_factory=list)


@dataclass
class ModuleMetrics:
    loc: int = 0
    sloc: int = 0
    n_functions: int = 0  # subroutines + functions
    n_common_blocks: int = 0
    max_nesting: int = 0
    cyclomatic: int = 0  # 1 + number of branch points over the whole module
    n_feature_hits: int = 0
    distinct_features: int = 0
    weighted_feature_score: float = 0.0  # sum of feature weights over all hits
    semantic_hits: int = 0  # hits whose feature severity is "semantic"
    has_syntax_errors: bool = (
        False  # tree-sitter ERROR/MISSING nodes that are not known legacy syntax
    )
    calls: tuple[str, ...] = ()  # external procedures called (CALL x / x(...)), lower case


@dataclass
class ParsedModule:
    path: str
    source: str
    units: list[CodeUnit]
    hits: list[FeatureHit]
    metrics: ModuleMetrics
    api: tuple[ApiSymbol, ...] = ()

    @property
    def feature_ids(self) -> list[str]:
        """Distinct feature ids in first-seen order."""
        return list(dict.fromkeys(h.feature_id for h in self.hits))


# --------------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------------


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage
    model: str
    latency_s: float = 0.0


@dataclass
class LLMCallRecord:
    """One metered model call (also the row written to the BigQuery ``llm_calls`` table)."""

    run_id: str
    timestamp: str  # ISO-8601 UTC
    module: str
    stage: str  # "convert" | "repair" | "test_gen" | ...
    model: str
    tier: str  # Tier value
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


# --------------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------------


@dataclass
class ContextSnippet:
    id: str
    title: str
    text: str
    score: float
    feature_ids: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    tier: Tier
    model: str
    difficulty: float  # 0..1
    threshold: float | None
    policy: str  # "always_frontier" | "always_open" | "complexity"
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Validation and testing
# --------------------------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    code: str  # "syntax_error" | "fortran_remnant" | "api_changed" | "empty_output" ...
    message: str
    line: int | None = None


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def feedback(self) -> str:
        return "\n".join(
            f"- [{i.code}]{f' line {i.line}' if i.line else ''}: {i.message}" for i in self.issues
        )


@dataclass
class TestRunResult:
    __test__ = False  # not a pytest test class

    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    timed_out: bool = False
    collection_error: bool = False
    output: str = ""
    duration_s: float = 0.0

    @property
    def all_passed(self) -> bool:
        return (
            self.total > 0
            and self.passed == self.total
            and not self.failed
            and not self.errors
            and not self.timed_out
            and not self.collection_error
        )

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


# --------------------------------------------------------------------------------------------
# Pipeline results
# --------------------------------------------------------------------------------------------


@dataclass
class Attempt:
    index: int
    tier: Tier
    model: str
    stage: str  # "convert" | "repair"
    usage: TokenUsage
    validation: ValidationReport
    test: TestRunResult | None = None


@dataclass
class ModuleResult:
    path: str
    success: bool  # validation ok AND every generated test passed
    converted_source: str
    generated_tests: str
    routing: RoutingDecision
    final_tier: Tier
    escalated: bool
    attempts: list[Attempt] = field(default_factory=list)
    test: TestRunResult | None = None
    usage_by_model: dict[str, TokenUsage] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for u in self.usage_by_model.values():
            total = total + u
        return total
