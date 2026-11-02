"""Request and response models of the HTTP API (Fortran 77 sources in, Python modules out)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..fileio import SOURCE_SUFFIXES
from ..models import ModuleResult, ParsedModule, RoutingDecision
from ..routing import RoutingPolicyName

MAX_PATH_CHARS = 255
MAX_SNIPPET_CHARS = 300
MAX_ERROR_CHARS = 1000
TEST_GENERATION_FAILED = "test generation failed"  # prefix of the pipeline's error text
SOURCE_SUFFIX_LIST = ", ".join(SOURCE_SUFFIXES)


def validate_path(path: str) -> str:
    """Accept only plain relative posix paths of Fortran source files; never normalise.

    The extension (``.f``, ``.for``, ``.f77`` or ``.ftn``, any letter case) must follow a
    non-empty file name. Only the name is checked, the source text is not inspected here.

    Rejecting instead of cleaning up means what the client sent is what is stored and reported.
    The service never writes these paths to disk, but they reach logs, metric attributes and
    BigQuery rows, so they must be tame.
    """
    if not path or len(path) > MAX_PATH_CHARS:
        raise ValueError(f"path must have 1 to {MAX_PATH_CHARS} characters")
    # isprintable also rejects Unicode control, line-separator and format characters (for example
    # U+0085 or the bidi overrides), which ord() < 32 would let through into logs and tables.
    if not path.isprintable():
        raise ValueError("path must not contain control or non-printable characters")
    if "\\" in path:
        raise ValueError("path must use '/' separators, backslashes are not allowed")
    if ":" in path:
        raise ValueError("path must not contain ':' (drive letters are not allowed)")
    if path.startswith("/"):
        raise ValueError("path must be relative")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("path must not contain empty, '.' or '..' segments")
    name = path.rsplit("/", 1)[-1].lower()
    if not any(name.endswith(suffix) and len(name) > len(suffix) for suffix in SOURCE_SUFFIXES):
        raise ValueError(f"path must name a Fortran source file ({SOURCE_SUFFIX_LIST})")
    return path


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description=f"Relative posix path of a Fortran file ({SOURCE_SUFFIX_LIST}).")
    source: str = Field(description="Fortran 77 source text, fixed form (free form is detected).")

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        return validate_path(value)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[SourceFile] = Field(
        min_length=1, description="Fortran modules, each with a unique path."
    )

    @model_validator(mode="after")
    def _no_duplicate_paths(self) -> AnalyzeRequest:
        seen: set[str] = set()
        duplicates: list[str] = []
        for file in self.files:
            if file.path in seen and file.path not in duplicates:
                duplicates.append(file.path)
            seen.add(file.path)
        if duplicates:
            raise ValueError("duplicate paths: " + ", ".join(duplicates[:5]))
        return self

    def pairs(self) -> list[tuple[str, str]]:
        return [(f.path, f.source) for f in self.files]


class ModernizeRequest(AnalyzeRequest):
    policy: RoutingPolicyName | None = Field(
        default=None, description="Routing policy for this request; the service default if unset."
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Difficulty threshold of the complexity policy; the service default if unset.",
    )
    open_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Target share of THIS request's modules for the open model (complexity policy only). "
            "The threshold is calibrated on the batch, so it takes precedence over `threshold`. "
            "The service default (MODERNIZER_OPEN_FRACTION) applies if unset. The share that "
            "was planned is reported as `planned_open_share` in the summary."
        ),
    )
    run_tests: bool = Field(
        default=True,
        description="Generate and run pytest tests; if false a module only needs to"
        " pass static validation.",
    )
    include_tests: bool = Field(
        default=False, description="Return the generated pytest source for every module."
    )


# --------------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------------


def _client_error(error: str | None) -> str | None:
    """The pipeline's error text, shortened and flattened to printable characters.

    The pipeline reports generic messages (an exception type, never its text), with one
    exception: a failed test generation appends the text of the error, which can be backend
    output (project names, addresses). That detail is dropped here; the field is also kept tame
    for logs and tables whatever it is given.
    """
    if not error:
        return None
    if error.startswith(TEST_GENERATION_FAILED):
        return TEST_GENERATION_FAILED
    return "".join(c if c.isprintable() else " " for c in error[:MAX_ERROR_CHARS])


class ValidationIssueOut(BaseModel):
    code: str
    message: str
    line: int | None = None


class PytestCountsOut(BaseModel):
    passed: int
    failed: int
    errors: int
    total: int


class ModuleResultOut(BaseModel):
    path: str
    success: bool
    converted_source: str  # the Python (numpy) module; empty if no candidate was produced
    routed_tier: str
    final_tier: str
    escalated: bool
    difficulty: float
    attempts: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None
    validation_issues: list[ValidationIssueOut]
    tests: PytestCountsOut | None
    # Left unset (and therefore omitted from the response) unless the client asked for it.
    generated_tests: str | None = None

    @classmethod
    def from_result(cls, result: ModuleResult, *, include_tests: bool) -> ModuleResultOut:
        usage = result.total_usage
        issues = result.attempts[-1].validation.issues if result.attempts else []
        extra: dict[str, Any] = {"generated_tests": result.generated_tests} if include_tests else {}
        return cls(
            path=result.path,
            success=result.success,
            converted_source=result.converted_source,
            routed_tier=result.routing.tier.value,
            final_tier=result.final_tier.value,
            escalated=result.escalated,
            difficulty=result.routing.difficulty,
            attempts=len(result.attempts),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=result.cost_usd,
            error=_client_error(result.error),
            validation_issues=[
                ValidationIssueOut(code=i.code, message=i.message, line=i.line) for i in issues
            ],
            tests=(
                PytestCountsOut(
                    passed=result.test.passed,
                    failed=result.test.failed,
                    errors=result.test.errors,
                    total=result.test.total,
                )
                if result.test is not None
                else None
            ),
            **extra,
        )


class ModernizeResponse(BaseModel):
    run_id: str
    # Realised numbers from the results (see ``report.RunSummary``) plus ``planned_open_share``:
    # the open share the batch calibration planned, or null when no calibration took place.
    summary: dict[str, float | int | None]
    results: list[ModuleResultOut]


class FeatureOut(BaseModel):
    id: str
    line: int
    snippet: str


class DifficultyOut(BaseModel):
    score: float
    reasons: list[str]


class RoutingOut(BaseModel):
    tier: str
    model: str
    difficulty: float
    threshold: float | None
    policy: str
    reasons: list[str]

    @classmethod
    def from_decision(cls, decision: RoutingDecision) -> RoutingOut:
        return cls(
            tier=decision.tier.value,
            model=decision.model,
            difficulty=decision.difficulty,
            threshold=decision.threshold,
            policy=decision.policy,
            reasons=list(decision.reasons),
        )


class FileAnalysis(BaseModel):
    path: str
    error: str | None = None
    features: list[FeatureOut] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    difficulty: DifficultyOut | None = None
    routing: RoutingOut | None = None

    @classmethod
    def from_parsed(
        cls,
        parsed: ParsedModule,
        *,
        score: float,
        reasons: list[str],
        decision: RoutingDecision,
    ) -> FileAnalysis:
        return cls(
            path=parsed.path,
            features=[
                FeatureOut(id=h.feature_id, line=h.line, snippet=h.snippet[:MAX_SNIPPET_CHARS])
                for h in parsed.hits
            ],
            metrics=asdict(parsed.metrics),
            difficulty=DifficultyOut(score=score, reasons=reasons),
            routing=RoutingOut.from_decision(decision),
        )


class AnalyzeResponse(BaseModel):
    files: list[FileAnalysis]
