"""Result types of an evaluation run, with lossless JSON (de)serialisation.

Counts in ``PolicyResult`` are always derived from its rows (``PolicyResult.from_rows``), so a
summary can never disagree with the per-module data. Token and cost totals are passed in
separately because they come from the OpenTelemetry meter, not from the rows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .policies import PolicySpec
from .render import DRY_RUN_WARNING, render_markdown
from .stats import wilson_interval

SCHEMA_VERSION = 2

OPEN = "open"
FRONTIER = "frontier"
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


@dataclass(frozen=True)
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {"input": self.input_tokens, "output": self.output_tokens, "total": self.total}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenCounts:
        return cls(int(data["input"]), int(data["output"]))


def _tokens_map(data: Mapping[str, Mapping[str, Any]]) -> dict[str, TokenCounts]:
    return {key: TokenCounts.from_dict(value) for key, value in data.items()}


@dataclass(frozen=True)
class ModuleRow:
    id: str
    difficulty: str  # the corpus label, not the router's score
    difficulty_score: float  # the router's score for the legacy source
    routed_tier: str
    final_tier: str
    escalated: bool
    generated_success: bool  # the pipeline's own verdict (LLM-written tests)
    reference_success: bool | None  # all held-out tests passed; None when not measured
    reference_passed: int | None
    reference_total: int | None
    input_tokens: int
    output_tokens: int
    tokens_excluding_testgen: int
    cost_usd: float
    cost_usd_excluding_testgen: float
    attempts: int
    error: str | None
    forced_frontier: bool = False  # the feature rule sent it to the frontier (complexity policy)

    @property
    def reference_fraction(self) -> float | None:
        if self.reference_passed is None or self.reference_total is None:
            return None
        return self.reference_passed / self.reference_total if self.reference_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModuleRow:
        return cls(**data)


@dataclass(frozen=True)
class PolicyResult:
    spec: PolicySpec
    n: int
    generated_pass: int
    reference_pass: int | None  # None when held-out scoring was skipped
    reference_test_fraction: float | None  # mean over modules of passed / total held-out tests
    routed_open: int
    final_open_successes: int  # finished on the open tier and passed the generated tests
    final_open_reference_successes: int | None  # ... and passed every held-out test
    escalated: int
    tokens: TokenCounts
    tokens_by_tier: dict[str, TokenCounts]
    tokens_by_stage: dict[str, TokenCounts]
    tokens_excluding_testgen: TokenCounts
    cost_usd_total: float
    cost_usd_by_tier: dict[str, float]
    cost_usd_by_stage: dict[str, float]
    cost_usd_excluding_testgen: float
    rows: list[ModuleRow]
    threshold_used: float | None = None  # only for the complexity policy
    generated_tests_run: bool = True  # False: "generated_pass" is static validation only
    # The routing parameter that was asked for (a policy input, not a result), the share the
    # router planned for the whole batch before any model call, and the counts that explain
    # the gap between them.
    target_open_fraction: float | None = None
    planned_open_share: float | None = None  # None unless the threshold was calibrated
    forced_frontier: int = 0  # modules the feature rule sent to the frontier
    final_open: int = 0  # modules that ended on the open tier (after escalation)

    @classmethod
    def from_rows(
        cls,
        spec: PolicySpec,
        rows: Sequence[ModuleRow],
        *,
        tokens: TokenCounts,
        tokens_excluding_testgen: TokenCounts,
        cost_usd_total: float,
        cost_usd_excluding_testgen: float,
        tokens_by_tier: Mapping[str, TokenCounts] | None = None,
        tokens_by_stage: Mapping[str, TokenCounts] | None = None,
        cost_usd_by_tier: Mapping[str, float] | None = None,
        cost_usd_by_stage: Mapping[str, float] | None = None,
        threshold_used: float | None = None,
        generated_tests_run: bool = True,
        target_open_fraction: float | None = None,
        planned_open_share: float | None = None,
    ) -> PolicyResult:
        rows = list(rows)
        measured = bool(rows) and all(r.reference_success is not None for r in rows)
        fractions = [r.reference_fraction or 0.0 for r in rows]
        return cls(
            spec=spec,
            n=len(rows),
            generated_pass=sum(r.generated_success for r in rows),
            reference_pass=sum(bool(r.reference_success) for r in rows) if measured else None,
            reference_test_fraction=sum(fractions) / len(rows) if measured else None,
            routed_open=sum(r.routed_tier == OPEN for r in rows),
            final_open_successes=sum(r.final_tier == OPEN and r.generated_success for r in rows),
            final_open_reference_successes=(
                sum(r.final_tier == OPEN and bool(r.reference_success) for r in rows)
                if measured
                else None
            ),
            escalated=sum(r.escalated for r in rows),
            tokens=tokens,
            tokens_by_tier=dict(tokens_by_tier or {}),
            tokens_by_stage=dict(tokens_by_stage or {}),
            tokens_excluding_testgen=tokens_excluding_testgen,
            cost_usd_total=cost_usd_total,
            cost_usd_by_tier=dict(cost_usd_by_tier or {}),
            cost_usd_by_stage=dict(cost_usd_by_stage or {}),
            cost_usd_excluding_testgen=cost_usd_excluding_testgen,
            rows=rows,
            threshold_used=threshold_used,
            generated_tests_run=generated_tests_run,
            target_open_fraction=target_open_fraction,
            planned_open_share=planned_open_share,
            forced_frontier=sum(r.forced_frontier for r in rows),
            final_open=sum(r.final_tier == OPEN for r in rows),
        )

    @property
    def generated_pass_rate(self) -> float:
        return self.generated_pass / self.n if self.n else 0.0

    @property
    def reference_pass_rate(self) -> float | None:
        if self.reference_pass is None:
            return None
        return self.reference_pass / self.n if self.n else 0.0

    @property
    def open_fraction(self) -> float:
        """Share of modules the router sent to the open model (before escalation)."""
        return self.routed_open / self.n if self.n else 0.0

    @property
    def final_open_share(self) -> float:
        """Share of modules that ended on the open tier; escalations lower it."""
        return self.final_open / self.n if self.n else 0.0

    def generated_interval(self) -> tuple[float, float]:
        return wilson_interval(self.generated_pass, self.n)

    def reference_interval(self) -> tuple[float, float] | None:
        if self.reference_pass is None:
            return None
        return wilson_interval(self.reference_pass, self.n)

    def by_difficulty(self) -> dict[str, dict[str, int | None]]:
        """Counts per corpus difficulty label, ordered easy, medium, hard."""
        out: dict[str, dict[str, int | None]] = {}
        labels = sorted({r.difficulty for r in self.rows}, key=difficulty_sort_key)
        for label in labels:
            group = [r for r in self.rows if r.difficulty == label]
            measured = all(r.reference_success is not None for r in group)
            out[label] = {
                "n": len(group),
                "generated_pass": sum(r.generated_success for r in group),
                "reference_pass": (
                    sum(bool(r.reference_success) for r in group) if measured else None
                ),
                "routed_open": sum(r.routed_tier == OPEN for r in group),
                "escalated": sum(r.escalated for r in group),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "n": self.n,
            "generated_pass": self.generated_pass,
            "reference_pass": self.reference_pass,
            "reference_test_fraction": self.reference_test_fraction,
            "routed_open": self.routed_open,
            "final_open_successes": self.final_open_successes,
            "final_open_reference_successes": self.final_open_reference_successes,
            "escalated": self.escalated,
            "tokens": self.tokens.to_dict(),
            "tokens_by_tier": {k: v.to_dict() for k, v in self.tokens_by_tier.items()},
            "tokens_by_stage": {k: v.to_dict() for k, v in self.tokens_by_stage.items()},
            "tokens_excluding_testgen": self.tokens_excluding_testgen.to_dict(),
            "cost_usd_total": self.cost_usd_total,
            "cost_usd_by_tier": dict(self.cost_usd_by_tier),
            "cost_usd_by_stage": dict(self.cost_usd_by_stage),
            "cost_usd_excluding_testgen": self.cost_usd_excluding_testgen,
            "threshold_used": self.threshold_used,
            "generated_tests_run": self.generated_tests_run,
            "target_open_fraction": self.target_open_fraction,
            "planned_open_share": self.planned_open_share,
            "forced_frontier": self.forced_frontier,
            "final_open": self.final_open,
            "rows": [r.to_dict() for r in self.rows],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyResult:
        return cls(
            spec=PolicySpec.from_dict(data["spec"]),
            n=data["n"],
            generated_pass=data["generated_pass"],
            reference_pass=data["reference_pass"],
            reference_test_fraction=data["reference_test_fraction"],
            routed_open=data["routed_open"],
            final_open_successes=data["final_open_successes"],
            final_open_reference_successes=data["final_open_reference_successes"],
            escalated=data["escalated"],
            tokens=TokenCounts.from_dict(data["tokens"]),
            tokens_by_tier=_tokens_map(data["tokens_by_tier"]),
            tokens_by_stage=_tokens_map(data["tokens_by_stage"]),
            tokens_excluding_testgen=TokenCounts.from_dict(data["tokens_excluding_testgen"]),
            cost_usd_total=data["cost_usd_total"],
            cost_usd_by_tier=dict(data["cost_usd_by_tier"]),
            cost_usd_by_stage=dict(data["cost_usd_by_stage"]),
            cost_usd_excluding_testgen=data["cost_usd_excluding_testgen"],
            rows=[ModuleRow.from_dict(r) for r in data["rows"]],
            threshold_used=data["threshold_used"],
            generated_tests_run=data["generated_tests_run"],
            target_open_fraction=data["target_open_fraction"],
            planned_open_share=data["planned_open_share"],
            forced_frontier=data["forced_frontier"],
            final_open=data["final_open"],
        )


def difficulty_sort_key(label: str) -> tuple[int, str]:
    return DIFFICULTY_ORDER.get(label, len(DIFFICULTY_ORDER)), label


@dataclass(frozen=True)
class PassComparison:
    """Paired comparison of one pass metric between a policy and the baseline."""

    metric: str  # "generated" | "reference"
    n: int
    baseline_pass: int
    policy_pass: int
    baseline_rate: float
    policy_rate: float
    delta: float  # policy minus baseline, as a fraction (not percentage points)
    baseline_interval: tuple[float, float]  # 95% Wilson
    policy_interval: tuple[float, float]
    only_baseline_passes: int  # discordant: baseline passes, policy fails
    only_policy_passes: int  # discordant: policy passes, baseline fails
    p_value: float  # exact two-sided sign test on the discordant pairs

    @property
    def discordant(self) -> int:
        return self.only_baseline_passes + self.only_policy_passes

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["baseline_interval"] = list(self.baseline_interval)
        data["policy_interval"] = list(self.policy_interval)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PassComparison:
        fields = dict(data)
        fields["baseline_interval"] = tuple(fields["baseline_interval"])
        fields["policy_interval"] = tuple(fields["policy_interval"])
        return cls(**fields)


@dataclass(frozen=True)
class Comparison:
    """A policy against the baseline policy on the same modules."""

    baseline: str
    policy: str
    n: int
    token_reduction_pct: float | None  # all calls; negative means more tokens than baseline
    token_reduction_pct_excluding_testgen: float | None
    cost_reduction_pct: float | None
    cost_reduction_pct_excluding_testgen: float | None
    generated: PassComparison
    reference: PassComparison | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "policy": self.policy,
            "n": self.n,
            "token_reduction_pct": self.token_reduction_pct,
            "token_reduction_pct_excluding_testgen": self.token_reduction_pct_excluding_testgen,
            "cost_reduction_pct": self.cost_reduction_pct,
            "cost_reduction_pct_excluding_testgen": self.cost_reduction_pct_excluding_testgen,
            "generated": self.generated.to_dict(),
            "reference": self.reference.to_dict() if self.reference else None,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Comparison:
        reference = data["reference"]
        return cls(
            baseline=data["baseline"],
            policy=data["policy"],
            n=data["n"],
            token_reduction_pct=data["token_reduction_pct"],
            token_reduction_pct_excluding_testgen=data["token_reduction_pct_excluding_testgen"],
            cost_reduction_pct=data["cost_reduction_pct"],
            cost_reduction_pct_excluding_testgen=data["cost_reduction_pct_excluding_testgen"],
            generated=PassComparison.from_dict(data["generated"]),
            reference=PassComparison.from_dict(reference) if reference else None,
            verdict=data["verdict"],
        )


@dataclass(frozen=True)
class EvaluationReport:
    dry_run: bool
    policies: list[PolicyResult]
    comparisons: list[Comparison]
    corpus: dict[str, Any]  # {"n": ..., "by_difficulty": {...}, "ids": [...]}
    settings: dict[str, Any]  # snapshot of model names, threshold, pipeline knobs
    baseline: str | None = None
    run_tests: bool = True
    generated_at: str = ""
    notes: list[str] = field(default_factory=list)  # extra caveats supplied by the caller

    def policy(self, name: str) -> PolicyResult:
        for result in self.policies:
            if result.spec.name == name:
                return result
        raise KeyError(f"no policy named {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dry_run": self.dry_run,
            "dry_run_warning": DRY_RUN_WARNING if self.dry_run else None,
            "generated_at": self.generated_at,
            "run_tests": self.run_tests,
            "baseline": self.baseline,
            "corpus": self.corpus,
            "settings": self.settings,
            "notes": list(self.notes),
            "policies": [p.to_dict() for p in self.policies],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationReport:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported report schema version {version!r}")
        return cls(
            dry_run=bool(data["dry_run"]),
            policies=[PolicyResult.from_dict(p) for p in data["policies"]],
            comparisons=[Comparison.from_dict(c) for c in data["comparisons"]],
            corpus=dict(data["corpus"]),
            settings=dict(data["settings"]),
            baseline=data["baseline"],
            run_tests=bool(data["run_tests"]),
            generated_at=data["generated_at"],
            notes=list(data["notes"]),
        )

    @classmethod
    def from_json(cls, text: str) -> EvaluationReport:
        return cls.from_dict(json.loads(text))

    def to_markdown(self) -> str:
        return render_markdown(self)
