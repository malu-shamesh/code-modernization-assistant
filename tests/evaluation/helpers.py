"""Synthetic rows, results and corpora for tests that must not run a pipeline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from modernizer.evaluation import (
    EvaluationReport,
    ModuleRow,
    PolicyResult,
    PolicySpec,
    TokenCounts,
    compare_results,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "f77"


def make_row(
    module_id: str,
    *,
    difficulty: str = "easy",
    score: float = 0.3,
    routed: str = "open",
    final: str | None = None,
    generated: bool = True,
    reference: bool | None = True,
    tokens: int = 100,
    tokens_excl: int | None = None,
    cost: float = 0.0,
    forced: bool = False,
) -> ModuleRow:
    final = final or routed
    return ModuleRow(
        id=module_id,
        difficulty=difficulty,
        difficulty_score=score,
        routed_tier=routed,
        final_tier=final,
        escalated=routed != final,
        generated_success=generated,
        reference_success=reference,
        reference_passed=None if reference is None else (8 if reference else 3),
        reference_total=None if reference is None else 8,
        input_tokens=tokens * 3 // 4,
        output_tokens=tokens - tokens * 3 // 4,
        tokens_excluding_testgen=tokens if tokens_excl is None else tokens_excl,
        cost_usd=cost,
        cost_usd_excluding_testgen=cost,
        attempts=1,
        error=None,
        forced_frontier=forced,
    )


def make_result(
    name: str,
    rows: Sequence[ModuleRow],
    *,
    policy: str | None = None,
    escalate: bool = False,
    threshold: float | None = None,
    testgen_tokens: int = 0,
    testgen_cost: float = 0.0,
    tests_run: bool = True,
    target: float | None = None,
    planned: float | None = None,
) -> PolicyResult:
    total_tokens = sum(r.input_tokens + r.output_tokens for r in rows) + testgen_tokens
    total_cost = sum(r.cost_usd for r in rows) + testgen_cost
    return PolicyResult.from_rows(
        PolicySpec(name, policy or name, threshold, escalate, target),  # type: ignore[arg-type]
        rows,
        tokens=TokenCounts(total_tokens, 0),
        tokens_excluding_testgen=TokenCounts(total_tokens - testgen_tokens, 0),
        cost_usd_total=total_cost,
        cost_usd_excluding_testgen=total_cost - testgen_cost,
        tokens_by_tier={"open": TokenCounts(total_tokens, 0)},
        cost_usd_by_tier={"open": total_cost},
        threshold_used=threshold,
        generated_tests_run=tests_run,
        target_open_fraction=target,
        planned_open_share=planned,
    )


def make_report(*, dry_run: bool = False, n: int = 30) -> EvaluationReport:
    """Baseline passes everything; the cheaper policy loses two modules."""
    labels = ["easy", "medium", "hard"]
    ids = [f"m{i:02d}" for i in range(n)]
    base_rows = [
        make_row(i, difficulty=labels[k % 3], routed="frontier", tokens=100, cost=1.0)
        for k, i in enumerate(ids)
    ]
    cheap_rows = [
        make_row(
            i,
            difficulty=labels[k % 3],
            routed="open",
            final="open" if k >= 2 else "frontier",
            reference=k >= 2,
            generated=True,
            tokens=60,
            cost=0.4,
        )
        for k, i in enumerate(ids)
    ]
    baseline = make_result("always_frontier", base_rows, testgen_tokens=40, testgen_cost=0.5)
    cheap = make_result(
        "complexity",
        cheap_rows,
        policy="complexity",
        escalate=True,
        threshold=0.5,
        testgen_tokens=40,
        testgen_cost=0.5,
    )
    comparison = compare_results(baseline, cheap, dry_run=dry_run)
    by_difficulty = {label: sum(r.difficulty == label for r in base_rows) for label in labels}
    return EvaluationReport(
        dry_run=dry_run,
        policies=[baseline, cheap],
        comparisons=[comparison],
        corpus={"n": n, "by_difficulty": by_difficulty, "ids": ids},
        settings={"frontier_model": "f", "open_model": "o", "threshold": 0.5},
        baseline="always_frontier",
        run_tests=True,
        generated_at="2026-01-01T00:00:00+00:00",
    )


def write_case(
    root: Path,
    case_id: str,
    *,
    difficulty: str = "easy",
    skip: str | None = None,
    meta: dict[str, object] | None = None,
) -> Path:
    """Create a minimal benchmark directory; ``skip`` names a file to leave out."""
    directory = root / case_id
    directory.mkdir(parents=True)
    contents = {
        "legacy.f": "      PROGRAM X\n      END\n",
        "reference.py": "def main():\n    return None\n",
        "test_reference.py": "import target_module\n\n\ndef test_x():\n    assert True\n",
        "meta.json": json.dumps(
            meta
            if meta is not None
            else {"id": case_id, "difficulty": difficulty, "features": ["f"], "notes": "n"}
        ),
    }
    for name, text in contents.items():
        if name != skip:
            (directory / name).write_text(text, encoding="utf-8")
    return directory
