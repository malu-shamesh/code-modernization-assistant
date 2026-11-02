"""The statement engine and an independent tree-sitter parse agree on units and nesting."""

from __future__ import annotations

from pathlib import Path

import pytest

from modernizer.parsing import analyze, normalize_source
from modernizer.parsing.grammar import error_spans, tree_outline

from .helpers import fixed
from .test_golden import LEGACY

CORPUS = Path(__file__).resolve().parents[2] / "benchmarks" / "f77"
PROGRAMS = sorted(CORPUS.glob("*/legacy.f")) if CORPUS.is_dir() else []

TANGLED = fixed(
    """
    SUBROUTINE T(N, X)
    DIMENSION X(N)
    DO 30 I = 1, N
    DO 20 J = 1, N
    DO 10 K = 1, N
    IF (X(K) .GT. 0.0) THEN
    IF (X(J) .GT. 0.0) X(J) = 0.0
    END IF
    10 CONTINUE
    20 CONTINUE
    30 X(I) = 1.0
    GO TO (40, 50), N
    40 CONTINUE
    50 RETURN
    END
    PROGRAM P
    CALL T(3, X)
    END
    """
)


def outline_matches(source: str) -> tuple[list, list, int, int]:
    module = analyze("x.f", source)
    normalized = normalize_source(source)
    outline = tree_outline(normalized.text)
    tree_units = [
        (kind, normalized.line_map[first], normalized.end_line_map[last])
        for kind, first, last in outline.units
    ]
    mine = [(u.kind, u.start_line, u.end_line) for u in module.units]
    return tree_units, mine, outline.max_depth, module.metrics.max_nesting


@pytest.mark.parametrize("source", [LEGACY, TANGLED], ids=["golden", "tangled"])
def test_units_and_nesting_agree_on_handwritten_modules(source):
    tree_units, mine, tree_depth, my_depth = outline_matches(source)
    assert tree_units == mine
    assert tree_depth == my_depth
    assert error_spans(normalize_source(source).text) == []


def test_tangled_module_has_the_expected_depth():
    assert outline_matches(TANGLED)[3] == 4


@pytest.mark.skipif(not PROGRAMS, reason="benchmarks/f77 is not present")
@pytest.mark.parametrize("path", PROGRAMS, ids=lambda p: p.parent.name)
def test_units_and_nesting_agree_on_the_benchmark_programs(path):
    tree_units, mine, tree_depth, my_depth = outline_matches(path.read_text(encoding="utf-8"))
    assert tree_units == mine
    assert tree_depth == my_depth
