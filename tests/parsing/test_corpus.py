"""The shared FORTRAN 77 benchmark programs analyse cleanly (skipped when they are absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

from modernizer.parsing import analyze

CORPUS = Path(__file__).resolve().parents[2] / "benchmarks" / "f77"
PROGRAMS = sorted(CORPUS.glob("*/legacy.f")) if CORPUS.is_dir() else []


@pytest.mark.skipif(not PROGRAMS, reason="benchmarks/f77 is not present")
@pytest.mark.parametrize("path", PROGRAMS, ids=lambda p: p.parent.name)
def test_benchmark_program_analyses_without_syntax_errors(path):
    module = analyze(str(path), path.read_text(encoding="utf-8"))
    assert module.units
    assert not module.metrics.has_syntax_errors
    assert module.hits == sorted(module.hits, key=lambda h: (h.line, h.feature_id, h.end_line))
    assert "fixed_form_source" in module.feature_ids
    for unit in module.units:
        assert 1 <= unit.start_line <= unit.end_line <= module.metrics.loc
