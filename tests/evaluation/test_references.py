"""Every hand-written reference port must pass the project's own static validator."""

from __future__ import annotations

import pytest

from modernizer.evaluation import load_corpus
from modernizer.parsing import analyze, validate_conversion

from .helpers import BENCHMARK_DIR


def _params() -> list:
    params = []
    for path in sorted(p for p in BENCHMARK_DIR.iterdir() if p.is_dir() and p.name[0] not in "._"):
        params.append(pytest.param(path.name, id=path.name))
    return params


@pytest.mark.parametrize("case_id", _params())
def test_reference_passes_static_validation_against_its_own_legacy_source(case_id):
    (case,) = load_corpus(BENCHMARK_DIR, ids=[case_id])
    report = validate_conversion(analyze(f"{case.id}.f", case.legacy_source), case.reference_source)
    assert report.ok, report.feedback()
