"""Every benchmark module under benchmarks/f77: it parses cleanly, its meta.json feature claims
are detected, and its hand-written reference port satisfies the calling convention."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from modernizer.parsing import analyze, validate_conversion

BENCH_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "f77"
BENCH_DIRS = (
    sorted(p for p in BENCH_ROOT.iterdir() if (p / "legacy.f").is_file())
    if BENCH_ROOT.is_dir()
    else []
)
# Coarse regression guard only: analysis of a few-KB module takes a small fraction of this.
MAX_ANALYZE_SECONDS = 2.0
MAX_ALL_MODULES_SECONDS = 20.0

pytestmark = pytest.mark.skipif(
    not BENCH_DIRS, reason=f"no benchmark modules found in {BENCH_ROOT}"
)

each_benchmark = pytest.mark.parametrize("bench", BENCH_DIRS, ids=[p.name for p in BENCH_DIRS])


def _legacy(bench: Path):
    return analyze(str(bench / "legacy.f"), (bench / "legacy.f").read_text(encoding="utf-8"))


@each_benchmark
def test_legacy_has_no_syntax_errors(bench: Path):
    assert not _legacy(bench).metrics.has_syntax_errors


@each_benchmark
def test_claimed_features_are_detected(bench: Path):
    meta = json.loads((bench / "meta.json").read_text(encoding="utf-8"))
    claimed = set(meta["features"])
    detected = {hit.feature_id for hit in _legacy(bench).hits}
    missing = sorted(claimed - detected)
    assert not missing, (
        f"{bench.name}: meta.json claims features the detector did not find: {missing} "
        f"(detected: {sorted(detected)})"
    )


@each_benchmark
def test_reference_port_follows_the_calling_convention(bench: Path):
    reference = (bench / "reference.py").read_text(encoding="utf-8")
    report = validate_conversion(_legacy(bench), reference)
    problems = [f"{i.code} (line {i.line}): {i.message}" for i in report.issues]
    assert report.ok, f"{bench.name}: reference.py fails validation:\n" + "\n".join(problems)


def _best_of(runs: int, func) -> float:
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def test_analyzing_the_largest_benchmark_stays_fast():
    largest = max(BENCH_DIRS, key=lambda p: (p / "legacy.f").stat().st_size)
    source = (largest / "legacy.f").read_text(encoding="utf-8")
    assert _best_of(3, lambda: analyze("legacy.f", source)) < MAX_ANALYZE_SECONDS


def test_analyzing_every_benchmark_as_one_long_module_stays_fast():
    source = "\n".join((p / "legacy.f").read_text(encoding="utf-8") for p in BENCH_DIRS)
    assert _best_of(1, lambda: analyze("all.f", source)) < MAX_ALL_MODULES_SECONDS
