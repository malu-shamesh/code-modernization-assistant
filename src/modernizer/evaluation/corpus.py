"""The benchmark corpus: FORTRAN 77 modules with a reference Python port and held-out tests.

The corpus lives in ``benchmarks/f77``. Layout of ``<root>/<id>/``: ``legacy.f`` (the fixed-form
input), ``reference.py`` (a hand-written port that follows ``modernizer.conventions``),
``test_reference.py`` (tests that import ``target_module``; they are never shown to the model
under evaluation) and ``meta.json`` (``id``, ``difficulty``, ``features``, ``notes``). The
difficulty label is the corpus author's judgement and is independent of the routing policy's own
difficulty score. Sources are read with ``modernizer.fileio.read_source`` (UTF-8, then latin-1).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..fileio import read_source

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

LEGACY_FILE = "legacy.f"
REFERENCE_FILE = "reference.py"
TESTS_FILE = "test_reference.py"
META_FILE = "meta.json"
_REQUIRED_FILES = (LEGACY_FILE, REFERENCE_FILE, TESTS_FILE, META_FILE)


class CorpusError(ValueError):
    """The benchmark directory is missing, incomplete or inconsistent."""


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    difficulty: str  # "easy" | "medium" | "hard"
    features: tuple[str, ...]
    notes: str
    legacy_source: str  # the FORTRAN 77 text of legacy.f
    reference_source: str  # the hand-written Python port (reference.py)
    reference_tests: Path  # test_reference.py, never shown to the model


def _load_meta(case_dir: Path) -> dict[str, object]:
    path = case_dir / META_FILE
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise CorpusError(f"{path}: expected a JSON object")
    missing = [k for k in ("id", "difficulty", "features", "notes") if k not in meta]
    if missing:
        raise CorpusError(f"{path}: missing field(s) {', '.join(missing)}")
    if meta["id"] != case_dir.name:
        raise CorpusError(f"{path}: id {meta['id']!r} does not match directory {case_dir.name!r}")
    if meta["difficulty"] not in DIFFICULTIES:
        raise CorpusError(
            f"{path}: difficulty {meta['difficulty']!r} is not one of {', '.join(DIFFICULTIES)}"
        )
    features = meta["features"]
    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        raise CorpusError(f"{path}: features must be a list of strings")
    if not isinstance(meta["notes"], str):
        raise CorpusError(f"{path}: notes must be a string")
    return meta


def _load_case(case_dir: Path) -> BenchmarkCase:
    missing = [name for name in _REQUIRED_FILES if not (case_dir / name).is_file()]
    if missing:
        raise CorpusError(f"{case_dir}: missing file(s) {', '.join(missing)}")
    meta = _load_meta(case_dir)
    return BenchmarkCase(
        id=str(meta["id"]),
        difficulty=str(meta["difficulty"]),
        features=tuple(meta["features"]),  # type: ignore[arg-type]
        notes=str(meta["notes"]),
        legacy_source=read_source(case_dir / LEGACY_FILE),
        reference_source=read_source(case_dir / REFERENCE_FILE),
        reference_tests=case_dir / TESTS_FILE,
    )


def load_corpus(
    root: Path, *, ids: Sequence[str] | None = None, limit: int | None = None
) -> list[BenchmarkCase]:
    """Load the benchmark cases under ``root``, sorted by id.

    ``ids`` selects cases (an unknown id is an error, not a silent skip); ``limit`` then keeps
    the first ``limit`` of the sorted selection. Directories starting with "." or "_" (tool
    caches) are ignored. Only the selected cases are validated, so a broken case does not block
    a run that does not use it.
    """
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f"benchmark directory not found: {root}")
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")

    available = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )
    if ids is not None:
        unknown = sorted(set(ids) - set(available))
        if unknown:
            raise CorpusError(f"unknown benchmark id(s): {', '.join(unknown)}")
        selected = sorted(set(ids))
    else:
        selected = available
    if limit is not None:
        selected = selected[:limit]
    return [_load_case(root / name) for name in selected]


def corpus_summary(cases: Sequence[BenchmarkCase]) -> dict[str, object]:
    """``{"n": total, "by_difficulty": {"easy": k, ...}}`` with every difficulty present."""
    counts = dict.fromkeys(DIFFICULTIES, 0)
    for case in cases:
        counts[case.difficulty] = counts.get(case.difficulty, 0) + 1
    return {"n": len(cases), "by_difficulty": counts}


def split_calibration(
    cases: Sequence[BenchmarkCase], fraction: float = 0.5
) -> tuple[list[BenchmarkCase], list[BenchmarkCase]]:
    """Deterministic, stratified split into ``(calibration_cases, evaluation_cases)``.

    Cases are sorted by id and grouped by difficulty; inside each group they are dealt out in
    alternation (for 0.5: calibration, evaluation, calibration, ...) so both halves keep the
    difficulty mix. Calibrate a routing threshold on the first half and report results on the
    second; a threshold tuned and scored on the same modules is optimistic.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    share = Fraction(fraction).limit_denominator(10_000)
    groups: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in sorted(cases, key=lambda c: c.id):
        groups[case.difficulty].append(case)

    calibration: list[BenchmarkCase] = []
    evaluation: list[BenchmarkCase] = []
    for group in groups.values():
        taken = 0
        for index, case in enumerate(group, start=1):
            # Bresenham-style: the calibration share of the first `index` cases is ceil(index * f).
            target = -((-index * share.numerator) // share.denominator)
            if target > taken:
                calibration.append(case)
                taken += 1
            else:
                evaluation.append(case)
    return sorted(calibration, key=lambda c: c.id), sorted(evaluation, key=lambda c: c.id)
