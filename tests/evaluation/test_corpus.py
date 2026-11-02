from __future__ import annotations

import json
from collections import Counter

import pytest

from modernizer.evaluation import (
    BenchmarkCase,
    CorpusError,
    corpus_summary,
    load_corpus,
    split_calibration,
)

from .helpers import BENCHMARK_DIR, write_case

DIFFICULTIES = ("easy", "medium", "hard")


@pytest.fixture(scope="module")
def corpus() -> list[BenchmarkCase]:
    return load_corpus(BENCHMARK_DIR)


def test_default_corpus_root_is_the_fortran_corpus():
    assert BENCHMARK_DIR.parts[-2:] == ("benchmarks", "f77")


def test_real_corpus_loads_sixty_sorted_cases(corpus):
    assert len(corpus) == 60
    ids = [c.id for c in corpus]
    assert ids == sorted(ids)
    assert len(set(ids)) == 60
    assert {i[0] for i in ids} == set("abcdef")
    assert {c.difficulty for c in corpus} <= set(DIFFICULTIES)


def test_real_corpus_cases_are_complete(corpus):
    for case in corpus:
        assert case.legacy_source.strip()
        assert case.legacy_source == (BENCHMARK_DIR / case.id / "legacy.f").read_bytes().decode(
            "utf-8"
        )
        assert case.reference_source.strip()
        assert case.reference_tests == BENCHMARK_DIR / case.id / "test_reference.py"
        assert case.reference_tests.is_file()
        assert case.features
        assert isinstance(case.features, tuple)
        assert case.notes


def test_summary_counts_match_the_meta_files(corpus):
    labels = [
        json.loads((BENCHMARK_DIR / case.id / "meta.json").read_text("utf-8"))["difficulty"]
        for case in corpus
    ]
    expected = Counter(labels)
    summary = corpus_summary(corpus)
    assert summary["n"] == len(labels) == 60
    assert summary["by_difficulty"] == {label: expected.get(label, 0) for label in DIFFICULTIES}
    assert sum(summary["by_difficulty"].values()) == 60
    assert [c.difficulty for c in corpus] == labels


def test_legacy_sources_are_decoded_like_the_pipeline_reads_files(tmp_path):
    write_case(tmp_path, "latin")
    (tmp_path / "latin" / "legacy.f").write_bytes("C     café\n      END\n".encode("latin-1"))
    (case,) = load_corpus(tmp_path)
    assert "café" in case.legacy_source


def test_ids_and_limit_select_in_sorted_order():
    picked = load_corpus(BENCHMARK_DIR, ids=["b01_blas1_strided_axpy_dot", "a02_gcd_lcm_euclid"])
    assert [c.id for c in picked] == ["a02_gcd_lcm_euclid", "b01_blas1_strided_axpy_dot"]
    assert [c.id for c in load_corpus(BENCHMARK_DIR, limit=3)] == [
        "a01_horner_poly",
        "a02_gcd_lcm_euclid",
        "a03_factorial_fibonacci",
    ]
    assert load_corpus(BENCHMARK_DIR, limit=0) == []


def test_unknown_id_and_missing_directory_are_errors(tmp_path):
    with pytest.raises(CorpusError, match="nope"):
        load_corpus(BENCHMARK_DIR, ids=["nope"])
    with pytest.raises(CorpusError, match="not found"):
        load_corpus(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_corpus(BENCHMARK_DIR, limit=-1)


def test_cache_directories_are_ignored(tmp_path):
    write_case(tmp_path, "a01")
    (tmp_path / ".ruff_cache").mkdir()
    (tmp_path / "__pycache__").mkdir()
    assert [c.id for c in load_corpus(tmp_path)] == ["a01"]


@pytest.mark.parametrize("name", ["legacy.f", "reference.py", "test_reference.py", "meta.json"])
def test_missing_file_is_named_in_the_error(tmp_path, name):
    write_case(tmp_path, "a01", skip=name)
    with pytest.raises(CorpusError, match=name.replace(".", r"\.")):
        load_corpus(tmp_path)


def test_bad_metadata_is_rejected(tmp_path):
    write_case(tmp_path, "bad_difficulty", difficulty="trivial")
    write_case(
        tmp_path,
        "wrong_id",
        meta={"id": "other", "difficulty": "easy", "features": [], "notes": ""},
    )
    write_case(tmp_path, "no_notes", meta={"id": "no_notes", "difficulty": "easy", "features": []})
    for case_id, message in (
        ("bad_difficulty", "difficulty"),
        ("wrong_id", "does not match"),
        ("no_notes", "notes"),
    ):
        with pytest.raises(CorpusError, match=message):
            load_corpus(tmp_path, ids=[case_id])


def test_only_selected_cases_are_validated(tmp_path):
    write_case(tmp_path, "good")
    write_case(tmp_path, "zbroken", skip="legacy.f")
    assert [c.id for c in load_corpus(tmp_path, ids=["good"])] == ["good"]
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_split_is_deterministic_stratified_and_a_partition(corpus):
    calibration, evaluation = split_calibration(corpus)
    assert split_calibration(list(reversed(corpus))) == (calibration, evaluation)
    assert {c.id for c in calibration}.isdisjoint(c.id for c in evaluation)
    assert len(calibration) + len(evaluation) == len(corpus)
    assert [c.id for c in calibration] == sorted(c.id for c in calibration)
    for label in DIFFICULTIES:
        total = sum(c.difficulty == label for c in corpus)
        in_calibration = sum(c.difficulty == label for c in calibration)
        assert abs(in_calibration - total / 2) <= 0.5


def test_split_alternates_inside_each_difficulty(tmp_path):
    for i in range(4):
        write_case(tmp_path, f"e{i}", difficulty="easy")
    calibration, evaluation = split_calibration(load_corpus(tmp_path))
    assert [c.id for c in calibration] == ["e0", "e2"]
    assert [c.id for c in evaluation] == ["e1", "e3"]


def test_split_fraction_and_validation(corpus):
    calibration, _ = split_calibration(corpus, 0.25)
    assert abs(len(calibration) - 0.25 * len(corpus)) <= 3
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_calibration(corpus, bad)
