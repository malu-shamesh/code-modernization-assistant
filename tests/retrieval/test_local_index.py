from __future__ import annotations

import threading

import numpy as np
import pytest

from modernizer.retrieval import LocalVectorIndex


def unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_query_returns_best_first():
    index = LocalVectorIndex(3)
    index.upsert(["x", "y", "xy"], np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0)]))
    result = index.query(unit(1, 0.1, 0), 3)
    assert [i for i, _ in result] == ["x", "xy", "y"]
    assert result[0][1] > result[1][1] > result[2][1]
    assert all(isinstance(score, float) for _, score in result)


def test_k_limits_and_non_positive_k():
    index = LocalVectorIndex(2)
    index.upsert(["a", "b", "c"], np.stack([unit(1, 0), unit(1, 1), unit(0, 1)]))
    assert len(index.query(unit(1, 0), 2)) == 2
    assert len(index.query(unit(1, 0), 10)) == 3
    assert index.query(unit(1, 0), 0) == []


def test_empty_index_returns_nothing():
    assert LocalVectorIndex(2).query(unit(1, 0), 5) == []


def test_upsert_replaces_existing_ids_in_place():
    index = LocalVectorIndex(2)
    index.upsert(["a", "b"], np.stack([unit(1, 0), unit(0, 1)]))
    index.upsert(["a"], unit(0, 1)[None, :])
    assert len(index) == 2
    scores = dict(index.query(unit(0, 1), 2))
    assert scores["a"] == pytest.approx(1.0)


def test_duplicate_ids_within_one_batch_keep_the_last_vector():
    index = LocalVectorIndex(2)
    index.upsert(["a", "a", "b"], np.stack([unit(1, 0), unit(0, 1), unit(1, 1)]))
    assert len(index) == 2
    assert dict(index.query(unit(0, 1), 2))["a"] == pytest.approx(1.0)


def test_ties_are_broken_by_insertion_order():
    index = LocalVectorIndex(2)
    index.upsert(["first", "second", "third"], np.stack([unit(1, 0)] * 3))
    assert [i for i, _ in index.query(unit(1, 0), 3)] == ["first", "second", "third"]


def test_shape_validation():
    index = LocalVectorIndex(3)
    with pytest.raises(ValueError, match="shape"):
        index.upsert(["a"], np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="ids"):
        index.upsert(["a", "b"], np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        index.query(np.zeros(5, dtype=np.float32), 1)
    with pytest.raises(ValueError, match="NaN"):
        index.upsert(["a"], np.full((1, 3), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        LocalVectorIndex(0)


def test_save_load_round_trip(tmp_path):
    index = LocalVectorIndex(4)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(5, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index.upsert([f"id-{i}" for i in range(5)], vectors)
    path = tmp_path / "nested" / "index.npz"
    index.save(path)
    assert path.exists() and not path.with_name(path.name + ".tmp").exists()

    loaded = LocalVectorIndex.load(path)
    assert loaded.dim == 4 and loaded.ids == index.ids
    query = vectors[2]
    assert loaded.query(query, 5) == index.query(query, 5)


def test_save_load_empty_index(tmp_path):
    path = tmp_path / "empty.npz"
    LocalVectorIndex(7).save(path)
    loaded = LocalVectorIndex.load(path)
    assert loaded.dim == 7 and len(loaded) == 0


def test_concurrent_upserts_and_queries_are_consistent():
    index = LocalVectorIndex(8)
    errors: list[BaseException] = []

    def writer(worker: int) -> None:
        try:
            rng = np.random.default_rng(worker)
            for i in range(40):
                vec = rng.normal(size=(1, 8)).astype(np.float32)
                index.upsert([f"w{worker}-{i}"], vec / np.linalg.norm(vec))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(100):
                index.query(unit(*([1.0] * 8)), 5)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(index) == 160
    assert len(index.ids) == len(set(index.ids))
