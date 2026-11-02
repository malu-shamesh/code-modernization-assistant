"""In-memory brute-force vector index, used offline and in tests."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class LocalVectorIndex:
    """Exact nearest-neighbour search by dot product.

    Vectors are expected to be L2-normalised already (every ``Embedder`` guarantees this), so
    the dot product equals cosine similarity. All methods are thread-safe.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._ids: list[str] = []
        self._row_of: dict[str, int] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    @property
    def ids(self) -> list[str]:
        with self._lock:
            return list(self._ids)

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (n, {self.dim}), got {matrix.shape}")
        if matrix.shape[0] != len(ids):
            raise ValueError(f"got {len(ids)} ids but {matrix.shape[0]} vectors")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors contain NaN or infinite values")
        with self._lock:
            stored = self._matrix.shape[0]
            new_rows: list[np.ndarray] = []
            for item_id, vector in zip(ids, matrix, strict=True):
                row = self._row_of.get(item_id)
                if row is None:
                    self._row_of[item_id] = len(self._ids)
                    self._ids.append(item_id)
                    new_rows.append(vector)
                elif row < stored:
                    self._matrix[row] = vector
                else:
                    new_rows[row - stored] = vector
            if new_rows:
                self._matrix = np.vstack([self._matrix, np.asarray(new_rows, dtype=np.float32)])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, similarity)`` pairs, best first, ties in insertion order."""
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query must have {self.dim} dimensions, got {query.shape[0]}")
        if k <= 0:
            return []
        with self._lock:
            if not self._ids:
                return []
            scores = self._matrix @ query
            order = np.lexsort((np.arange(len(scores)), -scores))[:k]
            return [(self._ids[i], float(scores[i])) for i in order]

    def save(self, path: str | Path) -> None:
        """Write the index to ``path`` (npz). Written atomically via a temporary file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with self._lock:
            ids = np.asarray(self._ids, dtype=np.str_)
            matrix = self._matrix.copy()
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, ids=ids, vectors=matrix, dim=np.int64(self.dim))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> LocalVectorIndex:
        with np.load(Path(path), allow_pickle=False) as data:
            ids = [str(i) for i in data["ids"].tolist()]
            vectors = data["vectors"]
            dim = int(data["dim"])
        index = cls(dim)
        if ids:
            index.upsert(ids, vectors)
        return index
