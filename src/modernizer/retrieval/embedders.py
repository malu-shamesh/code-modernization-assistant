"""Deterministic offline embedder (the Vertex AI embedder lives in ``vertex_embedder``)."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import numpy as np

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPACES = re.compile(r"\s+")
_NGRAM = 4

# Relative weight of each feature family before sublinear tf scaling.
_W_TOKEN = 1.0
_W_SUBTOKEN = 0.6
_W_NGRAM = 0.4


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return (matrix / safe).astype(np.float32, copy=False)


@lru_cache(maxsize=200_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dim, sign


def _split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for chunk in token.split("_"):
        if chunk:
            parts.extend(p for p in _CAMEL_BOUNDARY.split(chunk) if p)
    return [p.lower() for p in parts]


class HashingEmbedder:
    """Signed feature hashing over identifiers, their sub-tokens and character 4-grams.

    Fully deterministic and stable across processes and platforms (blake2b, not ``hash()``),
    needs no network and no model download. Output rows are L2-normalised float32; text with
    no features maps to the zero vector.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(
        self, texts: Sequence[str], *, task: Literal["document", "query"] = "document"
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            self._fill(out[row], text)
        return l2_normalise(out)

    def _fill(self, row: np.ndarray, text: str) -> None:
        counts: Counter[tuple[str, float]] = Counter()
        lowered = _SPACES.sub(" ", text.lower()).strip()
        for match in _IDENT.finditer(text):
            token = match.group(0)
            counts[(f"w:{token.lower()}", _W_TOKEN)] += 1
            subtokens = _split_identifier(token)
            if len(subtokens) > 1:
                for sub in subtokens:
                    counts[(f"s:{sub}", _W_SUBTOKEN)] += 1
        for i in range(len(lowered) - _NGRAM + 1):
            counts[(f"c:{lowered[i : i + _NGRAM]}", _W_NGRAM)] += 1
        for (feature, weight), count in counts.items():
            index, sign = _bucket(feature, self.dim)
            row[index] += sign * weight * (1.0 + math.log(count))
