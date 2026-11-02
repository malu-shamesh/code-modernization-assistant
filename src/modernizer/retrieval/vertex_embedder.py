"""Vertex AI text embeddings through the ``google-genai`` SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from .embedders import l2_normalise

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only without the gcp extra
    genai = None
    genai_types = None

_TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


class VertexEmbedder:
    """Vertex AI text embeddings through the ``google-genai`` SDK.

    ``client`` may be any object exposing ``models.embed_content(model=, contents=, config=)``
    (a ``genai.Client`` or a test fake). When omitted, a Vertex-backed client is created with
    Application Default Credentials.
    """

    def __init__(
        self,
        model: str = "text-embedding-005",
        *,
        project: str | None = None,
        location: str = "us-central1",
        dim: int = 768,
        client: Any = None,
        batch_size: int = 100,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if client is None:
            if genai is None:
                raise ImportError(
                    "VertexEmbedder needs google-genai: pip install 'code-modernization-assistant[gcp]'"
                )
            client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = client

    def embed(
        self, texts: Sequence[str], *, task: Literal["document", "query"] = "document"
    ) -> np.ndarray:
        if task not in _TASK_TYPES:
            raise ValueError(f"unknown task '{task}', expected 'document' or 'query'")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        config = self._config(_TASK_TYPES[task])
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            # The API rejects empty strings, so blank inputs are replaced by a placeholder.
            batch = [t if t.strip() else "empty" for t in texts[start : start + self.batch_size]]
            response = self._client.models.embed_content(
                model=self.model, contents=batch, config=config
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"embedding service returned {len(embeddings)} vectors for {len(batch)} inputs"
                )
            for embedding in embeddings:
                values = embedding.values
                if values is None or len(values) != self.dim:
                    got = "no values" if values is None else f"{len(values)} values"
                    raise RuntimeError(f"expected {self.dim}-dim embedding, got {got}")
                rows.append(list(values))
        return l2_normalise(np.asarray(rows, dtype=np.float32))

    def _config(self, task_type: str) -> Any:
        if genai_types is not None:
            return genai_types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self.dim
            )
        return {"task_type": task_type, "output_dimensionality": self.dim}
