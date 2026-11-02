"""Vertex AI Vector Search: a ``VectorIndex`` implementation and a provisioning helper.

Distance assumption: the index is created with ``DOT_PRODUCT_DISTANCE`` over L2-normalised
vectors (``provision_vector_search`` does exactly that). For this measure the service reports
the dot product itself in ``MatchNeighbor.distance``, so a larger value is a closer match and
it is used directly as the similarity score. The SDK enum comment describes the measure as
"a negative of the dot product"; if an index in your project reports negated values, pass
``negate_distance=True`` and the sign is flipped. Indexes built with any other measure
(squared L2, cosine distance) are not supported by this adapter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
    from google.cloud.aiplatform.matching_engine.matching_engine_index_config import (
        DistanceMeasureType,
    )
    from google.cloud.aiplatform_v1.types import IndexDatapoint
except ImportError:  # pragma: no cover - exercised only without the gcp extra
    MatchingEngineIndex = None
    MatchingEngineIndexEndpoint = None
    DistanceMeasureType = None
    IndexDatapoint = None

_INSTALL_HINT = "pip install 'code-modernization-assistant[gcp]'"
_UPSERT_BATCH = 100


def _require_sdk() -> None:
    if MatchingEngineIndex is None or MatchingEngineIndexEndpoint is None:
        raise ImportError(f"Vertex Vector Search needs google-cloud-aiplatform: {_INSTALL_HINT}")


class VertexVectorSearchIndex:
    """``VectorIndex`` backed by a deployed Vertex AI Vector Search streaming index.

    ``index`` and ``endpoint`` may be injected (any objects with the same ``upsert_datapoints``
    and ``find_neighbors`` methods); otherwise they are attached by resource name.
    """

    def __init__(
        self,
        *,
        index_name: str,
        endpoint_name: str,
        deployed_index_id: str,
        project: str | None = None,
        location: str = "us-central1",
        index: Any = None,
        endpoint: Any = None,
        negate_distance: bool = False,
    ) -> None:
        if index is None or endpoint is None:
            _require_sdk()
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.deployed_index_id = deployed_index_id
        self.negate_distance = negate_distance
        self._index = (
            index
            if index is not None
            else MatchingEngineIndex(index_name=index_name, project=project, location=location)
        )
        self._endpoint = (
            endpoint
            if endpoint is not None
            else MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_name, project=project, location=location
            )
        )

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {matrix.shape}")
        if matrix.shape[0] != len(ids):
            raise ValueError(f"got {len(ids)} ids but {matrix.shape[0]} vectors")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors contain NaN or infinite values")
        for start in range(0, len(ids), _UPSERT_BATCH):
            datapoints = [
                self._datapoint(item_id, vector.tolist())
                for item_id, vector in zip(
                    ids[start : start + _UPSERT_BATCH],
                    matrix[start : start + _UPSERT_BATCH],
                    strict=True,
                )
            ]
            self._index.upsert_datapoints(datapoints=datapoints)

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        batches = self._endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query.tolist()],
            num_neighbors=k,
        )
        if not batches:
            return []
        sign = -1.0 if self.negate_distance else 1.0
        scored = [(n.id, sign * float(n.distance)) for n in batches[0]]
        # Sort locally so callers get best-first order whatever the service returns.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    @staticmethod
    def _datapoint(item_id: str, values: list[float]) -> Any:
        if IndexDatapoint is None:
            raise ImportError(
                f"Vertex Vector Search needs google-cloud-aiplatform: {_INSTALL_HINT}"
            )
        return IndexDatapoint(datapoint_id=item_id, feature_vector=values)


def _sanitise_deployed_id(display_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", display_name)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"d_{cleaned}"
    return f"{cleaned}_deployed"


def provision_vector_search(
    *,
    project: str,
    location: str,
    dim: int,
    display_name: str = "modernizer-kb",
    deployed_index_id: str | None = None,
    approximate_neighbors_count: int = 20,
    machine_type: str = "e2-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
) -> dict[str, Any]:
    """Create, and deploy to a public endpoint, a streaming-update tree-AH index.

    The index uses ``DOT_PRODUCT_DISTANCE`` (matching the normalised embeddings the rest of the
    package produces). This creates billable resources: a deployed index bills for its
    replicas until it is undeployed. Index creation and deployment are synchronous and can take
    tens of minutes.
    """
    _require_sdk()
    if dim <= 0:
        raise ValueError("dim must be positive")
    deployed_id = deployed_index_id or _sanitise_deployed_id(display_name)
    index = MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        dimensions=dim,
        approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        index_update_method="STREAM_UPDATE",
        description="Fortran 77 to Python migration knowledge base for code-modernization-assistant",
        project=project,
        location=location,
    )
    endpoint = MatchingEngineIndexEndpoint.create(
        display_name=f"{display_name}-endpoint",
        public_endpoint_enabled=True,
        project=project,
        location=location,
    )
    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_id,
        display_name=f"{display_name}-deployed",
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )
    return {
        "project": project,
        "location": location,
        "dim": dim,
        "index_name": index.resource_name,
        "endpoint_name": endpoint.resource_name,
        "deployed_index_id": deployed_id,
        "public_endpoint_domain_name": getattr(endpoint, "public_endpoint_domain_name", None),
    }
