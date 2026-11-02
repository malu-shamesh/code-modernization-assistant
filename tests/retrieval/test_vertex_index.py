from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modernizer.protocols import VectorIndex
from modernizer.retrieval import VertexVectorSearchIndex, provision_vector_search
from modernizer.retrieval import vertex_index as vertex_module


class FakeIndex:
    def __init__(self) -> None:
        self.batches: list[list] = []

    def upsert_datapoints(self, datapoints, update_mask=None):
        self.batches.append(list(datapoints))
        return self


class FakeEndpoint:
    def __init__(self, neighbors: list[tuple[str, float]]) -> None:
        self.neighbors = neighbors
        self.calls: list[dict] = []

    def find_neighbors(self, **kwargs):
        self.calls.append(kwargs)
        return [[SimpleNamespace(id=i, distance=d) for i, d in self.neighbors]]


def make_index(neighbors=(), **kwargs):
    index, endpoint = FakeIndex(), FakeEndpoint(list(neighbors))
    adapter = VertexVectorSearchIndex(
        index_name="projects/p/locations/l/indexes/1",
        endpoint_name="projects/p/locations/l/indexEndpoints/2",
        deployed_index_id="dep_1",
        index=index,
        endpoint=endpoint,
        **kwargs,
    )
    return adapter, index, endpoint


def test_adapter_satisfies_vector_index_protocol():
    adapter, _, _ = make_index()
    assert isinstance(adapter, VectorIndex)


def test_upsert_builds_datapoints_with_ids_and_vectors():
    adapter, index, _ = make_index()
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    adapter.upsert(["a", "b"], vectors)
    (batch,) = index.batches
    assert [dp.datapoint_id for dp in batch] == ["a", "b"]
    assert list(batch[1].feature_vector) == [0.0, 1.0]


def test_upsert_splits_large_batches():
    adapter, index, _ = make_index()
    n = 250
    adapter.upsert([f"id{i}" for i in range(n)], np.ones((n, 3), dtype=np.float32))
    assert [len(b) for b in index.batches] == [100, 100, 50]
    assert index.batches[2][-1].datapoint_id == "id249"


def test_upsert_validates_input():
    adapter, index, _ = make_index()
    with pytest.raises(ValueError, match="ids"):
        adapter.upsert(["a"], np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        adapter.upsert(["a"], np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="NaN"):
        adapter.upsert(["a"], np.full((1, 3), np.inf, dtype=np.float32))
    assert index.batches == []


def test_upsert_of_nothing_makes_no_call():
    adapter, index, _ = make_index()
    adapter.upsert([], np.zeros((0, 3), dtype=np.float32))
    assert index.batches == []


def test_query_passes_arguments_and_returns_dot_product_as_score():
    adapter, _, endpoint = make_index([("a", 0.9), ("b", 0.4)])
    result = adapter.query(np.asarray([1.0, 0.0], dtype=np.float32), 2)
    assert result == [("a", pytest.approx(0.9)), ("b", pytest.approx(0.4))]
    (call,) = endpoint.calls
    assert call["deployed_index_id"] == "dep_1"
    assert call["num_neighbors"] == 2
    assert call["queries"] == [[1.0, 0.0]]


def test_query_sorts_best_first_and_truncates_to_k():
    adapter, _, _ = make_index([("low", 0.1), ("high", 0.8), ("mid", 0.5)])
    result = adapter.query(np.zeros(2, dtype=np.float32), 2)
    assert [i for i, _ in result] == ["high", "mid"]


def test_negate_distance_flips_sign_for_negated_dot_product_indexes():
    adapter, _, _ = make_index([("close", -0.9), ("far", -0.2)], negate_distance=True)
    result = adapter.query(np.zeros(2, dtype=np.float32), 5)
    assert result[0] == ("close", pytest.approx(0.9))
    assert result[1] == ("far", pytest.approx(0.2))


def test_query_handles_empty_responses_and_non_positive_k():
    adapter, _, endpoint = make_index([])
    assert adapter.query(np.zeros(2, dtype=np.float32), 3) == []
    assert adapter.query(np.zeros(2, dtype=np.float32), 0) == []
    assert len(endpoint.calls) == 1
    endpoint.find_neighbors = lambda **_: []
    assert adapter.query(np.zeros(2, dtype=np.float32), 3) == []


def test_constructor_attaches_by_resource_name_when_not_injected(monkeypatch):
    seen: dict[str, dict] = {}

    class StubIndex:
        def __init__(self, **kwargs):
            seen["index"] = kwargs

    class StubEndpoint:
        def __init__(self, **kwargs):
            seen["endpoint"] = kwargs

    monkeypatch.setattr(vertex_module, "MatchingEngineIndex", StubIndex)
    monkeypatch.setattr(vertex_module, "MatchingEngineIndexEndpoint", StubEndpoint)
    VertexVectorSearchIndex(
        index_name="idx", endpoint_name="ep", deployed_index_id="d", project="proj", location="eu"
    )
    assert seen["index"] == {"index_name": "idx", "project": "proj", "location": "eu"}
    assert seen["endpoint"] == {"index_endpoint_name": "ep", "project": "proj", "location": "eu"}


class StubResource:
    def __init__(self, resource_name: str) -> None:
        self.resource_name = resource_name
        self.public_endpoint_domain_name = "123.us-central1-456.vdb.vertexai.goog"
        self.deploy_kwargs: dict | None = None

    def deploy_index(self, **kwargs):
        self.deploy_kwargs = kwargs
        return self


def install_creation_stubs(monkeypatch):
    created: dict[str, dict] = {}
    index = StubResource("projects/p/locations/l/indexes/11")
    endpoint = StubResource("projects/p/locations/l/indexEndpoints/22")

    class StubIndexCls:
        @staticmethod
        def create_tree_ah_index(**kwargs):
            created["index"] = kwargs
            return index

    class StubEndpointCls:
        @staticmethod
        def create(**kwargs):
            created["endpoint"] = kwargs
            return endpoint

    monkeypatch.setattr(vertex_module, "MatchingEngineIndex", StubIndexCls)
    monkeypatch.setattr(vertex_module, "MatchingEngineIndexEndpoint", StubEndpointCls)
    return created, index, endpoint


def test_provision_creates_streaming_dot_product_index_and_deploys(monkeypatch):
    created, index, endpoint = install_creation_stubs(monkeypatch)
    info = provision_vector_search(project="proj", location="us-central1", dim=768)

    idx = created["index"]
    assert idx["dimensions"] == 768
    assert idx["index_update_method"] == "STREAM_UPDATE"
    assert idx["distance_measure_type"] == vertex_module.DistanceMeasureType.DOT_PRODUCT_DISTANCE
    assert idx["display_name"] == "modernizer-kb"
    assert idx["project"] == "proj" and idx["location"] == "us-central1"

    assert created["endpoint"]["public_endpoint_enabled"] is True
    assert created["endpoint"]["project"] == "proj"

    deploy = endpoint.deploy_kwargs
    assert deploy is not None and deploy["index"] is index
    assert deploy["deployed_index_id"] == "modernizer_kb_deployed"

    assert info["index_name"] == index.resource_name
    assert info["endpoint_name"] == endpoint.resource_name
    assert info["deployed_index_id"] == "modernizer_kb_deployed"
    assert info["public_endpoint_domain_name"].endswith("vertexai.goog")


def test_provision_honours_explicit_names(monkeypatch):
    created, _, endpoint = install_creation_stubs(monkeypatch)
    info = provision_vector_search(
        project="p", location="l", dim=8, display_name="9-my kb", deployed_index_id="custom_id"
    )
    assert info["deployed_index_id"] == "custom_id"
    assert endpoint.deploy_kwargs["deployed_index_id"] == "custom_id"
    assert created["index"]["display_name"] == "9-my kb"


def test_provision_sanitises_generated_deployed_id(monkeypatch):
    install_creation_stubs(monkeypatch)
    info = provision_vector_search(project="p", location="l", dim=8, display_name="9-my kb")
    assert info["deployed_index_id"] == "d_9_my_kb_deployed"


def test_provision_rejects_bad_dim_before_creating_anything(monkeypatch):
    created, _, _ = install_creation_stubs(monkeypatch)
    with pytest.raises(ValueError):
        provision_vector_search(project="p", location="l", dim=0)
    assert created == {}
