from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from modernizer.protocols import Embedder
from modernizer.retrieval import HashingEmbedder, VertexEmbedder

SAMPLE_TEXTS = [
    "      WRITE (6, 100) TOTAL\n  100 FORMAT (1X, F10.2)",
    "      SUBROUTINE FOOBAR(MY_VALUE)\n      COMMON /WORK/ GETVALUE, MY_VALUE",
    "",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_hashing_embedder_shape_dtype_and_norm():
    vectors = HashingEmbedder(128).embed(SAMPLE_TEXTS[:2])
    assert vectors.shape == (2, 128)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_hashing_embedder_satisfies_protocol_and_default_dim():
    embedder = HashingEmbedder()
    assert embedder.dim == 512
    assert isinstance(embedder, Embedder)


def test_empty_text_gives_zero_vector_and_empty_batch_gives_empty_matrix():
    embedder = HashingEmbedder(64)
    assert not embedder.embed([""]).any()
    assert not embedder.embed(["   \n\t"]).any()
    assert embedder.embed([]).shape == (0, 64)


def test_invalid_dim_is_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(0)


def test_same_input_same_output_and_task_is_irrelevant():
    embedder = HashingEmbedder(256)
    a = embedder.embed(SAMPLE_TEXTS, task="document")
    b = embedder.embed(SAMPLE_TEXTS, task="query")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:1], embedder.embed(SAMPLE_TEXTS[:1]))


def test_batch_rows_do_not_depend_on_batch_composition():
    embedder = HashingEmbedder(256)
    together = embedder.embed(SAMPLE_TEXTS)
    alone = embedder.embed([SAMPLE_TEXTS[1]])
    np.testing.assert_array_equal(together[1], alone[0])


def test_embeddings_are_stable_across_processes():
    script = (
        "import sys;"
        "from modernizer.retrieval.embedders import HashingEmbedder;"
        "texts = sys.argv[1:];"
        "sys.stdout.write(HashingEmbedder(64).embed(texts).tobytes().hex())"
    )
    local = HashingEmbedder(64).embed(SAMPLE_TEXTS).tobytes().hex()
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", script, *SAMPLE_TEXTS],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=120,
        )
        assert proc.stdout.strip() == local


def test_similar_code_is_closer_than_unrelated_code():
    embedder = HashingEmbedder(512)
    query, near, far = embedder.embed(
        [
            "      DO 10 I = 1, N\n         TOTAL = TOTAL + X(I)\n   10 CONTINUE",
            "      DO 20 K = 1, 10\n         COUNT = COUNT + X(K)\n   20 CONTINUE",
            "      SUBROUTINE CONFIG(PATH)\n      CHARACTER*64 PATH\n      RETURN\n      END",
        ]
    )
    assert cosine(query, near) > cosine(query, far)


def test_identifier_subtokens_connect_snake_and_camel_case():
    embedder = HashingEmbedder(512)
    snake, camel, other = embedder.embed(["read_file_header", "readFileHeader", "write_socket"])
    assert cosine(snake, camel) > 0.4
    assert cosine(snake, camel) > cosine(snake, other)


def test_case_is_ignored_for_whole_tokens():
    embedder = HashingEmbedder(256)
    a, b = embedder.embed(["CONTINUE", "continue"])
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-5)


class FakeModels:
    def __init__(self, dim: int, scale: float = 3.0) -> None:
        self.dim = dim
        self.scale = scale
        self.calls: list[dict] = []

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        embeddings = []
        for text in contents:
            values = [self.scale * ((len(text) + j) % 7 + 1) for j in range(self.dim)]
            embeddings.append(SimpleNamespace(values=values))
        return SimpleNamespace(embeddings=embeddings)


class FakeClient:
    def __init__(self, dim: int) -> None:
        self.models = FakeModels(dim)


def task_of(call: dict) -> str:
    config = call["config"]
    return config["task_type"] if isinstance(config, dict) else config.task_type


def dim_of(call: dict) -> int:
    config = call["config"]
    key = "output_dimensionality"
    return config[key] if isinstance(config, dict) else getattr(config, key)


def test_vertex_embedder_calls_client_with_task_type_and_dimensionality():
    client = FakeClient(16)
    embedder = VertexEmbedder("my-model", dim=16, client=client)
    doc = embedder.embed(["a", "bb"], task="document")
    query = embedder.embed(["ccc"], task="query")
    assert doc.shape == (2, 16) and query.shape == (1, 16)
    first, second = client.models.calls
    assert first["model"] == "my-model" and first["contents"] == ["a", "bb"]
    assert task_of(first) == "RETRIEVAL_DOCUMENT" and dim_of(first) == 16
    assert task_of(second) == "RETRIEVAL_QUERY"


def test_vertex_embedder_normalises_output():
    vectors = VertexEmbedder(dim=8, client=FakeClient(8)).embed(["alpha", "beta"])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_vertex_embedder_batches_requests_and_keeps_order():
    client = FakeClient(4)
    embedder = VertexEmbedder(dim=4, client=client, batch_size=3)
    texts = [f"text-{'x' * i}" for i in range(7)]
    vectors = embedder.embed(texts)
    assert [len(c["contents"]) for c in client.models.calls] == [3, 3, 1]
    assert [t for c in client.models.calls for t in c["contents"]] == texts
    single = VertexEmbedder(dim=4, client=FakeClient(4)).embed([texts[5]])
    np.testing.assert_allclose(vectors[5], single[0], atol=1e-6)


def test_vertex_embedder_replaces_blank_inputs_with_placeholder():
    client = FakeClient(4)
    VertexEmbedder(dim=4, client=client).embed(["", "  ", "real"])
    sent = client.models.calls[0]["contents"]
    assert sent[2] == "real" and all(t.strip() for t in sent)


def test_vertex_embedder_empty_input_makes_no_call():
    client = FakeClient(4)
    out = VertexEmbedder(dim=4, client=client).embed([])
    assert out.shape == (0, 4) and client.models.calls == []


def test_vertex_embedder_rejects_count_mismatch():
    client = FakeClient(4)
    client.models.embed_content = lambda **_: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0] * 4)]
    )
    with pytest.raises(RuntimeError, match="1 vectors for 2 inputs"):
        VertexEmbedder(dim=4, client=client).embed(["a", "b"])


def test_vertex_embedder_rejects_missing_embeddings():
    client = FakeClient(4)
    client.models.embed_content = lambda **_: SimpleNamespace(embeddings=None)
    with pytest.raises(RuntimeError):
        VertexEmbedder(dim=4, client=client).embed(["a"])


def test_vertex_embedder_rejects_wrong_dimension():
    with pytest.raises(RuntimeError, match="expected 8-dim"):
        VertexEmbedder(dim=8, client=FakeClient(4)).embed(["a"])


def test_vertex_embedder_rejects_bad_arguments():
    client = FakeClient(4)
    with pytest.raises(ValueError):
        VertexEmbedder(dim=0, client=client)
    with pytest.raises(ValueError):
        VertexEmbedder(dim=4, client=client, batch_size=0)
    with pytest.raises(ValueError, match="task"):
        VertexEmbedder(dim=4, client=client).embed(["a"], task="other")  # type: ignore[arg-type]


def test_vertex_embedder_defaults_and_protocol():
    embedder = VertexEmbedder(client=FakeClient(768))
    assert embedder.model == "text-embedding-005" and embedder.dim == 768
    assert embedder.batch_size == 100
    assert isinstance(embedder, Embedder)
