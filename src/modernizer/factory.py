"""Assembly of the pipeline from settings."""

from __future__ import annotations

from .llm import ModelSet, build_model_set
from .pipeline import Pipeline
from .protocols import ResultSink
from .retrieval import (
    HashingEmbedder,
    KnowledgeBase,
    LocalVectorIndex,
    Retriever,
    VertexEmbedder,
    VertexVectorSearchIndex,
)
from .routing import Router
from .settings import Settings
from .sinks import sink_from_spec
from .telemetry import Telemetry, setup_telemetry


def build_retriever(settings: Settings) -> Retriever:
    """Local numpy index by default; Vertex AI Vector Search when configured.

    With the Vertex backend the index must already hold the knowledge base (see
    ``modernizer index build``), so this function does not upsert.
    """
    kb = KnowledgeBase.default()
    if settings.retrieval == "local":
        embedder = HashingEmbedder()
        retriever = Retriever(kb, embedder, LocalVectorIndex(embedder.dim))
        retriever.build_index()
        return retriever
    vs = settings.vector_search
    missing = [
        name
        for name, value in (
            ("MODERNIZER_VS_INDEX", vs.index_name),
            ("MODERNIZER_VS_ENDPOINT", vs.endpoint_name),
            ("MODERNIZER_VS_DEPLOYED_ID", vs.deployed_index_id),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"the vertex retrieval backend needs {', '.join(missing)}")
    embedder = VertexEmbedder(
        vs.embedding_model,
        project=settings.models.project,
        location=vs.location,
        dim=vs.embedding_dim,
    )
    index = VertexVectorSearchIndex(
        index_name=vs.index_name or "",
        endpoint_name=vs.endpoint_name or "",
        deployed_index_id=vs.deployed_index_id or "",
        project=settings.models.project,
        location=vs.location,
    )
    return Retriever(kb, embedder, index)


def build_router(settings: Settings, models: ModelSet) -> Router:
    return Router(
        models.frontier.name,
        models.open.name,
        policy=settings.policy,
        threshold=settings.threshold,
    )


def build_pipeline(
    settings: Settings,
    *,
    models: ModelSet | None = None,
    retriever: Retriever | None = None,
    telemetry: Telemetry | None = None,
    sink: ResultSink | None = None,
    run_id: str | None = None,
) -> Pipeline:
    """Create a pipeline; anything not passed in is built from ``settings``."""
    models = models or build_model_set(settings.models)
    return Pipeline(
        models,
        build_router(settings, models),
        retriever if retriever is not None else build_retriever(settings),
        telemetry=telemetry
        or setup_telemetry(
            settings.telemetry,
            otlp_endpoint=settings.otlp_endpoint,
            project_id=settings.models.project,
        ),
        config=settings.pipeline,
        sink=sink if sink is not None else sink_from_spec(settings.sink),
        run_id=run_id,
    )
