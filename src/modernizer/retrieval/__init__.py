"""Retrieval layer: migration knowledge base, embedders, vector indexes and the retriever.

The Vertex AI classes are exported lazily because importing the Google Cloud SDKs takes
seconds; offline use (local index, hashing embedder) never pays for them.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .embedders import HashingEmbedder
from .knowledge import KnowledgeBase, KnowledgeEntry
from .local_index import LocalVectorIndex
from .retriever import Retriever

if TYPE_CHECKING:
    from .vertex_embedder import VertexEmbedder
    from .vertex_index import VertexVectorSearchIndex, provision_vector_search

_LAZY = {
    "VertexEmbedder": ".vertex_embedder",
    "VertexVectorSearchIndex": ".vertex_index",
    "provision_vector_search": ".vertex_index",
}

__all__ = [
    "HashingEmbedder",
    "KnowledgeBase",
    "KnowledgeEntry",
    "LocalVectorIndex",
    "Retriever",
    "VertexEmbedder",
    "VertexVectorSearchIndex",
    "provision_vector_search",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target, __name__), name)
    globals()[name] = value
    return value
