"""Feature-aware retrieval of migration knowledge for a parsed module."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ..features import FEATURES, is_semantic, weight_of
from ..models import ContextSnippet, ParsedModule
from ..protocols import Embedder, VectorIndex
from .knowledge import KnowledgeBase, KnowledgeEntry

logger = logging.getLogger(__name__)

_MAX_QUERY_LINES = 40
_MAX_QUERY_LINE_CHARS = 200
_MAX_QUERY_CHARS = 4000
_MIN_CANDIDATES = 30
_CANDIDATE_FACTOR = 5


@dataclass
class _Candidate:
    entry: KnowledgeEntry
    similarity: float
    score: float


class Retriever:
    """Combines semantic search with the structural signal from the AST feature hits.

    Ranking is ``similarity + feature_boost`` for entries that share a construct with the
    module, followed by a coverage pass that guarantees every detected construct that has a
    knowledge entry is represented in the result when ``k`` allows.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: Embedder,
        index: VectorIndex,
        *,
        feature_boost: float = 0.25,
    ) -> None:
        self.kb = kb
        self.embedder = embedder
        self.index = index
        self.feature_boost = feature_boost

    def build_index(self) -> int:
        """Embed every knowledge entry and upsert it into the index; returns the entry count."""
        entries = self.kb.entries
        if not entries:
            return 0
        vectors = self.embedder.embed([e.embedding_text() for e in entries], task="document")
        self.index.upsert([e.id for e in entries], vectors)
        return len(entries)

    def retrieve(
        self, module: ParsedModule, k: int = 6, max_chars: int = 6000
    ) -> list[ContextSnippet]:
        feature_ids = [f for f in module.feature_ids if f in FEATURES]
        query = self._module_query(module, feature_ids)
        return self._retrieve(query, feature_ids, k, max_chars)

    def retrieve_for_text(
        self,
        text: str,
        feature_ids: Sequence[str] = (),
        k: int = 6,
        *,
        max_chars: int = 6000,
    ) -> list[ContextSnippet]:
        unknown = [f for f in feature_ids if f not in FEATURES]
        if unknown:
            raise ValueError(f"unknown feature id(s) {unknown}")
        distinct = list(dict.fromkeys(feature_ids))
        parts = [FEATURES[f].description for f in distinct]
        parts.append(text[:_MAX_QUERY_CHARS])
        return self._retrieve("\n".join(p for p in parts if p), distinct, k, max_chars)

    def _module_query(self, module: ParsedModule, feature_ids: list[str]) -> str:
        parts = [FEATURES[f].description for f in feature_ids]
        lines: dict[str, None] = {}
        for hit in module.hits:
            text = hit.snippet or self._source_lines(module, hit.line, hit.end_line)
            for raw in text.splitlines():
                line = raw.strip()[:_MAX_QUERY_LINE_CHARS]
                if line:
                    lines.setdefault(line)
        parts.extend(list(lines)[:_MAX_QUERY_LINES])
        if not lines:
            parts.append(module.source[:_MAX_QUERY_CHARS])
        return "\n".join(parts)[:_MAX_QUERY_CHARS]

    @staticmethod
    def _source_lines(module: ParsedModule, line: int, end_line: int) -> str:
        source_lines = module.source.splitlines()
        return "\n".join(source_lines[max(line - 1, 0) : max(end_line, line)])

    def _retrieve(
        self, query: str, feature_ids: list[str], k: int, max_chars: int
    ) -> list[ContextSnippet]:
        if k <= 0 or max_chars <= 0 or len(self.kb) == 0:
            return []
        ranked = self._rank(query, feature_ids, k)
        selected = self._select(ranked, feature_ids, k)
        return self._fit(selected, max_chars)

    def _rank(self, query: str, feature_ids: list[str], k: int) -> list[_Candidate]:
        n_candidates = min(
            len(self.kb),
            max(_CANDIDATE_FACTOR * k, _CANDIDATE_FACTOR * len(feature_ids), _MIN_CANDIDATES),
        )
        vector = self.embedder.embed([query], task="query")[0]
        found = self.index.query(vector, n_candidates)
        if not found:
            logger.warning("vector index returned no candidates; was build_index() called?")

        wanted = set(feature_ids)
        candidates: dict[str, _Candidate] = {}
        for entry_id, similarity in found:
            if entry_id in candidates or entry_id not in self.kb:
                continue
            entry = self.kb.get(entry_id)
            boost = self.feature_boost if wanted.intersection(entry.feature_ids) else 0.0
            candidates[entry_id] = _Candidate(entry, similarity, similarity + boost)

        # Entries for detected constructs that the nearest-neighbour search missed are still
        # eligible for the coverage pass; their similarity is unknown, so use the pool floor.
        floor = min((c.similarity for c in candidates.values()), default=0.0)
        for feature_id in feature_ids:
            for entry in self.kb.by_feature(feature_id):
                if entry.id not in candidates:
                    candidates[entry.id] = _Candidate(entry, floor, floor + self.feature_boost)

        order = {e.id: i for i, e in enumerate(self.kb.entries)}
        return sorted(candidates.values(), key=lambda c: (-c.score, order[c.entry.id]))

    def _select(self, ranked: list[_Candidate], feature_ids: list[str], k: int) -> list[_Candidate]:
        by_id = {c.entry.id: c for c in ranked}
        # Entries added to the knowledge base after _rank ran are not in by_id; skip them.
        pools = {
            f: [by_id[e.id] for e in self.kb.by_feature(f) if e.id in by_id] for f in feature_ids
        }
        required = [f for f in feature_ids if pools[f]]
        # Semantic constructs first, then heavier ones; sort is stable so ties keep source order.
        required.sort(key=lambda f: (not is_semantic(f), -weight_of(f)))

        forced: dict[str, _Candidate] = {}
        covered: set[str] = set()
        for feature_id in required:
            if len(forced) >= k:
                break
            if feature_id in covered:
                continue
            best = max(pools[feature_id], key=lambda c: c.score)
            forced[best.entry.id] = best
            covered.update(best.entry.feature_ids)

        chosen = dict(forced)
        for candidate in ranked:
            if len(chosen) >= k:
                break
            chosen.setdefault(candidate.entry.id, candidate)
        return sorted(chosen.values(), key=lambda c: -c.score)

    @staticmethod
    def _fit(selected: list[_Candidate], max_chars: int) -> list[ContextSnippet]:
        texts = [c.entry.render() for c in selected]
        keep = len(selected)
        while keep > 1 and sum(len(t) for t in texts[:keep]) > max_chars:
            keep -= 1
        texts = texts[:keep]
        if keep == 1 and len(texts[0]) > max_chars:
            texts[0] = selected[0].entry.render(max_chars=max_chars)
        return [
            ContextSnippet(
                id=c.entry.id,
                title=c.entry.title,
                text=text,
                score=c.score,
                feature_ids=c.entry.feature_ids,
            )
            for c, text in zip(selected[:keep], texts, strict=True)
        ]
