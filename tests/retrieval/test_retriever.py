from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from modernizer.models import FeatureHit, ModuleMetrics, ParsedModule
from modernizer.retrieval import (
    HashingEmbedder,
    KnowledgeBase,
    KnowledgeEntry,
    LocalVectorIndex,
    Retriever,
)


def entry(
    entry_id: str,
    title: str,
    features: tuple[str, ...],
    legacy: str,
    modern: str,
    notes: str = "",
) -> KnowledgeEntry:
    return KnowledgeEntry(entry_id, title, features, legacy, modern, notes)


CORPUS = [
    entry(
        "format-write",
        "formatted WRITE with a FORMAT statement",
        ("format_statement", "io_formatted"),
        "      WRITE (6, 100) TOTAL\n  100 FORMAT (1X, 'TOTAL:', F10.2)",
        'print(f" TOTAL:{total:10.2f}")',
        "The first character of each record is carriage control, not data.",
    ),
    entry(
        "goto-computed",
        "computed GO TO dispatches on an index",
        ("goto_computed", "goto_assigned"),
        "      GO TO (10, 20, 30), K\n   10 CONTINUE\n   20 CONTINUE\n   30 CONTINUE",
        "if k == 1:\n    pass\nelif k == 2:\n    pass\nelif k == 3:\n    pass",
        "An index outside 1..n falls through to the next statement.",
    ),
    entry(
        "common-block",
        "COMMON block shares storage between routines",
        ("common_block",),
        "      COMMON /WORK/ A(10), N\n      IF (N .GT. 0) A(1) = 0.0",
        "class Work:\n    a = [0.0] * 10\n    n = 0",
    ),
    entry(
        "equivalence-alias",
        "EQUIVALENCE aliases two names onto one storage",
        ("equivalence", "save_statement"),
        "      REAL X(4), Y\n      EQUIVALENCE (X(1), Y)\n      SAVE X",
        "x = np.zeros(4, dtype=np.float32)\ny = x[0:1]",
        "Aliased storage must be a numpy view, because a copy hides later writes.",
    ),
    entry(
        "do-labeled",
        "labeled DO loop closed by CONTINUE",
        ("do_labeled",),
        "      DO 10 I = 1, 10\n         S = S + X(I)\n   10 CONTINUE",
        "for i in range(1, 11):\n    s += x[i - 1]",
    ),
    entry(
        "character-star",
        "CHARACTER*n strings are blank padded",
        ("character_star", "character_substring"),
        "      CHARACTER*8 NAME\n      NAME = 'AB'\n      IF (NAME(1:1) .EQ. 'A') N = 1",
        "name = 'AB'.ljust(8)\nif name[0:1] == 'A':\n    n = 1",
        "Assignment pads with blanks or truncates to the declared length.",
    ),
    entry(
        "integer-division",
        "integer division truncates toward zero",
        ("integer_division",),
        "      INTEGER N, M, Q\n      Q = N / M",
        "q = f77rt.idiv(n, m)",
        "Python // floors, so negative operands give a different result.",
    ),
    entry(
        "if-arithmetic",
        "arithmetic IF branches three ways",
        ("if_arithmetic", "if_logical_single"),
        "      IF (X - 1.0) 10, 20, 30\n   10 CONTINUE",
        "d = x - 1.0\nif d < 0:\n    pass\nelif d == 0:\n    pass\nelse:\n    pass",
        "Evaluate the expression once; branches may share a label.",
    ),
    entry(
        "data-init",
        "DATA statement initialises variables once",
        ("data_statement",),
        "      DATA N /5/, A /3*0.0/",
        "n = 5\na = [0.0] * 3",
    ),
    entry(
        "general-advice",
        "general advice: read the whole routine and add tests first",
        (),
        "C     NO SPECIFIC CONSTRUCT",
        "# pin behaviour with tests before converting",
        "Write characterisation tests before touching the code.",
    ),
]


def make_kb() -> KnowledgeBase:
    return KnowledgeBase(CORPUS)


def make_retriever(**kwargs) -> Retriever:
    embedder = HashingEmbedder(512)
    retriever = Retriever(make_kb(), embedder, LocalVectorIndex(embedder.dim), **kwargs)
    retriever.build_index()
    return retriever


def make_module(source: str = "", hits: Sequence[tuple[str, int, str]] = ()) -> ParsedModule:
    return ParsedModule(
        path="m.f",
        source=source,
        units=[],
        hits=[FeatureHit(fid, line, line, snippet) for fid, line, snippet in hits],
        metrics=ModuleMetrics(),
    )


def test_build_index_returns_entry_count_and_fills_index():
    embedder = HashingEmbedder(64)
    index = LocalVectorIndex(embedder.dim)
    retriever = Retriever(make_kb(), embedder, index)
    assert retriever.build_index() == len(CORPUS)
    assert len(index) == len(CORPUS)


def test_build_index_empty_kb_is_noop():
    embedder = HashingEmbedder(64)
    index = LocalVectorIndex(embedder.dim)
    assert Retriever(KnowledgeBase(), embedder, index).build_index() == 0
    assert len(index) == 0


def test_computed_goto_module_retrieves_goto_entry_first():
    module = make_module(
        "      GO TO (10, 20, 30), K\n   10 CONTINUE\n   20 CONTINUE\n   30 CONTINUE\n",
        [("goto_computed", 1, "GO TO (10, 20, 30), K")],
    )
    snippets = make_retriever().retrieve(module)
    assert snippets[0].id == "goto-computed"
    assert "Fortran 77:" in snippets[0].text and "Python:" in snippets[0].text
    assert snippets[0].feature_ids == ("goto_computed", "goto_assigned")


def test_results_sorted_by_descending_score_and_limited_to_k():
    module = make_module("x", [("do_labeled", 1, "DO 10 I = 1, 3")])
    snippets = make_retriever().retrieve(module, k=4)
    assert len(snippets) == 4
    scores = [s.score for s in snippets]
    assert scores == sorted(scores, reverse=True)
    assert snippets[0].id == "do-labeled"


def test_feature_boost_is_added_for_matching_entries():
    module = make_module("x", [("do_labeled", 1, "DO 10 I = 1, 3")])
    boosted = make_retriever(feature_boost=0.5).retrieve(module, k=3)
    plain = make_retriever(feature_boost=0.0).retrieve(module, k=3)
    by_id_boosted = {s.id: s.score for s in boosted}
    by_id_plain = {s.id: s.score for s in plain}
    assert by_id_boosted["do-labeled"] == pytest.approx(by_id_plain["do-labeled"] + 0.5, abs=1e-5)


def test_every_detected_feature_is_covered_when_k_allows():
    module = make_module(
        "x",
        [
            ("format_statement", 1, "100 FORMAT (1X, F10.2)"),
            ("goto_computed", 2, "GO TO (10, 20), K"),
            ("do_labeled", 3, "DO 10 I = 1, 3"),
            ("common_block", 4, "COMMON /WORK/ A, N"),
            ("integer_division", 5, "Q = N / M"),
            ("if_arithmetic", 6, "IF (X) 10, 20, 30"),
        ],
    )
    ids = {s.id for s in make_retriever().retrieve(module, k=6)}
    assert ids == {
        "format-write",
        "goto-computed",
        "do-labeled",
        "common-block",
        "integer-division",
        "if-arithmetic",
    }


def test_coverage_prioritises_semantic_features_when_k_is_small():
    module = make_module(
        "x",
        [
            ("do_labeled", 1, "DO 10 I = 1, 3"),
            ("data_statement", 2, "DATA N /5/"),
            ("integer_division", 3, "Q = N / M"),
            ("if_arithmetic", 4, "IF (X) 10, 20, 30"),
        ],
    )
    ids = {s.id for s in make_retriever().retrieve(module, k=3)}
    assert {"if-arithmetic", "integer-division"} <= ids
    assert len(ids) == 3


class FixedIndex:
    """Index stub that returns a canned ranking regardless of the query."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self.ranking = ranking

    def upsert(self, ids, vectors) -> None:
        pass

    def query(self, vector: np.ndarray, k: int):
        return self.ranking[:k]


def test_coverage_swaps_in_entry_that_semantic_search_ranked_low():
    ranking = [
        ("general-advice", 0.9),
        ("format-write", 0.8),
        ("equivalence-alias", 0.7),
        ("do-labeled", 0.6),
        ("integer-division", 0.2),
    ]
    retriever = Retriever(make_kb(), HashingEmbedder(32), FixedIndex(ranking), feature_boost=0.0)
    module = make_module("x", [("integer_division", 1, "Q = N / M")])
    snippets = retriever.retrieve(module, k=2)
    assert len(snippets) == 2
    assert "integer-division" in {s.id for s in snippets}
    assert snippets[0].id == "general-advice"


def test_coverage_uses_entries_missing_from_the_candidate_pool():
    ranking = [("general-advice", 0.9), ("format-write", 0.8)]
    retriever = Retriever(make_kb(), HashingEmbedder(32), FixedIndex(ranking), feature_boost=0.0)
    module = make_module("x", [("if_arithmetic", 1, "IF (X) 10, 20, 30")])
    ids = [s.id for s in retriever.retrieve(module, k=2)]
    assert "if-arithmetic" in ids


def test_never_returns_more_than_k_even_with_many_features():
    names = [
        "format_statement",
        "goto_computed",
        "do_labeled",
        "common_block",
        "integer_division",
    ]
    hits = [(fid, i + 1, "x") for i, fid in enumerate(names)]
    snippets = make_retriever().retrieve(make_module("x", hits), k=2)
    assert len(snippets) == 2


def test_max_chars_drops_lowest_ranked_snippets():
    module = make_module(
        "x",
        [
            ("format_statement", 1, "100 FORMAT (1X, F10.2)"),
            ("goto_computed", 2, "GO TO (1, 2), K"),
        ],
    )
    retriever = make_retriever()
    full = retriever.retrieve(module, k=6, max_chars=100_000)
    limit = len(full[0].text) + len(full[1].text) + 5
    limited = retriever.retrieve(module, k=6, max_chars=limit)
    assert 1 <= len(limited) < len(full)
    assert sum(len(s.text) for s in limited) <= limit
    assert [s.id for s in limited] == [s.id for s in full[: len(limited)]]


def test_max_chars_smaller_than_one_snippet_clips_the_top_snippet():
    module = make_module("x", [("goto_computed", 1, "GO TO (1, 2), K")])
    snippets = make_retriever().retrieve(module, k=6, max_chars=120)
    assert len(snippets) == 1
    assert len(snippets[0].text) <= 120
    assert snippets[0].id == "goto-computed"


def test_non_positive_limits_return_nothing():
    retriever = make_retriever()
    module = make_module("x", [("do_labeled", 1, "DO 10 I = 1, 3")])
    assert retriever.retrieve(module, k=0) == []
    assert retriever.retrieve(module, max_chars=0) == []


def test_module_without_hits_returns_top_scoring_general_entries():
    module = make_module("      FUNCTION F(X)\n      F = X + 1.0\n      RETURN\n      END\n")
    snippets = make_retriever().retrieve(module, k=3)
    assert len(snippets) == 3
    assert len({s.id for s in snippets}) == 3


def test_empty_source_and_no_hits_still_returns_entries():
    snippets = make_retriever().retrieve(make_module(""), k=4)
    assert len(snippets) == 4


def test_k_larger_than_corpus_returns_whole_corpus_once():
    snippets = make_retriever().retrieve(make_module("x"), k=50, max_chars=1_000_000)
    assert sorted(s.id for s in snippets) == sorted(e.id for e in CORPUS)


def test_empty_knowledge_base_returns_nothing():
    embedder = HashingEmbedder(32)
    retriever = Retriever(KnowledgeBase(), embedder, LocalVectorIndex(embedder.dim))
    assert retriever.retrieve(make_module("x")) == []


def test_unknown_feature_ids_in_hits_are_ignored():
    module = make_module("x", [("not_a_feature", 1, "CALL MYSTERY")])
    assert len(make_retriever().retrieve(module, k=2)) == 2


def test_stale_index_ids_are_skipped():
    retriever = make_retriever()
    retriever.index.upsert(["ghost"], HashingEmbedder(512).embed(["ghost DO 10 I = 1, N"]))
    source = "      DO 10 I = 1, N\n   10 CONTINUE\n"
    ids = [s.id for s in retriever.retrieve(make_module(source), k=9)]
    assert "ghost" not in ids


def test_unbuilt_index_still_serves_coverage_entries():
    embedder = HashingEmbedder(64)
    retriever = Retriever(make_kb(), embedder, LocalVectorIndex(embedder.dim))
    module = make_module("x", [("common_block", 1, "COMMON /WORK/ A, N")])
    assert [s.id for s in retriever.retrieve(module, k=1)] == ["common-block"]


class RecordingEmbedder:
    def __init__(self) -> None:
        self.inner = HashingEmbedder(64)
        self.dim = self.inner.dim
        self.queries: list[str] = []

    def embed(self, texts, *, task="document"):
        if task == "query":
            self.queries.extend(texts)
        return self.inner.embed(texts, task=task)


def test_query_contains_distinct_descriptions_and_deduped_hit_lines():
    embedder = RecordingEmbedder()
    retriever = Retriever(make_kb(), embedder, LocalVectorIndex(embedder.dim))
    retriever.build_index()
    module = make_module(
        "",
        [
            ("do_labeled", 1, "      DO 10 I = 1, 3"),
            ("do_labeled", 1, "      DO 10 I = 1, 3"),
            ("do_labeled", 9, "      DO 20 J = 1, 9"),
        ],
    )
    retriever.retrieve(module)
    (query,) = embedder.queries
    assert query.count("DO loop closed by a statement label") == 1
    assert query.count("DO 10 I = 1, 3") == 1
    assert "DO 20 J = 1, 9" in query


def test_query_is_capped():
    embedder = RecordingEmbedder()
    retriever = Retriever(make_kb(), embedder, LocalVectorIndex(embedder.dim))
    hits = [("do_labeled", i + 1, f"      DO 10 K{i} = 1, {i}" + "X" * 500) for i in range(500)]
    retriever.retrieve(make_module("", hits))
    assert len(embedder.queries[0]) <= 4000


def test_hit_without_snippet_falls_back_to_source_lines():
    embedder = RecordingEmbedder()
    retriever = Retriever(make_kb(), embedder, LocalVectorIndex(embedder.dim))
    source = "      A = 1.0\n      DO 10 I = 1, 3\n   10 CONTINUE\n"
    module = make_module(source, [("do_labeled", 2, "")])
    retriever.retrieve(module)
    assert "DO 10 I = 1, 3" in embedder.queries[0]


def test_retrieve_for_text_ranks_matching_entry_first():
    retriever = make_retriever()
    snippets = retriever.retrieve_for_text(
        "      COMMON /WORK/ A(10), N\n      IF (N .GT. 0) A(1) = 0.0", ["common_block"], k=3
    )
    assert snippets[0].id == "common-block"
    assert len(snippets) == 3


def test_retrieve_for_text_without_features_uses_similarity_only():
    text = "      DO 10 I = 1, 10\n         S = S + X(I)\n   10 CONTINUE"
    snippets = make_retriever().retrieve_for_text(text, k=2)
    assert snippets[0].id == "do-labeled"


def test_retrieve_for_text_rejects_unknown_feature_ids():
    with pytest.raises(ValueError, match="unknown feature"):
        make_retriever().retrieve_for_text("x", ["nope"])


def test_entries_added_after_ranking_do_not_break_coverage_selection():
    retriever = make_retriever()
    ranked = retriever._rank("      DO 10 I = 1, 3", ["do_labeled"], k=3)
    retriever.kb.add(
        entry("late", "late labeled DO note", ("do_labeled",), "      DO 5 J = 1, 1", "pass")
    )
    selected = retriever._select(ranked, ["do_labeled"], k=3)
    assert "do-labeled" in {c.entry.id for c in selected}
    assert "late" not in {c.entry.id for c in selected}


def test_packaged_corpus_covers_computed_goto_and_format_hits():
    kb = KnowledgeBase.default()
    wanted = ("goto_computed", "format_statement")
    for feature_id in wanted:
        assert kb.by_feature(feature_id), f"packaged corpus has no entry for {feature_id}"

    embedder = HashingEmbedder(512)
    retriever = Retriever(kb, embedder, LocalVectorIndex(embedder.dim))
    assert retriever.build_index() == len(kb)
    source = (
        "      SUBROUTINE DISP(K, TOTAL)\n"
        "      GO TO (10, 20), K\n"
        "   10 WRITE (6, 100) TOTAL\n"
        "  100 FORMAT (1X, 'TOTAL:', F10.2)\n"
        "   20 RETURN\n"
        "      END\n"
    )
    module = make_module(
        source,
        [
            ("goto_computed", 2, "GO TO (10, 20), K"),
            ("format_statement", 4, "100 FORMAT (1X, 'TOTAL:', F10.2)"),
        ],
    )
    snippets = retriever.retrieve(module, k=4)
    assert len(snippets) <= 4
    covered = {f for s in snippets for f in s.feature_ids}
    assert set(wanted) <= covered
    assert all("Fortran 77:" in s.text and "Python:" in s.text for s in snippets)
