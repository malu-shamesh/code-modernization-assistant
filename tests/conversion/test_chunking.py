from __future__ import annotations

from modernizer.conversion.chunking import (
    build_chunks,
    chunk_api,
    has_shared_state,
    other_chunks_note,
    pack_units,
    state_lines,
)
from modernizer.models import CodeUnit, FeatureHit, ModuleMetrics, ParsedModule

from .helpers import AXPY, BUMP, TOTAL, axpy_module, make_module, state_module


def texts(pairs):
    return [t for _, t in pairs]


def test_pack_units_groups_in_order_and_isolates_oversized_units():
    module = state_module()
    sizes = [len(u.source) for u in module.units]
    assert pack_units(module, 5) == [[0], [1], [2], [3]]
    assert pack_units(module, 10_000) == [[0, 1, 2, 3]]
    groups = pack_units(module, sizes[0] + 2 + sizes[1])
    assert groups[0] == [0, 1]
    assert [i for g in groups for i in g] == [0, 1, 2, 3]


def test_build_chunks_keeps_original_hit_lines_and_assigns_each_hit_once():
    module = state_module()
    chunks = build_chunks(module, [[0, 1], [2, 3]])
    first, second = chunks
    assert first.source == module.units[0].source + "\n\n" + module.units[1].source
    assert [(h.feature_id, h.line) for h in first.module.hits] == [
        ("block_data", 1),
        ("common_block", 2),
        ("data_statement", 3),
        ("common_block", 8),
    ]
    assert [h.line for h in second.module.hits] == [13, 22]
    assert first.names == {"init", "bump"} and second.names == {"total", "axpy"}
    assert len(first.module.hits) + len(second.module.hits) == len(module.hits)


def test_hit_outside_every_unit_goes_to_the_next_unit_or_the_last():
    module = make_module([("a", "subroutine", BUMP), ("b", "subroutine", TOTAL)])
    module.hits = [
        FeatureHit("common_block", 6, 6, "between units"),
        FeatureHit("common_block", 99, 99, "after everything"),
    ]
    first, second = build_chunks(module, [[0], [1]])
    assert [h.line for h in first.module.hits] == []
    assert [h.line for h in second.module.hits] == [6, 99]


def test_chunk_api_matches_unit_names_and_entry_points():
    module = state_module()
    chunks = build_chunks(module, [[0, 1], [2, 3]])
    assert [s.name for s in chunk_api(module, chunks[0])] == ["bump"]
    assert [s.name for s in chunk_api(module, chunks[1])] == ["total", "axpy"]

    entry_unit = CodeUnit(
        "calc", "subroutine", "      SUBROUTINE CALC(X)\n      ENTRY CALC2(X, Y)\n      END", 1, 3
    )
    entry_module = ParsedModule(
        "e.f", entry_unit.source, [entry_unit], [], ModuleMetrics(), api=module.api[:1]
    )
    entry_module.api = (
        *module.api[:1],
        type(module.api[0])("calc2", "subroutine"),
    )
    (chunk,) = build_chunks(entry_module, [[0]])
    assert [s.name for s in chunk_api(entry_module, chunk)] == ["calc2"]


def test_other_chunks_note_lists_signatures_of_the_other_chunks_only():
    module = state_module()
    chunks = build_chunks(module, [[0], [1, 2], [3]])
    note = other_chunks_note(module, chunks, 1)
    assert "def axpy(n, a, x, y, ierr)" in note
    assert "def bump" not in note and "def total" not in note
    # a chunk without API symbols falls back to its unit kinds and names
    assert "- block_data init" in note


def test_has_shared_state():
    assert has_shared_state(state_module())
    assert not has_shared_state(axpy_module())
    only_data = make_module(
        [("a", "subroutine", BUMP)], [FeatureHit("data_statement", 1, 1, "DATA X /1/")]
    )
    assert not has_shared_state(only_data)
    block = make_module([("init", "block_data", "      BLOCK DATA INIT\n      END")])
    assert has_shared_state(block)


def test_state_lines_keep_headers_declarations_and_state_statements():
    pairs = state_lines(state_module())
    assert pairs == [
        (1, "      BLOCK DATA INIT"),
        (2, "      COMMON /CNT/ NCALLS"),
        (3, "      DATA NCALLS /0/"),
        (6, "      SUBROUTINE BUMP(N)"),
        (7, "      INTEGER N"),
        (8, "      COMMON /CNT/ NCALLS"),
        (12, "      INTEGER FUNCTION TOTAL()"),
        (13, "      COMMON /CNT/ NCALLS"),
    ]


def test_state_lines_skip_executable_code_comments_and_assignments_that_look_like_decls():
    src = "\n".join(
        [
            "      SUBROUTINE S(N)",
            "C     DATA in a comment /1/",
            "      REAL W(10), V",
            "      SAVE W",
            "      REALX = 3.0",
            "      DATAX = N/2",
            "      DATA V / 1.5 /, K / 7 /",
            "      PARAMETER (NMAX = 10)",
            "      CHARACTER*8 NAME",
            "      DATA NAME / 'A=B!X' /",
            "      DATA (W(I), I = 1, 3)",
            "     &     / 1., 2., 3. /",
            "      DO 10 I = 1, N",
            "   10 CONTINUE",
            "      END",
        ]
    )
    module = make_module([("s", "subroutine", src)])
    numbers = [n for n, _ in state_lines(module)]
    assert numbers == [1, 3, 4, 7, 8, 9, 10, 11, 12]


def test_state_lines_use_hits_for_unusual_statements():
    src = "      SUBROUTINE S\n      ENTRY2 = 1\n      END"
    module = make_module([("s", "subroutine", src)], [FeatureHit("save_statement", 2, 2, "x")])
    assert [n for n, _ in state_lines(module)] == [1, 2]


def test_units_without_state_contribute_nothing():
    assert state_lines(axpy_module()) == []
    assert AXPY.startswith("      SUBROUTINE")
