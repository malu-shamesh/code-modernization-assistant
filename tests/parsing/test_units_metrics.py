"""Program units, dependencies and module metrics."""

from __future__ import annotations

import pytest

from modernizer.features import FEATURES, is_semantic
from modernizer.parsing import analyze, unit_dependencies

from .helpers import fixed

SOURCE = fixed(
    """
    C driver and one worker
    PROGRAM DEMO
    INTEGER I, N
    COMMON /A/ N
    COMMON /B/ X
    COMMON /A/ M
    CALL WORK(N)
    CALL EXTERN(X)
    Y = FARFN(X)
    DO 10 I = 1, N
    IF (I .GT. 2 .AND. N .LT. 9) THEN
    N = N - 1
    ELSE IF (I .EQ. 1) THEN
    N = N + 1
    END IF
    10 CONTINUE
    STOP
    END

    SUBROUTINE WORK(K)
    GO TO (10, 20, 30), K
    10 CONTINUE
    20 CONTINUE
    30 RETURN
    END
    """
)


@pytest.fixture(scope="module")
def module():
    return analyze("demo.f", SOURCE)


def test_units_names_kinds_and_line_ranges(module):
    assert [(u.name, u.kind) for u in module.units] == [("demo", "program"), ("work", "subroutine")]
    demo, work = module.units
    assert (demo.start_line, demo.end_line) == (2, 18)
    assert (work.start_line, work.end_line) == (20, 25)
    lines = SOURCE.splitlines()
    assert demo.source == "\n".join(lines[1:18])
    assert work.source.splitlines()[0].strip() == "SUBROUTINE WORK(K)"


def test_unit_hits_fall_inside_the_unit(module):
    demo, work = module.units
    assert {"common_block", "do_labeled", "stop_pause"} <= {h.feature_id for h in demo.hits}
    assert "goto_computed" in {h.feature_id for h in work.hits}
    for unit in module.units:
        assert all(unit.start_line <= h.line <= unit.end_line for h in unit.hits)


def test_unit_defines_and_references(module):
    demo, work = module.units
    assert {"i", "n", "x", "m", "y"} <= set(demo.defines)
    assert demo.references == ("work", "extern", "farfn")
    assert work.references == ()


def test_unit_dependencies_only_name_units_of_the_module(module):
    assert unit_dependencies(module) == {"demo": {"work"}, "work": set()}


def test_metrics(module):
    m = module.metrics
    assert m.loc == len(SOURCE.splitlines())
    assert m.sloc == m.loc - 2  # one comment line and one blank line
    assert m.n_functions == 1
    assert m.n_common_blocks == 2
    assert m.max_nesting == 2
    # 1 + DO + IF + .AND. + ELSE IF + computed GOTO (3 targets -> 2)
    assert m.cyclomatic == 1 + 1 + 1 + 1 + 1 + 2
    assert m.calls == ("extern", "farfn")
    assert not m.has_syntax_errors


def test_feature_metrics_are_consistent_with_the_registry(module):
    m = module.metrics
    assert m.n_feature_hits == len(module.hits)
    assert m.distinct_features == len(module.feature_ids)
    assert m.weighted_feature_score == pytest.approx(
        sum(FEATURES[h.feature_id].weight for h in module.hits)
    )
    assert m.semantic_hits == sum(1 for h in module.hits if is_semantic(h.feature_id))


def test_arithmetic_if_adds_two_branches_and_logical_if_one():
    base = analyze("a.f", fixed("X = 1\nEND")).metrics.cyclomatic
    arithmetic = analyze("a.f", fixed("IF (X) 1, 2, 3\n1 CONTINUE\n2 CONTINUE\n3 CONTINUE\nEND"))
    logical = analyze("a.f", fixed("IF (X .GT. 0) Y = 1\nEND"))
    assert arithmetic.metrics.cyclomatic == base + 2
    assert logical.metrics.cyclomatic == base + 1


def test_or_operator_counts_and_nested_loops_set_depth():
    source = fixed(
        """
        DO 20 I = 1, 3
        DO 10 J = 1, 3
        IF (I .EQ. 1 .OR. J .EQ. 2) X = 1
        10 CONTINUE
        20 CONTINUE
        END
        """
    )
    m = analyze("a.f", source).metrics
    assert m.max_nesting == 2
    assert m.cyclomatic == 1 + 2 + 1 + 1


def test_shared_terminal_loops_count_depth_correctly():
    source = fixed("DO 10 I = 1, 3\nDO 10 J = 1, 3\nDO 10 K = 1, 3\n10 X = 0\nEND")
    assert analyze("a.f", source).metrics.max_nesting == 3


def test_main_program_without_program_statement():
    module = analyze("a.f", fixed("X = 1.0\nPRINT *, X\nEND"))
    assert [(u.name, u.kind) for u in module.units] == [("main", "program")]


def test_block_data_unit():
    module = analyze("a.f", fixed("BLOCK DATA INIT\nCOMMON /B/ X\nDATA X /1.0/\nEND"))
    assert [(u.name, u.kind) for u in module.units] == [("init", "block_data")]
    assert module.metrics.n_common_blocks == 1
    assert module.metrics.n_functions == 0


def test_unit_without_end_statement_runs_to_the_last_line():
    module = analyze("a.f", fixed("SUBROUTINE A\nX = 1.0\nSUBROUTINE B\nY = 2.0\n"))
    assert [(u.name, u.start_line, u.end_line) for u in module.units] == [("a", 1, 2), ("b", 3, 4)]


@pytest.mark.parametrize(
    "text",
    [
        "IF (X .GT. 0) THEN\nX = 1\nEND",
        "X = = 3\nEND",
        "this is not fortran @@@\nEND",
        "DO 10 I = 1, 3\nX = 1\nEND",
        "CALL F(((1)\nEND",
        "ELSE\nX = 1\nEND",
        "END IF\nEND",
    ],
)
def test_syntax_errors_are_flagged(text):
    assert analyze("a.f", fixed(text)).metrics.has_syntax_errors


def test_legacy_constructs_are_not_syntax_errors():
    source = fixed(
        """
        SUBROUTINE S(A, *)
        COMMON // X
        CALL F(A, *10)
        ASSIGN 10 TO K
        GO TO K
        WRITE(6,20)
        20 FORMAT(5HHELLO)
        10 RETURN 1
        END
        """
    )
    assert not analyze("a.f", source).metrics.has_syntax_errors


def test_empty_module_is_not_a_syntax_error():
    module = analyze("a.f", "")
    assert module.units == [] and module.hits == []
    assert not module.metrics.has_syntax_errors


def test_analyze_takes_only_path_and_source():
    source = fixed("X = 1\nEND")
    assert analyze("a.f", source).units[0].name == "main"
    with pytest.raises(TypeError):
        analyze("a.f", source, sibling_modules={"b"})  # type: ignore[call-arg]


def test_calls_leave_out_entry_points_defined_in_the_same_file():
    source = fixed(
        """
        SUBROUTINE RST
        ENTRY ADD(X)
        END
        SUBROUTINE USER
        CALL RST
        CALL ADD(1.0)
        CALL ELSEWHERE
        END
        """
    )
    assert analyze("a.f", source).metrics.calls == ("elsewhere",)
