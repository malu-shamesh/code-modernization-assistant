"""extract_api: dummy arguments, intents, written scalars and function result types."""

from __future__ import annotations

from modernizer.models import ApiParam, ApiSymbol
from modernizer.parsing import analyze, extract_api, unit_dependencies

from .helpers import fixed


def api(source: str) -> dict[str, ApiSymbol]:
    return {s.name: s for s in extract_api(fixed(source))}


def intents(symbol: ApiSymbol) -> dict[str, str]:
    return {p.name: p.intent for p in symbol.params}


def test_symbols_are_lower_case_per_program_unit_in_file_order():
    source = "PROGRAM Main\nEND\nSUBROUTINE Sub(A)\nEND\nFUNCTION Fn(B)\nEND\nBLOCK DATA Init\nEND"
    symbols = extract_api(fixed(source))
    assert [(s.name, s.kind) for s in symbols] == [
        ("main", "program"),
        ("sub", "subroutine"),
        ("fn", "function"),
        ("init", "block_data"),
    ]


def test_param_types_ranks_and_order():
    symbol = api(
        """
        SUBROUTINE S(N, X, D, Z, L, C, A, B)
        INTEGER N
        REAL X
        DOUBLE PRECISION D
        COMPLEX Z
        LOGICAL L
        CHARACTER*4 C
        REAL A(N)
        DIMENSION B(2, 3)
        END
        """
    )["s"]
    assert symbol.params == (
        ApiParam("n", "integer", 0, "in"),
        ApiParam("x", "real", 0, "in"),
        ApiParam("d", "double", 0, "in"),
        ApiParam("z", "complex", 0, "in"),
        ApiParam("l", "logical", 0, "in"),
        ApiParam("c", "character", 0, "in"),
        ApiParam("a", "real", 1, "in"),
        ApiParam("b", "real", 2, "in"),
    )


def test_real_star_8_is_reported_as_double():
    symbol = api("SUBROUTINE S(X)\nREAL*8 X\nEND")["s"]
    assert symbol.params[0].type == "double"


def test_undeclared_dummies_use_implicit_rules_or_unknown_under_implicit_none():
    assert [p.type for p in api("SUBROUTINE S(I, X)\nEND")["s"].params] == ["integer", "real"]
    strict = api("SUBROUTINE S(I)\nIMPLICIT NONE\nEND")["s"]
    assert strict.params[0].type == "unknown"


def test_assigned_scalar_is_an_output_and_listed_in_returns_in_parameter_order():
    symbol = api(
        """
        SUBROUTINE S(A, B, C, D)
        B = A
        D = A
        END
        """
    )["s"]
    assert symbol.returns == ("b", "d")
    assert intents(symbol) == {"a": "in", "b": "out", "c": "in", "d": "out"}


def test_read_before_write_is_inout():
    symbol = api("SUBROUTINE S(A, B)\nA = A + 1.0\nB = 2.0\nEND")["s"]
    assert intents(symbol) == {"a": "inout", "b": "out"}


def test_do_index_and_read_targets_are_written():
    symbol = api(
        """
        SUBROUTINE S(I, N, T, IOS)
        DO 10 I = 1, 3
        10 CONTINUE
        READ(5,*) N
        READ(5,*,IOSTAT=IOS) T
        END
        """
    )["s"]
    assert symbol.returns == ("i", "n", "t", "ios")


def test_internal_file_write_modifies_the_character_dummy():
    symbol = api("SUBROUTINE S(BUF, N)\nCHARACTER*8 BUF\nWRITE(BUF,'(I8)') N\nEND")["s"]
    assert symbol.returns == ("buf",)
    assert intents(symbol)["n"] == "in"


def test_array_dummies_are_inout_when_an_element_is_assigned_and_never_returned():
    symbol = api(
        """
        SUBROUTINE S(A, B, N)
        REAL A(N), B(N)
        INTEGER N
        A(1) = B(1)
        END
        """
    )["s"]
    assert intents(symbol) == {"a": "inout", "b": "in", "n": "in"}
    assert symbol.returns == ()


def test_interprocedural_write_through_a_call_chain():
    symbols = api(
        """
        SUBROUTINE OUTER(X, Y, Z, W)
        REAL X, Y, Z, W
        CALL MIDDLE(X)
        CALL UNKNOWN(Z)
        W = 1.0
        END
        SUBROUTINE MIDDLE(P)
        REAL P
        CALL INNER(P)
        END
        SUBROUTINE INNER(R)
        REAL R
        R = 2.0
        END
        """
    )
    assert symbols["inner"].returns == ("r",)
    assert symbols["middle"].returns == ("p",)
    assert symbols["outer"].returns == ("x", "w")
    assert intents(symbols["outer"])["z"] == "in"  # unknown external: assumed not written


def test_callee_that_only_reads_does_not_make_the_actual_an_output():
    symbols = api(
        """
        SUBROUTINE CALLER(X)
        REAL X
        CALL READER(X)
        END
        SUBROUTINE READER(P)
        REAL P, T
        T = P
        END
        """
    )
    assert symbols["caller"].returns == ()


def test_array_passed_to_writing_callee_and_element_to_scalar_dummy():
    symbols = api(
        """
        SUBROUTINE OUTER(A, B)
        REAL A(3), B(3)
        CALL FILL(A)
        CALL SETONE(B(2))
        END
        SUBROUTINE FILL(V)
        REAL V(3)
        V(1) = 1.0
        END
        SUBROUTINE SETONE(S)
        REAL S
        S = 1.0
        END
        """
    )
    assert intents(symbols["outer"]) == {"a": "inout", "b": "inout"}
    assert symbols["outer"].returns == ()


def test_mutual_recursion_reaches_a_fixpoint():
    symbols = api(
        """
        SUBROUTINE PING(N, M)
        INTEGER N, M
        IF (N .GT. 0) CALL PONG(N - 1, M)
        END
        SUBROUTINE PONG(N, M)
        INTEGER N, M
        IF (N .GT. 0) CALL PING(N - 1, M)
        M = N
        END
        """
    )
    assert symbols["ping"].returns == ("m",)
    assert symbols["pong"].returns == ("m",)


def test_function_result_type_from_prefix_declaration_or_implicit_rule():
    symbols = api(
        """
        REAL FUNCTION A(X)
        A = X
        END
        FUNCTION B(I)
        INTEGER B
        B = I
        END
        FUNCTION C(I)
        C = I
        END
        FUNCTION KOUNT(I)
        KOUNT = I
        END
        DOUBLE PRECISION FUNCTION D(I)
        D = I
        END
        CHARACTER*8 FUNCTION E(I)
        E = 'X'
        END
        """
    )
    assert {n: s.result_type for n, s in symbols.items()} == {
        "a": "real",
        "b": "integer",
        "c": "real",
        "kount": "integer",
        "d": "double",
        "e": "character",
    }


def test_implicit_statement_changes_function_result_type():
    symbol = api("FUNCTION F(X)\nIMPLICIT DOUBLE PRECISION (A-H, O-Z)\nF = X\nEND")
    assert symbol["f"].result_type == "double"
    assert symbol["f"].params[0].type == "double"


def test_function_result_assignment_is_not_an_output():
    symbol = api("FUNCTION F(X)\nF = X\nEND")["f"]
    assert symbol.returns == () and symbol.kind == "function"


def test_function_with_output_argument():
    symbol = api("REAL FUNCTION F(X, IERR)\nF = X\nIERR = 0\nEND")["f"]
    assert symbol.returns == ("ierr",)


def test_alternate_returns_are_counted_and_not_parameters():
    symbol = api("SUBROUTINE S(A, *, B, *)\nIF (A .GT. 0.0) RETURN 1\nB = 1.0\nRETURN 2\nEND")["s"]
    assert symbol.alt_returns == 2
    assert [p.name for p in symbol.params] == ["a", "b"]
    assert symbol.returns == ("b",)


def test_procedure_dummies_are_external_and_never_returned():
    symbol = api(
        """
        SUBROUTINE S(F, G, X)
        EXTERNAL F
        Y = G(X)
        X = F(Y)
        END
        """
    )["s"]
    assert [p.type for p in symbol.params] == ["external", "external", "real"]
    assert symbol.returns == ("x",)


def test_program_and_block_data_have_no_parameters():
    symbols = api("PROGRAM P\nEND\nBLOCK DATA\nEND")
    assert symbols["p"].params == () and symbols["blockdata"].kind == "block_data"


def test_analyze_exposes_the_api():
    module = analyze("a.f", fixed("SUBROUTINE S(A)\nA = 1.0\nEND"))
    assert module.api == extract_api(module.source)
    assert module.api[0].returns == ("a",)


def test_extract_api_never_raises():
    for text in ("", "((", "SUBROUTINE", "SUBROUTINE S(", "\x00"):
        assert isinstance(extract_api(text), tuple)


ENTRY_SOURCE = """
SUBROUTINE RST
N = 0
RETURN
ENTRY ADD(X)
N = N + X
RETURN
ENTRY GET(CNT, *)
CNT = N
IF (N .EQ. 0) RETURN 1
END
SUBROUTINE USER
CALL RST
CALL ADD(1.0)
END
"""


def test_entry_points_get_their_own_api_symbols_after_the_host_unit():
    symbols = api(ENTRY_SOURCE)
    assert list(symbols) == ["rst", "add", "get", "user"]
    assert symbols["rst"].params == ()
    assert [(p.name, p.type, p.intent) for p in symbols["add"].params] == [("x", "real", "in")]
    assert symbols["get"].returns == ("cnt",) and symbols["get"].alt_returns == 1
    assert all(symbols[n].kind == "subroutine" for n in symbols)


def test_entry_dummies_count_as_dummy_arguments_for_detectors_and_dependencies():
    module = analyze("e.f", fixed(ENTRY_SOURCE))
    ids = {h.feature_id for h in module.hits}
    assert {"entry_statement", "alternate_return", "arg_output"} <= ids
    assert unit_dependencies(module) == {"rst": set(), "user": {"rst"}}


def test_open_specifiers_are_inputs_but_iostat_and_inquire_results_are_outputs():
    symbol = api(
        """
        SUBROUTINE S(FNAME, FMTSTR, IOS, FOUND, UNIT)
        CHARACTER*20 FNAME, FMTSTR
        LOGICAL FOUND
        INTEGER IOS, UNIT
        OPEN(UNIT=UNIT, FILE=FNAME, FORM=FMTSTR, IOSTAT=IOS)
        INQUIRE(FILE=FNAME, EXIST=FOUND)
        END
        """
    )["s"]
    assert symbol.returns == ("ios", "found")


def test_write_that_can_be_skipped_is_inout_because_the_caller_value_flows_back():
    symbol = api(
        """
        SUBROUTINE S(N, A, B, C, D, E, F, G)
        INTEGER N, A, B, C, D, E, F, G
        IF (N .LT. 0) A = 1
        IF (N .GT. 5) THEN
          B = 2
        ENDIF
        DO 10 I = 1, N
          C = I
        10 CONTINUE
        IF (N .EQ. 0) RETURN
        D = 5
        IF (N .EQ. 1) GO TO 20
        E = 7
        20 F = 9
        READ(5,*,END=30) G
        30 CONTINUE
        END
        """
    )["s"]
    assert intents(symbol) == {
        "n": "in",
        "a": "inout",
        "b": "inout",
        "c": "inout",
        "d": "inout",  # after a conditional RETURN
        "e": "inout",  # after a conditional jump
        "f": "inout",  # the label can be reached around the unconditional statements
        "g": "inout",  # every statement after the first jump is skippable
    }
    assert symbol.returns == ("a", "b", "c", "d", "e", "f", "g")


def test_unconditional_writes_before_any_branch_stay_out():
    symbol = api(
        """
        SUBROUTINE S(N, A, B, C)
        INTEGER N, A, B, C
        A = 1
        IF (N .EQ. 0) RETURN
        B = A
        END
        """
    )["s"]
    assert intents(symbol)["a"] == "out"
    assert intents(symbol)["b"] == "inout"


def test_substring_assignment_keeps_the_old_value_so_the_dummy_is_inout():
    symbols = api(
        """
        SUBROUTINE PART(S)
        CHARACTER*8 S
        S(1:2) = 'AB'
        END
        SUBROUTINE WHOLE(S)
        CHARACTER*8 S
        S = 'AB'
        END
        """
    )
    assert intents(symbols["part"]) == {"s": "inout"}
    assert intents(symbols["whole"]) == {"s": "out"}


def test_a_conditional_call_to_a_writing_callee_is_inout():
    symbols = api(
        """
        SUBROUTINE OUTER(N, X, Y)
        REAL X, Y
        CALL SETIT(X)
        IF (N .GT. 0) CALL SETIT(Y)
        END
        SUBROUTINE SETIT(P)
        REAL P
        P = 1.0
        END
        """
    )
    assert intents(symbols["outer"]) == {"n": "in", "x": "out", "y": "inout"}


def test_statement_function_formals_do_not_count_as_reads_of_unit_variables():
    symbol = api(
        """
        SUBROUTINE S(X, Y)
        REAL X, Y, F
        F(X) = X*2.0
        X = 1.0
        Y = F(X)
        END
        """
    )["s"]
    assert intents(symbol)["x"] == "out"
