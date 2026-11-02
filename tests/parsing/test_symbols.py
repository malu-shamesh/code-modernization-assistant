"""Per-unit symbol tables, implicit typing rules and expression types."""

from __future__ import annotations

from modernizer.parsing import detect_features, symbol_tables
from modernizer.parsing.exprs import parse_expr
from modernizer.parsing.symbols import expr_type

from .helpers import fixed

TABLE_SOURCE = fixed(
    """
    SUBROUTINE S(A, N, F)
    IMPLICIT DOUBLE PRECISION (A-H)
    INTEGER N
    DIMENSION A(N, 3)
    CHARACTER*8 NAME
    PARAMETER (PI = 3.14D0)
    EXTERNAL F
    INTRINSIC SIN
    COMMON /BLK/ X, K(10)
    SAVE K
    G(T) = T*T
    Z = 1
    END
    """
)


def table(source: str, unit: str = "s"):
    return symbol_tables(source)[unit]


def test_declared_names_and_attributes():
    scope = table(TABLE_SOURCE)
    assert scope.dummies == ["a", "n", "f"]
    assert scope.symbols["n"].type == "integer" and scope.symbols["n"].is_dummy
    assert scope.symbols["name"].type == "character" and scope.symbols["name"].char_len == "8"
    assert scope.symbols["pi"].is_parameter
    assert scope.symbols["f"].is_external
    assert scope.symbols["sin"].is_intrinsic
    assert scope.symbols["x"].common == "blk" and scope.symbols["k"].common == "blk"
    assert scope.symbols["k"].saved
    assert "g" in scope.stmt_functions and scope.stmt_functions["g"].dummies == ("t",)


def test_arrays_and_ranks():
    scope = table(TABLE_SOURCE)
    assert scope.is_array("a") and scope.symbols["a"].rank == 2
    assert scope.symbols["k"].rank == 1
    assert not scope.is_array("n")


def test_implicit_rules_default_and_overridden():
    scope = table(TABLE_SOURCE)
    assert scope.type_of_name("a") == "double"  # IMPLICIT DOUBLE PRECISION (A-H)
    assert scope.type_of_name("i") == "integer"  # default I-N rule survives
    assert scope.type_of_name("z") == "real"  # default rule outside the IMPLICIT range
    assert scope.type_of_name("name") == "character"  # explicit declaration wins


def test_default_implicit_typing_without_statement():
    scope = table(fixed("SUBROUTINE S\nEND"))
    assert [scope.type_of_name(n) for n in ("i", "n", "a", "x", "kount", "total")] == [
        "integer",
        "integer",
        "real",
        "real",
        "integer",
        "real",
    ]


def test_implicit_none_is_honoured():
    scope = table(fixed("SUBROUTINE S\nIMPLICIT NONE\nINTEGER I\nEND"))
    assert scope.implicit_none
    assert scope.type_of_name("i") == "integer"
    assert scope.type_of_name("x") is None


def test_function_result_type_sources():
    source = fixed(
        """
        REAL FUNCTION F(X)
        F = X
        END
        FUNCTION G(I)
        INTEGER G
        G = I
        END
        FUNCTION H(I)
        H = I
        END
        """
    )
    tables = symbol_tables(source)
    assert tables["f"].result_type == "real"
    assert tables["g"].result_type == "integer"
    assert tables["h"].result_type is None
    assert tables["h"].implicit_type("h") == "real"


def test_expression_types_follow_the_symbol_table():
    scope = table(fixed("SUBROUTINE S\nINTEGER N\nDOUBLE PRECISION D\nCOMPLEX Z\nEND"))
    cases = {
        "N": "integer",
        "N + 1": "integer",
        "N / 2": "integer",
        "N * 2.0": "real",
        "N + D": "double",
        "D * Z": "complex",
        "1.0D0": "double",
        "'abc'": "character",
        ".TRUE.": "logical",
        "N .GT. 2": "logical",
        "ABS(N)": "integer",
        "INT(D)": "integer",
        "FLOAT(N)": "real",
        "SQRT(D)": "double",
        "'a' // 'b'": "character",
        "N ** 2": "integer",
    }
    for text, expected in cases.items():
        assert expr_type(parse_expr(text), scope) == expected, text


def test_unknown_type_is_none_not_a_guess():
    scope = table(fixed("SUBROUTINE S\nIMPLICIT NONE\nEND"))
    assert expr_type(parse_expr("X + 1"), scope) is None


def implicit_hits(source: str) -> list:
    return [h for h in detect_features(fixed(source)) if h.feature_id == "implicit_typing"]


def test_implicit_typing_reports_the_first_use_of_each_undeclared_name_once():
    hits = implicit_hits("SUBROUTINE S\nINTEGER I\nX = 1.0\nY = X + 2.0\nX = Y\nEND")
    assert [h.line for h in hits] == [3, 4]


def test_implicit_typing_ignores_declared_names_functions_and_intrinsics():
    source = """
    REAL FUNCTION F(X)
    REAL X
    INTEGER I
    EXTERNAL G
    F = SQRT(X) + G(X)
    I = 1
    END
    """
    assert implicit_hits(source) == []


def test_implicit_typing_not_reported_under_implicit_none():
    assert implicit_hits("SUBROUTINE S\nIMPLICIT NONE\nINTEGER I\nI = 1\nEND") == []


def test_implicit_typing_counts_undeclared_dummies_and_parameter_names():
    hits = implicit_hits("SUBROUTINE S(N)\nPARAMETER (M = 2)\nK = N + M\nEND")
    assert [h.line for h in hits] == [2, 3]


def test_statement_function_dummies_are_not_variables():
    assert implicit_hits("SUBROUTINE S\nREAL F\nF(T) = T*2.0\nEND") == []


def test_unit_scoping_is_independent():
    source = "SUBROUTINE A\nX = 1.0\nEND\nSUBROUTINE B\nIMPLICIT NONE\nEND"
    assert len(implicit_hits(source)) == 1


def test_implicit_undefined_means_implicit_none():
    scope = table(fixed("SUBROUTINE S\nIMPLICIT UNDEFINED (A-Z)\nEND"))
    assert scope.implicit_none


def test_percent_val_is_parsed_and_reported_as_a_vendor_extension():
    source = fixed("SUBROUTINE S(N)\nINTEGER N\nCALL EXTRN(%VAL(N))\nEND")
    assert "vendor_extension" in {h.feature_id for h in detect_features(source)}
