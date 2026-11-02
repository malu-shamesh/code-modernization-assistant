"""validate_conversion: one test per issue code plus the accepted shapes."""

from __future__ import annotations

import textwrap

import pytest

from modernizer.models import ParsedModule
from modernizer.parsing import analyze, validate_conversion

from .helpers import fixed

LEGACY = analyze(
    "legacy.f",
    fixed(
        """
        SUBROUTINE SOLVE(A, N, X, IERR)
        REAL A(N), X
        INTEGER N, IERR
        X = 1.0
        IERR = 0
        END
        REAL FUNCTION AREA(R)
        AREA = 3.14*R*R
        END
        REAL FUNCTION BOTH(Z, K)
        BOTH = Z
        K = 1
        END
        SUBROUTINE LEAVE(A, *)
        IF (A .GT. 0.0) RETURN 1
        RETURN
        END
        SUBROUTINE QUIET(A)
        REAL A(3)
        A(1) = 0.0
        END
        PROGRAM MAIN
        END
        """
    ),
)

GOOD = textwrap.dedent(
    """\
    import math

    import numpy as np
    from f77rt import idiv

    def solve(a, n, x, ierr):
        x = 1.0
        ierr = 0
        return x, ierr

    def area(r):
        return 3.14 * r * r

    def both(z, k):
        k = 1
        return z, k

    def leave(a):
        if a > 0.0:
            return 1
        return 0

    def quiet(a):
        a[0] = 0.0

    def main():
        pass

    if __name__ == "__main__":
        main()
    """
)


@pytest.fixture(autouse=True)
def runtime_names(monkeypatch):
    monkeypatch.setattr(
        "modernizer.runtime.public_names", lambda: frozenset({"idiv", "FortranStop"})
    )


def codes(source: str, legacy: ParsedModule = LEGACY) -> list[str]:
    return [issue.code for issue in validate_conversion(legacy, source).issues]


def test_legacy_api_is_what_the_tests_assume():
    api = {s.name: s for s in LEGACY.api}
    assert api["solve"].returns == ("x", "ierr")
    assert api["both"].returns == ("k",) and api["both"].kind == "function"
    assert api["leave"].alt_returns == 1
    assert api["quiet"].returns == ()


def test_good_conversion_is_ok():
    report = validate_conversion(LEGACY, GOOD)
    assert report.ok and report.issues == []


@pytest.mark.parametrize("source", ["", "   \n\t\n", None, 42])
def test_empty_output(source):
    report = validate_conversion(LEGACY, source)  # type: ignore[arg-type]
    assert not report.ok
    assert [i.code for i in report.issues] == ["empty_output"]


def test_syntax_error_reports_the_line():
    report = validate_conversion(LEGACY, "x = 1\ndef solve(:\n    pass\n")
    assert [i.code for i in report.issues] == ["syntax_error"]
    assert report.issues[0].line == 2


def test_compile_time_syntax_errors_are_caught_too():
    assert codes("return 1\n") == ["syntax_error"]


def test_fortran_text_that_does_not_compile_is_a_syntax_error():
    assert codes("x = 1.0d0 .eq. y\n") == ["syntax_error"]
    assert codes("go to 10\n") == ["syntax_error"]


@pytest.mark.parametrize(
    "line",
    [
        "goto(10)",
        "go_to(20)",
        "write(6, 100)",
        "x = a.eq.b",
        "10",
    ],
)
def test_fortran_remnants_that_still_compile(line):
    source = GOOD + f"\ndef extra(a, b, x):\n    {line}\n"
    assert "fortran_remnant" in codes(source)


def test_remnant_checks_are_precise():
    benign = GOOD + textwrap.dedent(
        '''
        import operator

        def write(unit, text):
            return text

        def helper(a, b):
            """A .EQ. B means 10 CONTINUE."""
            # 10 CONTINUE
            s = "a .eq. b"
            write(6, s)
            return operator.eq(a, b) or operator.lt(a, b)
        '''
    )
    assert codes(benign) == []


def test_missing_function_is_api_changed():
    source = GOOD.replace("def area(r):", "def surface(r):")
    report = validate_conversion(LEGACY, source)
    assert [i.code for i in report.issues] == ["api_changed"]
    assert "area" in report.issues[0].message


def test_parameter_names_and_order_must_match_exactly():
    assert codes(GOOD.replace("def solve(a, n, x, ierr)", "def solve(a, n, ierr, x)")) == [
        "api_changed"
    ]
    assert codes(GOOD.replace("def solve(a, n, x, ierr)", "def solve(a, n, x)")) == ["api_changed"]
    assert codes(GOOD.replace("def solve(a, n, x, ierr)", "def solve(a, n, x, ierr, extra=0)")) == [
        "api_changed"
    ]
    assert codes(GOOD.replace("def solve(a, n, x, ierr)", "def solve(a, n, x, *ierr)")) == [
        "api_changed"
    ]


def test_tuple_return_must_have_the_right_length():
    assert codes(GOOD.replace("return x, ierr", "return x")) == []  # a bare name cannot be checked
    assert codes(GOOD.replace("return x, ierr", "return x, ierr, 0")) == ["api_changed"]
    assert codes(GOOD.replace("return z, k", "return z")) == []
    assert codes(GOOD.replace("return z, k", "return z, k, 1")) == ["api_changed"]


def test_function_must_return_a_value():
    assert codes(GOOD.replace("return 3.14 * r * r", "pass")) == ["api_changed"]
    assert codes(GOOD.replace("return 3.14 * r * r", "return None")) == ["api_changed"]


def test_bare_function_result_must_not_be_a_tuple():
    assert codes(GOOD.replace("return 3.14 * r * r", "return 3.14, r")) == ["api_changed"]


def test_subroutine_without_outputs_must_not_return_a_value():
    assert codes(GOOD.replace("a[0] = 0.0", "a[0] = 0.0\n    return 5")) == ["api_changed"]
    assert codes(GOOD.replace("a[0] = 0.0", "a[0] = 0.0\n    return")) == []
    assert codes(GOOD.replace("a[0] = 0.0", "a[0] = 0.0\n    return None")) == []


def test_subroutine_with_outputs_must_return_them():
    assert codes(GOOD.replace("return x, ierr", "pass")) == ["api_changed"]


def test_alternate_return_adds_one_trailing_value():
    # no outputs + one alternate return -> a bare int (one item)
    assert codes(GOOD.replace("return 1\n", "return 1, 2\n")) == ["api_changed"]
    legacy = analyze(
        "alt.f",
        fixed("SUBROUTINE ALT2(A, B, *)\nB = A\nIF (A .GT. 0.0) RETURN 1\nEND"),
    )
    assert legacy.api[0].returns == ("b",) and legacy.api[0].alt_returns == 1
    assert codes("def alt2(a, b):\n    return b, 0\n", legacy) == []
    assert codes("def alt2(a, b):\n    return b, 0, 1\n", legacy) == ["api_changed"]


def test_nested_function_returns_do_not_count():
    source = GOOD.replace(
        "def quiet(a):\n    a[0] = 0.0",
        "def quiet(a):\n    def inner():\n        return 5\n    a[0] = inner() * 0",
    )
    assert codes(source) == []


def test_program_and_block_data_do_not_need_a_function():
    legacy = analyze("m.f", fixed("PROGRAM P\nEND\nBLOCK DATA\nEND"))
    assert codes("x = 1\n", legacy) == []


def test_unknown_runtime_helper():
    assert codes("from f77rt import idiv, no_such_helper\n" + GOOD) == ["unknown_runtime_helper"]
    assert codes("from f77rt import *\n" + GOOD) == ["unknown_runtime_helper"]
    assert codes("import f77rt\nf77rt.idiv(1, 2)\n" + GOOD) == []
    assert codes("import f77rt as rt\nrt.missing(1)\n" + GOOD) == ["unknown_runtime_helper"]


def test_unknown_helper_check_uses_the_current_runtime_names(monkeypatch):
    monkeypatch.setattr("modernizer.runtime.public_names", lambda: frozenset({"other"}))
    assert codes(GOOD) == ["unknown_runtime_helper"]


@pytest.mark.parametrize(
    "line",
    ["import pandas", "from scipy import linalg", "import requests as r", "from . import sibling"],
)
def test_disallowed_import(line):
    assert codes(line + "\n" + GOOD) == ["disallowed_import"]


@pytest.mark.parametrize(
    "line",
    ["import os", "import sys", "import numpy.linalg", "from math import sqrt", "import f77rt"],
)
def test_allowed_imports(line):
    assert codes(line + "\n" + GOOD) == []


def test_issue_lines_and_feedback_text():
    report = validate_conversion(LEGACY, GOOD.replace("return x, ierr", "return x, ierr, 0"))
    assert report.issues[0].line is not None
    assert "api_changed" in report.feedback()


def test_ok_is_true_iff_there_are_no_issues():
    for source in (GOOD, "def solve(:", "", GOOD.replace("def area", "def other")):
        report = validate_conversion(LEGACY, source)
        assert report.ok == (not report.issues)


def test_never_raises_on_odd_input():
    for source in ("\x00", "def f(:" * 5000, "(" * 50000, "x = '" + "a" * 100000):
        assert validate_conversion(LEGACY, source).ok is False
    empty = ParsedModule("x", "", [], [], LEGACY.metrics)
    assert validate_conversion(empty, GOOD).ok


# --- the calling convention: dummies, keywords, ENTRY, BLOCK DATA, alternate returns ---------

CONVENTION = analyze(
    "conv.f",
    fixed(
        """
        SUBROUTINE PUREOUT(N, RES)
        INTEGER N
        RES = N * 2.0
        END
        SUBROUTINE STARS(A, *, B, *)
        IF (A .GT. 0.0) RETURN 1
        IF (A .LT. 0.0) RETURN 2
        B = A
        END
        SUBROUTINE ALTONLY(A, *)
        IF (A .GT. 0.0) RETURN 1
        END
        REAL FUNCTION FALT(A, K, *)
        FALT = A
        K = 2
        IF (A .GT. 0.0) RETURN 1
        END
        SUBROUTINE KW(IN, IS, X)
        INTEGER IN, IS
        X = IN + IS
        END
        SUBROUTINE FIRST(A)
        ENTRY SECOND(B, D)
        D = B
        END
        BLOCK DATA INIT
        COMMON /BLK/ Q
        DATA Q /1.0/
        END
        """
    ),
)


def test_convention_api_is_what_the_tests_assume():
    api = {s.name: s for s in CONVENTION.api}
    assert [p.name for p in api["pureout"].params] == ["n", "res"]
    assert api["pureout"].returns == ("res",)
    assert [p.name for p in api["stars"].params] == ["a", "b"] and api["stars"].alt_returns == 2
    assert api["altonly"].returns == () and api["altonly"].alt_returns == 1
    assert [p.name for p in api["kw"].params] == ["in", "is", "x"]
    assert {"first", "second"} <= set(api)
    assert api["init"].kind == "block_data"


CONVENTION_GOOD = textwrap.dedent(
    """\
    def pureout(n, res):
        return n * 2.0

    def stars(a, b):
        if a > 0.0:
            return b, 1
        if a < 0.0:
            return b, 2
        b = a
        return b, 0

    def altonly(a):
        if a > 0.0:
            return 1
        return 0

    def falt(a, k):
        k = 2
        if a > 0.0:
            return a, k, 1
        return a, k, 0

    def kw(in_, is_, x):
        return in_ + is_

    def first(a):
        pass

    def second(b, d):
        return b
    """
)


def test_convention_good_conversion_is_ok():
    report = validate_conversion(CONVENTION, CONVENTION_GOOD)
    assert report.issues == [] and report.ok


def test_pure_output_stays_a_parameter():
    assert codes(CONVENTION_GOOD.replace("def pureout(n, res)", "def pureout(n)"), CONVENTION) == [
        "api_changed"
    ]


def test_star_dummies_are_not_parameters():
    source = CONVENTION_GOOD.replace("def stars(a, b)", "def stars(a, star, b, star2)")
    assert codes(source, CONVENTION) == ["api_changed"]
    source = CONVENTION_GOOD.replace("def stars(a, b)", "def stars(a, b, *alt)")
    assert codes(source, CONVENTION) == ["api_changed"]


@pytest.mark.parametrize("spelling", ["in__", "in2", "_in", "in_x", "IN", "kwin"])
def test_keyword_dummy_accepts_only_the_trailing_underscore(spelling):
    source = CONVENTION_GOOD.replace("def kw(in_, ", f"def kw({spelling}, ")
    assert source != CONVENTION_GOOD
    assert codes(source.replace("in_ + is_", "0"), CONVENTION) == ["api_changed"]


def test_keyword_dummy_cannot_keep_its_bare_spelling():
    source = CONVENTION_GOOD.replace("def kw(in_, ", "def kw(in, ").replace("in_ + is_", "0")
    assert codes(source, CONVENTION) == ["syntax_error"]


def test_a_bare_return_in_a_unit_that_must_return_values_is_reported():
    source = CONVENTION_GOOD.replace("return a, k, 1", "return")
    assert codes(source, CONVENTION) == ["api_changed"]
    source = CONVENTION_GOOD.replace("return 1\n    return 0", "return\n    return 0")
    assert codes(source, CONVENTION) == ["api_changed"]


def test_keyword_dummy_underscore_is_required_for_every_keyword_dummy():
    source = CONVENTION_GOOD.replace("def kw(in_, is_, x)", "def kw(in_, is2, x)")
    assert codes(source, CONVENTION) == ["api_changed"]
    source = CONVENTION_GOOD.replace("def kw(in_, is_, x)", "def kw(in_, is_, x_)")
    assert codes(source, CONVENTION) == ["api_changed"]


def test_non_keyword_dummies_do_not_get_an_underscore():
    legacy = analyze("t.f", fixed("SUBROUTINE T(PRINT, MATCH, NONE)\nEND"))
    assert [p.name for p in legacy.api[0].params] == ["print", "match", "none"]
    assert codes("def t(print, match, none):\n    pass\n", legacy) == []
    assert codes("def t(print_, match, none):\n    pass\n", legacy) == ["api_changed"]


def test_every_entry_needs_its_own_function_with_its_own_parameters():
    assert codes(CONVENTION_GOOD.replace("def second(b, d)", "def other(b, d)"), CONVENTION) == [
        "api_changed"
    ]
    assert codes(
        CONVENTION_GOOD.replace("def second(b, d)", "def second(a, b, d)"), CONVENTION
    ) == ["api_changed"]


def test_block_data_needs_no_callable():
    assert "init" not in CONVENTION_GOOD
    assert codes(CONVENTION_GOOD, CONVENTION) == []


def test_alternate_return_only_subroutine_returns_a_bare_int():
    assert codes(CONVENTION_GOOD.replace("return 1\n    return 0", "return 1, 0"), CONVENTION) == [
        "api_changed"
    ]
    assert codes(CONVENTION_GOOD.replace("return 1\n    return 0", "return (1,)"), CONVENTION) == [
        "api_changed"
    ]
    source = CONVENTION_GOOD.replace("def altonly(a):\n", "def altonly(a):\n    return 0\n")
    assert codes(source, CONVENTION) == []


def test_outputs_plus_alternate_returns_need_outputs_plus_one_items():
    assert codes(CONVENTION_GOOD.replace("return b, 2", "return b, 2, 9"), CONVENTION) == [
        "api_changed"
    ]
    assert codes(CONVENTION_GOOD.replace("return b, 2", "return 2"), CONVENTION) == []
    assert codes(CONVENTION_GOOD.replace("return a, k, 1", "return a, 1"), CONVENTION) == [
        "api_changed"
    ]
    assert codes(CONVENTION_GOOD.replace("return a, k, 1", "return a, k, 1, 0"), CONVENTION) == [
        "api_changed"
    ]
