"""Garbage input never raises; concurrent use is safe; free-form and tab-form files work."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from modernizer.models import ParsedModule
from modernizer.parsing import (
    analyze,
    detect_features,
    extract_api,
    normalize_source,
    unit_dependencies,
    validate_conversion,
)

from .helpers import fixed
from .test_golden import CONVERTED, LEGACY

GARBAGE = [
    "",
    "\n\n\n",
    "\x00\x01\x02\x03",
    "\ufeff      X = 1\n",
    "(((((((((((((((((((((",
    ")))))",
    "'unterminated string",
    "      CALL F(" + "A," * 800 + "B)\n",
    "      X = " + "(" * 3000 + "1" + ")" * 3000 + "\n",
    "      X = " + "1 + " * 1500 + "1\n",
    "      DO 10 I = 1\n      END END END\n",
    "this is not fortran @@@ ### $$$",
    "      " + "9" * 3000 + "\n",
    "C" * 8000,
    "      GOTO\n      DO\n      IF\n      ELSE IF\n      FORMAT\n      DATA\n      COMMON\n",
    "      SUBROUTINE\n      FUNCTION\n      BLOCK DATA\n      PROGRAM\n      ENTRY\n",
    "      IF (((X) THEN\n      ENDIF\n      ENDDO\n",
    "      READ(\n      WRITE(6,\n      PRINT\n      OPEN(UNIT=\n",
    "      X(1:2:3) = Y(:)(:)\n      Z = .AND. .OR.\n",
    "      CHARACTER*(\n      REAL*\n      DOUBLE\n      IMPLICIT\n      IMPLICIT REAL (A-\n",
    "\t\t\t\t\n\t1 2\n",
    "      DATA X /\n      EQUIVALENCE (\n      PARAMETER (\n      SAVE /\n",
    "x" * 8000,
]


@pytest.mark.parametrize("text", GARBAGE)
def test_garbage_never_raises(text):
    module = analyze("g.f", text)
    assert isinstance(module, ParsedModule)
    assert module.path == "g.f"
    assert isinstance(detect_features(text), list)
    assert isinstance(extract_api(text), tuple)
    assert isinstance(normalize_source(text).text, str)
    assert isinstance(unit_dependencies(module), dict)
    assert isinstance(validate_conversion(module, text).ok, bool)


@pytest.mark.parametrize("value", [None, 42, b"      X = 1\n", ["a"]])
def test_non_text_input_never_raises(value):
    module = analyze("g.f", value)  # type: ignore[arg-type]
    assert module.metrics.has_syntax_errors
    assert detect_features(value) == []  # type: ignore[arg-type]
    assert extract_api(value) == ()  # type: ignore[arg-type]


def test_obvious_garbage_sets_has_syntax_errors():
    for text in ("this is not fortran @@@ ### $$$", "((((", "      X = = = 1\n", "'abc"):
        assert analyze("g.f", text).metrics.has_syntax_errors, text


def test_truncated_and_damaged_versions_of_a_real_module_never_raise():
    rng = random.Random(7)
    for cut in range(0, len(LEGACY), 211):
        analyze("t.f", LEGACY[:cut])
    for _ in range(12):
        chars = list(LEGACY)
        for _ in range(rng.randint(1, 25)):
            chars[rng.randrange(len(chars))] = rng.choice("()',*/=:.&!\n 0aZ$%")
        analyze("t.f", "".join(chars))


def test_results_are_independent_and_deterministic():
    first = analyze("a.f", LEGACY)
    second = analyze("a.f", LEGACY)
    assert first.hits == second.hits and first.metrics == second.metrics
    assert first.api == second.api


def test_thread_safety_under_a_thread_pool():
    expected = analyze("a.f", LEGACY)
    free = "program p\n  integer :: i\n  do i = 1, 3\n    print *, i\n  end do\nend program p\n"

    def work(n: int):
        module = analyze(f"{n}.f", LEGACY if n % 2 == 0 else free)
        report = validate_conversion(expected, CONVERTED)
        return n, module, report.ok

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(24)))
    for n, module, ok in results:
        assert ok
        if n % 2 == 0:
            assert module.hits == expected.hits
            assert module.metrics == expected.metrics
            assert module.api == expected.api
        else:
            assert [u.name for u in module.units] == ["p"]


def test_free_form_module_end_to_end():
    source = (
        "module m\n"
        "contains\n"
        "subroutine scale(x, n, k)\n"
        "  implicit none\n"
        "  integer, intent(in) :: n\n"
        "  real :: x(n)\n"
        "  integer :: k, i\n"
        "  do i = 1, n\n"
        "    x(i) = x(i) * 2.0 &\n"
        "           + 1.0\n"
        "  end do\n"
        "  k = n / 2  ! integer division\n"
        "end subroutine scale\n"
        "end module m\n"
    )
    module = analyze("m.f90", source)
    assert [(u.name, u.kind) for u in module.units] == [("scale", "subroutine")]
    ids = {h.feature_id for h in module.hits}
    assert {"integer_division", "arg_output", "array_adjustable", "vendor_extension"} <= ids
    assert "fixed_form_source" not in ids
    scale = module.api[0]
    assert scale.returns == ("k",)
    division = next(h for h in module.hits if h.feature_id == "integer_division")
    assert division.line == 12


def test_dec_tab_form_file_end_to_end():
    source = "\tSUBROUTINE S(A)\n\tREAL A\n10\tA = 1.0\n\tGO TO 10\n\tEND\n"
    module = analyze("s.for", source)
    ids = {h.feature_id for h in module.hits}
    assert {"goto_unconditional", "fixed_form_source"} <= ids
    assert module.api[0].returns == ("a",)
    assert not module.metrics.has_syntax_errors


def test_lowercase_and_mixed_case_sources():
    module = analyze("a.f", fixed("subroutine Foo(a)\nreal A\na = Abs(a)\nend"))
    assert module.units[0].name == "foo"
    assert module.api[0].returns == ("a",)


def test_hits_inside_a_string_or_comment_are_not_reported():
    source = fixed(
        """
        C GO TO 10 and STOP and COMMON /B/ X and D1MACH(1) .EQ. 1.0D0
        CHARACTER*40 S
        S = 'GO TO 10 STOP D1MACH(1) .EQ. 1.0D0 FORMAT(1X)'  ! COMMON /B/ X
        END
        """
    )
    ids = {h.feature_id for h in detect_features(source)}
    assert ids == {"fixed_form_source", "character_star"}
