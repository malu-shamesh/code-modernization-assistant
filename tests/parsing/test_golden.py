"""Golden test: a realistic fixed-form FORTRAN 77 module analysed end to end."""

from __future__ import annotations

import pytest

from modernizer.models import ParsedModule
from modernizer.parsing import analyze, unit_dependencies, validate_conversion

LEGACY = """C     NUMERICAL UTILITIES (EXCERPT).  COLUMN-MAJOR STORAGE THROUGHOUT.
      DOUBLE PRECISION FUNCTION TRAPZ(F, A, B, N)
      EXTERNAL F
      DOUBLE PRECISION F, A, B, H, S
      INTEGER N, I
      H = (B - A)/N
      S = 0.5D0*(F(A) + F(B))
      DO 10 I = 1, N - 1
         S = S + F(A + I*H)
   10 CONTINUE
      TRAPZ = S*H
      RETURN
      END
C
      SUBROUTINE MATVEC(A, LDA, M, N, X, Y)
      INTEGER LDA, M, N, I, J
      DOUBLE PRECISION A(LDA,*), X(*), Y(*)
      DO 20 I = 1, M
         Y(I) = 0.0D0
   20 CONTINUE
      DO 40 J = 1, N
         DO 30 I = 1, M
            Y(I) = Y(I) + A(I,J)*X(J)
   30    CONTINUE
   40 CONTINUE
      RETURN
      END
C
      SUBROUTINE CLASFY(IVAL, NAME, IERR, *)
      INTEGER IVAL, IERR
      CHARACTER*8 NAME
      COMMON /STATE/ NCALLS, LAST
      DATA NCALLS /0/
      NCALLS = NCALLS + 1
      LAST = IVAL
      IERR = 0
      GO TO (100, 200), MOD(IVAL, 3) + 1
      IF (IVAL) 300, 400, 500
  100 NAME = 'ZERO'
      RETURN
  200 NAME = 'ONE'
      RETURN
  300 NAME = 'NEG'
      IERR = -1
      RETURN 1
  400 NAME = 'ZED'
      RETURN
  500 NAME = 'POS' // ' ' // 'ITIVE'
      WRITE(6, 900) IVAL, NAME(1:3)
  900 FORMAT(1X, 'CLASS ', I5, 2X, A3)
      END
C
      DOUBLE PRECISION FUNCTION SQUARE(X)
      DOUBLE PRECISION X
      SQUARE = X*X
      END
C
      PROGRAM DRIVER
      DOUBLE PRECISION A(3,3), X(3), Y(3), TRAPZ, SQUARE
      EXTERNAL SQUARE
      CHARACTER*8 LABEL
      DO 5 I = 1, 3
         X(I) = I
         DO 5 J = 1, 3
    5 A(I,J) = 1.0D0/(I + J - 1)
      CALL MATVEC(A, 3, 3, 3, X, Y)
      CALL MATVEC(A(1,2), 3, 3, 2, X, Y)
      AREA = TRAPZ(SQUARE, 0.0D0, 1.0D0, 100)
      CALL CLASFY(7, LABEL, IERR, *99)
      PRINT *, AREA, Y(1)
   99 STOP
      END
"""

CONVERTED = """import numpy as np
from f77rt import FortranStop, idiv


def trapz(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s = s + f(a + i * h)
    return s * h


def matvec(a, lda, m, n, x, y):
    for i in range(m):
        y[i] = 0.0
    for j in range(n):
        for i in range(m):
            y[i] = y[i] + a[i, j] * x[j]


def clasfy(ival, name, ierr):
    ierr = 0
    return name, ierr, 0


def square(x):
    return x * x


def main():
    pass


if __name__ == "__main__":
    main()
"""


@pytest.fixture(scope="module")
def module() -> ParsedModule:
    return analyze("numerics.f", LEGACY)


def test_units(module):
    assert [(u.name, u.kind, u.start_line, u.end_line) for u in module.units] == [
        ("trapz", "function", 2, 13),
        ("matvec", "subroutine", 15, 27),
        ("clasfy", "subroutine", 29, 51),
        ("square", "function", 53, 56),
        ("driver", "program", 58, 72),
    ]
    assert unit_dependencies(module) == {
        "trapz": set(),
        "matvec": set(),
        "clasfy": set(),
        "square": set(),
        "driver": {"matvec", "trapz", "square", "clasfy"},
    }


def test_feature_hits_and_their_lines(module):
    def lines(feature: str) -> list[int]:
        return [h.line for h in module.hits if h.feature_id == feature]

    assert lines("do_labeled") == [8, 18, 21, 22, 62, 64]
    assert lines("array_adjustable") == [17]
    assert lines("array_assumed_size") == [17]
    assert lines("goto_computed") == [37]
    assert lines("if_arithmetic") == [38]
    assert lines("alternate_return") == [29, 45, 69]
    assert lines("common_block") == [32]
    assert lines("procedure_argument") == [3, 68]
    assert lines("sequence_association") == [67]
    assert lines("intrinsic_semantics") == [37]
    assert lines("character_substring") == [48, 49]
    assert lines("format_statement") == [50]
    assert lines("implicit_typing") == [32, 62, 64, 68, 69]
    shared = next(h for h in module.hits if h.feature_id == "do_shared_terminal")
    assert (shared.line, shared.end_line) == (62, 65)
    assert not lines("integer_division")  # 1.0D0/(I + J - 1) has a real operand
    assert module.hits == sorted(module.hits, key=lambda h: (h.line, h.feature_id, h.end_line))


def test_api(module):
    api = {s.name: s for s in module.api}
    trapz = api["trapz"]
    assert trapz.kind == "function" and trapz.result_type == "double"
    assert [(p.name, p.type, p.rank) for p in trapz.params] == [
        ("f", "external", 0),
        ("a", "double", 0),
        ("b", "double", 0),
        ("n", "integer", 0),
    ]
    assert trapz.returns == ()
    matvec = api["matvec"]
    assert [(p.name, p.rank, p.intent) for p in matvec.params] == [
        ("a", 2, "in"),
        ("lda", 0, "in"),
        ("m", 0, "in"),
        ("n", 0, "in"),
        ("x", 1, "in"),
        ("y", 1, "inout"),
    ]
    assert matvec.returns == ()
    clasfy = api["clasfy"]
    assert clasfy.returns == ("name", "ierr") and clasfy.alt_returns == 1
    assert api["driver"].kind == "program"


def test_metrics(module):
    m = module.metrics
    assert (m.loc, m.n_functions, m.n_common_blocks, m.max_nesting) == (72, 4, 1, 2)
    assert m.cyclomatic == 10
    assert m.sloc == m.loc - 5  # five "C" comment lines
    assert m.calls == ()
    assert not m.has_syntax_errors
    assert m.distinct_features == len(module.feature_ids)


def test_reference_conversion_passes_static_validation(module):
    report = validate_conversion(module, CONVERTED)
    assert report.ok, report.feedback()


def test_conversion_with_a_changed_signature_is_rejected(module):
    broken = CONVERTED.replace("def matvec(a, lda, m, n, x, y)", "def matvec(a, m, n, x, y)")
    report = validate_conversion(module, broken)
    assert [i.code for i in report.issues] == ["api_changed"]
