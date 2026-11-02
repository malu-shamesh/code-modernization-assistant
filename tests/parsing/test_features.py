"""One positive and one negative near-miss source per feature id."""

from __future__ import annotations

import pytest

from modernizer.features import FEATURE_IDS, FEATURES
from modernizer.parsing import detect_features

from .helpers import feature_ids, fixed

# feature id -> (source that must hit, similar-looking source that must NOT hit)
CASES: dict[str, tuple[str, str]] = {
    "goto_unconditional": (
        "GO TO 10\n10 CONTINUE\nEND",
        "GOTO10 = 5\nEND",
    ),
    "goto_computed": (
        "GO TO (10, 20), I\n10 CONTINUE\n20 CONTINUE\nEND",
        "GO TO 10\n10 CONTINUE\nEND",
    ),
    "goto_assigned": (
        "ASSIGN 10 TO K\nGO TO K\n10 CONTINUE\nEND",
        "K = 10\nGO TO 10\n10 CONTINUE\nEND",
    ),
    "if_arithmetic": (
        "IF (X) 10, 20, 30\n10 CONTINUE\n20 CONTINUE\n30 CONTINUE\nEND",
        "IF (X .GT. 0) THEN\nY = 1\nENDIF\nEND",
    ),
    "if_logical_single": (
        "IF (X .GT. 0) Y = 1\nEND",
        "IF (X .GT. 0) THEN\nY = 1\nENDIF\nEND",
    ),
    "do_labeled": (
        "DO 10 I = 1, 3\n10 CONTINUE\nEND",
        "DO10I = 1.5\nEND",
    ),
    "do_shared_terminal": (
        "DO 10 I = 1, 3\nDO 10 J = 1, 3\n10 X = 0\nEND",
        "DO 20 I = 1, 3\nDO 10 J = 1, 3\n10 X = 0\n20 CONTINUE\nEND",
    ),
    "do_real_index": (
        "REAL T\nDO 10 T = 1.0, 2.0, 0.5\n10 CONTINUE\nEND",
        "REAL T\nDO 10 I = 1, 2\n10 CONTINUE\nEND",
    ),
    "stop_pause": (
        "STOP\nEND",
        "CALL STOPIT(1)\nEND",
    ),
    "alternate_return": (
        "SUBROUTINE S(A, *)\nRETURN 1\nEND",
        "SUBROUTINE S(A)\nA = 1\nRETURN\nEND",
    ),
    "entry_statement": (
        "SUBROUTINE S(A)\nENTRY S2(A)\nEND",
        "SUBROUTINE S(A)\nENTRY2 = 1\nEND",
    ),
    "common_block": (
        "COMMON /B/ X\nEND",
        "COMMONX = 1\nEND",
    ),
    "equivalence": (
        "REAL A, B\nEQUIVALENCE (A, B)\nEND",
        "EQUIVALENCEX = 1\nEND",
    ),
    "block_data": (
        "BLOCK DATA INIT\nCOMMON /B/ X\nDATA X /1.0/\nEND",
        "SUBROUTINE INIT\nEND",
    ),
    "data_statement": (
        "REAL X\nDATA X /1.0/\nEND",
        "DATAX = 1.0\nEND",
    ),
    "save_statement": (
        "SUBROUTINE S\nSAVE N\nEND",
        "PROGRAM P\nINTEGER N\nDATA N /1/\nEND",
    ),
    "implicit_typing": (
        "SUBROUTINE S\nX = 1.0\nEND",
        "SUBROUTINE S\nIMPLICIT NONE\nREAL X\nX = 1.0\nEND",
    ),
    "implicit_statement": (
        "IMPLICIT REAL (A-H)\nEND",
        "IMPLICIT NONE\nEND",
    ),
    "parameter_statement": (
        "INTEGER N\nPARAMETER (N = 10)\nEND",
        "INTEGER N\nN = 10\nEND",
    ),
    "character_star": (
        "CHARACTER*8 S\nEND",
        "CHARACTER C\nEND",
    ),
    "real_kind_star": (
        "REAL*8 X\nEND",
        "CHARACTER*8 S\nREAL X\nEND",
    ),
    "double_precision": (
        "DOUBLE PRECISION X\nEND",
        "REAL X\nX = 1.0D0\nEND",
    ),
    "complex_type": (
        "COMPLEX Z\nEND",
        "REAL Z\nEND",
    ),
    "external_intrinsic_stmt": (
        "EXTERNAL F\nEND",
        "CALL F(X)\nEND",
    ),
    "statement_function": (
        "F(T) = T*2.0\nX = F(1.0)\nEND",
        "REAL A(3)\nA(1) = 2.0\nEND",
    ),
    "array_multidim": (
        "REAL A(3,4)\nEND",
        "REAL A(12)\nEND",
    ),
    "array_custom_bounds": (
        "REAL A(0:9)\nEND",
        "REAL A(1:10)\nEND",
    ),
    "array_adjustable": (
        "SUBROUTINE S(A, N)\nINTEGER N\nREAL A(N)\nEND",
        "SUBROUTINE S(A)\nREAL A(10)\nEND",
    ),
    "array_assumed_size": (
        "SUBROUTINE S(A)\nREAL A(*)\nEND",
        "SUBROUTINE S(A, N)\nINTEGER N\nREAL A(N)\nEND",
    ),
    "sequence_association": (
        "SUBROUTINE V(W, N)\nINTEGER N\nREAL W(N)\nEND\n"
        "SUBROUTINE T\nREAL A(3,4)\nCALL V(A(1,2), 3)\nEND",
        "SUBROUTINE V(W)\nREAL W\nW = 1.0\nEND\nSUBROUTINE T\nREAL A(3,4)\nCALL V(A(1,2))\nEND",
    ),
    "arg_output": (
        "SUBROUTINE S(A, B)\nREAL A, B\nB = A\nEND",
        "SUBROUTINE S(A, B)\nREAL A, B, T\nT = A + B\nEND",
    ),
    "procedure_argument": (
        "EXTERNAL F\nCALL SOLVE(F, X)\nEND",
        "EXTERNAL F\nY = F(X)\nEND",
    ),
    "machine_constants": (
        "EPS = D1MACH(4)\nEND",
        "PRINT *, 'D1MACH(4)'\nEND",
    ),
    "format_statement": (
        "WRITE(6,10) N\n10 FORMAT(1X,I5)\nEND",
        "WRITE(6,'(1X,I5)') N\nEND",
    ),
    "io_formatted": (
        "WRITE(6,100) X\n100 FORMAT(F8.2)\nEND",
        "WRITE(6,*) X\nEND",
    ),
    "io_list_directed": (
        "PRINT *, X\nEND",
        "PRINT 100, X\n100 FORMAT(F8.2)\nEND",
    ),
    "io_file_control": (
        "OPEN(UNIT=10, FILE='A')\nEND",
        "READ(10,*) X\nEND",
    ),
    "hollerith": (
        "WRITE(6,10)\n10 FORMAT(5HHELLO)\nEND",
        "Y = 2.0*H\nWRITE(6,10)\n10 FORMAT(1X,'HELLO')\nEND",
    ),
    "integer_division": (
        "INTEGER N, M\nM = N/2\nEND",
        "INTEGER N, M\nM = N/2.0\nX = FLOAT(N)/2\nEND",
    ),
    "intrinsic_specific": (
        "Y = DABS(X)\nEND",
        "Y = ABS(X)\nEND",
    ),
    "intrinsic_semantics": (
        "K = MOD(N, 3)\nEND",
        "REAL MOD(3)\nK = MOD(1)\nEND",
    ),
    "bit_intrinsics": (
        "K = IAND(M, 255)\nEND",
        "LOGICAL L, L2\nL = .NOT. L2\nEND",
    ),
    "logical_dotted": (
        "IF (A .GT. B) GOTO 10\n10 CONTINUE\nEND",
        "PRINT *, 'A .GT. B'\nEND",
    ),
    "character_substring": (
        "CHARACTER*10 S\nS(2:4) = 'AB'\nEND",
        "DIMENSION A(0:5)\nWRITE(6,10)\n10 FORMAT(1X,//,I5)\nEND",
    ),
    "exponent_literal_d": (
        "X = 1.0D0\nEND",
        "X = 1.0E0\nWRITE(6,10) X\n10 FORMAT(D12.4)\nEND",
    ),
    "fixed_form_source": (
        "C COMMENT\nX = 1\nEND",
        "",  # replaced by the same program in free form
    ),
    "vendor_extension": (
        "INCLUDE 'x.inc'\nEND",
        "INCLUDES = 1\nEND",
    ),
}
FREE_FORM = "program p\n  x = 1\nend program p\n"


def _sources(feature: str) -> tuple[str, str]:
    positive, negative = CASES[feature]
    if feature == "fixed_form_source":
        return fixed(positive), FREE_FORM
    return fixed(positive), fixed(negative)


def test_every_feature_has_a_detector_case():
    assert set(CASES) == set(FEATURE_IDS)
    assert len(FEATURES) == 47


@pytest.mark.parametrize("feature", FEATURE_IDS)
def test_positive(feature):
    positive, _ = _sources(feature)
    assert feature in feature_ids(positive), feature


@pytest.mark.parametrize("feature", FEATURE_IDS)
def test_negative_near_miss(feature):
    _, negative = _sources(feature)
    assert feature not in feature_ids(negative), feature


def test_hits_are_sorted_and_have_original_lines_and_snippets():
    source = fixed(
        """
        C a comment line
        INTEGER I
        DO 10 I = 1, 3
        &   , 5
        10 CONTINUE
        STOP
        END
        """
    )
    hits = detect_features(source)
    assert hits == sorted(hits, key=lambda h: (h.line, h.feature_id, h.end_line))
    do = next(h for h in hits if h.feature_id == "do_labeled")
    assert (do.line, do.end_line) == (3, 4)
    assert do.snippet.startswith("DO 10 I = 1, 3")
    stop = next(h for h in hits if h.feature_id == "stop_pause")
    assert stop.line == 6


def test_dec_type_and_accept_statements_are_vendor_extensions_but_print_is_not():
    assert "vendor_extension" in feature_ids(fixed("TYPE *, 'hi'\nACCEPT *, N\nEND"))
    assert "vendor_extension" not in feature_ids(fixed("PRINT *, 'hi'\nREAD *, N\nEND"))
