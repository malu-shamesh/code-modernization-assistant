"""Fixed-form / free-form reading and normalization."""

from __future__ import annotations

import pytest

from modernizer.parsing import NormalizedSource, analyze, detect_source_form, normalize_source
from modernizer.parsing.grammar import error_spans


def rows(source: str) -> list[str]:
    return normalize_source(source).text.splitlines()


def test_normalized_source_shape():
    result = normalize_source("      X = 1\n      END\n")
    assert isinstance(result, NormalizedSource)
    assert result.form == "fixed"
    assert result.text == "X = 1\nEND"
    assert len(result.text.split("\n")) == len(result.line_map) == len(result.end_line_map)
    assert result.line_map == [1, 2]


def test_comment_lines_blank_lines_and_inline_comments_are_dropped():
    source = (
        "C a comment\n"
        "c lower case comment\n"
        "* star comment\n"
        "! bang comment\n"
        "D     WRITE(*,*) 'debug line'\n"
        "\n"
        "      X = 1   ! trailing comment\n"
        "      Y = 2\n"
    )
    result = normalize_source(source)
    assert result.text.splitlines() == ["X = 1", "Y = 2"]
    assert result.line_map == [7, 8]


def test_continuation_lines_join_and_keep_both_line_numbers():
    source = "      X = A +\n     &    B +\n     $    C\n      Y = 1\n"
    result = normalize_source(source)
    assert result.text.splitlines()[0].replace(" ", "") == "X=A+B+C"
    assert result.line_map == [1, 4]
    assert result.end_line_map == [3, 4]


def test_comment_between_continuation_lines_is_skipped():
    source = "      CALL F(A,\nC     in the middle\n     &       B)\n"
    result = normalize_source(source)
    assert result.text.replace(" ", "").strip() == "CALLF(A,B)"
    assert (result.line_map, result.end_line_map) == ([1], [3])


def test_string_continuation_pads_to_column_72():
    source = "      S = 'AB\n     &CD'\n"
    assert rows(source)[0] == "S = 'AB" + " " * (66 - len("S = 'AB")) + "CD'"


def test_statement_labels_in_columns_1_to_5():
    source = "  100 CONTINUE\n   10 X = 1\n99999 CONTINUE\n"
    assert rows(source) == ["100 CONTINUE", "10 X = 1", "99999 CONTINUE"]


def test_dec_tab_form_lines():
    source = "10\tCONTINUE\n\tX = 1\n\tY = X +\n\t1 2\n"
    result = normalize_source(source)
    assert result.text.splitlines()[0] == "10 CONTINUE"
    assert result.text.splitlines()[1] == "X = 1"
    assert result.text.splitlines()[2].replace(" ", "") == "Y=X+2"
    assert result.end_line_map == [1, 2, 4]


def test_sequence_numbers_in_columns_73_to_80_are_dropped():
    source = f"{'      X = 1':<72}00000010\n{'      Y = 2':<72}ABC00020\n"
    assert rows(source) == ["X = 1", "Y = 2"]


def test_text_beyond_column_72_that_is_not_a_sequence_number_is_kept():
    source = "      Z = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15+ 16 + 17\n"
    assert rows(source)[0].endswith("+ 16 + 17")


def test_hollerith_constants_become_quoted_strings():
    source = (
        "      WRITE(6,10)\n"
        "   10 FORMAT(5HHELLO,1X,3HA'B,I5)\n"
        "      DATA A /4HABCD/, H /1.0/\n"
        "      CALL F(2HAB, 3)\n"
    )
    assert rows(source) == [
        "WRITE (6,10)",
        "10 FORMAT('HELLO',1X,'A''B',I5)",
        "DATA A /'ABCD'/, H /1.0/",
        "CALL F('AB', 3)",
    ]


def test_h_that_is_not_a_hollerith_marker_is_left_alone():
    source = "      X = 2.0*H\n      Y = 3*H\n      DO 5 H = 1, 3\n      Z(2,H) = 1\n"
    assert rows(source) == ["X = 2.0*H", "Y = 3*H", "DO 5 H = 1, 3", "Z(2,H) = 1"]


def test_hollerith_cut_by_a_continuation_is_blank_padded():
    source = "      WRITE(6,10)\n   10 FORMAT(1X,10HABC\n     &         ,I5)\n"
    assert "'ABC       '" in rows(source)[1]


def test_hollerith_count_takes_exactly_that_many_characters():
    assert rows("      DATA A /3HA/B,4/\n") == ["DATA A /'A/B',4/"]


@pytest.mark.parametrize(
    ("raw", "expected", "kind_prefix"),
    [
        ("DO10I=1,N", "DO 10 I=1,N", "DO"),
        ("GOTO10", "GOTO 10", "GOTO"),
        ("IF(X.GT.0)GOTO10", "IF(X.GT.0) GOTO 10", "IF"),
        ("REALX", "REAL X", "REAL"),
        ("CALLFOO(A)", "CALL FOO(A)", "CALL"),
        ("DOUBLEPRECISIONY", "DOUBLE PRECISION Y", "DOUBLE"),
        ("READ(5,*)A", "READ (5,*) A", "READ"),
        ("PRINT*,A", "PRINT *,A", "PRINT"),
        ("ELSEIF(X.GT.1)THEN", "ELSE IF(X.GT.1) THEN", "ELSE"),
        ("ENDIF", "END IF", "END"),
        ("ASSIGN10TOK", "ASSIGN 10 TO K", "ASSIGN"),
    ],
)
def test_blank_insensitive_keywords_are_split(raw, expected, kind_prefix):
    assert rows(f"      {raw}\n") == [expected]
    assert expected.startswith(kind_prefix)


def test_keyword_lookalike_assignments_are_not_split():
    source = "      DO10I=1.5\n      REALX=2.0\n      GOTO10=5\n      CALLX=1\n      IFX=3\n"
    assert rows(source) == ["DO10I=1.5", "REALX=2.0", "GOTO10=5", "CALLX=1", "IFX=3"]


def test_strings_with_doubled_quotes_and_bangs_survive():
    source = "      X = 'DON''T ! STOP'   ! real comment\n      Y = \"A!B\"\n"
    assert rows(source) == ["X = 'DON''T ! STOP'", 'Y = "A!B"']


def test_semicolon_separated_statements_are_split():
    result = normalize_source("      X = 1; Y = 2\n")
    assert result.text.splitlines() == ["X = 1", "Y = 2"]
    assert result.line_map == [1, 1]


@pytest.mark.parametrize(
    ("source", "form"),
    [
        ("      X = 1\n", "fixed"),
        ("C comment\n      X = 1\n", "fixed"),
        ("x = 1\n", "free"),
        ("program p\nend\n", "free"),
        ("  x = 1\n  y = 2\n", "free"),
        ("      x = 1 + &\n      2\n", "free"),
        ("      integer :: i\n", "free"),
        ("", "fixed"),
        ("C only comments\n", "fixed"),
        ("10\tCONTINUE\n", "fixed"),
    ],
)
def test_detect_source_form(source, form):
    assert detect_source_form(source) == form


def test_free_form_continuations_labels_and_semicolons():
    source = "program p\n  integer :: i\n  x = 1 + &\n      2\n  y = 3 ; z = 4\n10 continue\nend program p\n"
    result = normalize_source(source)
    assert result.form == "free"
    lines = result.text.splitlines()
    assert lines[2].replace(" ", "") == "x=1+2"
    assert "10 continue" in lines
    assert result.line_map[2:5] == [3, 5, 5]


def test_tree_text_rewrites_legacy_forms_the_grammar_rejects():
    source = (
        "      SUBROUTINE S(A, *, B, *)\n"
        "      COMMON // X\n"
        "      CALL F(A, *10)\n"
        "      ASSIGN 10 TO K\n"
        "      GO TO K\n"
        "   10 RETURN 1\n"
        "      END\n"
    )
    text = normalize_source(source).text
    assert "*" not in text.splitlines()[0]
    assert "//" not in text
    assert "*10" not in text
    assert error_spans(text) == []


def test_normalized_text_of_a_legacy_module_parses_without_errors():
    source = (
        "      SUBROUTINE S(A, N)\n"
        "      DIMENSION A(N)\n"
        "      DO10I=1,N\n"
        "      A(I)=I*2.5\n"
        "   10 CONTINUE\n"
        "      IF(N.GT.0)GOTO20\n"
        "      WRITE(6,100)A(1)\n"
        "  100 FORMAT(1X,5HHELLO,F10.3)\n"
        "   20 RETURN\n"
        "      END\n"
    )
    assert error_spans(normalize_source(source).text) == []


def test_normalize_source_never_raises_on_garbage():
    for garbage in ["", "\x00\x00", "))))", "'unterminated", "\t\t\t", "      \n" * 3, "9" * 200]:
        result = normalize_source(garbage)
        assert isinstance(result.text, str)
        assert len(result.line_map) == len(result.end_line_map)


def test_windows_line_endings_and_missing_final_newline():
    unix = "      X = 1\n     &    + 2\n      END"
    windows = unix.replace("\n", "\r\n")
    assert normalize_source(windows) == normalize_source(unix)
    assert normalize_source(unix).end_line_map == [2, 3]


def test_hollerith_after_a_dotted_operator_is_recognised_but_a_decimal_point_is_not():
    source = (
        "      IF (C .EQ. 1HA) N = 1\n      IF (C.NE.2HAB) N = 2\n      X = 3.5\n      Y = 1.5*H\n"
    )
    assert rows(source) == [
        "IF (C .EQ. 'A') N = 1",
        "IF (C.NE.'AB') N = 2",
        "X = 3.5",
        "Y = 1.5*H",
    ]


def test_blanks_in_an_assignment_target_are_not_significant():
    # the classic "DO 50 I = 1.5" typo is an assignment to the variable DO50I
    source = "      DO 50 I = 1.5\n      A B(1) = 2\n      A(I, J) = 3\n"
    assert rows(source) == ["DO50I= 1.5", "AB(1)= 2", "A(I, J) = 3"]
    module = analyze("a.f", "      DO 50 I = 1.5\n      END\n")
    assert not module.metrics.has_syntax_errors
    assert "do_labeled" not in module.feature_ids


def test_double_colon_and_trailing_ampersand_inside_strings_do_not_make_a_file_free_form():
    assert detect_source_form("      WRITE(6,*) 'A::B'\n") == "fixed"
    assert detect_source_form("      WRITE(6,*) 'A &'\n") == "fixed"
