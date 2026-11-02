import numpy as np
import pytest

import target_module as tm


def scan(line, maxtok=20):
    kind = np.zeros(maxtok + 2, dtype=np.int64)
    istart = np.zeros(maxtok + 2, dtype=np.int64)
    iend = np.zeros(maxtok + 2, dtype=np.int64)
    ntok, ierr = tm.token(line, maxtok, kind, istart, iend, 0, 0)
    toks = [(int(kind[k]), int(istart[k]), int(iend[k])) for k in range(ntok)]
    return toks, ierr, (kind, istart, iend)


def test_assignment_with_real_and_parentheses():
    toks, ierr, _ = scan("x1 = 3.25*(y+ 42)")
    assert ierr == 0
    assert toks == [
        (3, 1, 2),
        (4, 4, 4),
        (2, 6, 9),
        (4, 10, 10),
        (4, 11, 11),
        (3, 12, 12),
        (4, 13, 13),
        (1, 15, 16),
        (4, 17, 17),
    ]


def test_blank_padded_line_like_character_80():
    toks, ierr, _ = scan("sum = a1 + b2".ljust(80))
    assert ierr == 0
    assert toks == [(3, 1, 3), (4, 5, 5), (3, 7, 8), (4, 10, 10), (3, 12, 13)]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("((1,2))", [(4, 1, 1), (4, 2, 2), (1, 3, 3), (4, 4, 4), (1, 5, 5), (4, 6, 6), (4, 7, 7)]),
        ("  3.14159e0", [(2, 3, 9), (3, 10, 11)]),
        ("ABC+def", [(3, 1, 3), (4, 4, 4), (3, 5, 7)]),
        ("12ab", [(1, 1, 2), (3, 3, 4)]),
        ("abc12", [(3, 1, 5)]),
        ("7", [(1, 1, 1)]),
        ("7.", [(2, 1, 2)]),
        (".5+2", [(2, 1, 2), (4, 3, 3), (1, 4, 4)]),
        ("1.5.2", [(2, 1, 3), (2, 4, 5)]),
        ("1..2", [(2, 1, 2), (2, 3, 4)]),
        ("a/b^2", [(3, 1, 1), (4, 2, 2), (3, 3, 3), (4, 4, 4), (1, 5, 5)]),
    ],
)
def test_token_shapes(line, expected):
    toks, ierr, _ = scan(line)
    assert ierr == 0
    assert toks == expected


@pytest.mark.parametrize("line", ["", "     ", " " * 80])
def test_blank_line_has_no_tokens(line):
    toks, ierr, _ = scan(line)
    assert (toks, ierr) == ([], 0)


@pytest.mark.parametrize(
    "line, expected_tokens, expected_pos",
    [
        (".5+.x", [(2, 1, 2), (4, 3, 3)], 4),
        ("a.", [(3, 1, 1)], 2),
        ("X_1", [(3, 1, 1)], 2),
        ("a $ b", [(3, 1, 1)], 3),
        ("$", [], 1),
        ("2 + .", [(1, 1, 1), (4, 3, 3)], 5),
    ],
)
def test_unscannable_character_reports_position(line, expected_tokens, expected_pos):
    toks, ierr, _ = scan(line)
    assert toks == expected_tokens
    assert ierr == expected_pos


@pytest.mark.parametrize(
    "line, maxtok, expected_ntok, expected_ierr",
    [
        ("a+b+c", 3, 3, -1),
        ("a+b", 3, 3, 0),
        ("a+b   ", 3, 3, 0),
        ("a+b $", 3, 3, -1),
        ("a", 0, 0, -1),
        ("   ", 0, 0, 0),
    ],
)
def test_token_table_overflow(line, maxtok, expected_ntok, expected_ierr):
    toks, ierr, _ = scan(line, maxtok=maxtok)
    assert (len(toks), ierr) == (expected_ntok, expected_ierr)


def test_entries_beyond_ntok_are_untouched():
    toks, ierr, (kind, istart, iend) = scan("a+b")
    assert len(toks) == 3
    assert kind[3:].tolist() == [0] * (len(kind) - 3)
    assert istart[3:].tolist() == [0] * (len(istart) - 3)
    assert iend[3:].tolist() == [0] * (len(iend) - 3)


@pytest.mark.parametrize(
    "line, i1, i2, expected",
    [
        ("x1 = 3.25*(y+ 42)", 15, 16, 42),
        ("007", 1, 3, 7),
        ("90210", 1, 5, 90210),
        ("12345678", 3, 6, 3456),
        ("5", 2, 1, 0),
    ],
)
def test_tokint(line, i1, i2, expected):
    assert tm.tokint(line, i1, i2, 0) == expected


def test_tokint_of_scanned_integer_token():
    line = "f(100) + 7"
    toks, _, _ = scan(line)
    ints = [(s, e) for k, s, e in toks if k == 1]
    assert [tm.tokint(line, s, e, 0) for s, e in ints] == [100, 7]


@pytest.mark.parametrize(
    "ch, expected",
    [
        ("5", 1),
        ("a", 2),
        ("Z", 2),
        ("+", 3),
        (",", 3),
        ("(", 3),
        ("^", 3),
        (".", 4),
        (" ", 0),
        ("$", 0),
        ("_", 0),
        ("!", 0),
    ],
)
def test_iclass(ch, expected):
    assert tm.iclass(ch) == expected
