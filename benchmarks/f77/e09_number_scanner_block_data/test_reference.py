import pytest

import target_module as tm


def scan(text):
    return tm.numscn(text, False, 0, 0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("123", (True, 0, 3)),
        ("-12.5", (True, 0, 3)),
        ("+.5", (True, 0, 1)),
        ("1.", (True, 0, 1)),
        ("7  ", (True, 0, 1)),
        ("0.000", (True, 0, 4)),
        (".25", (True, 0, 2)),
        ("-0", (True, 0, 1)),
    ],
)
def test_numscn_accepts(text, expected):
    assert scan(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.2.3", (False, 4, 2)),
        ("--5", (False, 2, 0)),
        ("12a", (False, 3, 2)),
        ("  7", (False, 1, 0)),
        ("1 2", (False, 2, 1)),
        ("5-", (False, 2, 1)),
        ("+-1", (False, 2, 0)),
        ("..5", (False, 2, 0)),
        ("a", (False, 1, 0)),
        ("12e3", (False, 3, 2)),
    ],
)
def test_numscn_rejects_at_character(text, expected):
    assert scan(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (".", (False, 2, 0)),
        ("+", (False, 2, 0)),
        ("-", (False, 2, 0)),
        ("-.", (False, 3, 0)),
        ("   ", (False, 1, 0)),
        ("", (False, 1, 0)),
    ],
)
def test_numscn_incomplete_or_empty(text, expected):
    assert scan(text) == expected


def test_trailing_blanks_of_a_fixed_length_field_are_ignored():
    assert scan("-4.75".ljust(80)) == (True, 0, 3)
    assert scan("-4.7x".ljust(80)) == (False, 5, 2)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12.25", 12.25),
        ("-3.", -3.0),
        ("+7", 7.0),
        (".5", 0.5),
        ("-0.125", -0.125),
        ("007.50  ", 7.5),
        ("1000000", 1000000.0),
        ("-.375", -0.375),
        ("0.001", 0.001),
    ],
)
def test_numval_valid_numbers(text, expected):
    value, ierr = tm.numval(text, 0)
    assert ierr == 0
    assert value == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "text, expected_ierr",
    [("abc", 1), ("1.2.3", 4), ("  ", 1), ("4 5", 2), ("-", 2), ("9x", 2)],
)
def test_numval_invalid_text_gives_zero_and_error_position(text, expected_ierr):
    assert tm.numval(text, 99) == (0.0, expected_ierr)


def test_numval_fixed_length_field_with_trailing_blanks():
    value, ierr = tm.numval("-45.5".ljust(80), 0)
    assert ierr == 0
    assert value == pytest.approx(-45.5, rel=1e-9)


@pytest.mark.parametrize(
    "ch, expected",
    [("+", 1), ("-", 1), ("0", 2), ("9", 2), (".", 3), ("e", 4), (" ", 4), ("/", 4), (":", 4)],
)
def test_nclass(ch, expected):
    assert tm.nclass(ch) == expected
