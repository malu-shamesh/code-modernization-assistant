import target_module
import pytest


@pytest.mark.parametrize(
    "year, expected",
    [(1900, False), (2000, True), (2024, True), (2023, False), (2100, False), (2400, True)],
)
def test_isleap(year, expected):
    assert target_module.isleap(year) is expected


@pytest.mark.parametrize(
    "year, month, expected",
    [
        (2024, 2, 29),
        (2023, 2, 28),
        (1900, 2, 28),
        (2023, 4, 30),
        (2023, 12, 31),
        (2023, 1, 31),
        (2023, 13, 0),
        (2023, 0, 0),
    ],
)
def test_ndays(year, month, expected):
    assert target_module.ndays(year, month) == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        ((2024, 2, 29), 0),
        ((2023, 2, 29), 3),
        ((2023, 4, 31), 3),
        ((2023, 6, 0), 3),
        ((2023, 13, 1), 2),
        ((2023, 0, 1), 2),
        ((0, 1, 1), 1),
        ((10000, 1, 1), 1),
        ((1, 1, 1), 0),
        ((9999, 12, 31), 0),
    ],
)
def test_valdat(date, expected):
    assert target_module.valdat(*date) == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        ((2000, 1, 1), 6),  # Saturday
        ((2023, 1, 1), 0),  # Sunday
        ((1970, 1, 1), 4),  # Thursday
        ((2024, 2, 29), 4),  # Thursday
        ((2024, 3, 1), 5),  # Friday
        ((1900, 3, 1), 4),  # Thursday (1900 was not a leap year)
        ((2023, 12, 25), 1),  # Monday
        ((2023, 9, 30), 6),  # Saturday
    ],
)
def test_idow(date, expected):
    assert target_module.idow(*date) == expected


@pytest.mark.parametrize(
    "month, expected",
    [
        (1, 0),
        (2, 3),
        (3, 3),
        (4, 6),
        (5, 1),
        (6, 4),
        (7, 6),
        (8, 2),
        (9, 5),
        (10, 0),
        (11, 3),
        (12, 5),
    ],
)
def test_idow_every_month_on_the_15th_of_2023(month, expected):
    # 15 Jan 2023 was a Sunday (0); each month exercises its own table entry
    assert target_module.idow(2023, month, 15) == expected


def test_idow_century_leap_year_correction():
    # 1 Mar 2000 was a Wednesday: the year/400 term is 5 here
    assert target_module.idow(2000, 3, 1) == 3


def test_lastnb():
    assert target_module.lastnb("abc  ") == 3
    assert target_module.lastnb("x") == 1
    assert target_module.lastnb("   ") == 0


@pytest.mark.parametrize(
    "date, expected",
    [
        ((2024, 1, 1), 1),
        ((2024, 3, 1), 61),
        ((2023, 3, 1), 60),
        ((2023, 12, 31), 365),
        ((2024, 12, 31), 366),
        ((2023, 2, 10), 41),
    ],
)
def test_doy(date, expected):
    assert target_module.doy(*date) == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        ((1999, 5, 17), (1999, 5, 18)),
        ((2023, 12, 31), (2024, 1, 1)),
        ((2024, 2, 28), (2024, 2, 29)),
        ((2024, 2, 29), (2024, 3, 1)),
        ((2023, 2, 28), (2023, 3, 1)),
        ((2023, 4, 30), (2023, 5, 1)),
        ((2023, 1, 31), (2023, 2, 1)),
        ((2023, 11, 30), (2023, 12, 1)),
        ((2023, 12, 15), (2023, 12, 16)),
    ],
)
def test_nextdy(date, expected):
    assert target_module.nextdy(*date) == expected


@pytest.mark.parametrize(
    "date, text",
    [
        ((2024, 3, 5), "2024-03-05"),
        ((2023, 12, 31), "2023-12-31"),
        ((999, 1, 1), "0999-01-01"),
        ((1, 1, 1), "0001-01-01"),
        ((9999, 12, 31), "9999-12-31"),
    ],
)
def test_fmtiso(date, text):
    assert target_module.fmtiso(*date, "", 0) == (text, 0)


@pytest.mark.parametrize(
    "date, ierr",
    [((2023, 2, 29), 3), ((2023, 13, 1), 2), ((0, 5, 5), 1), ((2023, 6, 31), 3)],
)
def test_fmtiso_invalid(date, ierr):
    assert target_module.fmtiso(*date, "", 0) == (" " * 10, ierr)


@pytest.mark.parametrize(
    "date, text",
    [
        ((2000, 1, 1), "Saturday, January 1, 2000"),
        ((2024, 2, 29), "Thursday, February 29, 2024"),
        ((2023, 12, 25), "Monday, December 25, 2023"),
        ((2023, 9, 30), "Saturday, September 30, 2023"),
        ((2024, 7, 4), "Thursday, July 4, 2024"),
        ((2023, 5, 9), "Tuesday, May 9, 2023"),
        ((2023, 10, 10), "Tuesday, October 10, 2023"),
        # the year field is I4, so a three digit year gets a leading blank
        ((999, 1, 1), "Tuesday, January 1,  999"),
    ],
)
def test_fmtlng(date, text):
    got, ierr = target_module.fmtlng(*date, "", 0)
    assert ierr == 0
    assert got == text.ljust(40)


def test_fmtlng_invalid_date():
    assert target_module.fmtlng(2023, 2, 30, "", 0) == (" " * 40, 3)
    assert target_module.fmtlng(2023, 0, 10, "", 0) == (" " * 40, 2)


def test_output_placeholders_are_ignored():
    assert target_module.fmtiso(2024, 3, 5, "junk", 9) == ("2024-03-05", 0)
    assert target_module.fmtiso(2023, 2, 29, "junk", 9) == (" " * 10, 3)
    assert target_module.fmtlng(2000, 1, 1, "junk", 9) == ("Saturday, January 1, 2000".ljust(40), 0)
