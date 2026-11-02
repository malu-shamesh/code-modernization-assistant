import datetime

import pytest

from target_module import dayofy, dcount, idow, isleap, jdcal, jdn, mdays

# Offset between datetime's proleptic ordinal and the integer Julian day number.
ORDINAL_TO_JDN = 1721425


@pytest.fixture(autouse=True)
def clear_counts_afterwards():
    # No reset before the test: the first test below must see the BLOCK DATA state.
    yield
    dcount(0, 0, 1)


def test_block_data_starts_both_counters_at_zero():
    assert dcount(0, 0, 0) == (0, 0)


@pytest.mark.parametrize(
    "year, leap",
    [
        (2000, True),
        (1900, False),
        (2024, True),
        (2023, False),
        (2100, False),
        (1996, True),
        (0, True),
        (-4, True),
        (-100, False),
        (-400, True),
    ],
)
def test_leap_years(year, leap):
    assert isleap(year) is leap


def test_days_in_month():
    assert mdays(2024, 2) == 29
    assert mdays(2023, 2) == 28
    assert mdays(1900, 2) == 28
    assert mdays(2000, 2) == 29
    assert [mdays(2023, m) for m in (1, 3, 4, 6, 9, 11, 12)] == [31, 31, 30, 30, 30, 30, 31]
    assert mdays(2023, 0) == 0
    assert mdays(2023, 13) == 0


@pytest.mark.parametrize(
    "date, number",
    [
        ((2000, 1, 1), 2451545),
        ((1970, 1, 1), 2440588),
        ((1999, 12, 31), 2451544),
        ((2000, 2, 29), 2451604),
        ((2000, 3, 1), 2451605),
        ((2024, 2, 29), 2460370),
        ((1900, 1, 1), 2415021),
        ((1900, 3, 1), 2415080),
        ((1900, 2, 28), 2415079),
    ],
)
def test_julian_day_numbers_and_inverse(date, number):
    assert jdn(*date, 9) == (number, 0)
    assert jdcal(number, 0, 0, 0) == date


def test_invalid_dates_give_zero_and_error_code():
    assert jdn(2023, 13, 1, 9) == (0, 1)
    assert jdn(2023, 0, 1, 9) == (0, 1)
    assert jdn(2023, 2, 29, 9) == (0, 2)
    assert jdn(1900, 2, 29, 9) == (0, 2)
    assert jdn(2023, 4, 31, 9) == (0, 2)
    assert jdn(2023, 4, 0, 9) == (0, 2)
    assert jdn(2023, 13, 40, 9) == (0, 1)


def test_inverse_of_small_and_negative_numbers():
    assert jdcal(0, 9, 9, 9) == (-4713, 11, 24)
    assert jdcal(-1, 9, 9, 9) == (0, 0, 0)
    assert jdcal(-5000, 9, 9, 9) == (0, 0, 0)


@pytest.mark.parametrize("first_year", [1899, 1999, 2099])
def test_every_day_of_three_years_matches_datetime(first_year):
    day = datetime.date(first_year, 1, 1)
    end = datetime.date(first_year + 3, 1, 1)
    while day < end:
        expected = day.toordinal() + ORDINAL_TO_JDN
        number, ierr = jdn(day.year, day.month, day.day, 9)
        assert (number, ierr) == (expected, 0)
        assert jdcal(number, 0, 0, 0) == (day.year, day.month, day.day)
        assert dayofy(day.year, day.month, day.day) == day.timetuple().tm_yday
        day += datetime.timedelta(days=1)


def test_one_full_gregorian_cycle_matches_datetime():
    # 1601-01-01 .. 2000-12-31 is exactly one 400-year cycle (146097 days); the JDCAL
    # divisors only show their weaknesses on a few dates, so every day is checked
    first = datetime.date(1601, 1, 1).toordinal()
    last = datetime.date(2000, 12, 31).toordinal()
    assert last - first + 1 == 146097
    for ordinal in range(first, last + 1):
        day = datetime.date.fromordinal(ordinal)
        number, ierr = jdn(day.year, day.month, day.day, 9)
        assert ierr == 0
        assert number == ordinal + ORDINAL_TO_JDN
        assert jdcal(number, 0, 0, 0) == (day.year, day.month, day.day)


def test_consecutive_dates_are_consecutive_numbers():
    assert jdn(2000, 3, 1, 9)[0] - jdn(2000, 2, 29, 9)[0] == 1
    assert jdn(2001, 1, 1, 9)[0] - jdn(2000, 12, 31, 9)[0] == 1
    assert jdn(2001, 3, 1, 9)[0] - jdn(2001, 2, 28, 9)[0] == 1
    assert jdn(2001, 1, 1, 9)[0] - jdn(2000, 1, 1, 9)[0] == 366


@pytest.mark.parametrize(
    "jd, weekday",
    [(2451545, 6), (2440588, 4), (2415021, 1), (0, 1), (-1, 0), (-3, -2)],
)
def test_day_of_week_follows_fortran_mod(jd, weekday):
    # 0 = Sunday; MOD keeps the sign of the dividend, so JD = -3 gives -2
    assert idow(jd) == weekday


def test_day_of_year():
    assert dayofy(2000, 12, 31) == 366
    assert dayofy(1900, 12, 31) == 365
    assert dayofy(2023, 3, 1) == 60
    assert dayofy(2024, 3, 1) == 61
    assert dayofy(2023, 1, 1) == 1
    assert dayofy(2023, 2, 29) == 0
    assert dayofy(2023, 0, 1) == 0
    assert dayofy(2023, 1, 32) == 0


def test_conversion_counters_ignore_errors_and_reads():
    jdn(2000, 1, 1, 9)
    jdn(2000, 2, 30, 9)
    jdn(2000, 13, 1, 9)
    jdn(2001, 5, 5, 9)
    jdcal(2451545, 0, 0, 0)
    jdcal(-4, 0, 0, 0)
    dayofy(2000, 5, 5)
    assert dcount(0, 0, 0) == (2, 1)
    assert dcount(0, 0, 0) == (2, 1)
    assert dcount(0, 0, 1) == (2, 1)
    assert dcount(0, 0, 0) == (0, 0)
