import datetime
import importlib

import pytest

import target_module as tm

# Julian day numbers of well-known dates (noon-based JDN), derived by counting days
# from 2000-01-01 = 2451545.
KNOWN_DATES = [
    ((2000, 1, 1), 2451545),
    ((1999, 12, 31), 2451544),
    ((2000, 2, 29), 2451604),
    ((2000, 3, 1), 2451605),
    ((1900, 2, 28), 2415079),
    ((1900, 3, 1), 2415080),  # 1900 is not a leap year
    ((1970, 1, 1), 2440588),
    ((2024, 2, 29), 2460370),
    ((2024, 12, 31), 2460676),
    ((1858, 11, 17), 2400001),
]


def test_counters_start_at_zero_after_block_data_initialisation():
    importlib.reload(tm)
    assert tm.cntget(-1, -1) == (0, 0)


@pytest.mark.parametrize("ymd, jd", KNOWN_DATES)
def test_jday_known_dates(ymd, jd):
    assert tm.jday(*ymd) == jd


@pytest.mark.parametrize("ymd, jd", KNOWN_DATES)
def test_caldat_known_dates(ymd, jd):
    assert tm.caldat(jd, 0, 0, 0) == ymd


def test_jday_january_and_february_use_truncating_division():
    # (IM-14)/12 is -1 for IM = 1, 2 only if -13/12 truncates to -1 (floor would give -2)
    assert tm.jday(2001, 1, 1) - tm.jday(2000, 1, 1) == 366
    assert tm.jday(2000, 3, 1) - tm.jday(2000, 2, 1) == 29


def test_calendar_round_trip_against_datetime_for_three_years():
    first = datetime.date(1999, 6, 1).toordinal()
    for ordinal in range(first, first + 1100):
        d = datetime.date.fromordinal(ordinal)
        jd = ordinal + 1721425
        assert tm.jday(d.year, d.month, d.day) == jd
        assert tm.caldat(jd, 0, 0, 0) == (d.year, d.month, d.day)


@pytest.mark.parametrize(
    "n, m, k, expected",
    [
        (1, 10, 3, (13, 4)),  # trips 1, 4, 7, 10: the variable ends at 13, not 10
        (5, 1, 1, (5, 0)),  # zero trips: the variable keeps its start value
        (10, 1, -3, (-2, 4)),
        (1, 10, 1, (11, 10)),
        (3, 3, 1, (4, 1)),
        (1, 10, 20, (21, 1)),
        (0, -5, -1, (-6, 6)),
        (10, 2, -3, (1, 3)),
        (1, 5, -2, (1, 0)),
    ],
)
def test_lasti_loop_variable_after_loop(n, m, k, expected):
    assert tm.lasti(n, m, k, 99) == expected


@pytest.mark.parametrize(
    "x, expected",
    [(2.5, 3), (-2.5, -3), (0.5, 1), (-0.5, -1), (2.4, 2), (-2.4, -2), (1.5, 2), (-1.25, -1)],
)
def test_rndmod_nearest_rounds_halves_away_from_zero(x, expected):
    assert tm.rndmod(x, 1) == expected


@pytest.mark.parametrize("x, expected", [(2.9, 2), (-2.9, -2), (-0.5, 0), (7.0, 7), (-3.0, -3)])
def test_rndmod_toward_zero(x, expected):
    assert tm.rndmod(x, 2) == expected


@pytest.mark.parametrize(
    "x, expected", [(2.5, 2), (-2.5, -3), (-3.0, -3), (0.0, 0), (-0.125, -1), (7.0, 7)]
)
def test_rndmod_floor(x, expected):
    assert tm.rndmod(x, 3) == expected


@pytest.mark.parametrize("mode", [0, 4, -1, 100])
def test_rndmod_out_of_range_mode_falls_through_to_zero(mode):
    assert tm.rndmod(5.5, mode) == 0


@pytest.mark.parametrize(
    "n, maxit, expected",
    [
        (6, 100, 8),  # 6 3 10 5 16 8 4 2 1
        (3, 100, 7),
        (7, 100, 16),
        (27, 1000, 111),
        (1, 100, 0),
        (0, 100, 0),
        (-5, 100, 0),
        (2, 1, 1),
        (6, 8, 8),  # exactly enough steps
        (6, 7, -7),  # one too few
        (27, 5, -5),
    ],
)
def test_colltz_steps(n, maxit, expected):
    assert tm.colltz(n, maxit) == expected


def test_colltz_accumulates_counters_in_common():
    tm.cntrst()
    tm.colltz(6, 100)
    tm.colltz(1, 100)
    tm.colltz(27, 5)
    assert tm.cntget(0, 0) == (3, 13)  # 8 + 0 + 5 steps
    tm.jday(2000, 1, 1)
    tm.rndmod(1.5, 1)
    assert tm.cntget(0, 0) == (3, 13)


def test_cntrst_clears_the_counters():
    tm.colltz(7, 100)
    tm.cntrst()
    assert tm.cntget(5, 5) == (0, 0)
