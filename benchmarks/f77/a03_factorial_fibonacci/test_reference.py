import math

import numpy as np
import pytest

import target_module as tm


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (5, 120),
        (10, 3628800),
        (12, 479001600),
        (13, -1),
        (100, -1),
        (-1, -1),
        (-7, -1),
    ],
)
def test_ifact(n, expected):
    assert tm.ifact(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (-3, 1.0),
        (0, 1.0),
        (1, 1.0),
        (5, 120.0),
        (20, 2432902008176640000.0),
        (25, 1.5511210043330986e25),
    ],
)
def test_dfact(n, expected):
    assert tm.dfact(n) == pytest.approx(expected, rel=1e-9)


def test_fibtab_first_twelve():
    f = np.zeros(12, dtype=np.int64)
    tm.fibtab(12, f)
    assert f.tolist() == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def test_fibtab_short_tables_leave_the_tail_alone():
    f = np.full(5, -7, dtype=np.int64)
    tm.fibtab(2, f)
    assert f.tolist() == [1, 1, -7, -7, -7]
    g = np.full(3, -7, dtype=np.int64)
    tm.fibtab(1, g)
    assert g.tolist() == [1, -7, -7]


def test_fibtab_nonpositive_n_touches_nothing():
    f = np.full(3, 9, dtype=np.int64)
    tm.fibtab(0, f)
    tm.fibtab(-4, f)
    assert f.tolist() == [9, 9, 9]


def test_fibtab_large_entry():
    f = np.zeros(40, dtype=np.int64)
    tm.fibtab(40, f)
    assert int(f[39]) == 102334155


@pytest.mark.parametrize(
    "n, k, expected",
    [
        (5, 2, 10),
        (5, 3, 10),
        (10, 5, 252),
        (6, 0, 1),
        (6, 6, 1),
        (6, 7, 0),
        (6, -1, 0),
        (0, 0, 1),
        (20, 10, 184756),
        (30, 4, 27405),
    ],
)
def test_ibinom(n, k, expected):
    assert tm.ibinom(n, k) == expected


def test_ibinom_matches_math_comb_for_small_triangle():
    for n in range(0, 15):
        for k in range(0, n + 1):
            assert tm.ibinom(n, k) == math.comb(n, k)
