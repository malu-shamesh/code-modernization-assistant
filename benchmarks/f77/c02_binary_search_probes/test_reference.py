import numpy as np
import pytest

import target_module
from target_module import ibsrch, icount, ilbnd

EVEN = np.array([2, 4, 6, 8, 10, 12, 14], dtype=np.int64)
DUPS = np.array([1, 3, 3, 3, 5, 8], dtype=np.int64)


@pytest.mark.parametrize(
    ("key", "index", "probes"),
    [
        (10, 5, 3),  # mid 4 (8), mid 6 (12), mid 5 (10)
        (2, 1, 3),  # mid 4, mid 2, mid 1
        (14, 7, 3),  # mid 4, mid 6, mid 7
        (8, 4, 1),  # hit at the first probe
        (7, -4, 3),  # mid 4, mid 2, mid 3, then LO=4 > HI=3
        (1, -1, 3),  # below the table: insertion point 1
        (15, -8, 3),  # above the table: insertion point N+1
    ],
)
def test_ibsrch_on_even_table(key, index, probes):
    assert ibsrch(7, key, EVEN, 0) == (index, probes)


def test_ibsrch_negative_keys():
    table = np.array([-9, -4, 0, 3, 8], dtype=np.int64)
    assert ibsrch(5, -4, table, 0) == (2, 3)
    assert ibsrch(5, -5, table, 0) == (-2, 3)


def test_ibsrch_single_element_and_empty_table():
    one = np.array([5], dtype=np.int64)
    assert ibsrch(1, 5, one, 0) == (1, 1)
    assert ibsrch(1, 4, one, 0) == (-1, 1)
    assert ibsrch(1, 6, one, 0) == (-2, 1)
    assert ibsrch(0, 3, one, 0) == (-1, 0)


def test_ibsrch_stops_at_n_not_at_array_end():
    table = np.array([1, 2, 3, 100, 200], dtype=np.int64)
    assert ibsrch(3, 100, table, 0) == (-4, 2)
    assert ibsrch(5, 100, table, 0) == (4, 2)


@pytest.mark.parametrize(
    ("key", "expected"),
    [(3, 2), (4, 5), (0, 1), (1, 1), (8, 6), (9, 7), (2, 2), (5, 5)],
)
def test_ilbnd_with_duplicates(key, expected):
    assert ilbnd(6, key, DUPS) == expected


def test_ilbnd_empty_table_and_partial_n():
    assert ilbnd(0, 7, DUPS) == 1
    # only the first three elements 1 3 3 take part
    assert ilbnd(3, 4, DUPS) == 4
    assert ilbnd(3, 3, DUPS) == 2


@pytest.mark.parametrize(
    ("key", "expected"),
    [(3, 3), (1, 1), (2, 0), (5, 1), (8, 1), (100, 0), (-5, 0)],
)
def test_icount(key, expected):
    assert icount(6, key, DUPS) == expected


def test_icount_all_equal():
    table = np.full(9, 4, dtype=np.int64)
    assert icount(9, 4, table) == 9
    assert icount(9, 5, table) == 0


def test_ibsrch_agrees_with_ilbnd_on_present_keys():
    table = np.arange(3, 60, 4, dtype=np.int64)
    n = len(table)
    for position, key in enumerate(table, start=1):
        index, _ = ibsrch(n, int(key), table, 0)
        assert index == position == ilbnd(n, int(key), table)


def test_module_exposes_the_fortran_units():
    for name in ("ibsrch", "ilbnd", "icount"):
        assert callable(getattr(target_module, name))
