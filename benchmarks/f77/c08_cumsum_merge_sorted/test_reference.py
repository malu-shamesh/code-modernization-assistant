import numpy as np
import pytest

import target_module
from target_module import cumcol, cumsum, ifirst, mrgsrt, rcopy


def arr(values, dtype=np.float32):
    return np.array(values, dtype=dtype)


def test_cumsum_basic():
    x = arr([1, 2, 3, 4])
    s = np.zeros(4, dtype=np.float32)
    cumsum(4, x, s)
    assert s.tolist() == [1, 3, 6, 10]
    assert x.tolist() == [1, 2, 3, 4]
    # the running sum may overwrite its own input
    y = arr([0.5, -0.25, 1.0, 2.0])
    cumsum(4, y, y)
    assert y.tolist() == [0.5, 0.25, 1.25, 3.25]


def test_cumsum_only_first_n_and_empty():
    x = arr([1, 1, 1, 1])
    s = arr([-1, -1, -1, -1])
    cumsum(2, x, s)
    assert s.tolist() == [1, 2, -1, -1]
    cumsum(0, x, s)
    assert s.tolist() == [1, 2, -1, -1]


@pytest.mark.parametrize(
    ("thresh", "expected"),
    [
        (6.0, 3),  # 1 + 2 + 3 reaches 6 exactly
        (6.5, 4),
        (10.0, 4),
        (10.5, 0),
        (0.0, 1),
        (-5.0, 1),
    ],
)
def test_ifirst(thresh, expected):
    assert ifirst(4, arr([1, 2, 3, 4]), thresh) == expected


def test_ifirst_with_negative_terms_empty_input_and_n():
    x = arr([-1, -2, 5, 1])
    # running sums -1, -3, 2, 3
    assert ifirst(4, x, 3.0) == 4
    assert ifirst(4, x, 2.0) == 3
    assert ifirst(0, x, -100.0) == 0
    assert ifirst(2, arr([1, 1, 100]), 50.0) == 0


def test_cumcol_respects_leading_dimension():
    a = np.array(
        [[1, 10], [2, 20], [3, 30], [-1, -1]],
        dtype=np.float32,
    )
    cumcol(4, 3, 2, a)
    assert a.tolist() == [[1, 10], [3, 30], [6, 60], [-1, -1]]


def test_cumcol_only_first_n_columns_and_m_rows():
    a = np.ones((3, 3), dtype=np.float32)
    cumcol(3, 2, 2, a)
    assert a.tolist() == [[1, 1, 1], [2, 2, 1], [1, 1, 1]]


@pytest.mark.parametrize(("m", "n"), [(1, 3), (0, 3), (3, 0)])
def test_cumcol_degenerate_shapes_change_nothing(m, n):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    before = a.copy()
    cumcol(3, m, n, a)
    assert (a == before).all()


def test_rcopy_copies_n_elements_only():
    a = arr([1, 2, 3, 4])
    b = arr([9, 9, 9, 9])
    rcopy(3, a, b)
    assert b.tolist() == [1, 2, 3, 9]


def merged(a_values, b_values, pad=-1.0):
    a = arr(a_values)
    b = arr(b_values)
    c = np.full(len(a_values) + len(b_values) + 2, pad, dtype=np.float32)
    nc = mrgsrt(len(a_values), a, len(b_values), b, c, 0)
    return c.tolist(), nc


def test_merge_interleaved_with_duplicates():
    c, nc = merged([1, 4, 7], [2, 4, 9, 10])
    assert nc == 7
    assert c == [1, 2, 4, 4, 7, 9, 10, -1, -1]


def test_merge_copies_tail_of_a_when_b_runs_out():
    c, nc = merged([1, 5, 6, 7], [2])
    assert (c[:5], nc) == ([1, 2, 5, 6, 7], 5)


def test_merge_copies_tail_of_b_when_a_runs_out():
    c, nc = merged([1], [2, 3, 4])
    assert (c[:4], nc) == ([1, 2, 3, 4], 4)


def test_merge_with_empty_and_disjoint_inputs():
    assert merged([], [3, 5]) == ([3, 5, -1, -1], 2)
    assert merged([2, 8], []) == ([2, 8, -1, -1], 2)
    assert merged([], []) == ([-1, -1], 0)
    assert merged([1, 2], [3, 4])[0][:4] == [1, 2, 3, 4]
    assert merged([3, 4], [1, 2])[0][:4] == [1, 2, 3, 4]


def test_merge_does_not_read_past_na_and_nb():
    a = arr([1, 3, 0, 0])
    b = arr([2, 4, -9])
    c = np.full(6, 7.0, dtype=np.float32)
    nc = mrgsrt(2, a, 2, b, c, 0)
    assert nc == 4
    assert c.tolist() == [1, 2, 3, 4, 7, 7]


def test_merge_larger_vectors():
    evens = list(range(0, 20, 2))
    threes = list(range(0, 30, 3))
    c, nc = merged(evens, threes)
    assert nc == 20
    assert c[:20] == sorted(evens + threes)


def test_module_exposes_the_fortran_units():
    for name in ("cumsum", "ifirst", "cumcol", "rcopy", "mrgsrt"):
        assert callable(getattr(target_module, name))
