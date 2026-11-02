import numpy as np
import pytest

import target_module as tm


@pytest.mark.parametrize(
    "i, j, expected",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 2, 3),
        (1, 3, 4),
        (2, 3, 5),
        (3, 3, 6),
        (3, 2, 5),
        (4, 1, 7),
        (5, 5, 15),
    ],
)
def test_pkidx(i, j, expected):
    assert tm.pkidx(i, j) == expected


def make_full(lda, n):
    a = np.full((lda, n), -1.0)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            a[i - 1, j - 1] = 10 * i + j
    return a


def test_packut_column_order_with_leading_dimension():
    a = make_full(4, 3)
    ap = np.full(8, -9.0)
    tm.packut(3, a, 4, ap)
    assert ap.tolist() == [11, 12, 22, 13, 23, 33, -9, -9]


def test_packut_uses_only_leading_n_columns():
    a = make_full(3, 3)
    ap = np.zeros(3)
    tm.packut(2, a, 3, ap)
    assert ap.tolist() == [11, 12, 22]


def test_packut_does_not_modify_matrix():
    a = make_full(4, 3)
    before = a.copy()
    tm.packut(3, a, 4, np.zeros(6))
    assert np.array_equal(a, before)


def test_unpksy_fills_both_triangles_and_keeps_padding():
    ap = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = np.full((4, 3), -9.0)
    tm.unpksy(3, ap, a, 4)
    assert a[:3, :].tolist() == [[1, 2, 4], [2, 3, 5], [4, 5, 6]]
    assert a[3, :].tolist() == [-9, -9, -9]


def test_unpksy_two_by_two_inside_bigger_array():
    ap = np.array([7.0, 8.0, 9.0])
    a = np.zeros((3, 3))
    tm.unpksy(2, ap, a, 3)
    assert a.tolist() == [[7, 8, 0], [8, 9, 0], [0, 0, 0]]


def test_pack_unpack_round_trip():
    n = 4
    sym = np.array(
        [[2.0, -1.0, 0.5, 3.0], [-1.0, 4.0, 1.5, 0.0], [0.5, 1.5, -2.0, 7.0], [3.0, 0.0, 7.0, 1.0]]
    )
    ap = np.zeros(n * (n + 1) // 2)
    tm.packut(n, sym, n, ap)
    out = np.zeros((n, n))
    tm.unpksy(n, ap, out, n)
    assert np.array_equal(out, sym)


@pytest.mark.parametrize(
    "n, x, expected",
    [
        (3, [1.0, 1.0, 1.0], [7.0, 10.0, 15.0]),
        (3, [1.0, 0.0, -1.0], [-3.0, -3.0, -2.0]),
        (2, [1.0, 2.0], [5.0, 8.0]),
        (1, [3.0, 99.0], [3.0, 0.0]),
    ],
)
def test_spmv(n, x, expected):
    ap = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    xv = np.array(x)
    y = np.full(len(x), 0.0)
    tm.spmv(n, ap, xv, y)
    assert y.tolist() == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "n, lim, expected",
    [
        (3, 6, 6),
        (3, 100, 10),
        (4, 100, 20),
        (2, 5, 3),
        (3, 0, 0),
        (0, 10, 0),
        (1, 3, 1),
        (1, 2, 0),
        (4, 7, 10),
    ],
)
def test_ntrip(n, lim, expected):
    assert tm.ntrip(n, lim) == expected
