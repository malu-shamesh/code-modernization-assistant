import target_module as m
import numpy as np
import pytest


def arr(*values):
    return np.array(values, dtype=float)


def test_daxpy_unit_stride():
    dy = arr(10, 20, 30)
    m.daxpy(3, 2.0, arr(1, 2, 3), 1, dy, 1)
    assert dy.tolist() == [12.0, 24.0, 36.0]


def test_daxpy_strided_source_and_zero_stride_source():
    dy = arr(1, 1, 1)
    m.daxpy(3, -1.0, arr(1, 0, 2, 0, 3), 2, dy, 1)
    assert dy.tolist() == [0.0, -1.0, -2.0]
    # INCX = 0 reuses the first element of X for every update
    dz = arr(1, 1, 1)
    m.daxpy(3, 2.0, arr(4, 99, 99), 0, dz, 1)
    assert dz.tolist() == [9.0, 9.0, 9.0]


def test_daxpy_negative_stride_walks_backwards():
    dy = arr(0, 0, 0)
    m.daxpy(3, 1.0, arr(1, 2, 3), -1, dy, 1)
    assert dy.tolist() == [3.0, 2.0, 1.0]


def test_daxpy_negative_stride_two_starts_at_third_element():
    # N=2, INCX=-2: IX starts at (1-2)*(-2)+1 = 3, then 1
    dy = arr(0, 0)
    m.daxpy(2, 1.0, arr(5, 0, 7), -2, dy, 1)
    assert dy.tolist() == [7.0, 5.0]


def test_daxpy_noop_cases():
    dy = arr(1, 2)
    m.daxpy(0, 5.0, arr(9, 9), 1, dy, 1)
    m.daxpy(-3, 5.0, arr(9, 9), 1, dy, 1)
    m.daxpy(2, 0.0, arr(9, 9), 1, dy, 1)
    assert dy.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "n, expected",
    [(3, 12.0), (4, 20.0), (5, 30.0), (7, 56.0), (10, 110.0), (12, 156.0)],
)
def test_ddot_unit_stride_all_unroll_paths(n, expected):
    # x = 1..n, y = 2 everywhere: dot = n*(n+1)
    dx = np.arange(1.0, n + 1)
    dy = np.full(n, 2.0)
    assert m.ddot(n, dx, 1, dy, 1) == pytest.approx(expected, rel=1e-9)


def test_ddot_ignores_elements_beyond_n_and_handles_empty_vectors():
    assert m.ddot(3, arr(1, 2, 3, 100), 1, arr(4, 5, 6, 100), 1) == pytest.approx(32.0)
    assert m.ddot(0, arr(1), 1, arr(1), 1) == 0.0


def test_ddot_strided():
    dx = arr(1, 9, 2, 9, 3, 9)
    dy = arr(4, 5, 6)
    assert m.ddot(3, dx, 2, dy, 1) == pytest.approx(32.0, rel=1e-9)


def test_ddot_negative_stride_reverses_x():
    assert m.ddot(3, arr(1, 2, 3), -1, arr(4, 5, 6), 1) == pytest.approx(28.0, rel=1e-9)


def test_dscal_unit_stride_cleanup_plus_block_and_short_vector():
    dx = np.arange(1.0, 10.0)  # 9 elements, scale the first 7
    m.dscal(7, 3.0, dx, 1)
    assert dx.tolist() == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 8.0, 9.0]
    dy = arr(1, 2, 3, 4)
    m.dscal(3, 0.5, dy, 1)
    assert dy.tolist() == [0.5, 1.0, 1.5, 4.0]


def test_dscal_stride_two_touches_every_other_element():
    dx = arr(1, 2, 3, 4, 5, 6)
    m.dscal(3, 2.0, dx, 2)
    assert dx.tolist() == [2.0, 2.0, 6.0, 4.0, 10.0, 6.0]
    dy = np.arange(1.0, 11.0)
    m.dscal(5, 3.0, dy, 2)
    assert dy.tolist() == [3.0, 2.0, 9.0, 4.0, 15.0, 6.0, 21.0, 8.0, 27.0, 10.0]


def test_dscal_nonpositive_stride_is_noop():
    dx = arr(1, 2, 3)
    m.dscal(3, 2.0, dx, 0)
    m.dscal(3, 2.0, dx, -1)
    m.dscal(0, 2.0, dx, 1)
    assert dx.tolist() == [1.0, 2.0, 3.0]


def test_dcopy_with_strides_and_negative_stride():
    dy = arr(0, 9, 0, 9, 0)
    m.dcopy(3, arr(1, 2, 3), 1, dy, 2)
    assert dy.tolist() == [1.0, 9.0, 2.0, 9.0, 3.0]
    dz = arr(0, 0, 0)
    m.dcopy(3, arr(1, 2, 3), 1, dz, -1)
    assert dz.tolist() == [3.0, 2.0, 1.0]


def test_single_element_vectors_are_processed():
    dy = arr(10, 20)
    m.daxpy(1, 3.0, arr(2, 99), 1, dy, 1)
    assert dy.tolist() == [16.0, 20.0]
    dz = arr(0, 7)
    m.dcopy(1, arr(5, 99), 1, dz, 1)
    assert dz.tolist() == [5.0, 7.0]
    dx = arr(4, 5)
    m.dscal(1, 2.0, dx, 1)
    assert dx.tolist() == [8.0, 5.0]
    assert m.ddot(1, arr(3, 99), 1, arr(4, 99), 1) == pytest.approx(12.0, rel=1e-9)


@pytest.mark.parametrize("n", [5, 10, 12])
def test_dscal_unit_stride_whole_blocks_and_several_blocks(n):
    dx = np.arange(1.0, n + 3)
    m.dscal(n, 2.0, dx, 1)
    assert dx.tolist() == [2.0 * v for v in range(1, n + 1)] + [float(n + 1), float(n + 2)]


def test_ddot_unrolled_blocks_with_distinct_vectors():
    # sum i*(11-i) for i = 1..10 is 220; sum i*(8-i) for i = 1..7 is 84
    assert m.ddot(10, np.arange(1.0, 11.0), 1, np.arange(10.0, 0.0, -1.0), 1) == pytest.approx(
        220.0, rel=1e-9
    )
    assert m.ddot(7, np.arange(1.0, 8.0), 1, np.arange(7.0, 0.0, -1.0), 1) == pytest.approx(
        84.0, rel=1e-9
    )
