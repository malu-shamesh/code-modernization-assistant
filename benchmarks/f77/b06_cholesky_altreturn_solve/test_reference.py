import target_module
import math
import numpy as np
import pytest

SPD = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
L = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]


def lower_only(rows, upper_fill=999.0, lda=None):
    """Matrix whose strictly upper triangle holds garbage the code must never read."""
    n = len(rows)
    lda = n if lda is None else lda
    a = np.full((lda, n), upper_fill)
    for i in range(n):
        for j in range(i + 1):
            a[i, j] = rows[i][j]
    return a


def test_factor_matches_hand_computed_cholesky():
    a = lower_only(SPD)
    dmin, alt = target_module.choldc(a, 3, 3, 0.0)
    assert alt == 0
    assert dmin == pytest.approx(1.0, rel=1e-9)
    for i in range(3):
        for j in range(i + 1):
            assert a[i, j] == pytest.approx(L[i][j], rel=1e-9)


def test_strictly_upper_triangle_is_never_read_or_written():
    a = lower_only(SPD, upper_fill=np.nan)
    dmin, alt = target_module.choldc(a, 3, 3, 0.0)
    assert alt == 0
    assert np.isnan(a[0, 1]) and np.isnan(a[0, 2]) and np.isnan(a[1, 2])
    assert a[2, 0] == pytest.approx(-8.0, rel=1e-9)


def test_padded_leading_dimension():
    a = lower_only(SPD, lda=5)
    dmin, alt = target_module.choldc(a, 5, 3, 0.0)
    assert (dmin, alt) == (pytest.approx(1.0), 0)
    assert a[2, 2] == pytest.approx(3.0, rel=1e-9)
    assert a[3:, :].tolist() == [[999.0] * 3] * 2


def test_not_positive_definite_takes_alternate_return_with_pivot():
    a = lower_only([[1.0, 0.0], [2.0, 1.0]])
    dmin, alt = target_module.choldc(a, 2, 2, 0.0)
    assert alt == 1
    assert dmin == pytest.approx(-3.0, rel=1e-9)


def test_zero_pivot_is_rejected():
    a = np.zeros((1, 1))
    assert target_module.choldc(a, 1, 1, 7.0) == (0.0, 1)


def test_negative_first_pivot_is_reported_immediately():
    a = lower_only([[-4.0]])
    dmin, alt = target_module.choldc(a, 1, 1, 0.0)
    assert (dmin, alt) == (-4.0, 1)


def test_empty_matrix_is_trivially_factored():
    dmin, alt = target_module.choldc(np.zeros((1, 1)), 1, 0, 5.0)
    assert (dmin, alt) == (0.0, 0)


def test_minimum_diagonal_is_found_anywhere():
    a = lower_only([[9.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 16.0]])
    dmin, alt = target_module.choldc(a, 3, 3, 0.0)
    assert alt == 0
    assert dmin == pytest.approx(1.0, rel=1e-9)
    assert [a[0, 0], a[1, 1], a[2, 2]] == pytest.approx([3.0, 1.0, 4.0], rel=1e-9)


def test_cholsl_solves_from_the_factor():
    a = lower_only(L)
    b = np.array([-20.0, -43.0, 192.0])
    target_module.cholsl(a, 3, 3, b)
    assert b == pytest.approx([1.0, 2.0, 3.0], rel=1e-9)


def test_cholsl_two_by_two_by_hand():
    # L = [[2,0],[1,3]]: A = L*L' = [[4,2],[2,10]], x = (1,1) gives b = (6,12)
    a = lower_only([[2.0, 0.0], [1.0, 3.0]])
    b = np.array([6.0, 12.0])
    target_module.cholsl(a, 2, 2, b)
    assert b == pytest.approx([1.0, 1.0], rel=1e-9)


def test_spdsol_success_returns_dmin_and_zero():
    a = lower_only(SPD)
    b = np.array([-20.0, -43.0, 192.0])
    dmin, alt = target_module.spdsol(a, 3, 3, b, 0.0)
    assert alt == 0
    assert dmin == pytest.approx(1.0, rel=1e-9)
    assert b == pytest.approx([1.0, 2.0, 3.0], rel=1e-9)


def test_spdsol_passes_the_alternate_return_on_and_keeps_b():
    a = lower_only([[1.0, 0.0], [2.0, 1.0]])
    b = np.array([3.0, 4.0])
    dmin, alt = target_module.spdsol(a, 2, 2, b, 0.0)
    assert alt == 1
    assert dmin == pytest.approx(-3.0, rel=1e-9)
    assert b.tolist() == [3.0, 4.0]


def test_choldt_is_log_of_determinant():
    # det(SPD) = (2*1*3)**2 = 36
    assert target_module.choldt(np.array(L), 3, 3) == pytest.approx(math.log(36.0), rel=1e-9)
    assert target_module.choldt(np.array(L), 3, 0) == 0.0


def test_single_positive_pivot_is_the_minimum():
    a = lower_only([[9.0]])
    dmin, alt = target_module.choldc(a, 1, 1, 0.0)
    assert (dmin, alt) == (3.0, 0)
    assert a[0, 0] == 3.0


def test_minimum_diagonal_may_come_from_the_first_column():
    a = lower_only([[1.0, 0.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 81.0]])
    dmin, alt = target_module.choldc(a, 3, 3, 0.0)
    assert (dmin, alt) == (1.0, 0)
    assert [a[0, 0], a[1, 1], a[2, 2]] == [1.0, 4.0, 9.0]
