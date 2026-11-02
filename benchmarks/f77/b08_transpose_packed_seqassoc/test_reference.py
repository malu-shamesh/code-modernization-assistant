import target_module
import numpy as np
import pytest


def test_dswap_unit_stride():
    dx = np.array([1.0, 2.0, 3.0])
    dy = np.array([4.0, 5.0, 6.0])
    target_module.dswap(3, dx, 1, dy, 1)
    assert dx.tolist() == [4.0, 5.0, 6.0]
    assert dy.tolist() == [1.0, 2.0, 3.0]


def test_dswap_strided_and_empty():
    dx = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    dy = np.array([7.0, 8.0, 9.0])
    target_module.dswap(3, dx, 2, dy, 1)
    assert dx.tolist() == [7.0, 0.0, 8.0, 0.0, 9.0]
    assert dy.tolist() == [1.0, 2.0, 3.0]
    target_module.dswap(0, dx, 1, dy, 1)
    assert dy.tolist() == [1.0, 2.0, 3.0]


def test_dswap_negative_stride_pairs_first_with_last():
    dx = np.array([1.0, 2.0, 3.0])
    dy = np.array([4.0, 5.0, 6.0])
    target_module.dswap(3, dx, 1, dy, -1)
    assert dx.tolist() == [6.0, 5.0, 4.0]
    assert dy.tolist() == [3.0, 2.0, 1.0]


SQUARE = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
SQUARE_T = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]


@pytest.mark.parametrize("order", ["C", "F"])
def test_sqtrns_three_by_three_any_memory_layout(order):
    a = np.array(SQUARE, order=order)
    target_module.sqtrns(a, 3, 3)
    assert a.tolist() == SQUARE_T


def test_sqtrns_with_leading_dimension_larger_than_n():
    a = np.full((5, 4), -1.0)
    a[:3, :3] = SQUARE
    target_module.sqtrns(a, 5, 3)
    assert a[:3, :3].tolist() == SQUARE_T
    # everything outside the leading 3 by 3 block is untouched
    assert (a[3:, :] == -1.0).all()
    assert (a[:, 3] == -1.0).all()


def test_sqtrns_is_an_involution_for_four_by_four():
    base = np.arange(1.0, 17.0).reshape(4, 4)
    a = base.copy()
    target_module.sqtrns(a, 4, 4)
    assert a.tolist() == base.T.tolist()
    target_module.sqtrns(a, 4, 4)
    assert a.tolist() == base.tolist()


def test_sqtrns_trivial_sizes_and_two_by_two():
    a = np.array([[5.0, 6.0], [7.0, 8.0]])
    target_module.sqtrns(a, 2, 1)
    target_module.sqtrns(a, 2, 0)
    assert a.tolist() == [[5.0, 6.0], [7.0, 8.0]]
    target_module.sqtrns(a, 2, 2)
    assert a.tolist() == [[5.0, 7.0], [6.0, 8.0]]


def vec(*values):
    return np.array(values, dtype=float)


def test_rctrns_two_by_three_by_hand():
    # 2 by 3 matrix [[1,3,5],[2,4,6]] is stored 1,2,3,4,5,6; its transpose is stored 1,3,5,2,4,6
    a = vec(1, 2, 3, 4, 5, 6)
    mark = np.ones(6, dtype=bool)  # stale marks must be reset
    target_module.rctrns(a, 2, 3, mark)
    assert a.tolist() == [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]


def test_rctrns_three_by_two_and_two_by_two_by_hand():
    a = vec(1, 2, 3, 4, 5, 6)
    target_module.rctrns(a, 3, 2, np.zeros(6, dtype=bool))
    assert a.tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    b = vec(1, 2, 3, 4)
    target_module.rctrns(b, 2, 2, np.zeros(4, dtype=bool))
    assert b.tolist() == [1.0, 3.0, 2.0, 4.0]


@pytest.mark.parametrize("m, n", [(2, 3), (3, 2), (4, 3), (3, 4), (5, 5), (2, 6), (7, 3)])
def test_rctrns_matches_matrix_transpose_in_column_major_storage(m, n):
    mat = np.arange(1.0, m * n + 1).reshape(m, n)
    a = mat.ravel(order="F").copy()
    mark = np.zeros(m * n, dtype=bool)
    target_module.rctrns(a, m, n, mark)
    assert a.tolist() == mat.T.ravel(order="F").tolist()
    # every position was visited, so every mark is set on return
    assert mark.all()


def test_rctrns_vector_shapes_return_before_touching_anything():
    a = vec(1, 2, 3, 4)
    mark = np.array([False, True, False, True])
    target_module.rctrns(a, 1, 4, mark)
    target_module.rctrns(a, 4, 1, mark)
    target_module.rctrns(a, 0, 4, mark)
    assert a.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert mark.tolist() == [False, True, False, True]


def test_packup_orders_the_upper_triangle_by_columns():
    a = np.array([[1.0, 2.0, 4.0], [-9.0, 3.0, 5.0], [-9.0, -9.0, 6.0]])
    ap = np.zeros(8)
    target_module.packup(a, 3, 3, ap)
    assert ap.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]


def test_packup_positions_for_four_by_four_with_padding():
    a = np.zeros((6, 4))
    for i in range(4):
        for j in range(4):
            a[i, j] = 10 * (i + 1) + (j + 1)
    ap = np.zeros(10)
    target_module.packup(a, 6, 4, ap)
    assert ap.tolist() == [11.0, 12.0, 22.0, 13.0, 23.0, 33.0, 14.0, 24.0, 34.0, 44.0]


def test_unpkup_builds_a_symmetric_matrix():
    ap = vec(1, 2, 3, 4, 5, 6)
    a = np.full((4, 3), -7.0)
    target_module.unpkup(ap, a, 4, 3)
    assert a[:3, :].tolist() == [[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 5.0, 6.0]]
    assert (a[3, :] == -7.0).all()


def test_pack_then_unpack_round_trips_a_symmetric_matrix():
    base = np.array(
        [[2.0, -1.0, 0.0, 5.0], [-1.0, 3.0, 7.0, 1.0], [0.0, 7.0, 4.0, -2.0], [5.0, 1.0, -2.0, 9.0]]
    )
    ap = np.zeros(10)
    target_module.packup(base, 4, 4, ap)
    out = np.zeros((4, 4))
    target_module.unpkup(ap, out, 4, 4)
    assert out.tolist() == base.tolist()


def test_dswap_single_element_and_negative_strides_start_at_the_far_end():
    dx = np.array([1.0, 2.0])
    dy = np.array([3.0, 4.0])
    target_module.dswap(1, dx, 1, dy, 1)
    assert (dx.tolist(), dy.tolist()) == ([3.0, 2.0], [1.0, 4.0])
    dx = np.array([1.0, 2.0, 3.0])
    dy = np.array([4.0, 5.0, 6.0])
    target_module.dswap(3, dx, -1, dy, 1)
    assert (dx.tolist(), dy.tolist()) == ([6.0, 5.0, 4.0], [3.0, 2.0, 1.0])
    # N=2, INCX=-2: x elements are 3 then 1; INCY=-2 walks y the same way
    dx = np.array([1.0, 2.0, 3.0])
    dy = np.array([7.0, 8.0, 9.0])
    target_module.dswap(2, dx, -2, dy, -2)
    assert (dx.tolist(), dy.tolist()) == ([7.0, 2.0, 9.0], [1.0, 8.0, 3.0])
    dx = np.array([1.0, 2.0, 3.0])
    dy = np.array([7.0, 8.0, 9.0])
    target_module.dswap(2, dx, -2, dy, 1)
    assert (dx.tolist(), dy.tolist()) == ([8.0, 2.0, 7.0], [3.0, 1.0, 9.0])
