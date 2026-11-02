import numpy as np
import pytest

from target_module import (
    sblksm,
    sclear,
    smget,
    smset,
    sstash,
    strnsp,
    sunsth,
    svget,
    svset,
)

SEQ16 = np.arange(1.0, 17.0)


@pytest.fixture(autouse=True)
def clean_workspace():
    sclear()
    yield
    sclear()


def test_workspace_starts_cleared():
    assert [svget(k) for k in range(1, 17)] == [0.0] * 16
    assert smget(4, 4) == 0.0


def test_clear_resets_every_vector_element():
    svset(16, SEQ16)
    sclear()
    assert [svget(k) for k in range(1, 17)] == [0.0] * 16
    assert sblksm(4) == 0.0


def test_vector_load_is_visible_as_column_major_matrix():
    svset(16, SEQ16)
    # A(I,J) = V(I + 4*(J-1))
    assert smget(1, 1) == 1.0
    assert smget(4, 1) == 4.0
    assert smget(1, 2) == 5.0
    assert smget(2, 3) == 10.0
    assert smget(1, 4) == 13.0
    assert smget(4, 4) == 16.0


def test_matrix_store_is_visible_in_vector():
    smset(3, 2, -7.5)
    assert svget(7) == -7.5
    smset(1, 4, 2.25)
    assert svget(13) == 2.25
    assert svget(3) == 0.0


def test_partial_vector_load_leaves_rest_zero():
    svset(3, np.array([9.0, 8.0, 7.0, 6.0]))
    assert [svget(k) for k in range(1, 6)] == [9.0, 8.0, 7.0, 0.0, 0.0]
    assert smget(3, 1) == 7.0
    assert smget(4, 1) == 0.0


def test_vector_load_is_clipped_to_sixteen_elements():
    svset(20, np.arange(1.0, 21.0))
    assert svget(16) == 16.0
    assert sblksm(4) == pytest.approx(136.0, rel=1e-9)


def test_zero_length_load_changes_nothing():
    smset(2, 2, 3.5)
    svset(0, np.array([99.0]))
    assert smget(2, 2) == 3.5


def test_leading_block_sums():
    svset(16, SEQ16)
    assert sblksm(0) == 0.0
    assert sblksm(1) == pytest.approx(1.0, rel=1e-9)
    # A(1,1)+A(2,1)+A(1,2)+A(2,2) = 1+2+5+6
    assert sblksm(2) == pytest.approx(14.0, rel=1e-9)
    # first three rows of the first three columns: (1+2+3) + (5+6+7) + (9+10+11)
    assert sblksm(3) == pytest.approx(54.0, rel=1e-9)
    assert sblksm(4) == pytest.approx(136.0, rel=1e-9)


def test_block_sum_is_accumulated_in_double_precision():
    # 0.1 and 0.2 are not representable in single precision: a REAL accumulator would be
    # off by about 4e-8 relative
    smset(1, 1, 0.1)
    smset(2, 1, 0.2)
    assert sblksm(2) == pytest.approx(0.3, rel=1e-12)


def test_transpose_reorders_the_vector_view():
    svset(16, SEQ16)
    strnsp()
    assert smget(1, 2) == 2.0
    assert smget(2, 1) == 5.0
    assert smget(4, 3) == 15.0
    # the vector view sees the transposed matrix column by column
    assert [svget(k) for k in range(1, 17)] == [
        1.0,
        5.0,
        9.0,
        13.0,
        2.0,
        6.0,
        10.0,
        14.0,
        3.0,
        7.0,
        11.0,
        15.0,
        4.0,
        8.0,
        12.0,
        16.0,
    ]


def test_transpose_twice_is_identity():
    svset(16, SEQ16)
    strnsp()
    strnsp()
    assert [svget(k) for k in range(1, 17)] == list(SEQ16)


def test_transpose_keeps_diagonal_and_trace_block():
    svset(16, SEQ16)
    strnsp()
    assert [smget(i, i) for i in range(1, 5)] == [1.0, 6.0, 11.0, 16.0]
    assert sblksm(2) == pytest.approx(14.0, rel=1e-9)


def test_stash_zeroes_column_and_unstash_restores_it():
    svset(16, SEQ16)
    sstash(3)
    assert [smget(i, 3) for i in range(1, 5)] == [0.0] * 4
    assert svget(9) == 0.0
    assert svget(8) == 8.0 and svget(13) == 13.0
    sunsth(3)
    assert [smget(i, 3) for i in range(1, 5)] == [9.0, 10.0, 11.0, 12.0]


def test_tail_is_separate_from_vector_and_matrix():
    svset(16, SEQ16)
    sstash(3)
    # overwrite every element of the vector view, tail must survive
    svset(16, -SEQ16)
    sunsth(3)
    assert [smget(i, 3) for i in range(1, 5)] == [9.0, 10.0, 11.0, 12.0]
    assert smget(1, 1) == -1.0
    assert smget(4, 4) == -16.0


def test_unstash_moves_column_between_positions():
    svset(16, SEQ16)
    sstash(1)
    sunsth(4)
    assert [smget(i, 4) for i in range(1, 5)] == [1.0, 2.0, 3.0, 4.0]
    assert [smget(i, 1) for i in range(1, 5)] == [0.0] * 4


def test_clear_also_clears_the_tail():
    svset(16, SEQ16)
    sstash(2)
    sclear()
    sunsth(2)
    assert [smget(i, 2) for i in range(1, 5)] == [0.0] * 4
