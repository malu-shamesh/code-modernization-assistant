import numpy as np
import pytest

import target_module
from target_module import insidx, isort, shsort


def sorted_ints(values, idir):
    ia = np.array(values, dtype=np.int64)
    isort(len(values), ia, idir)
    return ia.tolist()


@pytest.mark.parametrize("idir", [1, 0, 7])
def test_isort_ascending_for_non_negative_direction(idir):
    # ISIGN(1, 0) is +1, so zero also means ascending
    assert sorted_ints([5, 2, 9, 1, 5, 6], idir) == [1, 2, 5, 5, 6, 9]


@pytest.mark.parametrize("idir", [-1, -3])
def test_isort_descending_for_negative_direction(idir):
    assert sorted_ints([5, 2, 9, 1, 5, 6], idir) == [9, 6, 5, 5, 2, 1]


def test_isort_negative_values_both_directions():
    assert sorted_ints([-3, 4, -8, 0], 1) == [-8, -3, 0, 4]
    assert sorted_ints([-3, 4, -8, 0], -1) == [4, 0, -3, -8]


def test_isort_degenerate_and_partial_lengths():
    assert sorted_ints([], 1) == []
    assert sorted_ints([42], -1) == [42]
    ia = np.array([3, 1, 2, -7, -9], dtype=np.int64)
    isort(3, ia, 1)
    assert ia.tolist() == [1, 2, 3, -7, -9]


def test_insidx_is_stable_and_leaves_data_alone():
    a = np.array([30, 10, 20, 10], dtype=np.float32)
    idx = np.zeros(4, dtype=np.int64)
    insidx(4, a, idx)
    assert idx.tolist() == [2, 4, 3, 1]
    assert a.tolist() == [30, 10, 20, 10]


def test_insidx_equal_keys_keep_original_order():
    a = np.array([5, 5, 5], dtype=np.float32)
    idx = np.array([9, 9, 9], dtype=np.int64)
    insidx(3, a, idx)
    assert idx.tolist() == [1, 2, 3]


def test_insidx_descending_input_and_partial_n():
    a = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    idx = np.full(4, -1, dtype=np.int64)
    insidx(3, a, idx)
    assert idx.tolist() == [3, 2, 1, -1]


def test_shsort_fully_reversed_four_elements():
    # N/3 = 1 so only gap 1 runs: 1 + 2 + 3 comparisons
    a = np.array([4.0, 3.0, 2.0, 1.0])
    assert shsort(4, a, 0) == 6
    assert a.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_shsort_seven_elements_uses_gap_four_then_one():
    # gap 4 costs 3 comparisons, gap 1 costs 10 more
    a = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    assert shsort(7, a, 0) == 13
    assert a.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_shsort_sorted_input_costs_one_comparison_per_element():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert shsort(5, a, 0) == 4
    assert a.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize("values", [[], [2.5]])
def test_shsort_trivial_lengths(values):
    a = np.array(values, dtype=np.float64)
    assert shsort(len(values), a, 99) == 0
    assert a.tolist() == values


@pytest.mark.parametrize("n", [14, 15, 40, 42])
def test_shsort_sorts_permutations_across_gap_boundaries(n):
    # start gaps: n = 14 -> 4, n = 15 and 40 -> 13, n = 42 -> 40
    values = [float((17 * i) % 43) for i in range(n)]
    a = np.array(values)
    shsort(n, a, 0)
    assert a.tolist() == sorted(values)


def test_shsort_touches_only_first_n_elements():
    a = np.array([3.0, 1.0, 2.0, -5.0, -6.0])
    shsort(3, a, 0)
    assert a.tolist() == [1.0, 2.0, 3.0, -5.0, -6.0]


def test_module_exposes_the_fortran_units():
    for name in ("isort", "insidx", "shsort"):
        assert callable(getattr(target_module, name))
