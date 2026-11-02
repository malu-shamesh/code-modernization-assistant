import numpy as np
import pytest

import target_module
from target_module import bsort, issrtd


def run(values):
    x = np.array(values, dtype=np.float32)
    nswap, npass = bsort(len(values), x, 0, 0)
    return x.tolist(), nswap, npass


def test_mixed_vector_sorts_and_counts_swaps():
    # inversions (5,1) (5,4) (5,2) (4,2); passes: 3 swaps, 1 swap, clean pass
    x, nswap, npass = run([5, 1, 4, 2, 8])
    assert x == [1, 2, 4, 5, 8]
    assert (nswap, npass) == (4, 3)


def test_reversed_vector_needs_a_final_empty_pass():
    # 10 inversions; the pass bound shrinks 5, 4, 3, 2 and the last pass has no trips
    x, nswap, npass = run([5, 4, 3, 2, 1])
    assert x == [1, 2, 3, 4, 5]
    assert (nswap, npass) == (10, 5)


def test_sorted_vector_is_one_clean_pass():
    x, nswap, npass = run([1, 2, 3])
    assert x == [1, 2, 3]
    assert (nswap, npass) == (0, 1)


@pytest.mark.parametrize("values", [[7.5], []])
def test_degenerate_lengths(values):
    x, nswap, npass = run(values)
    assert x == values
    assert (nswap, npass) == (0, 1)


def test_equal_keys_are_not_exchanged():
    x, nswap, npass = run([2, 2, 1])
    assert x == [1, 2, 2]
    assert (nswap, npass) == (2, 3)


def test_negative_and_fractional_values():
    x, nswap, npass = run([0.5, -1.25, 3.0, -1.25, 0.0])
    assert x == [-1.25, -1.25, 0.0, 0.5, 3.0]
    # inversions: (0.5,-1.25) (0.5,-1.25) (0.5,0.0) (3.0,-1.25) (3.0,0.0)
    assert nswap == 5
    # pass 1 swaps at i=1,3,4 -> bound 4; pass 2 swaps at i=2,3 -> bound 3; pass 3 clean
    assert npass == 3


def test_only_first_n_elements_are_touched():
    x = np.array([3, 2, 1, -9, -8], dtype=np.float32)
    nswap, npass = bsort(3, x, 0, 0)
    assert x.tolist() == [1, 2, 3, -9, -8]
    assert (nswap, npass) == (3, 3)


def test_two_elements_swapped_once():
    x, nswap, npass = run([2, 1])
    assert x == [1, 2]
    # pass 1 swaps and sets the bound to 1, pass 2 has no trips
    assert (nswap, npass) == (1, 2)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 2, 3], True),
        ([1, 3, 2], False),
        ([2, 1], False),
        ([4], True),
        ([], True),
    ],
)
def test_issrtd(values, expected):
    x = np.array(values, dtype=np.float32)
    assert issrtd(len(values), x) is expected


def test_issrtd_ignores_elements_past_n():
    x = np.array([1, 2, 3, 0], dtype=np.float32)
    assert issrtd(3, x) is True
    assert issrtd(4, x) is False


def test_sort_result_passes_issrtd():
    values = [(7 * i) % 11 for i in range(10)]
    x = np.array(values, dtype=np.float32)
    bsort(10, x, 0, 0)
    assert issrtd(10, x)
    assert x.tolist() == sorted(values)


def test_module_exposes_the_fortran_units():
    assert callable(target_module.bsort)
    assert callable(target_module.issrtd)
