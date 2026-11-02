import numpy as np
import pytest

import target_module
from target_module import qsortx

MIXED = [3, -1, -1, 2, 0, -5, 3, 3]


def run(values, kflag):
    x = np.array(values, dtype=np.float32)
    ndup, alt = qsortx(len(values), x, kflag, 0)
    return x.tolist(), ndup, alt


def test_ascending_with_duplicates():
    # adjacent equal pairs in the result: (-1,-1) (3,3) (3,3)
    assert run(MIXED, 1) == ([-5, -1, -1, 0, 2, 3, 3, 3], 3, 0)


def test_descending_with_duplicates():
    assert run(MIXED, -1) == ([3, 3, 3, 2, 0, -1, -1, -5], 3, 0)


def test_all_equal_elements():
    assert run([2.5] * 10, 1) == ([2.5] * 10, 9, 0)
    assert run([2.5] * 10, -1) == ([2.5] * 10, 9, 0)


def test_one_and_two_elements():
    assert run([7.0], 1) == ([7.0], 0, 0)
    assert run([7.0], -1) == ([7.0], 0, 0)
    assert run([2.0, 1.0], 1) == ([1.0, 2.0], 0, 0)
    assert run([1.0, 2.0], -1) == ([2.0, 1.0], 0, 0)
    assert run([1.0, 1.0], 1) == ([1.0, 1.0], 1, 0)


@pytest.mark.parametrize("kflag", [0, 2, -2, 5])
def test_invalid_flag_takes_alternate_return_and_leaves_data(kflag):
    assert run([3.0, 1.0, 2.0], kflag) == ([3.0, 1.0, 2.0], 0, 1)


@pytest.mark.parametrize("n", [0, -3])
def test_empty_or_negative_length_takes_alternate_return(n):
    x = np.array([3.0, 1.0], dtype=np.float32)
    assert qsortx(n, x, 1, 99) == (0, 1)
    assert x.tolist() == [3.0, 1.0]


def test_only_first_n_elements_are_sorted():
    x = np.array([4, 3, 2, 1, 0], dtype=np.float32)
    assert qsortx(3, x, 1, 0) == (0, 0)
    assert x.tolist() == [2, 3, 4, 1, 0]
    y = np.array([1, 2, 3, 9, 8], dtype=np.float32)
    assert qsortx(3, y, -1, 0) == (0, 0)
    assert y.tolist() == [3, 2, 1, 9, 8]


@pytest.mark.parametrize("kflag", [1, -1])
def test_permutation_of_one_to_forty(kflag):
    # 7 is a unit modulo 41, so (7 i) mod 41 for i = 1..40 permutes 1..40
    values = [float((7 * i) % 41) for i in range(1, 41)]
    expected = [float(v) for v in range(1, 41)]
    if kflag < 0:
        expected.reverse()
    assert run(values, kflag) == (expected, 0, 0)


@pytest.mark.parametrize("kflag", [1, -1])
def test_sorted_reversed_and_organ_pipe_inputs(kflag):
    ascending = [float(i) for i in range(100)]
    pipe = [float(min(i, 99 - i)) for i in range(100)]
    for values in (ascending, ascending[::-1], pipe):
        result, _, alt = run(values, kflag)
        assert result == sorted(values, reverse=kflag < 0)
        assert alt == 0
    # the organ pipe holds every value 0..49 twice, so 50 adjacent equal pairs
    assert run(pipe, kflag)[1] == 50
    assert run(ascending, kflag)[1] == 0


def test_many_duplicates_count_pairs():
    values = [float(i % 5) for i in range(50)]
    result, ndup, alt = run(values, 1)
    assert result == [float(v) for v in range(5) for _ in range(10)]
    assert (ndup, alt) == (45, 0)


def test_long_inputs_do_not_overflow_the_explicit_stack():
    n = 3000
    values = [float((1237 * i) % 3001) for i in range(1, n + 1)]
    result, ndup, alt = run(values, 1)
    assert result == sorted(values)
    assert (ndup, alt) == (0, 0)
    result, _, _ = run(list(range(n, 0, -1)), -1)
    assert result == [float(v) for v in range(n, 0, -1)]


def test_fractional_and_large_values():
    values = [0.5, -2.5, 1000.0, -0.125, 0.5, 1e-3]
    result, ndup, alt = run(values, -1)
    assert result == sorted(np.array(values, dtype=np.float32).tolist(), reverse=True)
    assert (ndup, alt) == (1, 0)


def test_module_exposes_the_fortran_unit():
    assert callable(target_module.qsortx)
