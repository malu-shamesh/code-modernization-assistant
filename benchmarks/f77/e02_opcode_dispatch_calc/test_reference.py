import numpy as np
import pytest

import target_module as tm


def run_calc(ops, args, nops=None):
    iop = np.array(ops, dtype=np.int64)
    arg = np.array(args, dtype=np.float64)
    return tm.calc(len(ops) if nops is None else nops, iop, arg, 0.0, 0)


@pytest.mark.parametrize(
    "ops, args, expected",
    [
        ([1, 1, 3], [2.0, 3.0, 4.0], 20.0),
        ([1, 2, 4], [10.0, 4.0, 2.0], 3.0),
        ([1, 2, 3, 4], [1.5, 0.5, 8.0, 4.0], 2.0),
        ([2, 6], [3.0, 0.0], 3.0),
        ([1, 5, 6, 5], [3.0, 0.0, 0.0, 0.0], -3.0),
        ([1, 3, 2], [1.5, 4.0, 1.0], 5.0),
    ],
)
def test_calc_normal_programs(ops, args, expected):
    acc, ierr = run_calc(ops, args)
    assert ierr == 0
    assert acc == pytest.approx(expected, rel=1e-5)


def test_calc_empty_program():
    acc, ierr = run_calc([1], [5.0], nops=0)
    assert (acc, ierr) == (0.0, 0)


def test_calc_runs_only_nops_instructions():
    acc, ierr = run_calc([1, 1, 1], [1.0, 2.0, 100.0], nops=2)
    assert ierr == 0
    assert acc == pytest.approx(3.0, rel=1e-5)


def test_calc_division_by_zero_keeps_accumulator():
    acc, ierr = run_calc([1, 4, 1], [5.0, 0.0, 9.0])
    assert ierr == 2
    assert acc == pytest.approx(5.0, rel=1e-5)


@pytest.mark.parametrize(
    "ops, args, expected_acc",
    [
        ([1, 7, 1], [3.0, 9.0, 9.0], 3.0),
        ([0], [1.0], 0.0),
        ([2, -1], [4.0, 1.0], -4.0),
        ([1, 99], [2.5, 0.0], 2.5),
    ],
)
def test_calc_unknown_opcode(ops, args, expected_acc):
    acc, ierr = run_calc(ops, args)
    assert ierr == 1
    assert acc == pytest.approx(expected_acc, rel=1e-5)


@pytest.mark.parametrize(
    "x, eps, expected",
    [
        (0.5, 0.5, 0),
        (-0.5, 0.5, 0),
        (0.75, 0.5, 1),
        (-0.75, 0.5, -1),
        (0.0, 0.0, 0),
        (0.001, 0.0, 1),
        (-0.001, 0.0, -1),
        (100.0, 1.0, 1),
    ],
)
def test_ksign(x, eps, expected):
    assert tm.ksign(x, eps) == expected


def test_tally_mixed_vector():
    x = np.array([0.1, -0.2, 0.0, 5.0, -7.0, 0.05])
    assert tm.tally(6, x, 0.1, 0, 0, 0) == (2, 3, 1)


def test_tally_only_looks_at_first_n():
    x = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    assert tm.tally(2, x, 0.0, 0, 0, 0) == (1, 0, 1)


def test_tally_zero_length():
    assert tm.tally(0, np.array([3.0]), 0.0, 0, 0, 0) == (0, 0, 0)


def test_tally_all_zero_band():
    x = np.array([0.01, -0.02, 0.0, 0.03])
    assert tm.tally(4, x, 0.05, 0, 0, 0) == (0, 4, 0)
