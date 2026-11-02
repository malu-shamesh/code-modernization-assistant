import numpy as np
import pytest

import target_module as tm


def sat(values, lo, hi, alo, ahi, n=None):
    ix = np.array(values, dtype=np.int64)
    return tm.satacc(len(values) if n is None else n, ix, lo, hi, alo, ahi, 0, 0)


def runs(values, n=None):
    ix = np.array(values, dtype=np.int64)
    return tm.runlen(len(values) if n is None else n, ix, 0, 0, 0)


def test_satacc_clips_samples_and_total():
    assert sat([5, 20, -3, 8, -30], -10, 10, 0, 15) == (3, 5)


def test_satacc_total_clipped_at_lower_bound_repeatedly():
    assert sat([-4, -4, -4], -10, 10, -7, 7) == (2, -7)


def test_satacc_degenerate_sample_range():
    assert sat([1, 5, 3], 3, 3, 0, 7) == (3, 7)


def test_satacc_nothing_to_clip():
    assert sat([1, 2, 3], -100, 100, -100, 100) == (0, 6)


def test_satacc_empty_input():
    assert sat([9], -1, 1, -1, 1, n=0) == (0, 0)


def test_satacc_only_first_n_samples():
    assert sat([5, 5, 5, 5], 0, 100, 0, 7, n=2) == (1, 7)


def test_satacc_sample_clip_and_total_clip_on_same_step():
    # first sample 50 -> 10 (one event); 10 stays inside [0, 20]; second 50 -> 10, total 20 fits
    assert sat([50, 50], 0, 10, 0, 20) == (2, 20)
    # third sample pushes the total to 30 -> 20 (another event)
    assert sat([50, 50, 50], 0, 10, 0, 20) == (4, 20)


def test_satacc_does_not_modify_input():
    ix = np.array([5, 20, -3], dtype=np.int64)
    tm.satacc(3, ix, -10, 10, 0, 15, 0, 0)
    assert ix.tolist() == [5, 20, -3]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([5, 5, 2, 2, 2, 7, 7, 7, 1], (4, 3, 3)),
        ([9], (1, 1, 1)),
        ([4, 4, 4, 4], (1, 4, 1)),
        ([1, 2, 1, 2], (4, 1, 1)),
        ([3, 3, 1, 1, 1], (2, 3, 3)),
        ([1, 1, 2, 2], (2, 2, 1)),
        ([-1, -1, -1, 0], (2, 3, 1)),
        ([0, 0, 0, 0, 0, 1], (2, 5, 1)),
    ],
)
def test_runlen(values, expected):
    assert runs(values) == expected


def test_runlen_empty():
    assert runs([7], n=0) == (0, 0, 0)


def test_runlen_only_first_n_elements():
    assert runs([1, 1, 2, 2, 2], n=3) == (2, 2, 1)
