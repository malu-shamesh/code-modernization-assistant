import numpy as np
import pytest

import target_module
from target_module import histo, main, prthst

HEADER = " BIN     LOW    HIGH  |          |COUNT\n"


def count(values, xmin, xmax, nb, prefill=0):
    x = np.array(values, dtype=np.float32)
    icnt = np.full(max(nb, 1), prefill, dtype=np.int64)
    nout = histo(len(values), x, xmin, xmax, nb, icnt, 0)
    return icnt.tolist(), nout


def test_histo_sample_data():
    values = [0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 9.0, 10.0, 11.0]
    # bin width 2: [0,2) holds 2, [2,4) 3, [4,6) 3, [6,8) 1, [8,10] 2; 11.0 is outside
    assert count(values, 0.0, 10.0, 5) == ([2, 3, 3, 1, 2], 1)


def test_histo_value_equal_to_xmax_goes_into_last_bin():
    assert count([10.0], 0.0, 10.0, 4) == ([0, 0, 0, 1], 0)


def test_histo_bin_edges_belong_to_the_upper_bin():
    assert count([0.0, 2.0, 4.0], 0.0, 8.0, 4) == ([1, 1, 1, 0], 0)


def test_histo_negative_range_and_outliers_on_both_sides():
    values = [-1.0, -0.75, -0.5, -0.0625, 0.0, 0.5, 1.0, 1.5, -1.25]
    # width 0.5: [-1,-.5) 2, [-.5,0) 2, [0,.5) 1, [.5,1] 2; 1.5 and -1.25 are outside
    assert count(values, -1.0, 1.0, 4) == ([2, 2, 1, 2], 2)


def test_histo_clears_previous_counts():
    assert count([1.0], 0.0, 4.0, 4, prefill=99) == ([0, 1, 0, 0], 0)


def test_histo_empty_sample():
    assert count([], 0.0, 1.0, 3, prefill=7) == ([0, 0, 0], 0)


def test_histo_only_first_n_values_count():
    x = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    icnt = np.zeros(2, dtype=np.int64)
    assert histo(2, x, 0.0, 4.0, 2, icnt, 0) == 0
    assert icnt.tolist() == [1, 1]


@pytest.mark.parametrize(("xmin", "xmax", "nb"), [(0.0, 1.0, 0), (1.0, 1.0, 3), (2.0, 1.0, 3)])
def test_histo_rejects_bad_ranges_and_keeps_counts(xmin, xmax, nb):
    x = np.array([0.5], dtype=np.float32)
    icnt = np.full(3, 42, dtype=np.int64)
    assert histo(1, x, xmin, xmax, nb, icnt, 0) == -1
    assert icnt.tolist() == [42, 42, 42]


def test_prthst_rounds_bar_lengths_half_away_from_zero(capsys):
    prthst(3, np.array([1, 4, 0], dtype=np.int64), 0.0, 3.0, 0)
    # 10*1/4 = 2.5 -> 3 stars (round half up), 4 -> 10 stars, 0 -> none
    assert capsys.readouterr().out == (
        HEADER
        + "   1    0.00    1.00  |***       |    1\n"
        + "   2    1.00    2.00  |**********|    4\n"
        + "   3    2.00    3.00  |          |    0\n"
        + " TOTAL=     5  OUTSIDE =    0\n"
    )


def test_prthst_negative_limits_and_outside_count(capsys):
    prthst(2, np.array([3, 1], dtype=np.int64), -1.0, 1.0, 2)
    assert capsys.readouterr().out == (
        HEADER
        + "   1   -1.00    0.00  |**********|    3\n"
        + "   2    0.00    1.00  |***       |    1\n"
        + " TOTAL=     4  OUTSIDE =    2\n"
    )


def test_prthst_all_zero_counts_draw_no_bars(capsys):
    prthst(2, np.array([0, 0], dtype=np.int64), 0.0, 1.0, 5)
    assert capsys.readouterr().out == (
        HEADER
        + "   1    0.00    0.50  |          |    0\n"
        + "   2    0.50    1.00  |          |    0\n"
        + " TOTAL=     0  OUTSIDE =    5\n"
    )


def test_prthst_without_bins(capsys):
    prthst(0, np.zeros(1, dtype=np.int64), 0.0, 1.0, 0)
    assert capsys.readouterr().out == " NO BINS\n"


def test_main_prints_the_sample_report(capsys):
    main()
    assert capsys.readouterr().out == (
        HEADER
        + "   1    0.00    2.00  |*******   |    2\n"
        + "   2    2.00    4.00  |**********|    3\n"
        + "   3    4.00    6.00  |**********|    3\n"
        + "   4    6.00    8.00  |***       |    1\n"
        + "   5    8.00   10.00  |*******   |    2\n"
        + " TOTAL=    11  OUTSIDE =    1\n"
    )


def test_module_exposes_the_fortran_units():
    for name in ("histo", "prthst", "main"):
        assert callable(getattr(target_module, name))
