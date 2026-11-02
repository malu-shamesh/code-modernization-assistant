import numpy as np
import target_module
import pytest


@pytest.mark.parametrize(
    "text, value, nused",
    [
        ("42", 42.0, 2),
        ("  42", 42.0, 4),
        ("-12.5", -12.5, 5),
        ("+.5E1", 5.0, 5),
        ("3.14159", 3.14159, 7),
        ("2.5e3", 2500.0, 5),
        ("1D2", 100.0, 3),
        ("1.E3", 1000.0, 4),
        ("12345.678e-2", 123.45678, 12),
        ("5.", 5.0, 2),
        (".5", 0.5, 2),
        ("0.125", 0.125, 5),
        ("1e300", 1e300, 5),
        ("1e-300", 1e-300, 6),
        # the scan stops at the first character that cannot continue a number
        ("7 8", 7.0, 1),
        ("12abc", 12.0, 2),
        ("1-2", 1.0, 1),
        ("1.5.2", 1.5, 3),
        ("1E5E3", 1e5, 3),
        ("1e5+2", 1e5, 3),
        ("12E5X", 1.2e6, 4),
        # an exponent letter without digits is not part of the number
        ("1E", 1.0, 1),
        ("1E+", 1.0, 1),
        ("1E+ 5", 1.0, 1),
        ("2.5D-", 2.5, 3),
    ],
)
def test_scannm_numbers(text, value, nused):
    got_value, got_nused, alt = target_module.scannm(text, 0.0, 0)
    assert alt == 0
    assert got_nused == nused
    assert got_value == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize(
    "text", ["    ", "abc", "-", "+", ".", "e5", ".e5", "--5", "- 5", "+-1", ",5"]
)
def test_scannm_no_number_takes_alternate_return_1(text):
    assert target_module.scannm(text, 0.0, 0) == (0.0, 0, 1)


@pytest.mark.parametrize(
    "text, nused",
    [("1e400", 5), ("5E-301", 6), ("1e99999", 7), ("0." + "0" * 300 + "1", 303)],
)
def test_scannm_scale_overflow_takes_alternate_return_2(text, nused):
    value, got_nused, alt = target_module.scannm(text, 0.0, 0)
    assert alt == 2
    assert value == 0.0
    assert got_nused == nused


def test_scannm_scale_limit_is_inclusive():
    value, nused, alt = target_module.scannm("0." + "0" * 299 + "1", 0.0, 0)
    assert alt == 0
    assert nused == 302
    assert value == pytest.approx(1e-300, rel=1e-9)


def test_scannm_exponent_digits_saturate_at_9999():
    # IEXP = MIN(10*IEXP+K, 9999): E19999 reads as 9999, not 19999, so the net scale
    # 9999 - 10000 fraction digits is -1 and in range (an unbounded exponent would give 9999)
    text = "0." + "0" * 9999 + "1E19999"
    value, nused, alt = target_module.scannm(text, 0.0, 0)
    assert alt == 0
    assert nused == len(text)
    assert value == pytest.approx(0.1, rel=1e-9)


def test_scannm_negative_zero_and_sign_handling():
    value, nused, alt = target_module.scannm("-0", 0.0, 0)
    assert (value == 0.0, nused, alt) == (True, 2, 0)
    assert target_module.scannm("-7e0", 0.0, 0)[0] == pytest.approx(-7.0)
    assert target_module.scannm("+7e+2", 0.0, 0)[0] == pytest.approx(700.0)
    assert target_module.scannm("7e-2", 0.0, 0)[0] == pytest.approx(0.07)


def test_scanls_list_with_blanks_and_commas():
    vals = np.zeros(5)
    nv, ierr, ibad = target_module.scanls("1.5, -2 3e2", vals, 5, 0, 0, 0)
    assert (nv, ierr, ibad) == (3, 0, 0)
    assert vals.tolist() == pytest.approx([1.5, -2.0, 300.0, 0.0, 0.0])


def test_scanls_empty_separators_are_skipped():
    vals = np.zeros(4)
    nv, ierr, ibad = target_module.scanls(" 10,20 ,30,,40", vals, 4, 0, 0, 0)
    assert (nv, ierr, ibad) == (4, 0, 0)
    assert vals.tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])


@pytest.mark.parametrize("text", [" ", "   ", ",,  ,"])
def test_scanls_nothing_to_read(text):
    vals = np.zeros(3)
    assert target_module.scanls(text, vals, 3, 0, 0, 0) == (0, 0, 0)
    assert vals.tolist() == [0.0, 0.0, 0.0]


def test_scanls_too_many_values():
    vals = np.zeros(2)
    nv, ierr, ibad = target_module.scanls("7 8 9", vals, 2, 0, 0, 0)
    assert (nv, ierr, ibad) == (2, 3, 5)
    assert vals.tolist() == pytest.approx([7.0, 8.0])


def test_scanls_exactly_maxv_values_is_fine():
    vals = np.zeros(2)
    assert target_module.scanls("1 2 ", vals, 2, 0, 0, 0) == (2, 0, 0)
    assert target_module.scanls("1 2,", vals, 2, 0, 0, 0) == (2, 0, 0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1, x", (1, 1, 4)),
        ("12abc", (1, 1, 3)),
        ("1e 5", (1, 1, 2)),
        ("x", (0, 1, 1)),
        ("4 1e999", (1, 2, 3)),
        ("1e400 5", (0, 2, 1)),
    ],
)
def test_scanls_errors_report_the_start_of_the_failing_item(text, expected):
    vals = np.zeros(4)
    assert target_module.scanls(text, vals, 4, 0, 0, 0) == expected


def test_scanls_keeps_values_read_before_an_error():
    vals = np.zeros(4)
    assert target_module.scanls("4 5 oops 6", vals, 4, 0, 0, 0) == (2, 1, 5)
    assert vals.tolist() == pytest.approx([4.0, 5.0, 0.0, 0.0])


def test_output_placeholders_are_ignored():
    assert target_module.scannm("12", 99.0, 99) == (12.0, 2, 0)
    assert target_module.scannm("abc", 99.0, 99) == (0.0, 0, 1)
    vals = np.zeros(2)
    assert target_module.scanls("1 2", vals, 2, 99, 99, 99) == (2, 0, 0)
