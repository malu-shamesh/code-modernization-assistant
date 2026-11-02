import pytest

import target_module as tm

BLANK = " " * 16


@pytest.mark.parametrize(
    "n, expected",
    [(0, 0), (7, 7), (12345, 15), (-9875, 29), (1000, 1), (999999, 54)],
)
def test_idigsm(n, expected):
    assert tm.idigsm(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [(123, 321), (120, 21), (-120, -21), (0, 0), (5, 5), (1000, 1), (12321, 12321), (-907, -709)],
)
def test_irev(n, expected):
    assert tm.irev(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (-5, 0),
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 2),
        (15, 3),
        (16, 4),
        (99, 9),
        (100, 10),
        (1000000, 1000),
        (1000000007, 31622),
    ],
)
def test_isqrti(n, expected):
    assert tm.isqrti(n) == expected


@pytest.mark.parametrize(
    "n, ibase, expected",
    [
        (255, 16, "FF"),
        (255, 2, "11111111"),
        (-10, 2, "-1010"),
        (0, 2, "0"),
        (1000, 10, "1000"),
        (4095, 16, "FFF"),
        (-255, 16, "-FF"),
        (100, 8, "144"),
        (2147483647, 16, "7FFFFFFF"),
    ],
)
def test_itob_is_left_justified_and_blank_padded(n, ibase, expected):
    out = tm.itob(n, ibase, "Z" * 16)
    assert len(out) == 16
    assert out == expected.ljust(16)


@pytest.mark.parametrize("ibase", [-1, 0, 1, 17, 36])
def test_itob_rejects_unsupported_bases(ibase):
    assert tm.itob(10, ibase, "Z" * 16) == "?" + " " * 15


@pytest.mark.parametrize(
    "n, expected",
    [
        (65535, "1" * 16),  # exactly sixteen digits fit
        (65536, "*" * 16),  # seventeen do not
        (-32767, "-" + "1" * 15),  # sign included, sixteen characters
        (-65535, "*" * 16),
    ],
)
def test_itob_width_limit_counts_the_sign(n, expected):
    assert tm.itob(n, 2, BLANK) == expected


@pytest.mark.parametrize(
    "text, ibase, expected",
    [
        ("FF", 16, (255, 0)),
        ("-1010", 2, (-10, 0)),
        ("12 34", 10, (12, 0)),
        ("-12", 10, (-12, 0)),
        ("", 10, (0, 0)),
        ("-", 10, (0, 0)),
        ("7FFFFFFF", 16, (2147483647, 0)),
        ("12G", 16, (0, 3)),
        ("102", 2, (0, 3)),
        ("-G", 16, (0, 2)),
        ("ff", 16, (0, 1)),
        ("7", 1, (0, -1)),
        ("7", 17, (0, -1)),
    ],
)
def test_istoi(text, ibase, expected):
    assert tm.istoi(text.ljust(16), ibase, 123) == expected


@pytest.mark.parametrize("n, ibase", [(0, 2), (1234, 10), (-1234, 7), (48879, 16), (-1, 3)])
def test_radix_round_trip(n, ibase):
    text = tm.itob(n, ibase, BLANK)
    assert tm.istoi(text, ibase, 0) == (n, 0)


@pytest.mark.parametrize(
    "n, ibase, expected",
    [
        (255, 16, " " + "       255" + " IN BASE 16 IS " + "FF".ljust(16) + "\n"),
        (-10, 2, " " + "       -10" + " IN BASE  2 IS " + "-1010".ljust(16) + "\n"),
        (5, 40, " " + "         5" + " IN BASE 40 IS " + "?".ljust(16) + "\n"),
    ],
)
def test_showb_output_layout(capsys, n, ibase, expected):
    tm.showb(n, ibase)
    assert capsys.readouterr().out == expected
