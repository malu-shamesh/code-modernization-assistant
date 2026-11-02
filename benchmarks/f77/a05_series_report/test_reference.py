import pytest

import target_module as tm


@pytest.mark.parametrize(
    "n, expected",
    [
        (-2, 0.0),
        (0, 0.0),
        (1, 1.0),
        (2, 1.5),
        (3, 11.0 / 6.0),
        (4, 25.0 / 12.0),
        (10, 7381.0 / 2520.0),
    ],
)
def test_harmon(n, expected):
    assert tm.harmon(n) == pytest.approx(expected, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize(
    "n, expected",
    [(-3, 0), (0, 0), (1, 1), (2, 5), (3, 14), (10, 385), (100, 338350)],
)
def test_isumsq(n, expected):
    assert tm.isumsq(n) == expected


@pytest.mark.parametrize(
    "a, r, n, expected",
    [
        (2.0, 3.0, 4, 80.0),  # 2 + 6 + 18 + 54
        (1.0, -1.0, 5, 1.0),
        (1.0, -1.0, 4, 0.0),
        (5.0, 2.0, 0, 0.0),
        (5.0, 2.0, 1, 5.0),
        (0.5, 0.5, 5, 0.96875),
        (1.0, 0.1, 3, 1.11),
    ],
)
def test_geosum(a, r, n, expected):
    assert tm.geosum(a, r, n) == pytest.approx(expected, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize(
    "n, m, expected",
    [
        (10, 3, 18),  # 3 + 6 + 9
        (100, 7, 735),  # K = 14
        (5, 7, 0),  # K = 0
        (7, 7, 7),
        (20, 1, 210),
        (99, 10, 450),
    ],
)
def test_isummu(n, m, expected):
    assert tm.isummu(n, m) == expected


def test_harmon_large_n():
    assert tm.harmon(1000) == pytest.approx(7.485470860550345, rel=1e-9)


def test_isumsq_matches_closed_form():
    for n in range(0, 40):
        assert tm.isumsq(n) == n * (n + 1) * (2 * n + 1) // 6


def test_geosum_with_ratio_one_is_a_times_n():
    assert tm.geosum(3.0, 1.0, 7) == pytest.approx(21.0, rel=1e-9)


def test_main_prints_seven_lines_and_returns_nothing(capsys):
    assert tm.main() is None
    assert len(capsys.readouterr().out.splitlines()) == 7


def test_main_prints_exact_table(capsys):
    tm.main()
    out = capsys.readouterr().out
    assert out.split("\n") == [
        " N     HARMONIC    SUMSQ    GEOM",
        "   1     1.000000        1    0.50000",
        "   2     1.500000        5    0.75000",
        "   3     1.833333       14    0.87500",
        "   4     2.083333       30    0.93750",
        "   5     2.283333       55    0.96875",
        " MULTIPLES OF 7 UP TO 100:   735",
        "",
    ]
