import pytest

import target_module as tm

EXPECTED_REPORT = [
    " EMP     GROSS       TAX       NET",
    " 101    500.00     62.50    437.50",
    " 102    920.00    205.00    715.00",
    " 103   1100.00    300.00    800.00",
    " 104  BAD GRADE  7",
    " 105     78.00      0.00     78.00",
    " TOT   2598.00    567.50   2030.50",
    " PAID  4  REJECTED  1",
    " AVG NET    507.63",
    " TOP NET    800.00",
]


def test_main_prints_the_exact_report(capsys):
    tm.main()
    out = capsys.readouterr().out
    assert out.split("\n") == EXPECTED_REPORT + [""]


def test_main_report_column_layout(capsys):
    tm.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10
    assert lines[1][:4] == " 101"
    assert len(lines[1]) == 34
    assert lines[5][24:34] == "     78.00"
    assert lines[8].endswith("507.63")
    assert lines[9] == " TOP NET    800.00"


@pytest.mark.parametrize(
    "hours, rate, grade, expected",
    [
        (40.0, 12.5, 1, (500.0, 62.5)),
        (44.0, 20.0, 2, (920.0, 205.0)),
        (42.0, 25.0, 3, (1100.0, 300.0)),
        (45.0, 10.0, 1, (450.0, 56.25)),
        (30.0, 20.0, 3, (600.0, 112.5)),
        (8.0, 9.75, 2, (78.0, 0.0)),
        (5.0, 20.0, 2, (100.0, 0.0)),
        (0.0, 15.0, 1, (0.0, 0.0)),
    ],
)
def test_payclc_valid_grades(hours, rate, grade, expected):
    gross, tax, alt = tm.payclc(hours, rate, grade, 0.0, 0.0)
    assert alt == 0
    assert gross == pytest.approx(expected[0], rel=1e-5)
    assert tax == pytest.approx(expected[1], rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("grade", [0, 4, 7, -1, 100])
def test_payclc_bad_grade_takes_alternate_return(grade):
    assert tm.payclc(40.0, 10.0, grade, 5.0, 5.0) == (0.0, 0.0, 1)


def test_payclc_overtime_multiplier_differs_by_grade():
    g1 = tm.payclc(50.0, 10.0, 1, 0.0, 0.0)[0]
    g2 = tm.payclc(50.0, 10.0, 2, 0.0, 0.0)[0]
    g3 = tm.payclc(50.0, 10.0, 3, 0.0, 0.0)[0]
    assert g1 == pytest.approx(500.0, rel=1e-5)
    assert g2 == pytest.approx(550.0, rel=1e-5)
    assert g3 == pytest.approx(600.0, rel=1e-5)


@pytest.mark.parametrize(
    "amt, expected",
    [
        (12.34, (1, 12, 34)),
        (-12.34, (-1, -12, 34)),
        (2.625, (1, 2, 63)),
        (-2.625, (-1, -2, 63)),
        (0.5, (1, 0, 50)),
        (-0.5, (-1, 0, 50)),
        (0.99, (1, 0, 99)),
        (-0.99, (-1, 0, 99)),
        (0.0, (1, 0, 0)),
        (0.004, (1, 0, 0)),
        (-0.004, (1, 0, 0)),
        (1234.5, (1, 1234, 50)),
        (1000000.0, (1, 1000000, 0)),
    ],
)
def test_splitc(amt, expected):
    assert tm.splitc(amt, 0, 0, 0) == expected


def test_splitc_rounds_half_cents_away_from_zero():
    assert tm.splitc(507.625, 0, 0, 0) == (1, 507, 63)
    assert tm.splitc(-507.625, 0, 0, 0) == (-1, -507, 63)
    assert tm.splitc(0.125, 0, 0, 0) == (1, 0, 13)


def test_main_is_repeatable(capsys):
    tm.main()
    first = capsys.readouterr().out
    tm.main()
    second = capsys.readouterr().out
    assert first == second
    assert first.split("\n")[:-1] == EXPECTED_REPORT
