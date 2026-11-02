import target_module
import pytest


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "I"),
        (3, "III"),
        (4, "IV"),
        (9, "IX"),
        (14, "XIV"),
        (40, "XL"),
        (49, "XLIX"),
        (90, "XC"),
        (400, "CD"),
        (1994, "MCMXCIV"),
        (2024, "MMXXIV"),
        (3999, "MMMCMXCIX"),
        (3888, "MMMDCCCLXXXVIII"),
    ],
)
def test_toromn(n, expected):
    roman, ierr = target_module.toromn(n, "", 0)
    assert ierr == 0
    assert roman == expected.ljust(16)


@pytest.mark.parametrize("n", [0, -5, 4000, 100000])
def test_toromn_out_of_range(n):
    roman, ierr = target_module.toromn(n, "", 0)
    assert ierr == 1
    assert roman == " " * 16


@pytest.mark.parametrize(
    "roman, value",
    [
        ("I", 1),
        ("XIV", 14),
        ("XIV   ", 14),
        ("MCMXCIV", 1994),
        ("MMXXIV", 2024),
        ("MMMCMXCIX", 3999),
        ("MMMDCCCLXXXVIII", 3888),
    ],
)
def test_fromrm_valid(roman, value):
    assert target_module.fromrm(roman, 0) == (value, 0)


@pytest.mark.parametrize(
    "roman, ierr",
    [
        ("ABC", 2),
        ("xiv", 2),
        ("    ", 2),
        (" XIV", 2),
        ("X I", 2),
        ("M" * 17, 2),
        ("IIII", 3),
        ("IIX", 3),
        ("VX", 3),
        ("IC", 3),
        ("MMMM", 3),
        ("M" * 16, 3),
        ("VV", 3),
    ],
)
def test_fromrm_invalid(roman, ierr):
    assert target_module.fromrm(roman, 0) == (0, ierr)


def test_roundtrip_sampled_numbers():
    for n in range(1, 4000, 7):
        roman, ierr = target_module.toromn(n, "", 0)
        assert ierr == 0
        assert target_module.fromrm(roman, 0) == (n, 0)


def row(n, roman):
    return " %5d  %-16s\n" % (n, roman)


def test_prtrom_simple_table(capsys):
    target_module.prtrom(1, 3, 1)
    assert capsys.readouterr().out == row(1, "I") + row(2, "II") + row(3, "III")


def test_prtrom_step(capsys):
    target_module.prtrom(1, 10, 4)
    assert capsys.readouterr().out == row(1, "I") + row(5, "V") + row(9, "IX")


def test_prtrom_negative_step(capsys):
    target_module.prtrom(10, 1, -3)
    expected = row(10, "X") + row(7, "VII") + row(4, "IV") + row(1, "I")
    assert capsys.readouterr().out == expected


def test_prtrom_zero_trip_loop_prints_nothing(capsys):
    target_module.prtrom(5, 1, 1)
    assert capsys.readouterr().out == ""


def test_prtrom_out_of_range_lines(capsys):
    target_module.prtrom(3998, 4001, 1)
    out = capsys.readouterr().out
    assert out == (
        row(3998, "MMMCMXCVIII")
        + row(3999, "MMMCMXCIX")
        + "  4000  OUT OF RANGE\n"
        + "  4001  OUT OF RANGE\n"
    )
    target_module.prtrom(0, 1, 1)
    assert capsys.readouterr().out == "     0  OUT OF RANGE\n" + row(1, "I")


def test_output_placeholders_are_ignored():
    assert target_module.toromn(4, "ZZZZ", 9) == ("IV".ljust(16), 0)
    assert target_module.toromn(0, "ZZZZ", 9) == (" " * 16, 1)
    assert target_module.fromrm("IV", 9) == (4, 0)
    assert target_module.fromrm("ABC", 9) == (0, 2)
