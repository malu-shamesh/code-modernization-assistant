import target_module
import pytest

W = 40


def pad(s):
    return s.ljust(W)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World 42", "HELLO, WORLD 42"),
        ("already UPPER", "ALREADY UPPER"),
        ("", ""),
        ("   ", ""),
        # characters just outside the letter ranges must not change
        ("@AZ[`az{", "@AZ[`AZ{"),
    ],
)
def test_touppr(text, expected):
    got = target_module.touppr(text, "")
    assert got == pad(expected)
    assert len(got) == W


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World 42", "hello, world 42"),
        ("ALL CAPS", "all caps"),
        ("@AZ[`az{", "@az[`az{"),
    ],
)
def test_tolowr(text, expected):
    assert target_module.tolowr(text, "") == pad(expected)


def test_swapcs():
    assert target_module.swapcs("Hello World 7!", "") == pad("hELLO wORLD 7!")
    assert target_module.swapcs("@aZ[", "") == pad("@Az[")
    # both ends of both letter ranges, and the characters just outside them
    assert target_module.swapcs("@AZ[`az{", "") == pad("@az[`AZ{")


def test_truncation_to_forty_characters():
    long_text = "abcdefghij" * 5
    assert target_module.touppr(long_text, "") == "ABCDEFGHIJ" * 4
    assert target_module.tolowr(long_text.upper(), "") == "abcdefghij" * 4
    assert target_module.swapcs(long_text, "") == "ABCDEFGHIJ" * 4


def test_nupper_counts_only_capitals():
    assert target_module.nupper("Hello World ABC") == 5
    assert target_module.nupper("abc xyz 123") == 0
    assert target_module.nupper("@AZ[") == 2
    assert target_module.nupper("   ") == 0


def test_nupper_counts_beyond_forty_characters():
    # unlike the converting routines, NUPPER scans the whole argument
    assert target_module.nupper("A" * 50) == 50


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the qUICK  brown fOX", "The Quick  Brown Fox"),
        ("  hello", "  Hello"),
        ("3rd place", "3rd Place"),
        ("ALL CAPS HERE", "All Caps Here"),
        ("a", "A"),
        ("o'neil", "O'neil"),
    ],
)
def test_titlec(text, expected):
    assert target_module.titlec(text, "") == pad(expected)


def test_titlec_range_boundaries():
    got = target_module.titlec("a` z{ b@ cA dZ d[ `a {z", "")
    assert got == pad("A` Z{ B@ Ca Dz D[ `a {z")


def test_titlec_truncates():
    got = target_module.titlec("word " * 10, "")
    assert got == "Word " * 8


def test_output_placeholder_is_ignored():
    # the caller's RES is overwritten, whatever it held before
    junk = "zzzzzzzz"
    assert target_module.touppr("abc", junk) == pad("ABC")
    assert target_module.tolowr("ABC", junk) == pad("abc")
    assert target_module.swapcs("aB", junk) == pad("Ab")
    assert target_module.titlec("ab cd", junk) == pad("Ab Cd")
