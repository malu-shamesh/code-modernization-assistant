import target_module
import pytest


@pytest.mark.parametrize(
    "text, expected",
    [
        ("abc", 3),
        ("abc   ", 3),
        ("   ", 0),
        ("  a b ", 5),
        ("x", 1),
        (" x", 2),
    ],
)
def test_lastnb(text, expected):
    assert target_module.lastnb(text) == expected


@pytest.mark.parametrize("c", list("09AZaz"))
def test_isalnm_true(c):
    assert target_module.isalnm(c) is True


@pytest.mark.parametrize("c", list("/:@[`{ _-"))
def test_isalnm_false(c):
    # neighbours of the digit and letter ranges, blank and punctuation
    assert target_module.isalnm(c) is False


@pytest.mark.parametrize(
    "c, expected",
    [("a", "A"), ("z", "Z"), ("Q", "Q"), ("5", "5"), ("{", "{"), ("`", "`"), (" ", " ")],
)
def test_foldup(c, expected):
    assert target_module.foldup(c) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("abc", "cba"),
        ("Hello World", "dlroW olleH"),
        ("  ab", "ba  "),
        ("ab   ", "ba"),
        ("x", "x"),
        ("     ", ""),
    ],
)
def test_revers(text, expected):
    got = target_module.revers(text, "")
    assert got == expected.ljust(40)
    assert len(got) == 40


def test_revers_truncates_to_forty():
    text = "0123456789" * 5
    got = target_module.revers(text, "")
    # only the first 40 characters are reversed
    assert got == ("9876543210" * 4)


@pytest.mark.parametrize(
    "text",
    [
        "racecar",
        "A man, a plan, a canal: Panama",
        "Was it a car or a cat I saw?",
        "12321",
        "a",
        "aa",
        "   ",
        "no 'x' in Nixon",
        "Madam, I'm Adam",
    ],
)
def test_ispaln_true(text):
    assert target_module.ispaln(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "ab",
        "12 31",
        "palindrome",
        "abca",
        "A man, a plan, a canal: Panamb",
    ],
)
def test_ispaln_false(text):
    assert target_module.ispaln(text) is False


def test_revers_output_placeholder_is_ignored():
    assert target_module.revers("abc", "x" * 40) == "cba".ljust(40)
