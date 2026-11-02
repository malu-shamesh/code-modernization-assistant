import target_module
import pytest


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("AB", "AB", 0),
        ("AB", "AB   ", 0),
        ("AB   ", "AB", 0),
        ("A", "B", -1),
        ("a", "B", 1),
        ("abc", "abd", -1),
        ("AB", "AB!", -1),  # '!' (33) is above the blank (32)
        ("AB", "AB\t", 1),  # a tab (9) is below the blank
        ("AB", "AB\x01", 1),
        ("AB  X", "AB", 1),
        ("AB", "AB\x01\x01", 1),
        ("Z", "a", -1),
    ],
)
def test_order_blank_padded(a, b, expected):
    assert target_module.order(a, b) == expected


def test_order_is_antisymmetric():
    for a, b in [("A", "B"), ("AB", "AB!"), ("AB\t", "AB")]:
        assert target_module.order(a, b) == -target_module.order(b, a)


@pytest.mark.parametrize(
    "text, mode, width, expected",
    [
        ("  hello  ", 1, 10, "hello"),
        ("  hello  ", 2, 10, "     hello"),
        ("  hello  ", 3, 10, "  hello"),
        ("  hello  ", 3, 11, "   hello"),
        ("ab", 3, 9, "   ab"),
        ("ab", 3, 10, "    ab"),
        ("ab", 3, 1, "a"),
        ("abcdefghij", 3, 4, "abcd"),
        ("abcdefghij", 2, 4, "abcd"),
        ("abcdefghij", 1, 4, "abcd"),
        ("x", 2, 60, " " * 39 + "x"),
        ("x", 3, 60, " " * 19 + "x"),
        ("hello", 1, 0, "h"),
        ("hello", 2, 0, "h"),
        ("hello", 9, 8, "hello"),
        ("hello", 0, 8, "hello"),
        ("   ", 2, 10, ""),
        ("a b  c", 2, 10, "    a b  c"),
    ],
)
def test_justfy(text, mode, width, expected):
    got = target_module.justfy(text, mode, width, "")
    assert got == expected.ljust(40)
    assert len(got) == 40


def strip(names):
    return [s.rstrip(" ") for s in names]


def test_sortnm_ascii_order():
    names = ["ZEBRA", "apple", "Mango", "banana", "10 pins", " lead"]
    target_module.sortnm(names, 6)
    assert strip(names) == [" lead", "10 pins", "Mango", "ZEBRA", "apple", "banana"]


def test_sortnm_only_first_n():
    names = ["delta", "alpha", "charlie", "bravo", "aardvark"]
    target_module.sortnm(names, 3)
    assert strip(names) == ["alpha", "charlie", "delta", "bravo", "aardvark"]


def test_sortnm_trivial_sizes_leave_data_alone():
    names = ["b", "a"]
    target_module.sortnm(names, 1)
    assert strip(names) == ["b", "a"]
    target_module.sortnm(names, 0)
    assert strip(names) == ["b", "a"]
    target_module.sortnm(names, 2)
    assert strip(names) == ["a", "b"]


def test_sortnm_blank_padded_comparison():
    names = ["AB!", "AB", "AB\t"]
    target_module.sortnm(names, 3)
    assert [s.rstrip(" ") for s in names] == ["AB\t", "AB", "AB!"]


def test_sortnm_already_sorted_and_duplicates():
    names = ["a", "b", "b", "c"]
    target_module.sortnm(names, 4)
    assert strip(names) == ["a", "b", "b", "c"]
    names = ["c", "b", "b", "a"]
    target_module.sortnm(names, 4)
    assert strip(names) == ["a", "b", "b", "c"]


def test_sortnm_reverse_order_of_nine():
    names = [str(d) for d in range(9, 0, -1)]
    target_module.sortnm(names, 9)
    assert strip(names) == [str(d) for d in range(1, 10)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  a   b  c ", "a b c"),
        ("one two", "one two"),
        ("a    b", "a b"),
        ("     ", ""),
        ("x", "x"),
        ("   lead", "lead"),
        ("a" * 29 + "  b", "a" * 29 + " "),
        ("a" * 30 + " b", "a" * 30),
        ("abcdefghij" * 4, "abcdefghij" * 3),
        ("tab\there  now", "tab\there now"),
    ],
)
def test_squeez(text, expected):
    got = target_module.squeez(text, "")
    assert got == expected.ljust(30)
    assert len(got) == 30


def test_output_placeholders_are_ignored():
    junk = "j" * 40
    assert target_module.justfy("hi", 2, 5, junk) == "   hi".ljust(40)
    assert target_module.justfy("   ", 2, 5, junk) == " " * 40
    assert target_module.squeez("a  b", "j" * 30) == "a b".ljust(30)
