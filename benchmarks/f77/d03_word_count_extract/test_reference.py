import target_module
import pytest


@pytest.mark.parametrize(
    "text, expected",
    [
        ("the quick  brown fox", (4, 5)),
        ("Hello, world! How are you?", (5, 5)),
        ("   ", (0, 0)),
        ("one", (1, 3)),
        ("a;b:c", (3, 1)),
        ("don't stop", (2, 5)),
        # a tab is not a delimiter, so this is one word of length 3
        ("a\tb", (1, 3)),
        ("  leading and trailing  ", (3, 8)),
        ("end.", (1, 3)),
        (",,,", (0, 0)),
        ("a,b", (2, 1)),
        ("a.b", (2, 1)),
        ("a;b", (2, 1)),
        ("a:b", (2, 1)),
        ("a!b", (2, 1)),
        ("what?why", (2, 4)),
    ],
)
def test_wordct(text, expected):
    assert target_module.wordct(text, 0, 0) == expected


@pytest.mark.parametrize(
    "text, k, word, wlen",
    [
        ("alpha beta gamma", 1, "alpha", 5),
        ("alpha beta gamma", 2, "beta", 4),
        # the last word has no closing delimiter
        ("alpha beta gamma", 3, "gamma", 5),
        ("alpha beta gamma", 4, "", 0),
        ("alpha beta gamma", 0, "", 0),
        ("alpha beta gamma", -1, "", 0),
        ("x, y", 2, "y", 1),
        (",,hi", 1, "hi", 2),
        ("   ", 1, "", 0),
        ("one two  ", 2, "two", 3),
    ],
)
def test_getwrd(text, k, word, wlen):
    got_word, got_len = target_module.getwrd(text, k, "", 0)
    assert got_word == word.ljust(20)
    assert got_len == wlen


def test_getwrd_long_word_is_truncated_but_length_is_true():
    word, wlen = target_module.getwrd("abcdefghijklmnopqrstuvwxyz extra", 1, "", 0)
    assert word == "abcdefghijklmnopqrst"
    assert wlen == 26
    word, wlen = target_module.getwrd("abcdefghijklmnopqrstuvwxyz", 1, "", 0)
    assert word == "abcdefghijklmnopqrst"
    assert wlen == 26


@pytest.mark.parametrize(
    "text, pat, expected",
    [
        ("the cat sat on the mat", "the", 2),
        ("aaaa", "aa", 2),
        ("aaa", "aa", 1),
        ("abc", "", 0),
        ("abc", "   ", 0),
        # trailing blanks of the pattern are not significant
        ("abcabc", "bc  ", 2),
        ("abc abc", "c a", 1),
        ("hello", "xyz", 0),
        ("ab", "abc", 0),
        ("abab", "ab", 2),
        ("a", "a", 1),
    ],
)
def test_cntsub(text, pat, expected):
    assert target_module.cntsub(text, pat) == expected


def test_output_placeholders_are_ignored():
    assert target_module.wordct("a bc", 99, 99) == (2, 2)
    assert target_module.getwrd("a bc", 2, "junk", 99) == ("bc".ljust(20), 2)
    assert target_module.getwrd("a bc", 3, "junk", 99) == (" " * 20, 0)
