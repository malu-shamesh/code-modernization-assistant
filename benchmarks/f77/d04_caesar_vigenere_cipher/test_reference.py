import numpy as np
import target_module
import pytest

W = 60


def pad(s):
    return s.ljust(W)


@pytest.mark.parametrize(
    "text, key, expected",
    [
        ("Hello, World!", 3, "Khoor, Zruog!"),
        ("Khoor, Zruog!", -3, "Hello, World!"),
        ("xyz XYZ", 3, "abc ABC"),
        ("abc ABC", -3, "xyz XYZ"),
        ("Hello, World!", 29, "Khoor, Zruog!"),
        ("Hello, World!", -29, "Ebiil, Tloia!"),
        ("Hello, World!", 0, "Hello, World!"),
        ("Hello, World!", 26, "Hello, World!"),
        ("Hello, World!", -26, "Hello, World!"),
        ("Hello, World!", 52, "Hello, World!"),
        ("@[`{ 09", 5, "@[`{ 09"),
        ("Zz", 1, "Aa"),
        # the smallest negative shift: MOD(-1, 26) is -1 and needs the fix-up
        ("Hello, World!", -1, "Gdkkn, Vnqkc!"),
        ("Aa", -1, "Zz"),
    ],
)
def test_caesar(text, key, expected):
    got = target_module.caesar(text, key, "")
    assert got == pad(expected)
    assert len(got) == W


def test_caesar_truncates_to_sixty():
    got = target_module.caesar("a" * 65, 1, "")
    assert got == "b" * 60


@pytest.mark.parametrize("key", [1, 7, 13, 25, -4, 40])
def test_caesar_roundtrip(key):
    text = "The Quick Brown Fox, 1 Jumps Over!"
    enc = target_module.caesar(text, key, "")
    assert target_module.caesar(enc, -key, "") == pad(text)


def test_vigenere_classic_example():
    enc = target_module.vigenr("ATTACKATDAWN", "LEMON", 1, "")
    assert enc == pad("LXFOPVEFRNHR")
    dec = target_module.vigenr("LXFOPVEFRNHR", "LEMON", -1, "")
    assert dec == pad("ATTACKATDAWN")


def test_vigenere_key_advances_only_on_letters_and_keeps_case():
    enc = target_module.vigenr("Attack at Dawn!", "lemon", 1, "")
    assert enc == pad("Lxfopv ef Rnhr!")
    assert target_module.vigenr("Lxfopv ef Rnhr!", "LEMON", -1, "") == pad("Attack at Dawn!")


def test_vigenere_ignores_non_letters_in_keyword():
    # only the letters L, E, m, N form the key
    a = target_module.vigenr("ABCDEFGH", "L-E m0N", 1, "")
    b = target_module.vigenr("ABCDEFGH", "LEMN", 1, "")
    assert a == b == pad("LFOQPJSU")


def test_vigenere_keyword_without_letters_copies_text():
    assert target_module.vigenr("Keep me", "123 !", 1, "") == pad("Keep me")
    assert target_module.vigenr("Keep me", "", 1, "") == pad("Keep me")


def test_vigenere_key_a_is_identity_and_b_shifts_by_one():
    assert target_module.vigenr("Same Text", "a", 1, "") == pad("Same Text")
    assert target_module.vigenr("Same Text", "B", 1, "") == pad("Tbnf Ufyu")
    assert target_module.vigenr("Tbnf Ufyu", "B", -1, "") == pad("Same Text")


def test_vigenere_only_first_twenty_key_letters_are_used():
    # key letters B, A x 19, then a 21st letter C that must be ignored
    key = "B" + "A" * 19 + "C"
    assert target_module.vigenr("C" * 21, key, 1, "") == pad("D" + "C" * 19 + "D")


def test_vigenere_decrypting_below_a_wraps_to_z():
    # (0 - 1) is -1 and MOD keeps the sign, so the result needs the fix-up
    assert target_module.vigenr("A", "B", -1, "") == pad("Z")
    assert target_module.vigenr("aB", "bb", -1, "") == pad("zA")


def test_vigenere_truncates_to_sixty():
    got = target_module.vigenr("a" * 70, "b", 1, "")
    assert got == "b" * 60


def test_letfrq_counts_case_insensitively_and_resets():
    freq = np.full(26, 99, dtype=np.int64)
    target_module.letfrq("Hello, World 123", freq)
    expected = np.zeros(26, dtype=np.int64)
    for ch in "helloworld":
        expected[ord(ch) - 97] += 1
    assert freq.tolist() == expected.tolist()
    assert freq[ord("l") - 97] == 3
    assert freq[0] == 0
    assert freq[25] == 0


def test_letfrq_range_boundaries():
    freq = np.zeros(26, dtype=np.int64)
    target_module.letfrq("AZaz@[`{", freq)
    assert freq[0] == 2
    assert freq[25] == 2
    assert freq.sum() == 4


def test_mostfq_first_maximum_wins_and_empty_gives_zero():
    freq = np.zeros(26, dtype=np.int64)
    assert target_module.mostfq(freq) == 0
    freq[2] = 4
    freq[7] = 4
    freq[25] = 1
    assert target_module.mostfq(freq) == 3
    freq[25] = 9
    assert target_module.mostfq(freq) == 26


def test_crack_recovers_shift():
    cipher = target_module.caesar("EEE ATE", 7, "")
    assert cipher.rstrip() == "LLL HAL"
    assert target_module.crack(cipher, 0) == 7


def test_crack_key_stays_non_negative_for_letters_before_e():
    # most frequent letter A: the shift from E is 22 (not -4)
    assert target_module.crack("AB", 0) == 22
    assert target_module.crack("a b a", 0) == 22
    assert target_module.crack("EEEEE", 0) == 0


def test_crack_without_letters():
    assert target_module.crack("123 !?", 0) == 0


def test_output_placeholders_are_ignored():
    junk = "q" * 60
    assert target_module.caesar("abc", 1, junk) == pad("bcd")
    assert target_module.vigenr("abc", "b", 1, junk) == pad("bcd")
    assert target_module.vigenr("abc", "", 1, junk) == pad("abc")
    assert target_module.crack("EEE", 17) == 0
    assert target_module.crack("123", 17) == 0
