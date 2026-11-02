import target_module
import pytest


@pytest.mark.parametrize(
    "n, text",
    [
        (0, "ZERO"),
        (7, "SEVEN"),
        (13, "THIRTEEN"),
        (19, "NINETEEN"),
        (20, "TWENTY"),
        (21, "TWENTY-ONE"),
        (40, "FORTY"),
        (99, "NINETY-NINE"),
        (100, "ONE HUNDRED"),
        (101, "ONE HUNDRED ONE"),
        (110, "ONE HUNDRED TEN"),
        (342, "THREE HUNDRED FORTY-TWO"),
        (1000, "ONE THOUSAND"),
        (1001, "ONE THOUSAND ONE"),
        (12345, "TWELVE THOUSAND THREE HUNDRED FORTY-FIVE"),
        (100000, "ONE HUNDRED THOUSAND"),
        (1000000, "ONE MILLION"),
        (1000010, "ONE MILLION TEN"),
        (2000500, "TWO MILLION FIVE HUNDRED"),
        (-1, "MINUS ONE"),
        (-5, "MINUS FIVE"),
        (-1000000, "MINUS ONE MILLION"),
        (
            999999999,
            "NINE HUNDRED NINETY-NINE MILLION NINE HUNDRED NINETY-NINE "
            "THOUSAND NINE HUNDRED NINETY-NINE",
        ),
    ],
)
def test_numwrd(n, text):
    words, nch, ierr = target_module.numwrd(n, "", 0, 0)
    assert ierr == 0
    assert words == text.ljust(120)
    assert nch == len(text)


def test_numwrd_longest_text_fits_the_buffer():
    # 373 is the longest group (27 characters); three of them give the longest text
    group = "THREE HUNDRED SEVENTY-THREE"
    text = "MINUS " + group + " MILLION " + group + " THOUSAND " + group
    words, nch, ierr = target_module.numwrd(-373373373, "", 0, 0)
    assert ierr == 0
    assert nch == len(text) == 106
    assert words == text.ljust(120)


def test_numwrd_nine_nines_is_not_the_longest():
    words, nch, ierr = target_module.numwrd(-999999999, "", 0, 0)
    assert ierr == 0
    assert words.startswith("MINUS NINE HUNDRED NINETY-NINE MILLION")
    assert nch == 97


@pytest.mark.parametrize("n", [1000000000, -1000000000, 2000000000])
def test_numwrd_out_of_range(n):
    assert target_module.numwrd(n, "", 0, 0) == (" " * 120, 0, 1)


def test_lastnb():
    assert target_module.lastnb("abc  ") == 3
    assert target_module.lastnb("   ") == 0
    assert target_module.lastnb("a b") == 3
    assert target_module.lastnb("x") == 1
    assert target_module.lastnb(" x") == 2


def test_addwrd_first_word_has_no_leading_blank():
    buf, pos = target_module.addwrd(" " * 120, 1, "HELLO")
    assert pos == 6
    assert buf == "HELLO".ljust(120)


def test_addwrd_separates_with_one_blank_and_trims_the_word():
    buf, pos = target_module.addwrd("HELLO".ljust(120), 6, "WORLD   ")
    assert pos == 12
    assert buf == "HELLO WORLD".ljust(120)


def test_addwrd_ignores_blank_word():
    buf0 = "HELLO".ljust(120)
    assert target_module.addwrd(buf0, 6, "    ") == (buf0, 6)


def test_addwrd_writes_by_position_not_by_clearing():
    buf, pos = target_module.addwrd("X" * 120, 1, "HI")
    assert pos == 3
    assert buf == "HI" + "X" * 118


def test_addwrd_one_letter_words():
    buf, pos = target_module.addwrd(" " * 120, 1, "A")
    assert (buf, pos) == ("A".ljust(120), 2)
    buf, pos = target_module.addwrd(buf, pos, "B")
    assert (buf, pos) == ("A B".ljust(120), 4)


def test_addwrd_writes_the_separator_blank_explicitly():
    buf, pos = target_module.addwrd("X" * 120, 6, "HI")
    assert pos == 9
    assert buf == "X" * 5 + " HI" + "X" * 112


def test_addwrd_fit_check_counts_the_separator_only_after_the_first_word():
    # a word of exactly the buffer length fits at position 1
    assert target_module.addwrd(" " * 120, 1, "A" * 120) == ("A" * 120, 121)
    # at position 2 the blank is needed too: 118 letters fit, 119 do not
    start = "A".ljust(120)
    buf, pos = target_module.addwrd(start, 2, "B" * 118)
    assert (buf, pos) == ("A " + "B" * 118, 121)
    assert target_module.addwrd(start, 2, "B" * 119) == (start, 2)


def test_addwrd_drops_word_that_does_not_fit():
    start = "A" * 117 + "   "
    # 117 used, pos 118: "AB" needs 3 positions (blank + 2) and just fits
    buf, pos = target_module.addwrd(start, 118, "AB")
    assert pos == 121
    assert buf == "A" * 117 + " AB"
    # "ABC" needs 4 positions (blank + 3) and does not fit
    assert target_module.addwrd(start, 118, "ABC") == (start, 118)
    # "ABCD" needs 5 positions and is dropped without touching the buffer
    buf2, pos2 = target_module.addwrd(start, 118, "ABCD")
    assert (buf2, pos2) == (start, 118)


def test_output_placeholders_are_ignored():
    assert target_module.numwrd(21, "junk", 77, 5) == ("TWENTY-ONE".ljust(120), 10, 0)
    assert target_module.numwrd(10**9, "junk", 77, 5) == (" " * 120, 0, 1)
