import pytest

from target_module import scinit, scint, scnext, scpos, scsum, sctall


def pad(text):
    return text.ljust(12)


def scan_all(line):
    """Tokens of a line as (text, type) pairs, collected through repeated SCNEXT calls."""
    scinit(line)
    found = []
    for _ in range(100):
        tok, itype, alt = scnext("junk", 7)
        if alt == 1:
            assert itype == 0
            assert tok == " " * 12
            return found
        assert alt == 0
        found.append((tok.rstrip(), itype))
    raise AssertionError("scanner never reached the end of the line")


def test_tokens_positions_and_end_of_line():
    scinit("  AB  CD")
    assert scpos() == 1
    assert scnext("junk", 7) == (pad("AB"), 2, 0)
    assert scpos() == 5
    assert scnext("junk", 7) == (pad("CD"), 2, 0)
    assert scpos() == 9
    assert scnext("junk", 7) == (" " * 12, 0, 1)
    assert scpos() == 9
    assert scnext("junk", 7) == (" " * 12, 0, 1)


def test_expression_line_is_split_into_classified_tokens():
    assert scan_all("X1 = 25+(Y*3)") == [
        ("X1", 2),
        ("=", 3),
        ("25", 1),
        ("+", 3),
        ("(", 3),
        ("Y", 2),
        ("*", 3),
        ("3", 1),
        (")", 3),
    ]


def test_scan_position_after_each_token():
    scinit("X1 = 25+(Y*3)")
    positions = []
    for _ in range(9):
        scnext("junk", 7)
        positions.append(scpos())
    assert positions == [3, 5, 8, 9, 10, 11, 12, 13, 14]


def test_blank_and_empty_lines_have_no_tokens():
    for line in ("", "     ", " " * 60):
        scinit(line)
        assert scpos() == 1
        assert scnext("junk", 7) == (" " * 12, 0, 1)
        # no text at all: the position is never advanced
        assert scpos() == 1
        assert sctall(line, 9, 9, 9) == (0, 0, 0)


@pytest.mark.parametrize(
    "line, expected",
    [
        ("25ABC", [("25", 1), ("ABC", 2)]),
        ("A1B2", [("A1B2", 2)]),
        ("3.14", [("3", 1), (".", 3), ("14", 1)]),
        ("-7", [("-", 3), ("7", 1)]),
        ("_a+b9", [("_", 3), ("a", 2), ("+", 3), ("b9", 2)]),
        ("x y", [("x", 2), ("y", 2)]),
        ("1 2", [("1", 1), ("2", 1)]),
        ("12 34", [("12", 1), ("34", 1)]),
        # characters next to the ASCII ranges '0'-'9', 'A'-'Z' and 'a'-'z' are operators
        ("/9:", [("/", 3), ("9", 1), (":", 3)]),
        ("@Z[", [("@", 3), ("Z", 2), ("[", 3)]),
        ("`z{", [("`", 3), ("z", 2), ("{", 3)]),
        ("0A9Zaz", [("0", 1), ("A9Zaz", 2)]),
        ("A/B:C@D[E`F{G", [(c, 2 if c.isalpha() else 3) for c in "A/B:C@D[E`F{G"]),
    ],
)
def test_token_classes(line, expected):
    assert scan_all(line) == expected


def test_token_text_is_cut_to_twelve_but_position_moves_past_it():
    scinit("ABCDEFGHIJKLMNOP Q")
    assert scnext("junk", 7) == (pad("ABCDEFGHIJKL"), 2, 0)
    assert scpos() == 17
    assert scnext("junk", 7) == (pad("Q"), 2, 0)


def test_long_digit_run_is_truncated_as_well():
    scinit("123456789012345 7")
    tok, itype, alt = scnext("junk", 7)
    assert (tok, itype, alt) == ("123456789012", 1, 0)
    assert scnext("junk", 7) == (pad("7"), 1, 0)


def test_line_is_cut_to_forty_characters():
    line = "X" * 38 + " Y ZZ"
    assert scan_all(line) == [("X" * 12, 2), ("Y", 2)]
    # the 40th character is the last one seen, the following blank and ZZ are lost
    assert sctall(line, 0, 0, 0) == (0, 2, 0)
    scinit("A" * 40 + "B")
    assert scnext("junk", 7)[0] == "A" * 12
    assert scpos() == 41
    assert scnext("junk", 7)[2] == 1


def test_scinit_restarts_the_scan():
    scinit("AAA BBB")
    scnext("junk", 7)
    scinit("CCC")
    assert scpos() == 1
    assert scnext("junk", 7) == (pad("CCC"), 2, 0)


def test_tally_of_token_classes():
    assert sctall("X1 = 25+(Y*3)", 0, 0, 0) == (2, 2, 5)
    assert sctall("   7  8  9  ", 0, 0, 0) == (3, 0, 0)
    assert sctall("CALL FOO(A, B)", 0, 0, 0) == (0, 4, 3)
    # outputs from a previous call must not leak into the next one
    assert sctall("1", 5, 5, 5) == (1, 0, 0)


def test_tally_resets_the_position_for_later_scans():
    sctall("A B C", 0, 0, 0)
    assert scnext("junk", 7)[2] == 1
    scinit("Z")
    assert scnext("junk", 7) == (pad("Z"), 2, 0)


@pytest.mark.parametrize(
    "text, value",
    [
        ("25", 25),
        ("123456789   ", 123456789),
        ("007", 7),
        ("", 0),
        ("   ", 0),
        ("7 8", 7),
        ("12A", -1),
        ("12:", -1),
        ("1/2", -1),
        ("-5", -1),
        ("A", -1),
    ],
)
def test_digit_string_value(text, value):
    assert scint(text) == value


def test_sum_of_integer_tokens():
    assert scsum("10 + 20 * x3 - 5", 0) == 35
    assert scsum("A1+1", 99) == 1
    assert scsum("no numbers here", 5) == 0
    assert scsum("999999999 1", 0) == 1000000000
    assert scsum("", 0) == 0
