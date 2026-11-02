"""Character helpers (1-based, blank padded) and DO loop helpers."""

from __future__ import annotations

import numpy as np
import pytest

from modernizer.runtime import f77rt as rt


class TestCharCodes:
    def test_ichar(self):
        assert rt.ichar("A") == 65
        assert rt.ichar("abc") == 97
        assert rt.ichar(" ") == 32

    def test_ichar_errors(self):
        with pytest.raises(ValueError):
            rt.ichar("")
        with pytest.raises(TypeError):
            rt.ichar(65)

    def test_char(self):
        assert rt.char_(65) == "A"
        assert rt.char_(0) == "\x00"
        assert rt.char_(255) == "\xff"

    @pytest.mark.parametrize("bad", [-1, 256, 1000])
    def test_char_out_of_range(self, bad):
        with pytest.raises(ValueError):
            rt.char_(bad)

    def test_ichar_char_round_trip(self):
        assert all(rt.ichar(rt.char_(i)) == i for i in range(256))


class TestIndexAndLength:
    def test_index_is_one_based(self):
        assert rt.index_("abcabc", "bc") == 2
        assert rt.index_("abcabc", "a") == 1

    def test_index_back_finds_last_occurrence(self):
        assert rt.index_("abcabc", "bc", back=True) == 5
        assert rt.index_("abcabc", "bc", True) == 5

    def test_index_absent_is_zero(self):
        assert rt.index_("abc", "z") == 0
        assert rt.index_("", "a") == 0

    def test_index_of_empty_substring(self):
        assert rt.index_("abc", "") == 1
        assert rt.index_("abc", "", back=True) == 4

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("ab   ", 2), ("   ", 0), ("", 0), ("  a b  ", 5), ("a\t ", 2), ("abc", 3)],
    )
    def test_len_trim_only_removes_blanks(self, text, expected):
        assert rt.len_trim(text) == expected


class TestComparison:
    def test_equality_ignores_trailing_blanks(self):
        assert rt.str_eq("abc", "abc   ")
        assert rt.str_eq("", "   ")
        assert not rt.str_eq("abc", "abd")
        assert not rt.str_eq(" abc", "abc")

    def test_str_cmp_signs(self):
        assert rt.str_cmp("a", "b") == -1
        assert rt.str_cmp("b", "a") == 1
        assert rt.str_cmp("a ", "a") == 0

    def test_padding_uses_a_blank_not_a_smaller_character(self):
        # 'a' pads to 'a ' (blank = 32) which is greater than 'a' followed by chr(1).
        assert rt.str_cmp("a", "a\x01") == 1
        assert rt.lgt("a", "a\x01")

    @pytest.mark.parametrize(
        ("func", "a", "b", "expected"),
        [
            (rt.lge, "abc", "abc ", True),
            (rt.lge, "abd", "abc", True),
            (rt.lge, "abb", "abc", False),
            (rt.lgt, "B", "A", True),
            (rt.lgt, "abc", "abc", False),
            (rt.lle, "abc", "abd", True),
            (rt.lle, "abc", "abc  ", True),
            (rt.lle, "abd", "abc", False),
            (rt.llt, "A", "a", True),
            (rt.llt, "9", "A", True),
            (rt.llt, "Z", "a", True),
            (rt.llt, "a", "A", False),
            (rt.llt, "abc", "abc", False),
        ],
    )
    def test_ascii_collating_functions(self, func, a, b, expected):
        assert func(a, b) is expected


class TestAssignAndSubstrings:
    def test_char_assign_pads_and_truncates(self):
        assert rt.char_assign("abc", 5) == "abc  "
        assert rt.char_assign("abcdef", 3) == "abc"
        assert rt.char_assign("", 2) == "  "
        assert rt.char_assign("abc", 3) == "abc"
        assert rt.char_assign("abc", 0) == ""

    def test_char_assign_errors(self):
        with pytest.raises(ValueError):
            rt.char_assign("abc", -1)
        with pytest.raises(TypeError):
            rt.char_assign(5, 3)

    @pytest.mark.parametrize(
        ("text", "i", "j", "expected"),
        [
            ("abcdef", 2, 4, "bcd"),
            ("abcdef", 1, 1, "a"),
            ("abcdef", 1, 6, "abcdef"),
            ("abcdef", 4, 3, ""),
            ("abc", 2, 5, "bc  "),
            ("abc", 5, 6, "  "),
        ],
    )
    def test_substr(self, text, i, j, expected):
        assert rt.substr(text, i, j) == expected

    @pytest.mark.parametrize("start", [0, -1])
    def test_substr_start_below_one_is_an_error(self, start):
        with pytest.raises(IndexError):
            rt.substr("abc", start, 2)

    def test_set_substr_pads_value(self):
        assert rt.set_substr("abcdef", 2, 4, "XY") == "aXY ef"

    def test_set_substr_truncates_value(self):
        assert rt.set_substr("abcdef", 2, 3, "XYZ") == "aXYdef"

    def test_set_substr_extends_short_strings_with_blanks(self):
        assert rt.set_substr("abc", 2, 5, "wxyz") == "awxyz"
        assert rt.set_substr("a", 3, 3, "z") == "a z"

    def test_set_substr_empty_range_is_a_no_op(self):
        assert rt.set_substr("abc", 3, 2, "q") == "abc"

    def test_set_substr_start_below_one_is_an_error(self):
        with pytest.raises(IndexError):
            rt.set_substr("abc", 0, 2, "q")

    def test_set_substr_does_not_mutate_length_inside_bounds(self):
        assert len(rt.set_substr("abcdef", 1, 6, "xy")) == 6


class TestDoTrips:
    @pytest.mark.parametrize(
        ("start", "limit", "step", "expected"),
        [
            (1, 10, 1, 10),
            (1, 10, 3, 4),  # 1 4 7 10
            (10, 1, -3, 4),  # 10 7 4 1
            (1, 0, 1, 0),
            (5, 1, 1, 0),
            (1, 10, -1, 0),
            (1, 2, 5, 1),  # (2-1+5)/5 = 1.2 truncates to 1
            (2, 1, 5, 0),  # (1-2+5)/5 = 0.8 truncates to 0
            (3, 1, 5, 0),
            (10, 1, -20, 1),  # (1-10-20)/-20 = 1.45 truncates to 1
            (1, 1, 1, 1),
            (1, 1, -1, 1),
            (-5, 5, 2, 6),  # -5 -3 -1 1 3 5
            (0, 100, 7, 15),  # 0 7 ... 98
        ],
    )
    def test_integer_trip_counts(self, start, limit, step, expected):
        assert rt.do_trips(start, limit, step) == expected

    def test_default_step_is_one(self):
        assert rt.do_trips(1, 4) == 4

    def test_real_loops(self):
        assert rt.do_trips(1.0, 2.0, 0.5) == 3
        assert rt.do_trips(0.0, 0.3, 0.1) == 4
        assert rt.do_trips(0.0, 1.0, 0.1) == 11
        assert rt.do_trips(1.0, 0.0, 0.5) == 0

    def test_zero_step_is_an_error(self):
        with pytest.raises(ValueError):
            rt.do_trips(1, 10, 0)

    def test_numpy_integers(self):
        assert rt.do_trips(np.int64(1), np.int64(10), np.int64(3)) == 4


class TestDoRange:
    def test_integer_ranges(self):
        assert list(rt.do_range(1, 5)) == [1, 2, 3, 4, 5]
        assert list(rt.do_range(10, 1, -3)) == [10, 7, 4, 1]
        assert list(rt.do_range(1, 0)) == []
        assert list(rt.do_range(1, 10, 4)) == [1, 5, 9]

    def test_integer_values_are_python_ints(self):
        assert all(type(v) is int for v in rt.do_range(np.int64(1), np.int64(3)))

    def test_real_values_are_produced_by_repeated_addition(self):
        accumulated = []
        x = 0.0
        for _ in range(11):
            accumulated.append(x)
            x += 0.1
        values = list(rt.do_range(0.0, 1.0, 0.1))
        assert values == accumulated
        # Repeated addition and multiplication differ in the last bits for step 0.1.
        assert values != [k * 0.1 for k in range(11)]
        assert values[-1] == 0.9999999999999999

    def test_real_loop_trip_count_is_computed_once_not_from_the_drifting_value(self):
        # the accumulated value ends at 0.9999999999999999, but the count is fixed up front
        assert len(list(rt.do_range(0.0, 1.0, 0.1))) == rt.do_trips(0.0, 1.0, 0.1) == 11

    def test_real_values_with_negative_step(self):
        expected = []
        x = 1.0
        for _ in range(rt.do_trips(1.0, 0.0, -0.1)):
            expected.append(x)
            x += -0.1
        assert list(rt.do_range(1.0, 0.0, -0.1)) == expected

    def test_real_values_with_integer_arguments_are_floats(self):
        values = list(rt.do_range(1.0, 3, 1))
        assert values == [1.0, 2.0, 3.0]
        assert all(type(v) is float for v in values)

    def test_a_real_limit_makes_the_loop_real_even_with_integer_start_and_step(self):
        values = list(rt.do_range(1, 2.5))
        assert values == [1.0, 2.0]
        assert all(type(v) is float for v in values)
        assert type(rt.do_final(1, 2.5)) is float
        assert rt.do_final(1, 2.5) == 3.0

    def test_real_step_with_integer_start_gives_floats(self):
        assert list(rt.do_range(1, 2, 0.5)) == [1.0, 1.5, 2.0]

    def test_real_range_with_quarter_steps(self):
        assert list(rt.do_range(0.0, 1.0, 0.25)) == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_returns_an_iterator(self):
        it = rt.do_range(1, 3)
        assert iter(it) is it
        assert next(it) == 1

    def test_zero_step_fails_immediately(self):
        with pytest.raises(ValueError):
            rt.do_range(1, 5, 0)

    def test_trip_count_is_fixed_when_the_loop_starts(self):
        limit = 5
        seen = []
        for i in rt.do_range(1, limit):
            limit = 2
            seen.append(i)
        assert seen == [1, 2, 3, 4, 5]


class TestDoFinal:
    @pytest.mark.parametrize(
        ("start", "limit", "step", "expected"),
        [(1, 10, 1, 11), (1, 10, 3, 13), (10, 1, -3, -2), (1, 0, 1, 1), (5, 1, 1, 5)],
    )
    def test_integer_loops(self, start, limit, step, expected):
        assert rt.do_final(start, limit, step) == expected

    def test_real_loop(self):
        assert rt.do_final(1.0, 2.0, 0.5) == 2.5

    @pytest.mark.parametrize(
        ("start", "limit", "step"),
        [(0.0, 1.0, 0.1), (0.0, 0.5, 0.1), (1.0, 0.0, -0.1), (0.3, 2.0, 0.7), (0.0, 10.0, 0.1)],
    )
    def test_real_loop_final_value_matches_an_independent_loop(self, start, limit, step):
        trips = int((limit - start + step) / step)
        x = start
        for _ in range(trips):
            x += step
        assert rt.do_final(start, limit, step) == x
        assert rt.do_trips(start, limit, step) == trips

    def test_real_loop_final_value_differs_from_multiplication(self):
        assert rt.do_final(0.0, 1.0, 0.1) == 1.0999999999999999
        assert rt.do_final(0.0, 1.0, 0.1) != 0.0 + 11 * 0.1

    def test_empty_real_loop_keeps_the_start_value(self):
        assert rt.do_final(2.0, 1.0, 0.5) == 2.0

    def test_final_value_follows_the_last_yielded_value(self):
        for start, limit, step in [(0.0, 1.0, 0.1), (1.0, 0.0, -0.1), (0.0, 2.0, 0.3)]:
            values = list(rt.do_range(start, limit, step))
            assert rt.do_final(start, limit, step) == values[-1] + step

    def test_matches_python_loop_end_for_non_empty_loops(self):
        for start, limit, step in [(1, 7, 2), (20, 3, -4), (0, 9, 9)]:
            last = list(rt.do_range(start, limit, step))[-1]
            assert rt.do_final(start, limit, step) == last + step
