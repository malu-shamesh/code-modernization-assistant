"""Formatted, list-directed and internal reads, and the list-directed writer."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from modernizer.runtime import f77rt as rt

ri = rt.read_internal


class TestIntegerInput:
    @pytest.mark.parametrize(
        ("text", "fmt", "expected"),
        [
            ("  12", "(I4)", 12),
            ("-12 ", "(I4)", -12),
            ("+7  ", "(I4)", 7),
            ("    ", "(I4)", 0),  # an all-blank field is zero
            ("12", "(I4)", 12),  # short records are blank padded
            ("  1 2", "(I4)", 1),  # only the first four columns are the field: '  1 '
            ("1 2 ", "(I4)", 12),  # embedded blanks are ignored (BN)
            ("12345", "(I3)", 123),
        ],
    )
    def test_values(self, text, fmt, expected):
        assert ri(text, fmt) == [expected]

    def test_blank_zero_mode(self):
        assert ri("1 2 ", "(BZ,I4)") == [1020]
        assert ri("1 2 ", "(BN,I4)") == [12]
        assert ri("1 2 ", "(BZ,BN,I4)") == [12]

    @pytest.mark.parametrize("text", ["ab", "1.5", "1-2", "--1"])
    def test_invalid_integer(self, text):
        with pytest.raises(ValueError):
            ri(text.ljust(3), "(I3)")

    def test_zero_width_is_rejected_for_input(self):
        with pytest.raises(ValueError):
            ri("12", "(I0)")


class TestRealInput:
    @pytest.mark.parametrize(
        ("text", "fmt", "expected"),
        [
            ("  3.14", "(F6.2)", 3.14),
            ("314", "(F3.2)", 3.14),  # no point: the last d digits are fractional
            ("-314", "(F4.2)", -3.14),
            ("3.14", "(F4.0)", 3.14),  # an explicit point overrides d
            ("1.5E3", "(F5.1)", 1500.0),
            ("1.5D3", "(F5.1)", 1500.0),
            ("1.5e3", "(F5.1)", 1500.0),
            ("1.5+3", "(F5.1)", 1500.0),  # sign without a letter introduces the exponent
            ("1.5-3", "(F5.1)", 0.0015),
            ("15E2", "(F4.1)", 150.0),  # implied point applies to the mantissa: 1.5E2
            ("    ", "(F4.1)", 0.0),
            (".5", "(F2.1)", 0.5),
            ("5.", "(F2.1)", 5.0),
            ("1E-2", "(F4.0)", 0.01),
            ("5", "(F1.0)", 5.0),
            ("7", "(F1.3)", 0.007),  # leading zeros are supplied
            ("-0.5", "(F4.1)", -0.5),
            ("1 234.5", "(F7.1)", 1234.5),  # embedded blank ignored
        ],
    )
    def test_f_values(self, text, fmt, expected):
        (value,) = ri(text, fmt)
        assert value == pytest.approx(expected, rel=1e-15, abs=0)

    def test_e_and_d_descriptors_read_the_same_way(self):
        assert ri("0.1235E+05", "(E10.4)") == [pytest.approx(12350.0)]
        assert ri("1235", "(E4.4)") == [pytest.approx(0.1235)]
        assert ri("0.5D+01", "(D7.1)") == [pytest.approx(5.0)]
        assert ri("2.5", "(G3.1)") == [2.5]

    def test_special_values(self):
        assert ri("Inf", "(F3.0)") == [math.inf]
        assert ri("-inf", "(F4.0)") == [-math.inf]
        assert math.isnan(ri("NaN", "(F3.0)")[0])

    def test_scale_factor_applies_only_without_an_exponent(self):
        assert ri("1.5", "(1PF3.1)") == [pytest.approx(0.15)]
        assert ri("1.5E2", "(1PF5.1)") == [pytest.approx(150.0)]
        assert ri("1.5", "(-1PF3.1)") == [pytest.approx(15.0)]

    @pytest.mark.parametrize("text", ["x", "1.2.3", "E5", ".", "1e"])
    def test_invalid_real(self, text):
        with pytest.raises(ValueError):
            ri(text.ljust(5), "(F5.1)")

    def test_result_is_exactly_rounded(self):
        assert ri("0.1", "(F3.1)") == [0.1]
        assert ri("123456789012345678", "(F18.0)") == [123456789012345678.0]


class TestCharacterAndLogicalInput:
    def test_character(self):
        assert ri("hello", "(A5)") == ["hello"]
        assert ri("hi", "(A5)") == ["hi   "]
        assert ri("hello world", "(A)") == ["hello world"]
        assert ri("abcdef", "(A2,A2)") == ["ab", "cd"]
        assert ri("", "(A3)") == ["   "]

    @pytest.mark.parametrize(
        ("text", "width", "expected"),
        [
            ("T", 1, True),
            (".false.", 7, False),
            (" t", 2, True),
            (".TRUE.", 6, True),
            ("F", 1, False),
        ],
    )
    def test_logical(self, text, width, expected):
        assert ri(text, f"(L{width})") == [expected]

    def test_bad_logical(self):
        with pytest.raises(ValueError):
            ri("x", "(L1)")


class TestPositionAndRecords:
    def test_x_and_tabs(self):
        assert ri("abcdef", "(2X,A2)") == ["cd"]
        assert ri("abcdef", "(T4,A2)") == ["de"]
        assert ri("abcdef", "(A2,TL2,A2)") == ["ab", "ab"]
        assert ri("abcdef", "(A2,TR1,A2)") == ["ab", "de"]

    def test_mixed_fields(self):
        assert ri("  12  3.5ab", "(I4,F5.1,A2)") == [12, 3.5, "ab"]

    def test_slash_reads_the_next_record(self):
        assert ri(["12", "34"], "(I2,/,I2)") == [12, 34]
        assert ri(["12", "99", "34"], "(I2,//,I2)") == [12, 34]

    def test_missing_record_is_end_of_file(self):
        with pytest.raises(EOFError):
            ri("12", "(I2,/,I2)")

    def test_default_count_is_the_number_of_descriptors(self):
        assert ri("123", "(3I1)") == [1, 2, 3]
        assert ri("123456", "(2(I1,1X,I1))") == [1, 3, 4, 6]

    def test_explicit_count_limits_the_values(self):
        assert ri("123", "(3I1)", n=2) == [1, 2]
        assert ri("123", "(3I1)", n=0) == []

    def test_larger_count_reverts_the_format_on_a_new_record(self):
        assert ri(["1", "2", "3"], "(I1)", n=3) == [1, 2, 3]
        assert ri(["12", "34"], "(2I1)", n=4) == [1, 2, 3, 4]
        assert ri(["1 2", "3 4"], "(I1,1X,I1)", n=4) == [1, 2, 3, 4]

    def test_reversion_past_the_last_record_is_end_of_file(self):
        with pytest.raises(EOFError):
            ri(["1"], "(I1)", n=2)

    def test_format_without_descriptors_cannot_read(self):
        with pytest.raises(ValueError):
            ri("abc", "(3X)", n=1)

    def test_empty_internal_file_list_is_end_of_file(self):
        with pytest.raises(EOFError):
            ri([], "(I1)")

    def test_colon_ends_input_early(self):
        assert ri("12", "(I1,:,I1,I1)", n=1) == [1]

    def test_literal_in_input_format_is_skipped(self):
        assert ri("x=12", "('x=',I2)") == [12]


class TestStandardInput:
    def test_unit_5_reads_sys_stdin_at_call_time(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("  12\n  34\n"))
        assert rt.read_formatted(5, "(I4)") == [12]
        assert rt.read_formatted(5, "(I4)") == [34]

    @pytest.mark.parametrize("unit", [None, "*"])
    def test_default_input_designators(self, monkeypatch, unit):
        monkeypatch.setattr("sys.stdin", io.StringIO("3.5\n"))
        assert rt.read_formatted(unit, "(F3.1)") == [3.5]

    def test_end_of_file(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("1\n"))
        rt.read_formatted(5, "(I1)")
        with pytest.raises(EOFError):
            rt.read_formatted(5, "(I1)")

    def test_empty_stdin_is_end_of_file(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        with pytest.raises(EOFError):
            rt.read_formatted(5, "(I1)")

    def test_windows_line_endings_are_stripped(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("ab\r\ncd"))
        assert rt.read_formatted(5, "(A)") == ["ab"]
        assert rt.read_formatted(5, "(A)") == ["cd"]

    def test_blank_line_is_a_record(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("\n7\n"))
        assert rt.read_formatted(5, "(I2)") == [0]
        assert rt.read_formatted(5, "(I2)") == [7]

    def test_read_with_slash_consumes_two_lines(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("1\n2\n3\n"))
        assert rt.read_formatted(5, "(I1,/,I1)") == [1, 2]
        assert rt.read_formatted(5, "(I1)") == [3]

    @pytest.mark.parametrize("unit", [6, 0])
    def test_output_units_cannot_be_read(self, unit):
        with pytest.raises(ValueError):
            rt.read_formatted(unit, "(I1)")
        with pytest.raises(ValueError):
            rt.read_list(unit, 1)


class TestListDirectedInput:
    @staticmethod
    def read(monkeypatch, text, n, types=None):
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
        return rt.read_list(5, n, types)

    def test_blank_separated_integers(self, monkeypatch):
        assert self.read(monkeypatch, "1 2 3\n", 3) == [1, 2, 3]

    def test_comma_and_blank_separators(self, monkeypatch):
        assert self.read(monkeypatch, "1, 2.5 ,abc\n", 3) == [1, 2.5, "abc"]
        assert self.read(monkeypatch, "  4\t5  \n", 2) == [4, 5]

    def test_value_inference(self, monkeypatch):
        values = self.read(monkeypatch, "5 5. 1e3 1.5D2 T .FALSE. word\n", 7)
        assert values == [5, 5.0, 1000.0, 150.0, True, False, "word"]
        assert type(values[0]) is int
        assert type(values[1]) is float

    def test_declared_types_coerce(self, monkeypatch):
        values = self.read(monkeypatch, "1 2\n", 2, float)
        assert values == [1.0, 2.0]
        assert all(type(v) is float for v in values)
        assert self.read(monkeypatch, "7 8.5 T abc\n", 4, [int, float, bool, str]) == [
            7,
            8.5,
            True,
            "abc",
        ]

    def test_declared_int_rejects_reals(self, monkeypatch):
        with pytest.raises(ValueError):
            self.read(monkeypatch, "1.5\n", 1, int)

    def test_declared_string_keeps_tokens_that_look_like_numbers(self, monkeypatch):
        assert self.read(monkeypatch, "12 T\n", 2, str) == ["12", "T"]

    def test_types_length_must_match(self, monkeypatch):
        with pytest.raises(ValueError):
            self.read(monkeypatch, "1 2\n", 2, [int])

    def test_quoted_strings(self, monkeypatch):
        values = self.read(monkeypatch, "'hello world' \"it's\" 'it''s' 'a,b'\n", 4)
        assert values == ["hello world", "it's", "it's", "a,b"]

    def test_repeat_counts(self, monkeypatch):
        assert self.read(monkeypatch, "3*7\n", 3) == [7, 7, 7]
        assert self.read(monkeypatch, "2*1.5 4\n", 3) == [1.5, 1.5, 4]
        assert self.read(monkeypatch, "2*'ab'\n", 2) == ["ab", "ab"]

    def test_null_values(self, monkeypatch):
        assert self.read(monkeypatch, "1,,3\n", 3) == [1, None, 3]
        assert self.read(monkeypatch, "1, ,3\n", 3) == [1, None, 3]
        assert self.read(monkeypatch, ",5\n", 2) == [None, 5]
        assert self.read(monkeypatch, "3*\n", 3) == [None, None, None]

    def test_slash_terminates_and_leaves_the_rest_null(self, monkeypatch):
        assert self.read(monkeypatch, "1 / 3\n", 3) == [1, None, None]

    def test_values_continue_on_following_records(self, monkeypatch):
        assert self.read(monkeypatch, "1 2\n3\n", 3) == [1, 2, 3]

    def test_rest_of_the_record_is_discarded(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("1 2\n3 4\n5\n"))
        assert rt.read_list(5, 1) == [1]
        assert rt.read_list(5, 1) == [3]
        assert rt.read_list(5, 1) == [5]

    def test_zero_items_still_consume_a_record(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("skipped\n9\n"))
        assert rt.read_list(5, 0) == []
        assert rt.read_list(5, 1) == [9]

    def test_blank_records_are_skipped_while_values_are_missing(self, monkeypatch):
        assert self.read(monkeypatch, "\n\n7\n", 1) == [7]

    def test_end_of_file(self, monkeypatch):
        with pytest.raises(EOFError):
            self.read(monkeypatch, "1\n", 2)
        with pytest.raises(EOFError):
            self.read(monkeypatch, "", 1)

    def test_complex(self, monkeypatch):
        assert self.read(monkeypatch, "(1.0,2.0) 3\n", 2) == [1 + 2j, 3]
        assert self.read(monkeypatch, "(1.0, -2.5)\n", 1, complex) == [1 - 2.5j]

    def test_zero_repeat_count_is_an_error(self, monkeypatch):
        with pytest.raises(ValueError):
            self.read(monkeypatch, "0*5\n", 1)

    def test_negative_count_is_an_error(self):
        with pytest.raises(ValueError):
            rt.read_list(5, -1)


class TestListDirectedOutput:
    @staticmethod
    def written(capsys, *values):
        rt.write_list(6, *values)
        return capsys.readouterr().out

    def test_integers_are_right_aligned_after_a_blank(self, capsys):
        assert self.written(capsys, 1) == " " * 11 + "1\n"
        assert self.written(capsys, -42) == " " * 9 + "-42\n"
        assert self.written(capsys, 1, 2) == " " * 11 + "1" + " " * 11 + "2\n"
        assert self.written(capsys, 2147483647) == "  2147483647\n"

    def test_strings_are_written_as_is_after_a_blank(self, capsys):
        assert self.written(capsys, "abc") == " abc\n"
        assert self.written(capsys, "a", "b") == " a b\n"
        assert self.written(capsys, "x=", 5, "ok") == " x=" + " " * 11 + "5 ok\n"

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (3.5, " 3.5"),
            (0.0, " 0.0"),
            (-0.0, " -0.0"),
            (0.1, " 0.1"),
            (100.0, " 100.0"),
            (-2.5, " -2.5"),
            (1234567.0, " 1234567.0"),
            (1e7, " 1.0E+07"),
            (1.5e-5, " 1.5E-05"),
            (0.05, " 5.0E-02"),
            (1e100, " 1.0E+100"),
            (1.2345e-100, " 1.2345E-100"),
            (math.inf, " Infinity"),
            (-math.inf, " -Infinity"),
            (math.nan, " NaN"),
        ],
    )
    def test_reals_use_the_shortest_round_trip_digits(self, capsys, value, text):
        assert self.written(capsys, value) == text + "\n"

    def test_logicals_and_complex(self, capsys):
        assert self.written(capsys, True, False) == " T F\n"
        assert self.written(capsys, 1 + 2j) == " (1.0,2.0)\n"

    def test_numpy_scalars_and_arrays(self, capsys):
        assert self.written(capsys, np.float32(0.1)) == " 0.1\n"
        assert self.written(capsys, np.array([1.5, 2.5])) == " 1.5 2.5\n"
        assert self.written(capsys, np.int64(3)) == " " * 11 + "3\n"
        assert self.written(capsys, np.True_) == " T\n"

    def test_no_values_is_an_empty_record(self, capsys):
        assert self.written(capsys) == "\n"

    def test_default_output_designator(self, capsys):
        rt.write_list(None, "hi")
        assert capsys.readouterr().out == " hi\n"

    def test_unsupported_type(self, capsys):
        with pytest.raises(TypeError):
            rt.write_list(6, None)
        assert capsys.readouterr().out == ""
