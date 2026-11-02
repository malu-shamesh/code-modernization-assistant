"""Formatted output: expected text is derived by hand from the Fortran 77 standard.

There is no Fortran compiler here, so every expectation below was worked out on paper (and the
decimal expansions of the tie cases cross-checked with the decimal module).
"""

from __future__ import annotations

import io
import math
import random
from decimal import Decimal

import numpy as np
import pytest

from modernizer.runtime import f77rt as rt

fr = rt.format_record


class TestIntegerEditing:
    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(I5)", -42, "  -42"),
            ("(I5)", 42, "   42"),
            ("(I3)", 0, "  0"),
            ("(I3)", -99, "-99"),
            ("(I3)", -100, "***"),
            ("(I3)", 12345, "***"),
            ("(I1)", 10, "*"),
            ("(I5.3)", 7, "  007"),
            ("(I5.3)", -7, " -007"),
            ("(I3.3)", -5, "***"),  # '-005' needs 4 columns
            ("(I6.4)", 12345, " 12345"),
            ("(I3.0)", 0, "   "),  # Iw.0 of zero is all blanks
            ("(I2.0)", 5, " 5"),
            ("(I0)", 123, "123"),
            ("(I0)", -5, "-5"),
            ("(I4)", 2147483647, "****"),
            ("(I11)", 2147483647, " 2147483647"),
            ("(SP,I4)", 5, "  +5"),
            ("(SP,I4)", -5, "  -5"),
            ("(SP,I4)", 0, "  +0"),
            ("(SP,SS,I4)", 5, "   5"),
            ("(SP,S,I4)", 5, "   5"),
            ("(i5)", 12, "   12"),
        ],
    )
    def test_values(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    def test_repeat_count(self):
        assert fr("(2I3)", [1, 2]) == "  1  2"
        assert fr("(3I2)", [1, 2, 3]) == " 1 2 3"

    def test_numpy_integers(self):
        assert fr("(I4)", [np.int32(-7)]) == "  -7"

    def test_rejects_float_and_bool(self):
        with pytest.raises(TypeError):
            fr("(I4)", [1.0])
        with pytest.raises(TypeError):
            fr("(I4)", [True])


class TestFixedPoint:
    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(F10.3)", 3.14159, "     3.142"),
            ("(F8.2)", -3.14159, "   -3.14"),
            ("(F6.1)", 0.05, "   0.1"),  # exact binary value is just above 0.05
            ("(F6.1)", 0.04, "   0.0"),
            ("(F5.1)", 9.96, " 10.0"),
            ("(F4.1)", 99.96, "****"),  # rounds to 100.0, five columns
            ("(F6.2)", 0.0, "  0.00"),
            ("(F6.2)", -0.001, " -0.00"),  # the minus sign survives rounding to zero
            ("(F6.2)", -0.0, " -0.00"),
            ("(F7.2)", 123.456, " 123.46"),
            ("(F9.4)", 1.0, "   1.0000"),
            ("(F20.2)", 1e15, " 1000000000000000.00"),
            ("(F9.3)", 1e300, "*********"),
            ("(F5.2)", 1e-320, " 0.00"),
            ("(F10.0)", 1e9, "**********"),  # '1000000000.' has eleven characters
            ("(F11.0)", 1e9, "1000000000."),
            ("(F6.0)", 123.0, "  123."),
            ("(SP,F7.2)", 3.0, "  +3.00"),
            ("(SP,F7.2)", -3.0, "  -3.00"),
            ("(F20.17)", 0.1, " 0.10000000000000001"),  # exact value 0.1000000000000000055511...
        ],
    )
    def test_values(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(F4.3)", 0.5, ".500"),  # the optional leading zero is dropped to fit
            ("(F3.3)", 0.5, "***"),
            ("(F5.3)", -0.5, "-.500"),
            ("(F4.3)", -0.5, "****"),
            ("(F5.3)", 0.0, "0.000"),
            ("(F4.2)", 0.5, "0.50"),
            ("(F0.3)", 0.5, "0.500"),
            ("(F0.1)", -12.34, "-12.3"),
        ],
    )
    def test_leading_zero_and_zero_width(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(F5.0)", 0.5, "   0."),  # exact ties round half to even (gfortran / glibc)
            ("(F3.0)", 0.5, " 0."),
            ("(F5.0)", 1.5, "   2."),
            ("(F5.0)", 2.5, "   2."),
            ("(F5.0)", 3.5, "   4."),
            ("(F5.0)", -0.5, "  -0."),
            ("(F5.0)", -2.5, "  -2."),
            ("(F5.2)", 0.125, " 0.12"),
            ("(F5.2)", -0.125, "-0.12"),
            ("(F5.2)", 0.375, " 0.38"),
            ("(F6.1)", 0.25, "   0.2"),
            ("(F6.1)", 0.75, "   0.8"),
            ("(F6.1)", 0.35, "   0.3"),  # 0.35 is 0.34999999999999997...: not a tie
            ("(F5.2)", 0.145, " 0.14"),  # 0.14499999999999999...: not a tie
            ("(F5.2)", 0.135, " 0.14"),  # 0.13500000000000000888...: above the tie
            ("(F6.2)", 2.675, "  2.67"),  # 2.67499999999999982...
            ("(F6.2)", 1.005, "  1.00"),  # 1.00499999999999989...
        ],
    )
    def test_exact_ties_round_half_to_even_on_the_exact_binary_value(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    def test_tie_cases_really_are_exact_or_not(self):
        # Cross-check the premises of the table above with exact decimal expansions.
        assert Decimal(0.125) == Decimal("0.125")
        assert Decimal(0.375) == Decimal("0.375")
        assert Decimal(0.25) == Decimal("0.25")
        assert Decimal(0.35) < Decimal("0.35")
        assert Decimal(0.145) < Decimal("0.145")
        assert Decimal(0.135) > Decimal("0.135")
        assert Decimal(2.675) < Decimal("2.675")
        assert Decimal(1.005) < Decimal("1.005")
        assert Decimal(0.05) > Decimal("0.05")

    def test_integers_are_accepted_for_real_descriptors(self):
        assert fr("(F6.1)", [3]) == "   3.0"
        assert fr("(E10.2)", [5]) == "  0.50E+01"

    def test_matches_correctly_rounded_c_formatting_off_ties(self):
        rng = random.Random(20260502)
        for _ in range(300):
            x = rng.uniform(-1, 1) * 10 ** rng.randint(-4, 8)
            assert fr("(F20.3)", [x]) == f"{x:20.3f}"
            assert fr("(F24.9)", [x]) == f"{x:24.9f}"


class TestScientific:
    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(E12.4)", 12345.678, "  0.1235E+05"),
            ("(E12.4)", -0.000123456, " -0.1235E-03"),
            ("(E12.4)", 0.0, "  0.0000E+00"),
            ("(E12.4)", -0.0, " -0.0000E+00"),
            ("(E12.4)", 1.0, "  0.1000E+01"),
            ("(E12.4)", 0.5, "  0.5000E+00"),
            ("(E12.4)", 9.9996, "  0.1000E+02"),  # rounding carries into the exponent
            ("(E10.2)", 9.96, "  0.10E+02"),
            ("(E9.4)", 12345.678, ".1235E+05"),  # leading zero dropped to fit
            ("(E8.4)", 12345.678, "********"),
            ("(E10.4)", -12345.678, "-.1235E+05"),
            ("(SP,E12.4)", 1.0, " +0.1000E+01"),
            ("(E25.17)", 0.1, "  0.10000000000000001E+00"),
        ],
    )
    def test_two_digit_exponents(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(E10.3)", 1e100, " 0.100+101"),  # three-digit exponent drops the letter
            ("(E10.3)", 1e-100, " 0.100E-99"),  # 1e-100 = 0.1E-99
            ("(E10.3)", 1e-101, " 0.100-100"),
            ("(E10.3)", 2.5e300, " 0.250+301"),
            ("(E10.3)", 1.5e-5, " 0.150E-04"),
            ("(E10.3)", 1e99, " 0.100+100"),
            ("(E10.3)", 1e98, " 0.100E+99"),
        ],
    )
    def test_exponent_width_changes_with_magnitude(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(E12.4E1)", 1.0, "   0.1000E+1"),  # one-digit exponent field
            ("(E12.4E1)", 1e10, "************"),  # exponent 11 does not fit in one digit
            ("(E12.4E1)", 1e-9, "   0.1000E-8"),
            ("(E14.4E3)", 1.0, "   0.1000E+001"),
            ("(E14.4E3)", 1e100, "   0.1000E+101"),
            ("(E12.4E2)", 1.0, "  0.1000E+01"),
        ],
    )
    def test_explicit_exponent_digits(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(D12.4)", 1234.567, "  0.1235D+04"),
            ("(D10.3)", 1e150, " 0.100+151"),
            ("(D10.3)", -1e-5, "-0.100D-04"),
        ],
    )
    def test_d_edit_descriptor(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(E10.2)", 0.125, "  0.12E+00"),  # exact ties round half to even like F
            ("(E10.2)", 0.375, "  0.38E+00"),
            ("(E10.1)", 0.25, "   0.2E+00"),
            ("(E10.1)", 0.75, "   0.8E+00"),
            ("(E10.1)", -0.25, "  -0.2E+00"),
            ("(D10.2)", 0.125, "  0.12D+00"),
            ("(D10.2)", 0.375, "  0.38D+00"),
            ("(E10.2)", 1250.0, "  0.12E+04"),
            ("(E10.2)", 1350.0, "  0.14E+04"),
        ],
    )
    def test_exact_ties_round_half_to_even_in_e_and_d(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    def test_matches_python_scientific_formatting_off_ties(self):
        rng = random.Random(7)
        for _ in range(300):
            x = rng.uniform(0.1, 0.999) * 10 ** rng.randint(-200, 200) * rng.choice([-1, 1])
            mantissa, exp = f"{abs(x):.5e}".split("e")
            digits = mantissa.replace(".", "")
            expo = int(exp) + 1
            sign = "-" if x < 0 else ""
            tail = (
                f"E{'-' if expo < 0 else '+'}{abs(expo):02d}"
                if abs(expo) <= 99
                else (f"{'-' if expo < 0 else '+'}{abs(expo):03d}")
            )
            assert fr("(E20.6)", [x]).strip() == f"{sign}0.{digits}{tail}"

    def test_d_must_be_positive(self):
        with pytest.raises(ValueError):
            fr("(E10.0)", [1.0])


class TestScaleFactor:
    def test_f_scale_multiplies(self):
        assert fr("(2PF8.3)", [1.5]) == " 150.000"
        assert fr("(-1PF8.3)", [1.5]) == "   0.150"

    def test_e_scale_positive(self):
        # 1P: one digit before the point, d-k+1 = 4 after, exponent reduced by one.
        assert fr("(1PE12.4)", [12345.678]) == "  1.2346E+04"

    def test_e_scale_negative(self):
        # -1P: one leading zero after the point, d+k = 3 significant digits.
        assert fr("(-1PE12.4)", [12345.678]) == "  0.0123E+06"

    def test_scale_persists_until_changed(self):
        assert fr("(1P,2E12.4,0P,E12.4)", [1.0, 2.0, 3.0]) == (
            "  1.0000E+00  2.0000E+00  0.3000E+01"
        )

    def test_scale_out_of_range(self):
        with pytest.raises(ValueError):
            fr("(9PE12.4)", [1.0])


class TestGeneralEditing:
    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(G12.5)", 123.456, "  123.46    "),  # F8.2 followed by four blanks
            ("(G12.5)", 0.0, "  0.0000    "),
            ("(G14.5)", 1e10, "   0.10000E+11"),
            ("(G12.5)", 0.05, " 0.50000E-01"),  # below 0.1 uses the E form
            ("(G10.3)", 0.5, " 0.500    "),
            ("(G10.3)", 123.0, "  123.    "),  # exactly d digits before the point: F6.0
            ("(G10.3)", 999.9, " 0.100E+04"),  # rounds to 1000, which needs d+1 digits
            ("(G12.5E3)", 123.456, " 123.46     "),  # E3: five trailing blanks
            ("(G10.3)", 1.5, "  1.50    "),
        ],
    )
    def test_real_values(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    def test_other_types(self):
        assert fr("(G5.1)", [42]) == "   42"
        assert fr("(G3.1)", [True]) == "  T"
        assert fr("(G4.1)", ["ab"]) == "  ab"

    def test_requires_digits(self):
        with pytest.raises(ValueError):
            fr("(G10.0)", [1.0])


class TestLogicalAndCharacter:
    def test_logical(self):
        assert fr("(L3)", [True]) == "  T"
        assert fr("(L1)", [False]) == "F"
        assert fr("(L5)", [np.True_]) == "    T"

    def test_logical_rejects_integers(self):
        with pytest.raises(TypeError):
            fr("(L1)", [1])

    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(A5)", "abc", "  abc"),
            ("(A2)", "abc", "ab"),  # output keeps the leftmost w characters
            ("(A)", "abc", "abc"),
            ("(A3)", "abc", "abc"),
            ("(A)", "", ""),
            ("(A4)", "", "    "),
        ],
    )
    def test_character(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected

    def test_character_rejects_numbers(self):
        with pytest.raises(TypeError):
            fr("(A)", [5])

    def test_trailing_blanks_of_the_value_are_part_of_it(self):
        assert fr("(A,'|')", ["ab  "]) == "ab  |"


class TestPositionAndLiterals:
    def test_x_skips_columns(self):
        assert fr("(3X,I2)", [5]) == "    5"
        assert fr("(1X,A,1X,I3)", ["ab", 5]) == " ab   5"
        assert fr("(X,I2)", [5]) == "  5"

    def test_trailing_x_writes_nothing(self):
        assert fr("(I2,3X)", [5]) == " 5"

    def test_tab_positions(self):
        assert fr("(T5,I2)", [3]) == "     3"
        assert fr("(I2,TR3,I2)", [1, 2]) == " 1    2"
        assert fr("(A,T1,A)", ["abcde", "XY"]) == "XYcde"
        assert fr("(I4,TL2,I2)", [1234, 56]) == "1256"
        assert fr("(TL5,I2)", [7]) == " 7"  # TL cannot go before column 1
        assert fr("('x',T5,'y')", []) == "x   y"

    def test_quoted_literals(self):
        assert fr("('Total: ',I3)", [5]) == "Total:   5"
        assert fr("('it''s',1X,A)", ["ok"]) == "it's ok"
        assert fr('("say ""hi""")', []) == 'say "hi"'
        assert fr('("a\'b")', []) == "a'b"

    def test_hollerith(self):
        assert fr("(5HHello,1X,I2)", [7]) == "Hello  7"
        assert fr("(3H a ,I1)", [2]) == " a 2"

    def test_slash_starts_new_records(self):
        assert fr("(I2,/,I2)", [1, 2]) == " 1\n 2"
        assert fr("(I2,//,I2)", [1, 2]) == " 1\n\n 2"
        assert fr("(2/,I2)", [1]) == "\n\n 1"
        assert fr("(I2,/)", [1]) == " 1\n"
        assert fr("(/I2)", [1]) == "\n 1"

    def test_colon_stops_when_the_list_is_exhausted(self):
        assert fr("(I2,:,' end')", [1]) == " 1"
        assert fr("(I2,:,' end',I2)", [1, 2]) == " 1 end 2"
        assert fr("(I2,' end')", [1]) == " 1 end"

    def test_trailing_literals_stop_at_the_next_data_descriptor(self):
        assert fr("(I3,' end',I3)", [1]) == "  1 end"

    def test_empty_format_is_one_blank_record(self):
        assert fr("()", []) == ""
        assert fr("(1X)", []) == ""

    def test_fewer_values_than_descriptors(self):
        assert fr("(3I3)", [1]) == "  1"
        assert fr("(I3,A,F5.1)", [1, "xy"]) == "  1xy"

    def test_dollar_suppresses_the_final_newline(self, capsys):
        rt.write_formatted(6, "(A,$)", "Enter: ")
        assert capsys.readouterr().out == "Enter: "


class TestGroupsAndReversion:
    def test_nested_group_with_repeat(self):
        assert fr("(2(I2,F5.1))", [1, 2.0, 3, 4.0]) == " 1  2.0 3  4.0"
        assert fr("(2(I2,1X))", [1, 2]) == " 1  2"
        assert fr("(2(1X,2(I1,','),A))", [1, 2, "a", 3, 4, "b"]) == " 1,2,a 3,4,b"

    def test_single_item_list_reuses_the_format(self):
        assert fr("(I3)", [1, 2, 3]) == "  1\n  2\n  3"
        assert fr("(2I3)", [1, 2, 3, 4, 5]) == "  1  2\n  3  4\n  5"

    def test_reversion_without_group_restarts_from_the_beginning(self):
        assert fr("(A,I2)", ["a", 1, "b", 2]) == "a 1\nb 2"

    def test_reversion_to_the_last_top_level_group(self):
        # Standard 13.5.2: restart at the group ending at the last level-1 parenthesis,
        # including its repeat count; the I3 in front is only used once.
        assert fr("(I3,2(I2))", [1, 2, 3, 4, 5]) == "  1 2 3\n 4 5"

    def test_reversion_uses_only_the_last_group_not_later_items(self):
        assert fr("(2(I2),F5.1)", [1, 2, 3.0, 4, 5]) == " 1 2  3.0\n 4 5"

    def test_reversion_with_group_and_trailing_literal(self):
        assert fr("('x=',(I2,','))", [1, 2, 3]) == "x= 1,\n 2,\n 3,"

    def test_literal_only_group_cannot_consume_values(self):
        with pytest.raises(ValueError):
            fr("(I3,('x'))", [1, 2])

    def test_values_without_any_data_descriptor(self):
        with pytest.raises(ValueError):
            fr("('x')", [1])

    def test_array_values_expand_in_column_major_order(self):
        a = np.array([[1, 2], [3, 4]])
        assert fr("(4I2)", [a]) == " 1 3 2 4"

    def test_lists_and_mixed_arguments(self):
        assert fr("(3I2,A)", [[1, 2, 3], "z"]) == " 1 2 3z"
        assert fr("(A,A)", [["ab", "cd"]]) == "abcd"

    def test_complex_consumes_two_real_descriptors(self):
        assert fr("(2F6.1)", [1 + 2j]) == "   1.0   2.0"
        assert fr("(2F6.1,I2)", [3 - 1j, 7]) == "   3.0  -1.0 7"


class TestSpecialValues:
    @pytest.mark.parametrize(
        ("fmt", "value", "expected"),
        [
            ("(F10.3)", math.inf, "  Infinity"),
            ("(F10.3)", -math.inf, " -Infinity"),
            ("(F5.1)", math.inf, "  Inf"),
            ("(F4.1)", -math.inf, "-Inf"),
            ("(F3.1)", -math.inf, "***"),
            ("(F5.1)", math.nan, "  NaN"),
            ("(F2.1)", math.nan, "**"),
            ("(SP,F9.1)", math.inf, "+Infinity"),
            ("(SP,F8.1)", math.inf, "    +Inf"),
            ("(E12.4)", math.nan, "         NaN"),
            ("(G12.4)", -math.inf, "   -Infinity"),
        ],
    )
    def test_infinity_and_nan(self, fmt, value, expected):
        assert fr(fmt, [value]) == expected


class TestFormatSyntax:
    def test_case_blanks_and_missing_outer_parentheses(self):
        assert fr(" ( i5 , f8.2 ) ", [1, 2.0]) == "    1    2.00"
        assert fr("I5,F8.2", [1, 2.0]) == "    1    2.00"
        assert fr("(I5 F8.2)", [1, 2.0]) == "    1    2.00"

    def test_commas_are_optional_around_slash_and_p(self):
        assert fr("(I2/I2)", [1, 2]) == " 1\n 2"
        assert fr("(1PE12.4)", [1.0]) == "  1.0000E+00"

    @pytest.mark.parametrize(
        "fmt",
        ["(I5", "I5)", "(Q5)", "(F8)", "(F8.)", "(I)", "('abc)", "(2T5)", "(0I3)", "(ES12.4)",
         "(5Hab)", "(+3I2)", "(I5)(I5)", "(T0)"],
    )  # fmt: skip
    def test_malformed_formats(self, fmt):
        with pytest.raises(ValueError):
            fr(fmt, [1])

    def test_format_must_be_a_string(self):
        with pytest.raises(TypeError):
            fr(["I5"], [1])

    def test_formats_are_cached_but_independent(self):
        assert fr("(I3)", [1]) == "  1"
        assert fr("(I3)", [2]) == "  2"
        assert fr("(I3)", [1, 2]) == "  1\n  2"


class TestWriteFormattedStreams:
    def test_unit_6_goes_to_stdout_at_call_time(self, capsys):
        rt.write_formatted(6, "(I3)", 5)
        assert capsys.readouterr().out == "  5\n"

    @pytest.mark.parametrize("unit", [None, "*", " * "])
    def test_default_output_designators(self, capsys, unit):
        rt.write_formatted(unit, "(A)", "hi")
        assert capsys.readouterr().out == "hi\n"

    def test_unit_0_goes_to_stderr(self, capsys):
        rt.write_formatted(0, "(A)", "oops")
        captured = capsys.readouterr()
        assert captured.err == "oops\n"
        assert captured.out == ""

    def test_multiple_records(self, capsys):
        rt.write_formatted(6, "(I3)", 1, 2)
        assert capsys.readouterr().out == "  1\n  2\n"

    def test_trailing_slash_gives_a_blank_line(self, capsys):
        rt.write_formatted(6, "(I2,/)", 1)
        assert capsys.readouterr().out == " 1\n\n"

    def test_replaced_sys_stdout_is_honoured(self, monkeypatch):
        buffer = io.StringIO()
        monkeypatch.setattr("sys.stdout", buffer)
        rt.write_formatted(6, "(A)", "x")
        assert buffer.getvalue() == "x\n"

    def test_a_bad_format_writes_nothing(self, capsys):
        with pytest.raises(ValueError):
            rt.write_formatted(6, "(I3,F)", 1, 2.0)
        assert capsys.readouterr().out == ""

    def test_a_bad_value_writes_nothing(self, capsys):
        with pytest.raises(TypeError):
            rt.write_formatted(6, "(I3,I3)", 1, "x")
        assert capsys.readouterr().out == ""

    def test_unit_5_cannot_be_written(self):
        with pytest.raises(ValueError):
            rt.write_formatted(5, "(I3)", 1)

    def test_no_values_writes_the_literals(self, capsys):
        rt.write_formatted(6, "('hello')")
        assert capsys.readouterr().out == "hello\n"


class TestInternalWrite:
    def test_write_internal(self):
        assert rt.write_internal("(I3,F6.2)", 5, 3.14159) == "  5  3.14"
        assert rt.write_internal("(A)", "x") == "x"
        assert rt.write_internal("(I5.5)", 42) == "00042"

    def test_internal_file_holds_one_record(self):
        with pytest.raises(ValueError):
            rt.write_internal("(I2,/,I2)", 1, 2)
