"""Numeric intrinsics, bit operations, machine constants, complex and array helpers."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from modernizer.runtime import f77rt as rt


class TestIdiv:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(7, 2, 3), (-7, 2, -3), (7, -2, -3), (-7, -2, 3), (0, 5, 0), (6, 3, 2), (-1, 2, 0),
         (1, -2, 0), (-6, 3, -2), (2**40 + 1, 2, 2**39)],
    )  # fmt: skip
    def test_truncates_toward_zero(self, a, b, expected):
        assert rt.idiv(a, b) == expected

    def test_zero_divisor(self):
        with pytest.raises(ZeroDivisionError):
            rt.idiv(5, 0)

    def test_rejects_floats(self):
        with pytest.raises(TypeError):
            rt.idiv(5.0, 2)

    def test_numpy_integers_return_python_int(self):
        result = rt.idiv(np.int64(-7), np.int32(2))
        assert result == -3
        assert type(result) is int


class TestFmod:
    @pytest.mark.parametrize(
        ("a", "p", "expected"),
        [(7, 3, 1), (-7, 3, -1), (7, -3, 1), (-7, -3, -1), (6, 3, 0), (-6, 3, 0), (2, 5, 2)],
    )
    def test_integers_take_sign_of_dividend(self, a, p, expected):
        result = rt.fmod(a, p)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        ("a", "p", "expected"),
        [(5.5, 2.0, 1.5), (-5.5, 2.0, -1.5), (5.5, -2.0, 1.5), (-5.5, -2.0, -1.5),
         (0.0, 3.0, 0.0), (7, 2.5, 2.0), (1.0, 4.0, 1.0)],
    )  # fmt: skip
    def test_floats_follow_math_fmod(self, a, p, expected):
        result = rt.fmod(a, p)
        assert result == expected
        assert isinstance(result, float)

    @pytest.mark.parametrize("args", [(5, 0), (5.0, 0.0), (5, 0.0), (0, 0)])
    def test_zero_modulus(self, args):
        with pytest.raises(ZeroDivisionError):
            rt.fmod(*args)

    def test_infinite_dividend_is_nan(self):
        assert math.isnan(rt.fmod(math.inf, 2.0))

    def test_infinite_modulus_returns_dividend(self):
        assert rt.fmod(5.0, math.inf) == 5.0


class TestSign:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(3, -1, -3), (-3, 1, 3), (-3, 0, 3), (3, 0, 3), (3, 5, 3), (-3, -5, -3), (0, -1, 0)],
    )
    def test_integers(self, a, b, expected):
        assert rt.sign(a, b) == expected
        assert isinstance(rt.sign(a, b), int)

    def test_floats(self):
        assert rt.sign(2.5, -1.0) == -2.5
        assert rt.sign(-2.5, 1.0) == 2.5
        assert rt.sign(2.5, 0.0) == 2.5

    def test_negative_zero_counts_as_negative(self):
        assert rt.sign(2.5, -0.0) == -2.5

    def test_zero_magnitude_keeps_sign_bit(self):
        assert math.copysign(1.0, rt.sign(0.0, -1.0)) == -1.0
        assert math.copysign(1.0, rt.sign(0.0, 1.0)) == 1.0


class TestRounding:
    @pytest.mark.parametrize(
        ("x", "expected"),
        [(2.5, 3), (-2.5, -3), (0.5, 1), (-0.5, -1), (2.4, 2), (-2.4, -2), (2.6, 3), (-2.6, -3),
         (0.0, 0), (-0.4, 0), (1e15 + 0.5, 1000000000000001), (0.49999999999999994, 0), (7, 7)],
    )  # fmt: skip
    def test_nint_half_away_from_zero(self, x, expected):
        assert rt.nint(x) == expected
        assert isinstance(rt.nint(x), int)

    def test_nint_of_infinity_is_an_error(self):
        with pytest.raises(OverflowError):
            rt.nint(math.inf)

    @pytest.mark.parametrize(
        ("x", "expected"),
        [(2.5, 3.0), (-2.5, -3.0), (2.4, 2.0), (-7.0, -7.0), (0.5, 1.0), (1e300, 1e300)],
    )
    def test_anint(self, x, expected):
        assert rt.anint(x) == expected
        assert isinstance(rt.anint(x), float)

    def test_anint_keeps_the_sign_of_zero(self):
        assert math.copysign(1.0, rt.anint(-0.3)) == -1.0

    def test_anint_passes_through_non_finite(self):
        assert rt.anint(math.inf) == math.inf
        assert math.isnan(rt.anint(math.nan))

    @pytest.mark.parametrize(
        ("x", "expected"), [(2.9, 2.0), (-2.9, -2.0), (3.0, 3.0), (0.99, 0.0), (-0.99, -0.0)]
    )
    def test_aint(self, x, expected):
        assert rt.aint(x) == expected
        assert isinstance(rt.aint(x), float)

    def test_aint_keeps_the_sign_of_zero(self):
        assert math.copysign(1.0, rt.aint(-0.5)) == -1.0

    @pytest.mark.parametrize(
        ("x", "expected"),
        [(2.9, 2), (-2.9, -2), (5, 5), (-0.0, 0), (3 + 4j, 3), (-3.7 + 1j, -3), (1e10, 10**10)],
    )
    def test_int_truncates(self, x, expected):
        assert rt.int_(x) == expected
        assert isinstance(rt.int_(x), int)


class TestDim:
    @pytest.mark.parametrize(("a", "b", "expected"), [(5, 3, 2), (3, 5, 0), (4, 4, 0), (-2, -5, 3)])
    def test_integers(self, a, b, expected):
        assert rt.dim(a, b) == expected
        assert isinstance(rt.dim(a, b), int)

    def test_floats(self):
        assert rt.dim(5.5, 2.0) == 3.5
        result = rt.dim(2.0, 5.5)
        assert result == 0.0
        assert isinstance(result, float)

    def test_mixed_is_float(self):
        assert isinstance(rt.dim(5, 2.5), float)
        assert rt.dim(5, 2.5) == 2.5


class TestIpow:
    @pytest.mark.parametrize(
        ("base", "exp", "expected"),
        [(2, 10, 1024), (-2, 3, -8), (-2, 4, 16), (5, 0, 1), (0, 0, 1), (0, 5, 0),
         (2, -1, 0), (2, -3, 0), (-2, -1, 0), (7, -2, 0), (1, -5, 1), (-1, -3, -1), (-1, -4, 1),
         (1, -1, 1)],
    )  # fmt: skip
    def test_values(self, base, exp, expected):
        assert rt.ipow(base, exp) == expected

    def test_zero_to_negative_power(self):
        with pytest.raises(ZeroDivisionError):
            rt.ipow(0, -1)

    def test_rejects_float_operands(self):
        with pytest.raises(TypeError):
            rt.ipow(2.0, 2)
        with pytest.raises(TypeError):
            rt.ipow(2, 2.0)


class TestBitOperations:
    def test_and_or_xor(self):
        assert rt.iand(12, 10) == 8
        assert rt.ior(12, 10) == 14
        assert rt.ieor(12, 10) == 6

    def test_negative_operands_use_twos_complement(self):
        assert rt.iand(-1, 255) == 255
        assert rt.ior(-256, 255) == -1
        assert rt.ieor(-1, 1) == -2

    def test_results_wrap_to_signed_32_bits(self):
        assert rt.ior(0x7FFFFFFF, -(2**31)) == -1
        assert rt.ior(0x40000000, 0x40000000) == 0x40000000
        assert rt.iand(0xFFFFFFFF, 0xFFFFFFFF) == -1

    def test_not(self):
        assert rt.not_(0) == -1
        assert rt.not_(-1) == 0
        assert rt.not_(5) == -6
        assert rt.not_(-(2**31)) == 2**31 - 1

    @pytest.mark.parametrize(
        ("value", "shift", "expected"),
        [(1, 3, 8), (8, -3, 1), (-1, -1, 2147483647), (1, 31, -2147483648), (1, 32, 0),
         (-8, -1, 2147483644), (5, 0, 5), (1, -32, 0), (-1, 1, -2), (3, -1, 1),
         (0x40000000, 1, -(2**31))],
    )  # fmt: skip
    def test_ishft(self, value, shift, expected):
        assert rt.ishft(value, shift) == expected

    def test_bit_queries(self):
        assert rt.btest(5, 0) is True
        assert rt.btest(5, 1) is False
        assert rt.btest(-1, 31) is True
        assert rt.ibset(0, 31) == -(2**31)
        assert rt.ibset(0, 3) == 8
        assert rt.ibclr(7, 1) == 5
        assert rt.ibits(0b110100, 2, 3) == 5

    def test_bit_position_is_validated(self):
        with pytest.raises(ValueError):
            rt.btest(1, 32)
        with pytest.raises(ValueError):
            rt.ibits(1, 30, 5)

    def test_rejects_non_integers(self):
        with pytest.raises(TypeError):
            rt.iand(1.0, 1)


class TestMachineConstants:
    def test_d1mach(self):
        assert rt.d1mach(1) == 2.2250738585072014e-308
        assert rt.d1mach(2) == 1.7976931348623157e308
        assert rt.d1mach(3) == 2.0**-53
        assert rt.d1mach(4) == 2.0**-52
        assert rt.d1mach(5) == pytest.approx(0.301029995663981195, rel=1e-15)

    def test_r1mach(self):
        assert rt.r1mach(1) == 2.0**-126
        assert rt.r1mach(2) == 3.4028234663852886e38
        assert rt.r1mach(3) == 2.0**-24
        assert rt.r1mach(4) == 2.0**-23
        single = struct.unpack("f", struct.pack("f", math.log10(2.0)))[0]
        assert rt.r1mach(5) == single

    @pytest.mark.parametrize(
        ("i", "expected"),
        [(1, 5), (2, 6), (3, 7), (4, 0), (5, 32), (6, 4), (7, 2), (8, 31), (9, 2147483647),
         (10, 2), (11, 24), (12, -125), (13, 128), (14, 53), (15, -1021), (16, 1024)],
    )  # fmt: skip
    def test_i1mach(self, i, expected):
        assert rt.i1mach(i) == expected

    @pytest.mark.parametrize(
        ("func", "bad"),
        [(rt.d1mach, 0), (rt.d1mach, 6), (rt.r1mach, 0), (rt.r1mach, 6), (rt.i1mach, 0),
         (rt.i1mach, 17), (rt.i1mach, -1)],
    )  # fmt: skip
    def test_out_of_range_index(self, func, bad):
        with pytest.raises(ValueError):
            func(bad)


class TestComplexAndArrays:
    def test_cmplx(self):
        assert rt.cmplx(1, 2) == 1 + 2j
        assert rt.cmplx(3) == 3 + 0j
        assert rt.cmplx(1.5, -0.5) == complex(1.5, -0.5)
        assert rt.cmplx(1 + 2j) == 1 + 2j
        with pytest.raises(TypeError):
            rt.cmplx(1 + 2j, 1.0)

    def test_zeros_f_shape_dtype_order(self):
        a = rt.zeros_f(2, 3)
        assert a.shape == (2, 3)
        assert a.dtype == np.float64
        assert a.flags.f_contiguous
        assert not a.any()

    def test_zeros_f_accepts_dtype_and_tuple(self):
        assert rt.zeros_f(4, dtype=int).dtype.kind == "i"
        assert rt.zeros_f((2, 2), dtype=complex).dtype.kind == "c"
        assert rt.zeros_f(2, 2, 2).shape == (2, 2, 2)

    def test_zeros_f_empty_and_negative_extents(self):
        assert rt.zeros_f(0).shape == (0,)
        assert rt.zeros_f(-3, 2).shape == (0, 2)

    def test_zeros_f_rejects_non_integer_extent(self):
        with pytest.raises(TypeError):
            rt.zeros_f(2.5)


class TestFortranStop:
    def test_is_an_exception_with_code(self):
        assert issubclass(rt.FortranStop, Exception)
        assert rt.FortranStop().code is None
        assert rt.FortranStop(3).code == 3
        assert rt.FortranStop(np.int64(4)).code == 4

    def test_numeric_string_code(self):
        assert rt.FortranStop("7").code == 7

    def test_text_message(self):
        stop = rt.FortranStop("all done")
        assert stop.code is None
        assert stop.message == "all done"
        assert "all done" in str(stop)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(rt.FortranStop) as info:
            raise rt.FortranStop(2)
        assert info.value.code == 2
