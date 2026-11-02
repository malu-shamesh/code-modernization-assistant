"""Fortran 77 runtime helpers that converted code imports as ``from f77rt import ...``.

This is ONE standalone file (standard library and numpy only, no imports from ``modernizer``) so it
can be copied next to converted modules and into the test sandbox. Behaviour follows Fortran 77
and gfortran:

* numeric intrinsics: ``idiv fmod sign nint anint aint int_ dim ipow``
* characters (1-based, blank padded): ``ichar char_ index_ len_trim lge lgt lle llt str_eq
  str_cmp char_assign substr set_substr``
* DO loops: ``do_trips do_range do_final``
* formatted I/O: ``format_record write_formatted write_list read_formatted read_list
  write_internal read_internal`` and the unit table ``open_unit close_unit rewind_unit
  backspace_unit endfile_unit close_all_units``
* bit operations (32-bit two's complement): ``iand ior ieor ishft not_ btest ibset ibclr ibits``
* machine constants ``d1mach r1mach i1mach``, ``cmplx``, ``zeros_f`` and ``FortranStop``.

Output decimal conversion is done on the exact binary value of a float with Decimal. An exact
decimal tie (for example F5.2 of 0.125 or F5.0 of 2.5) rounds half to EVEN, which is what gfortran
produces because it delegates to glibc printf: 0.125 -> 0.12, 2.5 -> 2., 0.5 -> 0. . Other
compilers (or a compatible-rounding option) may round ties away from zero instead. NINT and ANINT
are unrelated intrinsics and keep half-away-from-zero semantics.

DO loops: the trip count is computed once from the initial parameters. An INTEGER loop variable
is exact (start, limit and step all ints). A REAL loop variable (any float among start, limit,
step) is produced by repeated addition (x += step), as every compiler does, so accumulated
rounding error is reproduced. The helper never guesses types: for an INTEGER loop variable with a
REAL limit or step, the caller (the converter) must convert the limit and step to int first.
"""

from __future__ import annotations

import atexit
import decimal
import functools
import math
import numbers
import operator
import os
import re
import struct
import sys
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np

__all__ = [
    "FortranStop",
    "aint",
    "anint",
    "backspace_unit",
    "btest",
    "char_",
    "char_assign",
    "close_all_units",
    "close_unit",
    "cmplx",
    "d1mach",
    "dim",
    "do_final",
    "do_range",
    "do_trips",
    "endfile_unit",
    "fmod",
    "format_record",
    "i1mach",
    "iand",
    "ibclr",
    "ibits",
    "ibset",
    "ichar",
    "idiv",
    "ieor",
    "index_",
    "int_",
    "ior",
    "ipow",
    "ishft",
    "len_trim",
    "lge",
    "lgt",
    "lle",
    "llt",
    "nint",
    "not_",
    "open_unit",
    "r1mach",
    "read_formatted",
    "read_internal",
    "read_list",
    "rewind_unit",
    "set_substr",
    "sign",
    "str_cmp",
    "str_eq",
    "substr",
    "write_formatted",
    "write_internal",
    "write_list",
    "zeros_f",
]


# --------------------------------------------------------------------------------------------
# Small type helpers
# --------------------------------------------------------------------------------------------


def _is_int(x: object) -> bool:
    return isinstance(x, numbers.Integral)


def _int(x: Any, what: str) -> int:
    if isinstance(x, numbers.Integral):
        return int(x)
    raise TypeError(f"{what} must be an integer, got {type(x).__name__}")


def _is_negative(x: Any) -> bool:
    """Sign bit test: -0.0 counts as negative (like gfortran SIGN), integer 0 does not."""
    if _is_int(x):
        return int(x) < 0
    return math.copysign(1.0, float(x)) < 0.0


class FortranStop(Exception):
    """Raised for STOP; ``code`` is the exit status (None when STOP carries no number)."""

    def __init__(self, code: int | str | None = None) -> None:
        self.message: str | None = None
        self.code: int | None
        if isinstance(code, str):
            text = code.strip()
            if text.isascii() and text.isdigit():
                self.code = int(text)
            else:
                self.code = None
                self.message = code
        elif code is None:
            self.code = None
        else:
            self.code = operator.index(code)
        super().__init__(self.message if self.message is not None else self.code)


# --------------------------------------------------------------------------------------------
# Numeric intrinsics
# --------------------------------------------------------------------------------------------


def idiv(a: int, b: int) -> int:
    """Integer division truncating toward zero (Fortran I/J); ZeroDivisionError when b is 0."""
    ia = _int(a, "idiv dividend")
    ib = _int(b, "idiv divisor")
    if ib == 0:
        raise ZeroDivisionError("integer division by zero")
    q = abs(ia) // abs(ib)
    return -q if (ia < 0) != (ib < 0) else q


def fmod(a: int | float, p: int | float) -> int | float:
    """MOD/AMOD/DMOD: a - int(a/p)*p, the result has the sign of a; p == 0 raises."""
    if _is_int(a) and _is_int(p):
        ia, ip = _int(a, "MOD argument"), _int(p, "MOD argument")
        if ip == 0:
            raise ZeroDivisionError("MOD with a zero divisor")
        return ia - idiv(ia, ip) * ip
    x, y = float(a), float(p)
    if y == 0.0:
        raise ZeroDivisionError("MOD with a zero divisor")
    if math.isnan(x) or math.isnan(y) or math.isinf(x):
        return math.nan
    return math.fmod(x, y)


def sign(a: int | float, b: int | float) -> int | float:
    """SIGN(a, b): |a| with the sign of b (b == 0 is positive, -0.0 is negative)."""
    negative = _is_negative(b)
    if _is_int(a):
        magnitude_i = abs(_int(a, "SIGN argument"))
        return -magnitude_i if negative else magnitude_i
    magnitude = abs(float(a))
    return -magnitude if negative else magnitude


def nint(x: float) -> int:
    """NINT: nearest integer, halves rounded away from zero."""
    if _is_int(x):
        return _int(x, "NINT argument")
    v = float(x)
    a = abs(v)
    r = math.floor(a)
    if a - r >= 0.5:
        r += 1
    return -r if v < 0 else r


def anint(x: float) -> float:
    """ANINT: nearest whole number as a float, halves rounded away from zero."""
    v = float(x)
    if not math.isfinite(v) or abs(v) >= 2.0**52:
        return v
    return math.copysign(float(nint(v)), v)


def aint(x: float) -> float:
    """AINT: truncate toward zero, result stays a float."""
    v = float(x)
    if not math.isfinite(v) or abs(v) >= 2.0**52:
        return v
    return math.copysign(float(math.trunc(v)), v)


def int_(x: float) -> int:
    """INT/IFIX/IDINT: truncate toward zero to an integer (complex uses the real part)."""
    if isinstance(x, complex):
        x = x.real
    if _is_int(x):
        return _int(x, "INT argument")
    return math.trunc(float(x))


def dim(a: int | float, b: int | float) -> int | float:
    """DIM: positive difference, max(a - b, 0)."""
    if _is_int(a) and _is_int(b):
        d = _int(a, "DIM argument") - _int(b, "DIM argument")
        return d if d > 0 else 0
    f = float(a) - float(b)
    return f if f > 0.0 else 0.0


def ipow(base: int, exp: int) -> int:
    """Integer power; a negative exponent truncates 1/base**|exp| toward zero (0 unless base is +-1)."""
    b = _int(base, "ipow base")
    e = _int(exp, "ipow exponent")
    if e >= 0:
        return b**e
    if b == 0:
        raise ZeroDivisionError("zero raised to a negative integer power")
    if b == 1:
        return 1
    if b == -1:
        return -1 if e % 2 else 1
    return 0


# --------------------------------------------------------------------------------------------
# Character helpers (1-based, blank padded)
# --------------------------------------------------------------------------------------------


def ichar(c: str) -> int:
    """ICHAR: code of the first character (an empty string is an error)."""
    if not isinstance(c, str):
        raise TypeError(f"ichar needs a str, got {type(c).__name__}")
    if not c:
        raise ValueError("ichar of an empty string")
    return ord(c[0])


def char_(i: int) -> str:
    """CHAR: the character with code i (0..255)."""
    code = _int(i, "char_ argument")
    if not 0 <= code <= 255:
        raise ValueError(f"CHAR argument {code} is outside 0..255")
    return chr(code)


def index_(s: str, sub: str, back: bool = False) -> int:
    """INDEX: 1-based position of sub in s (last occurrence when back), 0 if absent."""
    return (s.rfind(sub) if back else s.find(sub)) + 1


def len_trim(s: str) -> int:
    """LEN_TRIM: length without trailing blanks."""
    return len(s.rstrip(" "))


def _pad_pair(a: str, b: str) -> tuple[str, str]:
    n = max(len(a), len(b))
    return a.ljust(n), b.ljust(n)


def str_cmp(a: str, b: str) -> int:
    """Compare two strings after padding the shorter with blanks: -1, 0 or 1."""
    x, y = _pad_pair(a, b)
    return (x > y) - (x < y)


def str_eq(a: str, b: str) -> bool:
    """Character .EQ.: equality ignoring trailing blanks."""
    return str_cmp(a, b) == 0


def lge(a: str, b: str) -> bool:
    """LGE: a >= b in the ASCII collating sequence (blank padded)."""
    return str_cmp(a, b) >= 0


def lgt(a: str, b: str) -> bool:
    """LGT: a > b in the ASCII collating sequence (blank padded)."""
    return str_cmp(a, b) > 0


def lle(a: str, b: str) -> bool:
    """LLE: a <= b in the ASCII collating sequence (blank padded)."""
    return str_cmp(a, b) <= 0


def llt(a: str, b: str) -> bool:
    """LLT: a < b in the ASCII collating sequence (blank padded)."""
    return str_cmp(a, b) < 0


def char_assign(value: str, n: int) -> str:
    """Assign to CHARACTER*n: truncate on the right or pad with blanks to length n."""
    if not isinstance(value, str):
        raise TypeError(f"char_assign needs a str, got {type(value).__name__}")
    length = _int(n, "char_assign length")
    if length < 0:
        raise ValueError("CHARACTER length must not be negative")
    return value[:length].ljust(length)


def _check_substr_start(i: int) -> int:
    start = _int(i, "substring start")
    if start < 1:
        raise IndexError(f"substring start {start} is below 1")
    return start


def substr(s: str, i: int, j: int) -> str:
    """s(i:j): 1-based inclusive substring; empty when j < i, blank padded past the end of s."""
    start = _check_substr_start(i)
    stop = _int(j, "substring end")
    if stop < start:
        return ""
    return s[start - 1 : stop].ljust(stop - start + 1)


def set_substr(s: str, i: int, j: int, value: str) -> str:
    """s(i:j) = value: return the new string (value padded or truncated to j-i+1 characters)."""
    start = _check_substr_start(i)
    stop = _int(j, "substring end")
    if stop < start:
        return s
    padded = s.ljust(stop)
    return padded[: start - 1] + char_assign(value, stop - start + 1) + padded[stop:]


# --------------------------------------------------------------------------------------------
# DO loops
# --------------------------------------------------------------------------------------------


def do_trips(start: int | float, limit: int | float, step: int | float = 1) -> int:
    """DO trip count max(0, int((limit - start + step) / step)), truncating, computed once."""
    if step == 0:
        raise ValueError("DO loop step must not be zero")
    if _is_int(start) and _is_int(limit) and _is_int(step):
        n = idiv(_int(limit, "DO") - _int(start, "DO") + _int(step, "DO"), _int(step, "DO"))
    else:
        n = math.trunc((float(limit) - float(start) + float(step)) / float(step))
    return max(0, n)


def do_range(start: int | float, limit: int | float, step: int | float = 1) -> Iterator[Any]:
    """Iterator over a DO loop: exact for integers, repeated addition for reals, trips fixed once."""
    trips = do_trips(start, limit, step)
    if _is_int(start) and _is_int(limit) and _is_int(step):
        s0, s1 = _int(start, "DO"), _int(step, "DO")
        return (s0 + k * s1 for k in range(trips))
    return _real_do_values(float(start), float(step), trips)


def _real_do_values(start: float, step: float, trips: int) -> Iterator[float]:
    x = start
    for _ in range(trips):
        yield x
        x += step


def do_final(start: int | float, limit: int | float, step: int | float = 1) -> int | float:
    """Value of the DO variable after normal completion (repeated addition for reals)."""
    trips = do_trips(start, limit, step)
    if _is_int(start) and _is_int(limit) and _is_int(step):
        return _int(start, "DO") + trips * _int(step, "DO")
    x, inc = float(start), float(step)
    for _ in range(trips):
        x += inc
    return x


# --------------------------------------------------------------------------------------------
# Bit operations (32-bit two's complement, signed results)
# --------------------------------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF


def _wrap32(x: int) -> int:
    x &= _MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def iand(a: int, b: int) -> int:
    """IAND: bitwise and of 32-bit integers."""
    return _wrap32(_int(a, "iand") & _int(b, "iand"))


def ior(a: int, b: int) -> int:
    """IOR: bitwise inclusive or of 32-bit integers."""
    return _wrap32(_int(a, "ior") | _int(b, "ior"))


def ieor(a: int, b: int) -> int:
    """IEOR: bitwise exclusive or of 32-bit integers."""
    return _wrap32(_int(a, "ieor") ^ _int(b, "ieor"))


def not_(a: int) -> int:
    """NOT: bitwise complement of a 32-bit integer."""
    return _wrap32(~_int(a, "not_"))


def ishft(i: int, shift: int) -> int:
    """ISHFT: logical shift of a 32-bit integer, left for shift > 0 and right for shift < 0."""
    bits = _int(i, "ishft value") & _MASK32
    n = _int(shift, "ishft shift")
    if abs(n) >= 32:
        return 0
    return _wrap32(bits << n if n >= 0 else bits >> -n)


def _bit_position(pos: int) -> int:
    p = _int(pos, "bit position")
    if not 0 <= p < 32:
        raise ValueError(f"bit position {p} is outside 0..31")
    return p


def btest(i: int, pos: int) -> bool:
    """BTEST: True when bit pos (0 = least significant) of i is set."""
    return bool((_int(i, "btest") >> _bit_position(pos)) & 1)


def ibset(i: int, pos: int) -> int:
    """IBSET: i with bit pos set."""
    return _wrap32(_int(i, "ibset") | (1 << _bit_position(pos)))


def ibclr(i: int, pos: int) -> int:
    """IBCLR: i with bit pos cleared."""
    return _wrap32(_int(i, "ibclr") & ~(1 << _bit_position(pos)))


def ibits(i: int, pos: int, length: int) -> int:
    """IBITS: the length bits of i starting at bit pos, right justified."""
    p = _bit_position(pos)
    n = _int(length, "ibits length")
    if n < 0 or p + n > 32:
        raise ValueError("IBITS field does not fit in 32 bits")
    return _wrap32(((_int(i, "ibits") & _MASK32) >> p) & ((1 << n) - 1))


# --------------------------------------------------------------------------------------------
# Machine constants, complex, arrays
# --------------------------------------------------------------------------------------------


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def d1mach(i: int) -> float:
    """D1MACH(i): double precision machine constants, i = 1 tiny, 2 huge, 3 eps/2, 4 eps, 5 log10(2)."""
    table = {
        1: sys.float_info.min,
        2: sys.float_info.max,
        3: sys.float_info.epsilon / 2.0,
        4: sys.float_info.epsilon,
        5: math.log10(2.0),
    }
    key = _int(i, "d1mach index")
    if key not in table:
        raise ValueError(f"D1MACH index {key} is outside 1..5")
    return table[key]


def r1mach(i: int) -> float:
    """R1MACH(i): single precision machine constants, same layout as d1mach."""
    table = {
        1: 2.0**-126,
        2: (2.0 - 2.0**-23) * 2.0**127,
        3: 2.0**-24,
        4: 2.0**-23,
        5: _f32(math.log10(2.0)),
    }
    key = _int(i, "r1mach index")
    if key not in table:
        raise ValueError(f"R1MACH index {key} is outside 1..5")
    return table[key]


_I1MACH = {
    1: 5,
    2: 6,
    3: 7,
    4: 0,
    5: 32,
    6: 4,
    7: 2,
    8: 31,
    9: 2147483647,
    10: 2,
    11: 24,
    12: -125,
    13: 128,
    14: 53,
    15: -1021,
    16: 1024,
}


def i1mach(i: int) -> int:
    """I1MACH(i): integer machine constants (units, word size, float parameters), i = 1..16."""
    key = _int(i, "i1mach index")
    if key not in _I1MACH:
        raise ValueError(f"I1MACH index {key} is outside 1..16")
    return _I1MACH[key]


def cmplx(a: float | complex, b: float = 0.0) -> complex:
    """CMPLX(a, b): complex number from real and imaginary parts (a complex a is returned as is)."""
    if isinstance(a, complex):
        if b != 0.0:
            raise TypeError("CMPLX with a complex first argument takes no second argument")
        return a
    return complex(float(a), float(b))


def zeros_f(*shape: int | Sequence[int], dtype: Any = float) -> np.ndarray:
    """Zero-filled numpy array in Fortran (column-major) order; extents below 0 count as 0."""
    dims: Sequence[Any] = shape
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        dims = shape[0]
    extents = tuple(max(0, _int(e, "array extent")) for e in dims)
    return np.zeros(extents, dtype=dtype, order="F")


# --------------------------------------------------------------------------------------------
# FORMAT parsing
# --------------------------------------------------------------------------------------------

_DATA_KINDS = frozenset({"I", "F", "E", "D", "G", "L", "A"})
_REPEATED_KINDS = _DATA_KINDS | {"/"}
# Longest names first so that "TL" wins over "T" and "SP" over "S".
_WORDS = ("TL", "TR", "SP", "SS", "BN", "BZ", "ES", "EN", "I", "F", "E", "D", "G", "L", "A", "X")
_WORDS += ("T", "S", "P", "H")


@dataclass(frozen=True, slots=True)
class _Item:
    """One parsed FORMAT element. X/T/TL/TR keep their count in w, P keeps its value in w."""

    kind: str
    rep: int = 1
    w: int | None = None
    d: int | None = None
    m: int | None = None
    e: int | None = None
    text: str = ""
    items: tuple[_Item, ...] = ()


class _FormatParser:
    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0

    def error(self, message: str) -> ValueError:
        return ValueError(f"invalid FORMAT {self.s!r} at position {self.i}: {message}")

    def parse(self) -> tuple[_Item, ...]:
        self._skip_blanks()
        if self.i < len(self.s) and self.s[self.i] == "(":
            self.i += 1
            items = self._items(nested=True)
            self._skip_blanks()
            if self.i != len(self.s):
                raise self.error("unexpected text after the closing parenthesis")
            return items
        return self._items(nested=False)

    def _skip_blanks(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def _skip_separators(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t,":
            self.i += 1

    def _items(self, nested: bool) -> tuple[_Item, ...]:
        items: list[_Item] = []
        while True:
            self._skip_separators()
            if self.i >= len(self.s):
                if nested:
                    raise self.error("missing closing parenthesis")
                return tuple(items)
            if self.s[self.i] == ")":
                if not nested:
                    raise self.error("unbalanced closing parenthesis")
                self.i += 1
                return tuple(items)
            items.append(self._item())

    def _quoted(self) -> str:
        s = self.s
        quote = s[self.i]
        self.i += 1
        out: list[str] = []
        while self.i < len(s):
            ch = s[self.i]
            if ch == quote:
                if s[self.i + 1 : self.i + 2] == quote:
                    out.append(quote)
                    self.i += 2
                    continue
                self.i += 1
                return "".join(out)
            out.append(ch)
            self.i += 1
        raise self.error("unterminated character literal")

    def _uint(self) -> int | None:
        m = re.compile(r"[0-9]+").match(self.s, self.i)
        if m is None:
            return None
        self.i = m.end()
        return int(m.group())

    def _required_uint(self, what: str) -> int:
        value = self._uint()
        if value is None:
            raise self.error(f"{what} expected")
        return value

    def _item(self) -> _Item:
        s = self.s
        c = s[self.i]
        if c in "'\"":
            return _Item("lit", text=self._quoted())
        if c in "/:$":
            self.i += 1
            return _Item(c)
        if c == "(":
            self.i += 1
            return _Item("group", items=self._items(nested=True))

        signed = c in "+-"
        negative = c == "-"
        if signed:
            self.i += 1
        num = self._uint()
        if signed and num is None:
            raise self.error("sign without a number")
        if negative and num is not None:
            num = -num
        if self.i >= len(s):
            raise self.error("count without an edit descriptor")
        c = s[self.i]
        if num is not None and not signed:
            if c == "/":
                self.i += 1
                return _Item("/", rep=self._positive(num))
            if c == "(":
                self.i += 1
                return _Item("group", rep=self._positive(num), items=self._items(nested=True))
        word = self._word()
        if signed and word != "P":
            raise self.error("a signed count is only allowed before P")
        return self._dispatch(word, num)

    def _positive(self, num: int) -> int:
        if num < 1:
            raise self.error("repeat count must be positive")
        return num

    def _word(self) -> str:
        for word in _WORDS:
            if self.s[self.i : self.i + len(word)].upper() == word:
                self.i += len(word)
                return word
        raise self.error(f"unknown edit descriptor {self.s[self.i]!r}")

    def _dispatch(self, word: str, num: int | None) -> _Item:
        if word == "H":
            if num is None or num < 1:
                raise self.error("Hollerith literal needs a positive length")
            text = self.s[self.i : self.i + num]
            if len(text) < num:
                raise self.error("Hollerith literal is shorter than its length")
            self.i += num
            return _Item("lit", text=text)
        if word == "P":
            if num is None:
                raise self.error("scale factor P needs a value")
            return _Item("P", w=num)
        if word in ("ES", "EN"):
            raise self.error(f"edit descriptor {word} is not supported")
        if word == "X":
            return _Item("X", w=1 if num is None else self._positive(num))
        if word in ("T", "TL", "TR"):
            if num is not None:
                raise self.error(f"{word} takes no repeat count")
            n = self._required_uint(f"{word} position")
            if n < 1 and word == "T":
                raise self.error("T position must be at least 1")
            return _Item(word, w=n)
        if word in ("S", "SP", "SS", "BN", "BZ"):
            if num is not None:
                raise self.error(f"{word} takes no repeat count")
            return _Item(word)
        return self._data(word, 1 if num is None else self._positive(num))

    def _data(self, word: str, rep: int) -> _Item:
        w = self._uint()
        if word == "A":
            return _Item("A", rep, w=w)
        if w is None:
            raise self.error(f"{word} needs a field width")
        if word == "L":
            return _Item("L", rep, w=w)
        if word == "I":
            m = None
            if self.s[self.i : self.i + 1] == ".":
                self.i += 1
                m = self._required_uint("minimum digit count")
            return _Item("I", rep, w=w, m=m)
        if self.s[self.i : self.i + 1] != ".":
            raise self.error(f"{word} needs the form {word}w.d")
        self.i += 1
        d = self._required_uint("digit count")
        e = None
        if word in ("E", "G") and self.s[self.i : self.i + 1] in ("E", "e"):
            self.i += 1
            e = self._required_uint("exponent digit count")
        return _Item(word, rep, w=w, d=d, e=e)


@functools.lru_cache(maxsize=512)
def _compile_cached(fmt: str) -> tuple[_Item, ...]:
    return _FormatParser(fmt).parse()


def _compile(fmt: str) -> tuple[_Item, ...]:
    if not isinstance(fmt, str):
        raise TypeError(f"a FORMAT must be a str, got {type(fmt).__name__}")
    return _compile_cached(fmt)


def _has_data(items: Sequence[_Item]) -> bool:
    return any(
        it.kind in _DATA_KINDS or (it.kind == "group" and _has_data(it.items)) for it in items
    )


def _count_data(items: Sequence[_Item]) -> int:
    total = 0
    for it in items:
        if it.kind in _DATA_KINDS:
            total += it.rep
        elif it.kind == "group":
            total += it.rep * _count_data(it.items)
    return total


def _expand(items: Sequence[_Item]) -> Iterator[_Item]:
    for it in items:
        if it.kind == "group":
            for _ in range(it.rep):
                yield from _expand(it.items)
        elif it.kind in _REPEATED_KINDS:
            for _ in range(it.rep):
                yield it
        else:
            yield it


_NEW_RECORD = _Item("/")


def _walk(items: Sequence[_Item], has_more: Callable[[], bool]) -> Iterator[_Item]:
    """Atomic FORMAT items in execution order, including format reversion.

    When the format ends while list items remain, a record break is yielded and execution
    restarts at the last top-level parenthesised group (with its repeat count), or at the start
    of the format if there is none. The consumer stops iterating when the list is exhausted.
    """
    yield from _expand(items)
    revert: Sequence[_Item] = items
    for it in reversed(items):
        if it.kind == "group":
            revert = (it,)
            break
    if has_more() and not _has_data(revert):
        raise ValueError("FORMAT has no data edit descriptor for the remaining list items")
    while has_more():
        yield _NEW_RECORD
        yield from _expand(revert)


# --------------------------------------------------------------------------------------------
# Output editing
# --------------------------------------------------------------------------------------------


def _decimal_context(prec: int) -> decimal.Context:
    return decimal.Context(
        prec=prec,
        rounding=decimal.ROUND_HALF_EVEN,
        Emax=decimal.MAX_EMAX,
        Emin=decimal.MIN_EMIN,
    )


def _as_int(v: Any, descriptor: str) -> int:
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{descriptor} edit descriptor needs an integer, got {type(v).__name__}")
    return int(v)


def _as_float(v: Any, descriptor: str) -> float:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{descriptor} edit descriptor needs a real, got {type(v).__name__}")
    return float(v)


def _sign_text(negative: bool, plus: bool) -> str:
    return "-" if negative else "+" if plus else ""


def _fit(sign: str, body: str, w: int) -> str:
    """Right-justify sign+body in w columns; drop the optional leading zero; else asterisks."""
    text = sign + body
    if w == 0 or len(text) == w:
        return text
    if len(text) < w:
        return text.rjust(w)
    if body.startswith("0."):
        text = sign + body[1:]
        if len(text) <= w:
            return text.rjust(w)
    return "*" * w


def _edit_i(v: int, w: int, m: int | None, plus: bool) -> str:
    digits = str(abs(v))
    if m is not None:
        digits = "" if m == 0 and v == 0 else digits.zfill(m)
    text = _sign_text(v < 0, plus) + digits
    if w == 0:
        return text
    return text.rjust(w) if len(text) <= w else "*" * w


def _edit_l(v: Any, w: int) -> str:
    if not isinstance(v, bool | np.bool_):
        raise TypeError(f"L edit descriptor needs a logical, got {type(v).__name__}")
    return ("T" if v else "F").rjust(w)


def _edit_a(v: Any, w: int | None) -> str:
    if not isinstance(v, str):
        raise TypeError(f"A edit descriptor needs a character value, got {type(v).__name__}")
    if w is None or w == 0:
        return v
    return v[:w] if len(v) >= w else v.rjust(w)


def _edit_nonfinite(x: float, w: int, plus: bool) -> str:
    if math.isnan(x):
        return "NaN".rjust(w) if w >= 3 or w == 0 else "*" * w
    sign = _sign_text(x < 0, plus)
    if w == 0:
        return sign + "Inf"
    if len(sign) + 8 <= w:
        return (sign + "Infinity").rjust(w)
    if len(sign) + 3 <= w:
        return (sign + "Inf").rjust(w)
    return "*" * w


def _edit_f(x: float, w: int, d: int, scale: int, plus: bool) -> str:
    if not math.isfinite(x):
        return _edit_nonfinite(x, w, plus)
    exact = decimal.Decimal(abs(x))
    ctx = _decimal_context(max(exact.adjusted(), 0) + d + abs(scale) + 20)
    if scale:
        exact = exact.scaleb(scale, context=ctx)
    quantum = decimal.Decimal((0, (1,), -d))
    body = format(exact.quantize(quantum, rounding=decimal.ROUND_HALF_EVEN, context=ctx), "f")
    if d == 0:
        body += "."
    return _fit(_sign_text(math.copysign(1.0, x) < 0, plus), body, w)


def _sig_digits(x: float, sig: int) -> tuple[str, int]:
    """Round |x| to sig digits; return (digits, e) with |x| = 0.DIGITS * 10**e (zero: e = 0)."""
    if x == 0.0:
        return "0" * sig, 0
    rounded = _decimal_context(sig).plus(decimal.Decimal(abs(x)))
    digits = "".join(map(str, rounded.as_tuple().digits)).ljust(sig, "0")[:sig]
    return digits, rounded.adjusted() + 1


def _exponent_text(value: int, e: int | None, letter: str) -> str | None:
    sign = "-" if value < 0 else "+"
    magnitude = abs(value)
    if e is None:
        if magnitude <= 99:
            return f"{letter}{sign}{magnitude:02d}"
        if magnitude <= 999:
            return f"{sign}{magnitude:03d}"
        return None
    digits = str(magnitude).zfill(e)
    return None if len(digits) > e else f"{letter}{sign}{digits}"


def _edit_e(x: float, w: int, d: int, e: int | None, scale: int, plus: bool, letter: str) -> str:
    if not math.isfinite(x):
        return _edit_nonfinite(x, w, plus)
    if d < 1:
        raise ValueError("E/D/G edit descriptors need at least one digit after the point")
    if scale > 0:
        if scale > d + 1:
            raise ValueError("scale factor too large for the E edit descriptor")
        sig = d + 1
    else:
        sig = d + scale
        if sig < 1:
            raise ValueError("scale factor too small for the E edit descriptor")
    digits, ev = _sig_digits(x, sig)
    if scale > 0:
        mantissa = digits[:scale] + "." + digits[scale:]
    else:
        mantissa = "0." + "0" * (-scale) + digits
    exponent = _exponent_text(0 if x == 0.0 else ev - scale, e, letter)
    sign = _sign_text(math.copysign(1.0, x) < 0, plus)
    if exponent is None:
        return "*" * w
    return _fit(sign, mantissa + exponent, w)


def _edit_g(x: float, w: int, d: int, e: int | None, scale: int, plus: bool) -> str:
    if not math.isfinite(x):
        return _edit_nonfinite(x, w, plus)
    if d < 1:
        raise ValueError("G edit descriptor needs at least one significant digit")
    magnitude = 1 if x == 0.0 else _sig_digits(x, d)[1]
    if 0 <= magnitude <= d:
        n = 4 if e is None else e + 2
        if w == 0:
            return _edit_f(x, 0, d - magnitude, 0, plus)
        if w - n < 1:
            return "*" * w
        return _edit_f(x, w - n, d - magnitude, 0, plus) + " " * n
    return _edit_e(x, w, d, e, scale, plus, "E")


def _render(it: _Item, v: Any, plus: bool, scale: int) -> str:
    kind = it.kind
    w = it.w
    if kind == "I":
        return _edit_i(_as_int(v, "I"), w or 0, it.m, plus)
    if kind == "F":
        return _edit_f(_as_float(v, "F"), w or 0, it.d or 0, scale, plus)
    if kind in ("E", "D"):
        return _edit_e(_as_float(v, kind), w or 0, it.d or 0, it.e, scale, plus, kind)
    if kind == "G":
        if isinstance(v, bool | np.bool_):
            return _edit_l(v, w or 0)
        if isinstance(v, str):
            return _edit_a(v, w)
        if isinstance(v, numbers.Integral):
            return _edit_i(int(v), w or 0, None, plus)
        return _edit_g(_as_float(v, "G"), w or 0, it.d or 0, it.e, scale, plus)
    if kind == "L":
        return _edit_l(v, w or 0)
    return _edit_a(v, w)


def _flatten(values: Sequence[Any], split_complex: bool) -> list[Any]:
    """Expand arrays (column-major) and lists; complex numbers become two reals if asked."""
    out: list[Any] = []
    for v in values:
        if isinstance(v, np.ndarray):
            out.extend(_flatten(v.ravel(order="F").tolist(), split_complex))
        elif isinstance(v, list | tuple):
            out.extend(_flatten(v, split_complex))
        elif split_complex and isinstance(v, complex | np.complexfloating):
            out.extend((float(v.real), float(v.imag)))
        else:
            out.append(v)
    return out


class _RecordBuffer:
    """The record being built; supports tab positioning and overwriting."""

    def __init__(self) -> None:
        self.records: list[str] = []
        self.chars: list[str] = []
        self.pos = 0

    def put(self, text: str) -> None:
        if len(self.chars) < self.pos:
            self.chars.extend(" " * (self.pos - len(self.chars)))
        end = self.pos + len(text)
        self.chars[self.pos : end] = list(text)
        self.pos = end

    def new_record(self) -> None:
        self.records.append("".join(self.chars))
        self.chars = []
        self.pos = 0


def _format_records(fmt: str, values: Sequence[Any]) -> tuple[list[str], bool]:
    items = _compile(fmt)
    pending = deque(_flatten(values, split_complex=True))
    out = _RecordBuffer()
    plus = False
    scale = 0
    advance = True
    for it in _walk(items, lambda: bool(pending)):
        kind = it.kind
        if kind in _DATA_KINDS:
            if not pending:
                break
            out.put(_render(it, pending.popleft(), plus, scale))
        elif kind == "lit":
            out.put(it.text)
        elif kind == "/":
            out.new_record()
        elif kind == ":":
            if not pending:
                break
        elif kind in ("X", "TR"):
            out.pos += it.w or 0
        elif kind == "TL":
            out.pos = max(0, out.pos - (it.w or 0))
        elif kind == "T":
            out.pos = (it.w or 1) - 1
        elif kind == "SP":
            plus = True
        elif kind in ("S", "SS"):
            plus = False
        elif kind == "P":
            scale = it.w or 0
        elif kind == "$":
            advance = False
    out.new_record()
    return out.records, advance


def format_record(fmt: str, values: Sequence[Any]) -> str:
    """Format values with a Fortran FORMAT and return the output text (records joined by newline).

    Supports Iw[.m], Fw.d, Ew.d[Ee], Dw.d, Gw.d[Ee], Lw, A[w], nX, Tn/TLn/TRn, '/', ':', quoted
    and nH literals, repeat counts, nested groups, format reversion, S/SP/SS and kP. A value that
    does not fit its field prints as w asterisks. F/E/D/G also accept integers and arrays are
    expanded in column-major order; a complex value consumes two real descriptors. Exact decimal
    ties round half to even on the exact binary value (gfortran behaviour).
    """
    return "\n".join(_format_records(fmt, values)[0])


# --------------------------------------------------------------------------------------------
# Input editing
# --------------------------------------------------------------------------------------------

_REAL_RE = re.compile(r"([+-]?)([0-9]*)(?:\.([0-9]*))?(?:[EeDdQq]([+-]?[0-9]+)|([+-][0-9]+))?")
_INT_RE = re.compile(r"[+-]?[0-9]+")
_SPECIAL_RE = re.compile(r"([+-]?)(inf|infinity|nan)", re.IGNORECASE)


def _parse_special(text: str) -> float | None:
    m = _SPECIAL_RE.fullmatch(text)
    if m is None:
        return None
    if m.group(2).lower() == "nan":
        return math.nan
    return -math.inf if m.group(1) == "-" else math.inf


def _parse_real(text: str, decimals: int, scale: int) -> float:
    """Parse a real input field; without a decimal point the last ``decimals`` digits are fractional."""
    special = _parse_special(text)
    if special is not None:
        return special
    m = _REAL_RE.fullmatch(text)
    if m is None or not (m.group(2) or m.group(3)):
        raise ValueError(f"invalid real input {text!r}")
    sign, int_digits, frac_digits, exp1, exp2 = m.groups()
    exponent = exp1 if exp1 is not None else exp2
    if frac_digits is None:
        digits = int_digits.rjust(decimals + 1, "0")
        cut = len(digits) - decimals
        mantissa = digits[:cut] + "." + digits[cut:]
    else:
        mantissa = (int_digits or "0") + "." + (frac_digits or "0")
    exp_value = int(exponent) if exponent is not None else -scale
    return float(f"{sign}{mantissa}e{exp_value}")


def _parse_logical(text: str) -> bool:
    t = text.strip().upper()
    if t.startswith("."):
        t = t[1:]
    if t[:1] == "T":
        return True
    if t[:1] == "F":
        return False
    raise ValueError(f"invalid logical input {text!r}")


def _numeric_field(field: str, blank_zero: bool) -> str:
    return field.replace(" ", "0" if blank_zero else "")


def _read_item(it: _Item, rec: str, pos: int, blank_zero: bool, scale: int) -> tuple[Any, int]:
    kind = it.kind
    if kind == "A":
        if it.w:
            return rec[pos : pos + it.w].ljust(it.w), pos + it.w
        return rec[pos:], max(pos, len(rec))
    w = it.w or 0
    if w == 0:
        raise ValueError(f"{kind} edit descriptor needs a positive width for input")
    field = rec[pos : pos + w].ljust(w)
    pos += w
    if kind == "L":
        return _parse_logical(field), pos
    text = _numeric_field(field.strip() if not blank_zero else field, blank_zero)
    if kind == "I":
        if not text:
            return 0, pos
        if _INT_RE.fullmatch(text) is None:
            raise ValueError(f"invalid integer input {field!r}")
        return int(text), pos
    if not text:
        return 0.0, pos
    return _parse_real(text, it.d or 0, scale), pos


def _read_formatted_values(fmt: str, n: int | None, next_record: Callable[[], str]) -> list[Any]:
    items = _compile(fmt)
    count = _count_data(items) if n is None else n
    if count < 0:
        raise ValueError("number of values to read must not be negative")
    if count > 0 and not _has_data(items):
        raise ValueError("FORMAT has no data edit descriptor to read into")
    rec = next_record()
    pos = 0
    out: list[Any] = []
    blank_zero = False
    scale = 0
    for it in _walk(items, lambda: len(out) < count):
        kind = it.kind
        if kind in _DATA_KINDS:
            if len(out) >= count:
                break
            value, pos = _read_item(it, rec, pos, blank_zero, scale)
            out.append(value)
        elif kind == "/":
            rec = next_record()
            pos = 0
        elif kind == ":":
            if len(out) >= count:
                break
        elif kind in ("X", "TR"):
            pos += it.w or 0
        elif kind == "lit":
            pos += len(it.text)
        elif kind == "TL":
            pos = max(0, pos - (it.w or 0))
        elif kind == "T":
            pos = (it.w or 1) - 1
        elif kind == "BZ":
            blank_zero = True
        elif kind == "BN":
            blank_zero = False
        elif kind == "P":
            scale = it.w or 0
    return out


# --------------------------------------------------------------------------------------------
# List-directed input
# --------------------------------------------------------------------------------------------

_REPEAT_RE = re.compile(r"([0-9]+)\*(.*)", re.DOTALL)
_BOOL_TOKENS = {
    "T": True,
    ".T.": True,
    "TRUE": True,
    ".TRUE.": True,
    "F": False,
    ".F.": False,
    "FALSE": False,
    ".FALSE.": False,
}
_Token = tuple[str, bool]  # (text, was quoted); None stands for a null value


def _read_quoted(line: str, i: int) -> tuple[str, int]:
    quote = line[i]
    j = i + 1
    out: list[str] = []
    while j < len(line):
        ch = line[j]
        if ch == quote:
            if line[j + 1 : j + 2] == quote:
                out.append(quote)
                j += 2
                continue
            return "".join(out), j + 1
        out.append(ch)
        j += 1
    return "".join(out), j


def _scan_list_record(line: str, state: list[bool]) -> tuple[list[_Token | None], bool]:
    """Split one record into list-directed items; state[0] says a value preceded the last separator."""
    out: list[_Token | None] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
        elif c == ",":
            if not state[0]:
                out.append(None)
            state[0] = False
            i += 1
        elif c == "/":
            return out, True
        else:
            token: _Token | None
            repeat = 1
            if c in "'\"":
                text, i = _read_quoted(line, i)
                token = (text, True)
            elif c == "(":
                j = line.find(")", i)
                j = n - 1 if j < 0 else j
                token = (line[i : j + 1], False)
                i = j + 1
            else:
                j = i
                while j < n and line[j] not in " \t,/":
                    if line[j] in "'\"" and re.fullmatch(r"[0-9]+\*", line[i:j]):
                        break
                    j += 1
                text = line[i:j]
                i = j
                m = _REPEAT_RE.fullmatch(text)
                if m is None:
                    token = (text, False)
                else:
                    repeat = int(m.group(1))
                    if repeat < 1:
                        raise ValueError("zero repeat count in list-directed input")
                    if m.group(2):
                        token = (m.group(2), False)
                    elif i < n and line[i] in "'\"":
                        text, i = _read_quoted(line, i)
                        token = (text, True)
                    else:
                        token = None
            out.extend([token] * repeat)
            state[0] = True
    return out, False


def _convert_token(token: _Token | None, typ: type | None) -> Any:
    if token is None:
        return None
    text, quoted = token
    if typ is str:
        return text
    if quoted:
        if typ is None:
            return text
        raise ValueError(f"quoted string {text!r} cannot be read as {typ.__name__}")
    if typ is bool:
        return _parse_logical(text)
    if typ is int:
        if _INT_RE.fullmatch(text) is None:
            raise ValueError(f"invalid integer input {text!r}")
        return int(text)
    if typ is float:
        return _parse_real(text, 0, 0)
    if typ is complex:
        return _parse_complex(text)
    # No declared type: infer from the token.
    if text.upper() in _BOOL_TOKENS:
        return _BOOL_TOKENS[text.upper()]
    if _INT_RE.fullmatch(text):
        return int(text)
    if text.startswith("("):
        return _parse_complex(text)
    try:
        return _parse_real(text, 0, 0)
    except ValueError:
        return text


def _parse_complex(text: str) -> complex:
    parts = text.strip().strip("()").split(",")
    if len(parts) != 2:
        raise ValueError(f"invalid complex input {text!r}")
    return complex(_parse_real(parts[0].strip(), 0, 0), _parse_real(parts[1].strip(), 0, 0))


def _read_list_values(
    n: int, types: Sequence[type | None], next_record: Callable[[], str]
) -> list[Any]:
    state = [False]
    tokens: list[_Token | None] = []
    while True:
        new, slash = _scan_list_record(next_record(), state)
        tokens.extend(new)
        if slash:
            tokens.extend([None] * (n - len(tokens)))
        if len(tokens) >= n:
            break
    return [_convert_token(tokens[k], types[k]) for k in range(n)]


def _normalise_types(types: Any, n: int) -> list[type | None]:
    if types is None:
        return [None] * n
    if isinstance(types, type):
        return [types] * n
    kinds = list(types)
    if len(kinds) != n:
        raise ValueError(f"types has {len(kinds)} entries for {n} values")
    return kinds


# --------------------------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------------------------


class _Unit:
    """A file connected to a unit number: sequential formatted records in a binary file."""

    def __init__(
        self, number: int, path: str, handle: Any, scratch: bool, read_only: bool, at_eof: bool
    ) -> None:
        self.number = number
        self.path = path
        self.handle = handle
        self.scratch = scratch
        self.read_only = read_only
        self.at_eof = at_eof  # file position is known to be at the physical end
        self.eof_mark = False  # positioned after the endfile record
        self.starts: list[int] = []  # offsets of the records passed so far (for BACKSPACE)

    def _require_writable(self) -> None:
        if self.read_only:
            raise OSError(f"unit {self.number} is connected read-only")

    def write_record(self, text: str) -> None:
        self._require_writable()
        if not self.at_eof:
            self.handle.truncate()  # a sequential write destroys everything after it
            self.at_eof = True
        start = self.handle.tell()
        self.handle.write(text.encode("utf-8") + b"\n")
        self.handle.flush()
        self.starts.append(start)
        self.eof_mark = False

    def read_record(self) -> str:
        if self.eof_mark:
            raise EOFError(f"end of file on unit {self.number}")
        start = self.handle.tell()
        line = self.handle.readline()
        if not line:
            self.at_eof = True
            self.eof_mark = True
            raise EOFError(f"end of file on unit {self.number}")
        self.starts.append(start)
        text = line.decode("utf-8", errors="replace")
        if text.endswith("\n"):
            text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
        return text

    def rewind(self) -> None:
        self.handle.seek(0)
        self.starts.clear()
        self.at_eof = False
        self.eof_mark = False

    def backspace(self) -> None:
        if self.eof_mark:
            self.eof_mark = False
            return
        if not self.starts:
            return
        self.handle.seek(self.starts.pop())
        self.at_eof = False

    def endfile(self) -> None:
        self._require_writable()
        self.handle.truncate()
        self.handle.flush()
        self.at_eof = True
        self.eof_mark = True

    def close(self, delete: bool) -> None:
        try:
            self.handle.close()
        finally:
            if delete:
                try:
                    os.remove(self.path)
                except OSError:
                    pass


_UNITS: dict[int, _Unit] = {}
_LOCK = threading.RLock()
_STATUSES = ("old", "new", "replace", "scratch", "unknown")
_POSITIONS = ("asis", "rewind", "append")


def _is_star(unit: Any) -> bool:
    return unit is None or (isinstance(unit, str) and unit.strip() == "*")


def _unit_number(unit: Any) -> int:
    try:
        n = operator.index(unit)
    except TypeError:
        raise TypeError(f"a unit must be an integer, got {unit!r}") from None
    if n < 0:
        raise ValueError(f"unit number {n} is negative")
    return n


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _open_locked(n: int, file: Any, status: str, form: str, access: str, position: str) -> _Unit:
    status, form, access, position = (s.strip().lower() for s in (status, form, access, position))
    if status not in _STATUSES:
        raise ValueError(f"STATUS must be one of {_STATUSES}, got {status!r}")
    if form not in ("formatted", "unformatted"):
        raise ValueError(f"FORM must be 'formatted' or 'unformatted', got {form!r}")
    if form == "unformatted":
        raise NotImplementedError("unformatted files are not supported by the runtime")
    if access not in ("sequential", "direct"):
        raise ValueError(f"ACCESS must be 'sequential' or 'direct', got {access!r}")
    if access == "direct":
        raise NotImplementedError("direct access files are not supported by the runtime")
    if position not in _POSITIONS:
        raise ValueError(f"POSITION must be one of {_POSITIONS}, got {position!r}")
    name = None if file is None else os.fspath(file).rstrip(" ")
    if name == "":
        name = None

    existing = _UNITS.get(n)
    if existing is not None:
        if name is None or _same_file(existing.path, name):
            return existing
        _close_locked(n, None)

    read_only = False
    if status == "scratch":
        if name is not None:
            raise ValueError("a SCRATCH file must not have a name")
        fd, path = tempfile.mkstemp(prefix=f"f77_unit{n}_", suffix=".tmp")
        handle = os.fdopen(fd, "w+b")
        fresh = True
    else:
        path = name if name is not None else f"fort.{n}"
        for other in _UNITS.values():
            if _same_file(other.path, path):
                raise ValueError(f"file {path!r} is already connected to unit {other.number}")
        exists = os.path.exists(path)
        if status == "old" and not exists:
            raise FileNotFoundError(f"STATUS='OLD' but file {path!r} does not exist")
        mode = {"old": "r+b", "new": "x+b", "replace": "w+b"}.get(status)
        if mode is None:
            mode = "r+b" if exists else "w+b"
        fresh = mode != "r+b"
        try:
            handle = open(path, mode)
        except PermissionError:
            if mode != "r+b":
                raise
            handle = open(path, "rb")
            read_only = True
    unit = _Unit(n, path, handle, status == "scratch", read_only, at_eof=fresh)
    if position == "append" and not fresh:
        while True:
            start = handle.tell()
            if not handle.readline():
                break
            unit.starts.append(start)
        unit.at_eof = True
    _UNITS[n] = unit
    return unit


def _close_locked(n: int, status: str | None) -> None:
    unit = _UNITS.pop(n, None)
    if unit is None:
        return
    keyword = None if status is None else status.strip().lower()
    if keyword not in (None, "keep", "delete"):
        _UNITS[n] = unit
        raise ValueError(f"close STATUS must be 'keep' or 'delete', got {status!r}")
    if keyword == "keep" and unit.scratch:
        _UNITS[n] = unit
        raise ValueError("a SCRATCH file cannot be kept")
    unit.close(delete=unit.scratch or keyword == "delete")


def open_unit(
    unit: int,
    file: str | os.PathLike[str] | None = None,
    status: str = "unknown",
    form: str = "formatted",
    access: str = "sequential",
    position: str = "asis",
) -> None:
    """OPEN: connect a unit to a sequential formatted file (status old/new/replace/scratch/unknown)."""
    n = _unit_number(unit)
    with _LOCK:
        _open_locked(n, file, status, form, access, position)


def close_unit(unit: int, status: str | None = None) -> None:
    """CLOSE: disconnect a unit (no effect on an unconnected unit); status 'delete' removes the file."""
    n = _unit_number(unit)
    with _LOCK:
        _close_locked(n, status)


def close_all_units() -> None:
    """Close every connected file unit (also run automatically at interpreter exit)."""
    with _LOCK:
        for n in list(_UNITS):
            _close_locked(n, None)


atexit.register(close_all_units)


def _positioning_unit(unit: Any) -> _Unit | None:
    """Unit for REWIND/BACKSPACE/ENDFILE; the standard streams have nothing to reposition."""
    if _is_star(unit):
        return None
    n = _unit_number(unit)
    found = _UNITS.get(n)
    if found is None and n not in (0, 5, 6):
        found = _open_locked(n, None, "unknown", "formatted", "sequential", "asis")
    return found


def rewind_unit(unit: int) -> None:
    """REWIND: position the unit at its first record (no effect on units 0, 5 and 6)."""
    with _LOCK:
        target = _positioning_unit(unit)
        if target is not None:
            target.rewind()


def backspace_unit(unit: int) -> None:
    """BACKSPACE: move back one record (no effect at the first record or on units 0, 5, 6)."""
    with _LOCK:
        target = _positioning_unit(unit)
        if target is not None:
            target.backspace()


def endfile_unit(unit: int) -> None:
    """ENDFILE: truncate the file at the current position and mark the end of file."""
    with _LOCK:
        target = _positioning_unit(unit)
        if target is not None:
            target.endfile()


def _write_target(unit: Any) -> _Unit | TextIO:
    """Resolve an output unit; the caller holds the lock. Standard streams are looked up now."""
    if _is_star(unit):
        return sys.stdout
    n = _unit_number(unit)
    found = _UNITS.get(n)
    if found is not None:
        return found
    if n == 6:
        return sys.stdout
    if n == 0:
        return sys.stderr
    if n == 5:
        raise ValueError("unit 5 is the standard input and cannot be written")
    return _open_locked(n, None, "unknown", "formatted", "sequential", "asis")


def _emit(unit: Any, records: list[str], advance: bool) -> None:
    with _LOCK:
        target = _write_target(unit)
        if isinstance(target, _Unit):
            for record in records:
                target.write_record(record)
            return
    target.write("\n".join(records) + ("\n" if advance else ""))
    if not advance:
        target.flush()


def _stream_reader(stream: TextIO) -> Callable[[], str]:
    def next_record() -> str:
        line = stream.readline()
        if line == "":
            raise EOFError("end of file on standard input")
        if line.endswith("\n"):
            line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
        return line

    return next_record


def _read_from(unit: Any, parse: Callable[[Callable[[], str]], Any]) -> Any:
    """Run parse(next_record) against the record source of an input unit."""
    if not _is_star(unit):
        n = _unit_number(unit)
        with _LOCK:
            found = _UNITS.get(n)
            if found is None and n in (0, 6):
                raise ValueError(f"unit {n} is an output unit and cannot be read")
            if found is None and n != 5:
                found = _open_locked(n, None, "old", "formatted", "sequential", "asis")
            if found is not None:
                return parse(found.read_record)
    return parse(_stream_reader(sys.stdin))


def _internal_reader(text: str | Sequence[str]) -> Callable[[], str]:
    records = [text] if isinstance(text, str) else list(text)
    position = iter(records)

    def next_record() -> str:
        try:
            return next(position)
        except StopIteration:
            raise EOFError("end of internal file") from None

    return next_record


# --------------------------------------------------------------------------------------------
# Public formatted and list-directed I/O
# --------------------------------------------------------------------------------------------


def write_formatted(unit: int | str | None, fmt: str, *values: Any) -> None:
    """WRITE(unit, fmt): format the values and write the records; unit 6, None or '*' is stdout, 0 stderr."""
    records, advance = _format_records(fmt, values)
    _emit(unit, records, advance)


def _shortest_real(x: float) -> str:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    negative = math.copysign(1.0, x) < 0
    a = abs(x)
    if a == 0.0:
        body = "0.0"
    else:
        dec = decimal.Decimal(repr(a))
        if 0.1 <= a < 1e7:
            body = format(dec, "f")
            if "." not in body:
                body += ".0"
        else:
            digits = "".join(map(str, dec.as_tuple().digits)).rstrip("0") or "0"
            exp = dec.adjusted()
            mantissa = digits[0] + "." + (digits[1:] or "0")
            body = f"{mantissa}E{'-' if exp < 0 else '+'}{abs(exp):02d}"
    return "-" + body if negative else body


def _list_item(v: Any) -> str:
    if isinstance(v, bool | np.bool_):
        return "T" if v else "F"
    if isinstance(v, numbers.Integral):
        return str(int(v)).rjust(11)
    if isinstance(v, np.floating) and not isinstance(v, float):
        return _shortest_real(float(str(v)))
    if isinstance(v, numbers.Real):
        return _shortest_real(float(v))
    if isinstance(v, complex | np.complexfloating):
        return f"({_shortest_real(float(v.real))},{_shortest_real(float(v.imag))})"
    if isinstance(v, str):
        return v
    raise TypeError(f"cannot write a {type(v).__name__} with list-directed output")


def _list_record(values: Sequence[Any]) -> str:
    return "".join(" " + _list_item(v) for v in _flatten(values, split_complex=False))


def write_list(unit: int | str | None, *values: Any) -> None:
    """WRITE(unit, *) / PRINT *: approximate gfortran list-directed output (not byte exact).

    Each item is preceded by one blank; integers are right aligned in 11 columns, reals use the
    shortest round-trip digits (fixed form for 0.1 <= |x| < 1e7, else E form), logicals are T/F,
    complex is (re,im) and strings are written as is. Real field widths and the 80 column line
    wrapping of gfortran are not reproduced.
    """
    _emit(unit, [_list_record(values)], True)


def read_formatted(unit: int | str | None, fmt: str, n: int | None = None) -> list[Any]:
    """READ(unit, fmt): list of n values parsed with the edit descriptors; EOFError at end of file.

    I gives int, F/E/D/G float (a field without a decimal point uses the d implied digits), L
    bool, A str (blank padded to w; without w the rest of the record). n defaults to the number
    of data descriptors in the format; a larger n triggers format reversion. Unit 5 and None
    read sys.stdin at call time. Blanks inside numeric fields are ignored (BN) unless BZ is set.
    """
    return _read_from(unit, lambda next_record: _read_formatted_values(fmt, n, next_record))


def read_list(
    unit: int | str | None, n: int, types: type | Sequence[type] | None = None
) -> list[Any]:
    """READ(unit, *): list of n values split on blanks, commas and slashes; EOFError at end of file.

    Supports repeat counts (r*c), quoted strings and (re,im) complex. A null value (empty
    between commas, r* or after a slash) is None. Without types each value is inferred (int,
    float, bool, str); types gives one type or a list of types (int, float, bool, str, complex)
    to coerce the values exactly as the receiving variables would.
    """
    count = _int(n, "read_list count")
    if count < 0:
        raise ValueError("number of values to read must not be negative")
    kinds = _normalise_types(types, count)
    return _read_from(unit, lambda next_record: _read_list_values(count, kinds, next_record))


def write_internal(fmt: str, *values: Any) -> str:
    """WRITE to an internal file: the formatted record as a str (one record only)."""
    records, _ = _format_records(fmt, values)
    if len(records) != 1:
        raise ValueError(f"format produced {len(records)} records for a single internal record")
    return records[0]


def read_internal(text: str | Sequence[str], fmt: str, n: int | None = None) -> list[Any]:
    """READ from an internal file (a string, or a list of strings as records) with a FORMAT."""
    return _read_formatted_values(fmt, n, _internal_reader(text))
