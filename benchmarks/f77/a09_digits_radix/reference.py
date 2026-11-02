"""Reference port of legacy.f (decimal digit utilities and radix conversion)."""

from __future__ import annotations

_DIGITS = "0123456789ABCDEF"


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _mod(a: int, p: int) -> int:
    return a - p * _idiv(a, p)


def _isign(a: int, b: int) -> int:
    return abs(a) if b >= 0 else -abs(a)


def idigsm(n):
    m = abs(n)
    total = 0
    while m != 0:
        total = total + _mod(m, 10)
        m = _idiv(m, 10)
    return total


def irev(n):
    m = abs(n)
    ir = 0
    while m != 0:
        ir = ir * 10 + _mod(m, 10)
        m = _idiv(m, 10)
    return _isign(ir, n)


def isqrti(n):
    if n <= 0:
        return 0
    ix = n
    while True:
        iy = _idiv(ix + _idiv(n, ix), 2)
        if iy >= ix:
            break
        ix = iy
    return ix


def itob(n, ibase, str):
    if ibase < 2 or ibase > 16:
        return "?".ljust(16)
    m = abs(n)
    buf = []
    while True:
        buf.append(_DIGITS[_mod(m, ibase)])
        m = _idiv(m, ibase)
        if m <= 0:
            break
    if n < 0:
        buf.append("-")
    if len(buf) > 16:
        return "*" * 16
    return "".join(reversed(buf)).ljust(16)


def istoi(str, ibase, ierr):
    ierr = 0
    if ibase < 2 or ibase > 16:
        ierr = -1
        return 0, ierr
    s = str.ljust(16)[:16]
    start = 2 if s[0] == "-" else 1
    iv = 0
    for i in range(start, 17):
        ch = s[i - 1]
        if ch == " ":
            break
        idx = _DIGITS[:ibase].find(ch) + 1
        if idx == 0:
            ierr = i
            return 0, ierr
        iv = iv * ibase + (idx - 1)
    result = -iv if start == 2 else iv
    return result, ierr


def showb(n, ibase):
    s = itob(n, ibase, " " * 16)
    print(f" {n:10d} IN BASE {ibase:2d} IS {s:16s}")
