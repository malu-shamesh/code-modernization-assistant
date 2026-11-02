"""Python port of ROMAN (Roman numerals and a printed table)."""

from __future__ import annotations

_VAL = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
_SYM = ("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")
_DIGITS = "IVXLCDM"
_DV = (1, 5, 10, 50, 100, 500, 1000)


def _idiv(a: int, b: int) -> int:
    """Fortran integer division: truncate toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _i_edit(n: int, w: int) -> str:
    s = str(n)
    return "*" * w if len(s) > w else s.rjust(w)


def toromn(n, roman, ierr):
    if n < 1 or n > 3999:
        return " " * 16, 1
    m = n
    parts = []
    for i in range(13):
        k = m // _VAL[i]
        parts.append(_SYM[i] * k)
        m = m % _VAL[i]
    return "".join(parts).ljust(16), 0


def fromrm(roman, ierr):
    nch = 0
    for i in range(len(roman), 0, -1):
        if roman[i - 1] != " ":
            nch = i
            break
    if nch == 0 or nch > 16:
        return 0, 2
    v = []
    for i in range(nch):
        k = _DIGITS.find(roman[i]) + 1
        if k == 0:
            return 0, 2
        v.append(_DV[k - 1])
    v.append(0)
    total = 0
    for i in range(nch):
        if v[i] < v[i + 1]:
            total -= v[i]
        else:
            total += v[i]
    check, ie2 = toromn(total, "", 0)
    if ie2 != 0 or check.rstrip(" ") != roman.rstrip(" "):
        return 0, 3
    return total, 0


def prtrom(n1, n2, nstep):
    trips = max(0, _idiv(n2 - n1 + nstep, nstep))
    n = n1
    for _ in range(trips):
        roman, ierr = toromn(n, "", 0)
        if ierr == 0:
            print(" " + _i_edit(n, 5) + "  " + roman)
        else:
            print(" " + _i_edit(n, 5) + "  OUT OF RANGE")
        n += nstep
