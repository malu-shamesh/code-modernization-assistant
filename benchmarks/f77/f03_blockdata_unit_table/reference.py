"""Reference port of legacy.f (unit conversion table initialised by BLOCK DATA)."""

from __future__ import annotations

_MAXU = 8
_NBASE = 6


class _UnitCommon:
    """COMMON /UNAMES/ UNAME(8), /UFACTS/ FACT(8), /UNCNT/ NUNIT"""

    def __init__(self) -> None:
        self.uname = [""] * _MAXU
        self.fact = [0.0] * _MAXU
        self.nunit = 0


_com = _UnitCommon()


def _unitbd() -> None:
    """BLOCK DATA UNITBD"""
    names = ["M", "KM", "CM", "MM", "IN", "FT"]
    facts = [1.0, 1.0e3, 1.0e-2, 1.0e-3, 2.54e-2, 3.048e-1]
    names += [""] * (_MAXU - len(names))
    facts += [0.0] * (_MAXU - len(facts))
    _com.uname = [n.ljust(4) for n in names]
    _com.fact = facts
    _com.nunit = _NBASE


_unitbd()


def _str_eq(a, b):
    # Fortran compares as if the shorter operand were padded with blanks
    return a.rstrip(" ") == b.rstrip(" ")


def ucnt():
    return _com.nunit


def ufind(name):
    for i in range(1, _com.nunit + 1):
        if _str_eq(_com.uname[i - 1], name):
            return i
    return 0


def uconv(x, cfrom, cto, ierr):
    ierr = 0
    i = ufind(cfrom)
    if i == 0:
        return 0.0, 1
    j = ufind(cto)
    if j == 0:
        return 0.0, 2
    result = x * _com.fact[i - 1] / _com.fact[j - 1]
    return result, ierr


def uadd(name, factor, ierr):
    ierr = 0
    if factor <= 0.0:
        ierr = 2
    elif ufind(name) != 0:
        ierr = 3
    elif _com.nunit >= _MAXU:
        ierr = 1
    else:
        _com.nunit += 1
        _com.uname[_com.nunit - 1] = name[:4].ljust(4)
        _com.fact[_com.nunit - 1] = factor
    return ierr


def ureset():
    for i in range(7, 9):
        _com.uname[i - 1] = " " * 4
        _com.fact[i - 1] = 0.0
    _com.nunit = _NBASE
