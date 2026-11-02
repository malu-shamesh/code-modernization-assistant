"""Reference port of legacy.f (Julian day number calendar utilities)."""

from __future__ import annotations


class _Dcnt:
    """COMMON /DCNT/ NFWD, NREV"""

    def __init__(self) -> None:
        self.nfwd = 0
        self.nrev = 0


_dcnt = _Dcnt()  # BLOCK DATA DCNTBD

# SAVE NDAY with DATA
_NDAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _idiv(a, b):
    """Fortran integer division: truncation toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _fmod(a, p):
    """Fortran MOD: the result has the sign of the dividend."""
    return a - p * _idiv(a, p)


def isleap(iy):
    return (_fmod(iy, 4) == 0 and _fmod(iy, 100) != 0) or _fmod(iy, 400) == 0


def mdays(iy, im):
    if im < 1 or im > 12:
        return 0
    result = _NDAY[im - 1]
    if im == 2 and isleap(iy):
        result = 29
    return result


def jdn(iy, im, id, ierr):
    result = 0
    ierr = 0
    if im < 1 or im > 12:
        return result, 1
    if id < 1 or id > mdays(iy, im):
        return result, 2
    ll = _idiv(im - 14, 12)
    result = (
        id
        - 32075
        + _idiv(1461 * (iy + 4800 + ll), 4)
        + _idiv(367 * (im - 2 - ll * 12), 12)
        - _idiv(3 * _idiv(iy + 4900 + ll, 100), 4)
    )
    _dcnt.nfwd += 1
    return result, ierr


def jdcal(jd, iy, im, id):
    iy = 0
    im = 0
    id = 0
    if jd < 0:
        return iy, im, id
    ll = jd + 68569
    n = _idiv(4 * ll, 146097)
    ll = ll - _idiv(146097 * n + 3, 4)
    i = _idiv(4000 * (ll + 1), 1461001)
    ll = ll - _idiv(1461 * i, 4) + 31
    j = _idiv(80 * ll, 2447)
    k = ll - _idiv(2447 * j, 80)
    ll = _idiv(j, 11)
    j = j + 2 - 12 * ll
    i = 100 * (n - 49) + i + ll
    iy = i
    im = j
    id = k
    _dcnt.nrev += 1
    return iy, im, id


def idow(jd):
    return _fmod(jd + 1, 7)


def dayofy(iy, im, id):
    if im < 1 or im > 12:
        return 0
    if id < 1 or id > mdays(iy, im):
        return 0
    n = id
    for i in range(1, im):
        n = n + mdays(iy, i)
    return n


def dcount(nf, nr, ireset):
    nf = _dcnt.nfwd
    nr = _dcnt.nrev
    if ireset != 0:
        _dcnt.nfwd = 0
        _dcnt.nrev = 0
    return nf, nr
