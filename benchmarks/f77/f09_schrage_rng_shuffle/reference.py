"""Reference port of legacy.f (Park-Miller generator with Schrage's method, shuffle, sampling)."""

from __future__ import annotations

_IA = 16807
_IM = 2147483647
_IQ = 127773
_IR = 2836


class _RngSt:
    """COMMON /RNGST/ IX, NDRAW"""

    def __init__(self) -> None:
        self.ix = 0
        self.ndraw = 0


_rngst = _RngSt()


def _rngbd() -> None:
    """BLOCK DATA RNGBD: DATA IX /1/, NDRAW /0/"""
    _rngst.ix = 1
    _rngst.ndraw = 0


_rngbd()


def _idiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _fmod(a, p):
    return a - p * _idiv(a, p)


def rseed(iseed):
    _rngst.ix = _fmod(abs(iseed), _IM)
    if _rngst.ix == 0:
        _rngst.ix = 1
    _rngst.ndraw = 0


def rnext():
    k = _idiv(_rngst.ix, _IQ)
    _rngst.ix = _IA * (_rngst.ix - k * _IQ) - _IR * k
    if _rngst.ix < 0:
        _rngst.ix = _rngst.ix + _IM
    _rngst.ndraw = _rngst.ndraw + 1
    return _rngst.ix


def rbound(n):
    if n < 1:
        return 0
    return 1 + _fmod(rnext(), n)


def rstat(ndr):
    ndr = _rngst.ndraw
    return ndr


def rshuf(n, iperm):
    for i in range(1, n + 1):
        iperm[i - 1] = i
    for i in range(n, 1, -1):
        j = rbound(i)
        itmp = iperm[i - 1]
        iperm[i - 1] = iperm[j - 1]
        iperm[j - 1] = itmp


def rsamp(n, m, isamp, nsel):
    nsel = 0
    if m < 1 or m > n:
        return nsel
    for it in range(0, n):
        if rbound(n - it) > m - nsel:
            continue
        nsel = nsel + 1
        isamp[nsel - 1] = it + 1
        if nsel == m:
            return nsel
    return nsel
