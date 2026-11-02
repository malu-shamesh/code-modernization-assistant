"""Reference port of legacy.f (32-bit word utilities and a packed bit set)."""

from __future__ import annotations

_MASK = 0xFFFFFFFF


def _s32(x):
    x &= _MASK
    return x - (1 << 32) if x & 0x80000000 else x


def _ishft(i, shift):
    """Logical shift of a 32-bit two-complement word; |shift| >= 32 gives 0."""
    u = i & _MASK
    if shift >= 32 or shift <= -32:
        return 0
    if shift >= 0:
        return _s32(u << shift)
    return _s32(u >> -shift)


def _idiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _fmod(a, p):
    return a - p * _idiv(a, p)


def bitcnt(iw):
    result = 0
    it = iw
    while it != 0:
        if (it & 1) != 0:
            result = result + 1
        it = _ishft(it, -1)
    return result


def bitrev(iw, nb):
    nn = max(0, min(nb, 32))
    it = iw
    result = 0
    for _ in range(1, nn + 1):
        result = _s32(_ishft(result, 1) | (it & 1))
        it = _ishft(it, -1)
    return result


def graycd(iw):
    return _s32(iw ^ _ishft(iw, -1))


def graydc(ig):
    result = ig
    for i in range(0, 5):
        result = _s32(result ^ _ishft(result, -(2**i)))
    return result


def rotl32(iw, ish):
    n = _fmod(ish, 32)
    if n < 0:
        n = n + 32
    return _s32(_ishft(iw, n) | _ishft(iw, n - 32))


def _out_of_range(nw, k):
    return k < 0 or _idiv(k, 32) >= nw


def bsset(ibits, nw, k, ierr):
    ierr = 0
    if _out_of_range(nw, k):
        return 1
    iwd = _idiv(k, 32) + 1
    ibits[iwd - 1] = _s32(int(ibits[iwd - 1]) | _ishft(1, _fmod(k, 32)))
    return ierr


def bsclr(ibits, nw, k, ierr):
    ierr = 0
    if _out_of_range(nw, k):
        return 1
    iwd = _idiv(k, 32) + 1
    ibits[iwd - 1] = _s32(int(ibits[iwd - 1]) & ~_ishft(1, _fmod(k, 32)))
    return ierr


def bstest(ibits, nw, k):
    if _out_of_range(nw, k):
        return False
    iwd = _idiv(k, 32) + 1
    return (int(ibits[iwd - 1]) & _ishft(1, _fmod(k, 32))) != 0


def bscnt(ibits, nw):
    result = 0
    for i in range(1, nw + 1):
        result = result + bitcnt(int(ibits[i - 1]))
    return result


def bsnext(ibits, nw, k0):
    for k in range(max(k0, 0), 32 * nw):
        if bstest(ibits, nw, k):
            return k
    return -1
