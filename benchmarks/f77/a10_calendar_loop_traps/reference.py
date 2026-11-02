"""Reference port of legacy.f (calendar arithmetic and F77 semantic traps)."""

from __future__ import annotations

import math


class _Cnt:
    """COMMON /CNT/ NCALL, NSTEP, initialised by BLOCK DATA CNTINI."""

    def __init__(self) -> None:
        self.ncall = 0
        self.nstep = 0


cnt = _Cnt()


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def jday(iy, im, id):
    ma = _idiv(im - 14, 12)
    return (
        id
        - 32075
        + _idiv(1461 * (iy + 4800 + ma), 4)
        + _idiv(367 * (im - 2 - ma * 12), 12)
        - _idiv(3 * _idiv(iy + 4900 + ma, 100), 4)
    )


def caldat(jd, iy, im, id):
    l = jd + 68569
    n = _idiv(4 * l, 146097)
    l = l - _idiv(146097 * n + 3, 4)
    i = _idiv(4000 * (l + 1), 1461001)
    l = l - _idiv(1461 * i, 4) + 31
    j = _idiv(80 * l, 2447)
    k = l - _idiv(2447 * j, 80)
    l = _idiv(j, 11)
    j = j + 2 - 12 * l
    i = 100 * (n - 49) + i + l
    return i, j, k


def lasti(n, m, k, ntrip):
    trips = max(0, _idiv(m - n + k, k))
    return n + trips * k, trips


def _nint(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def rndmod(x, mode):
    if mode == 1:
        return _nint(x)
    if mode == 2:
        return int(x)
    if mode == 3:
        r = int(x)
        if float(r) > x:
            r = r - 1
        return r
    return 0


def colltz(n, maxit):
    cnt.ncall += 1
    m = n
    k = 0
    while m > 1:
        if k >= maxit:
            cnt.nstep += k
            return -k
        if m % 2 != 0:
            m = 3 * m + 1
        else:
            m = _idiv(m, 2)
        k += 1
    cnt.nstep += k
    return k


def cntget(nc, ns):
    return cnt.ncall, cnt.nstep


def cntrst():
    cnt.ncall = 0
    cnt.nstep = 0
