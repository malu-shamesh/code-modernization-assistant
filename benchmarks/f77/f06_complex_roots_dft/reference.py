"""Reference port of legacy.f (complex quadratic roots, small DFT, call statistics)."""

from __future__ import annotations

import cmath
import math


class _QStat:
    """COMMON /QSTAT/ NCALL, NCPLX"""

    def __init__(self) -> None:
        self.ncall = 0
        self.ncplx = 0


_qstat = _QStat()  # BLOCK DATA QSTBD


def quadrt(a, b, c, z1, z2, ikind):
    _qstat.ncall += 1
    z1 = complex(0.0, 0.0)
    z2 = complex(0.0, 0.0)
    if a == 0.0:
        if b == 0.0:
            ikind = -1
        else:
            ikind = 3
            z1 = complex(-c / b, 0.0)
        return z1, z2, ikind
    d = b * b - 4.0 * a * c
    if d > 0.0:
        # Fortran SIGN(x, 0.0) is +x
        q = -0.5 * (b + math.copysign(math.sqrt(d), b if b != 0.0 else 1.0))
        z1 = complex(q / a, 0.0)
        z2 = complex(c / q, 0.0)
        ikind = 0
    elif d == 0.0:
        z1 = complex(-b / (2.0 * a), 0.0)
        z2 = z1
        ikind = 1
    else:
        re = -b / (2.0 * a)
        aim = math.sqrt(-d) / (2.0 * a)
        z1 = complex(re, abs(aim))
        z2 = z1.conjugate()
        ikind = 2
        _qstat.ncplx += 1
    return z1, z2, ikind


def cquad(ca, cb, cc, z1, z2, ierr):
    ierr = 0
    z1 = complex(0.0, 0.0)
    z2 = complex(0.0, 0.0)
    if ca == complex(0.0, 0.0):
        ierr = 1
        return z1, z2, ierr
    d = cb * cb - 4.0 * ca * cc
    s = cmath.sqrt(d)
    z1 = (-cb + s) / (2.0 * ca)
    z2 = (-cb - s) / (2.0 * ca)
    return z1, z2, ierr


def cdft(n, x, xf, isign):
    pi2 = 6.283185307
    for k in range(0, n):
        s = complex(0.0, 0.0)
        for j in range(0, n):
            ang = float(isign) * pi2 * float((j * k) % n) / float(n)
            s = s + x[j] * complex(math.cos(ang), math.sin(ang))
        if isign > 0:
            s = s / float(n)
        xf[k] = s


def cpower(n, x):
    def zsq(z):
        return z.real**2 + z.imag**2

    total = 0.0
    for i in range(1, n + 1):
        total = total + zsq(complex(x[i - 1]))
    return total


def qstats(nc, nx, ireset):
    nc = _qstat.ncall
    nx = _qstat.ncplx
    if ireset != 0:
        _qstat.ncall = 0
        _qstat.ncplx = 0
    return nc, nx
