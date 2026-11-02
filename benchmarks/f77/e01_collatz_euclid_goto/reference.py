"""Hand-written Python port of legacy.f (NUMTHY integer sequence routines)."""

from __future__ import annotations


def _idiv(a: int, b: int) -> int:
    """Fortran integer division: truncates toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a: int, p: int) -> int:
    """Fortran MOD: the result takes the sign of the dividend."""
    return a - _idiv(a, p) * p


def colltz(n, maxstp, nstep, ipeak, ierr):
    nstep = 0
    ierr = 0
    ipeak = n
    if n <= 0:
        return nstep, ipeak, 1
    k = n
    while k != 1:
        if nstep >= maxstp:
            return nstep, ipeak, 2
        if _mod(k, 2) != 0:
            k = 3 * k + 1
        else:
            k = _idiv(k, 2)
        nstep += 1
        if k > ipeak:
            ipeak = k
    return nstep, ipeak, ierr


def igcd(m, n):
    ia = abs(m)
    ib = abs(n)
    while ib != 0:
        ia, ib = ib, _mod(ia, ib)
    return ia


def ilcm(m, n):
    if m == 0 or n == 0:
        return 0
    return abs(_idiv(m, igcd(m, n)) * n)


def gcdseq(n, a, g, nused):
    g = 0
    nused = 0
    i = 1
    while i <= n:
        g = igcd(g, int(a[i - 1]))
        nused = i
        if g == 1:
            break
        i += 1
    return g, nused
