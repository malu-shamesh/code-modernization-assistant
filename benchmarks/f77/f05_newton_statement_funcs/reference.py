"""Reference port of legacy.f (Newton cube root with statement functions)."""

from __future__ import annotations

import sys


def _d1mach(i):
    """Double precision machine constants (PORT library numbering)."""
    info = sys.float_info
    if i == 1:
        return info.min
    if i == 2:
        return info.max
    if i == 3:
        return info.epsilon / 2.0
    if i == 4:
        return info.epsilon
    raise ValueError("d1mach index out of range")


def reltol():
    return 10.0 * _d1mach(4)


def cbrtnw(a, x, nmax, iter, ierr):
    def f(z):
        return z * z * z - a

    def df(z):
        return 3.0 * z * z

    tol = reltol() * max(1.0, abs(a))
    ierr = 0
    x = a
    if abs(a) < 1.0:
        x = 1.0
    for iter in range(1, nmax + 1):
        if abs(f(x)) <= tol:
            return x, iter, ierr
        x = x - f(x) / df(x)
    # a DO variable ends one step past the last trip; with zero trips it is the start value
    iter = max(nmax, 0) + 1
    ierr = 1
    return x, iter, ierr


def dcbrt(a):
    x, _iter, _ierr = cbrtnw(a, 0.0, 60, 0, 0)
    return x


def cbrtv(n, a, x):
    count = 0
    for i in range(1, n + 1):
        x[i - 1], _iter, ierr = cbrtnw(a[i - 1], x[i - 1], 60, 0, 0)
        if ierr != 0:
            count = count + 1
    return count
