"""Reference port of legacy.f (trapezoid and Simpson quadrature)."""

from __future__ import annotations


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def trapz(f, a, b, n):
    h = (b - a) / float(n)
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s = s + f(a + float(i) * h)
    return s * h


def simpsn(f, a, b, n):
    if 2 * _idiv(n, 2) != n:
        n = n + 1
    h = (b - a) / float(n)
    s = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 1:
            s = s + 4.0 * f(a + float(i) * h)
        else:
            s = s + 2.0 * f(a + float(i) * h)
    return s * h / 3.0, n


def quadex(f, a, b, tol, nmax, val, nused):
    n = 2
    old, n = simpsn(f, a, b, n)
    while True:
        n = 2 * n
        if n > nmax:
            nused = 0
            val = old
            return val, nused
        val, n = simpsn(f, a, b, n)
        if abs(val - old) <= tol:
            nused = n
            return val, nused
        old = val
