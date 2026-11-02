"""Reference port of legacy.f (factorials, Fibonacci numbers, binomial coefficients)."""

from __future__ import annotations

MAXFAC = 12


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def ifact(n):
    if n < 0 or n > MAXFAC:
        return -1
    p = 1
    for k in range(2, n + 1):
        p = p * k
    return p


def dfact(n):
    p = 1.0
    for k in range(2, n + 1):
        p = p * float(k)
    return p


def fibtab(n, f):
    if n < 1:
        return
    f[0] = 1
    if n == 1:
        return
    f[1] = 1
    for i in range(3, n + 1):
        f[i - 1] = f[i - 2] + f[i - 3]


def ibinom(n, k):
    if k < 0 or k > n:
        return 0
    kk = min(k, n - k)
    r = 1
    for i in range(1, kk + 1):
        r = _idiv(r * (n - kk + i), i)
    return r
