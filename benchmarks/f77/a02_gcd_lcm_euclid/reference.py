"""Reference port of legacy.f (gcd, lcm and related integer utilities)."""

from __future__ import annotations

import math


def _idiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _mod(a: int, p: int) -> int:
    return int(math.fmod(a, p))


def igcd(m, n):
    a = abs(m)
    b = abs(n)
    while b != 0:
        a, b = b, _mod(a, b)
    return a


def ilcm(m, n):
    if m == 0 or n == 0:
        return 0
    g = igcd(m, n)
    return abs(_idiv(m, g) * n)


def igcdv(n, ia):
    result = 0
    for i in range(1, n + 1):
        result = igcd(result, ia[i - 1])
        if result == 1:
            break
    return result


def imodp(a, p):
    r = _mod(a, p)
    if r < 0:
        r = r + p
    return r


def reduce(num, den):
    g = igcd(num, den)
    if g == 0:
        return num, den
    num = _idiv(num, g)
    den = _idiv(den, g)
    if den < 0:
        num = -num
        den = -den
    return num, den
