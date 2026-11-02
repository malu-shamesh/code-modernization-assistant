"""Reference port of legacy.f (Horner polynomial utilities)."""

from __future__ import annotations


def horner(n, c, x):
    p = c[n]
    for i in range(n, 0, -1):
        p = p * x + c[i - 1]
    return p


def polder(n, c, x, p, dp):
    p = c[n]
    dp = 0.0
    for i in range(n, 0, -1):
        dp = dp * x + p
        p = p * x + c[i - 1]
    return p, dp


def deflat(n, c, r, q, rem):
    q[n - 1] = c[n]
    for i in range(n - 1, 0, -1):
        q[i - 1] = c[i] + r * q[i]
    rem = c[0] + r * q[0]
    return rem
