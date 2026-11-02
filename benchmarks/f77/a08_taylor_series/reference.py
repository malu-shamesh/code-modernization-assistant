"""Reference port of legacy.f (Taylor series with convergence tests)."""

from __future__ import annotations

import math

TWOPI = 6.28318530717958648
HALFPI = 1.57079632679489662


def texp(x, tol, nterms):
    term = 1.0
    total = 1.0
    nterms = 1
    for k in range(1, 201):
        term = term * x / float(k)
        total = total + term
        nterms = nterms + 1
        if abs(term) < tol * abs(total):
            break
    return total, nterms


def tsin(x, eps, nterms):
    y = math.fmod(x, TWOPI)
    term = y
    total = y
    nterms = 1
    for k in range(1, 101):
        term = -term * y * y / float((2 * k) * (2 * k + 1))
        total = total + term
        nterms = nterms + 1
        if abs(term) < eps:
            break
    return total, nterms


def tcos(x, eps, nterms):
    result, nterms = tsin(x + HALFPI, eps, nterms)
    return result, nterms


def exptab(x0, x1, dx, tol, tab, n):
    n = 0
    x = x0
    trips = max(0, int((x1 - x0 + dx) / dx))
    for _ in range(trips):
        n = n + 1
        tab[n - 1] = texp(x, tol, 0)[0]
        x = x + dx
    return n
