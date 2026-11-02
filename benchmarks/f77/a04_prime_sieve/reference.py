"""Reference port of legacy.f (trial division and sieve of Eratosthenes)."""

from __future__ import annotations

import math

import numpy


def isprim(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if int(math.fmod(n, 2)) == 0:
        return False
    lim = int(math.sqrt(float(n)))
    for i in range(3, lim + 1, 2):
        if int(math.fmod(n, i)) == 0:
            return False
    return True


def sieve(n, isp, np):
    np = 0
    if n < 1:
        return np
    isp[0] = False
    for i in range(2, n + 1):
        isp[i - 1] = True
    for i in range(2, n + 1):
        if not isp[i - 1]:
            continue
        np = np + 1
        for j in range(i * i, n + 1, i):
            isp[j - 1] = False
    return np


def nprime(n):
    if n > 1000:
        return -1
    work = numpy.zeros(1000, dtype=bool)
    return sieve(n, work, 0)
