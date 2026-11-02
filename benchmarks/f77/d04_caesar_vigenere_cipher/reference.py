"""Python port of CIPHER (Caesar and Vigenere ciphers, letter statistics)."""

from __future__ import annotations

import numpy as np

_UPALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOALPH = "abcdefghijklmnopqrstuvwxyz"
_WIDTH = 60


def _mod(a: int, p: int) -> int:
    """Fortran MOD: the result takes the sign of the dividend."""
    r = abs(a) % abs(p)
    return -r if a < 0 else r


def _index(s: str, sub: str) -> int:
    return s.find(sub) + 1


def caesar(text, key, res):
    res = [" "] * _WIDTH
    k = _mod(key, 26)
    if k < 0:
        k += 26
    n = min(len(text), _WIDTH)
    for i in range(1, n + 1):
        ic = ord(text[i - 1])
        if 65 <= ic <= 90:
            ic = 65 + _mod(ic - 65 + k, 26)
        elif 97 <= ic <= 122:
            ic = 97 + _mod(ic - 97 + k, 26)
        res[i - 1] = chr(ic)
    return "".join(res)


def vigenr(text, keywd, idir, res):
    kshift = []
    for ch in keywd:
        k = _index(_UPALPH, ch)
        if k == 0:
            k = _index(_LOALPH, ch)
        if k > 0 and len(kshift) < 20:
            kshift.append(k - 1)
    res = list(text[:_WIDTH].ljust(_WIDTH))
    nk = len(kshift)
    if nk == 0:
        return "".join(res)
    j = 0
    for i in range(1, min(len(text), _WIDTH) + 1):
        k = _index(_UPALPH, text[i - 1])
        ibase = 65
        if k == 0:
            k = _index(_LOALPH, text[i - 1])
            ibase = 97
        if k > 0:
            j += 1
            if j > nk:
                j = 1
            k = _mod(k - 1 + idir * kshift[j - 1], 26)
            if k < 0:
                k += 26
            res[i - 1] = chr(ibase + k)
    return "".join(res)


def letfrq(text, ifreq):
    for i in range(26):
        ifreq[i] = 0
    for ch in text:
        ic = ord(ch)
        k = 0
        if 65 <= ic <= 90:
            k = ic - 64
        if 97 <= ic <= 122:
            k = ic - 96
        if k > 0:
            ifreq[k - 1] += 1


def mostfq(ifreq):
    ibest = 0
    result = 0
    for i in range(1, 27):
        if ifreq[i - 1] > ibest:
            ibest = int(ifreq[i - 1])
            result = i
    return result


def crack(text, key):
    ifreq = np.zeros(26, dtype=np.int64)
    letfrq(text, ifreq)
    m = mostfq(ifreq)
    if m == 0:
        return 0
    return _mod(m - 5 + 26, 26)
