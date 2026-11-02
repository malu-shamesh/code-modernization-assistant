"""Python port of NUMWRD (English words for an integer)."""

from __future__ import annotations

_ONES = (
    "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE "
    "THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN"
).split()
_TENS = "TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()  # index t - 2
_SCALE = ("", "THOUSAND", "MILLION")
_BUFLEN = 120


def _mod(a: int, p: int) -> int:
    r = abs(a) % abs(p)
    return -r if a < 0 else r


def lastnb(s):
    for i in range(len(s), 0, -1):
        if s[i - 1] != " ":
            return i
    return 0


def addwrd(buf, pos, word):
    n = lastnb(word)
    if n == 0:
        return buf, pos
    need = n + 1 if pos > 1 else n
    if pos + need - 1 > _BUFLEN:
        return buf, pos
    chars = list(buf[:_BUFLEN].ljust(_BUFLEN))
    if pos > 1:
        chars[pos - 1] = " "
        pos += 1
    chars[pos - 1 : pos - 1 + n] = word[:n]
    pos += n
    return "".join(chars), pos


def numwrd(n, words, nch, ierr):
    words = " " * _BUFLEN
    if abs(n) > 999999999:
        return words, 0, 1
    pos = 1
    if n < 0:
        words, pos = addwrd(words, pos, "MINUS")
    if n == 0:
        words, pos = addwrd(words, pos, _ONES[0])
    m = abs(n)
    g = [0, 0, 0]
    for i in range(3):
        g[i] = _mod(m, 1000)
        m = m // 1000
    for i in (2, 1, 0):
        if g[i] == 0:
            continue
        h = g[i] // 100
        r = _mod(g[i], 100)
        if h > 0:
            words, pos = addwrd(words, pos, _ONES[h])
            words, pos = addwrd(words, pos, "HUNDRED")
        if r >= 20:
            t = r // 10
            u = _mod(r, 10)
            if u == 0:
                word = _TENS[t - 2]
            else:
                word = _TENS[t - 2] + "-" + _ONES[u]
            words, pos = addwrd(words, pos, word)
        elif r > 0:
            words, pos = addwrd(words, pos, _ONES[r])
        if i > 0:
            words, pos = addwrd(words, pos, _SCALE[i])
    return words, pos - 1, 0
