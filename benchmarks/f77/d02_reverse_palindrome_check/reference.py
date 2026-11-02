"""Python port of STRUTL (string reversal and palindrome test)."""

from __future__ import annotations


def lastnb(s):
    for i in range(len(s), 0, -1):
        if s[i - 1] != " ":
            return i
    return 0


def isalnm(c):
    ic = ord((c + " ")[0])
    return (48 <= ic <= 57) or (65 <= ic <= 90) or (97 <= ic <= 122)


def foldup(c):
    c = (c + " ")[0]
    ic = ord(c)
    if 97 <= ic <= 122:
        ic -= 32
    return chr(ic)


def revers(text, res):
    res = [" "] * 40
    n = min(lastnb(text), 40)
    for i in range(1, n + 1):
        res[i - 1] = text[n - i]
    return "".join(res)


def ispaln(text):
    i = 1
    j = len(text)
    while True:
        if i >= j:
            return True
        if not isalnm(text[i - 1]):
            i += 1
            continue
        if not isalnm(text[j - 1]):
            j -= 1
            continue
        if foldup(text[i - 1]) != foldup(text[j - 1]):
            return False
        i += 1
        j -= 1
