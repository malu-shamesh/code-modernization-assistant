"""Python port of WORDS (word counting and extraction)."""

from __future__ import annotations

_DELIM = " ,.;:!?"


def wordct(text, nwords, maxlen):
    nwords = 0
    maxlen = 0
    curlen = 0
    inword = False
    for ch in text:
        if ch in _DELIM:
            inword = False
            curlen = 0
        else:
            if not inword:
                inword = True
                nwords += 1
            curlen += 1
            if curlen > maxlen:
                maxlen = curlen
    return nwords, maxlen


def getwrd(text, k, word, wlen):
    n = 0
    inword = False
    start = 0
    for i in range(1, len(text) + 1):
        if text[i - 1] in _DELIM:
            if inword and n == k:
                return text[start - 1 : i - 1][:20].ljust(20), i - start
            inword = False
        elif not inword:
            inword = True
            n += 1
            start = i
    if inword and n == k:
        return text[start - 1 :][:20].ljust(20), len(text) - start + 1
    return " " * 20, 0


def cntsub(text, pat):
    np_ = 0
    for i in range(len(pat), 0, -1):
        if pat[i - 1] != " ":
            np_ = i
            break
    ncnt = 0
    if np_ == 0:
        return ncnt
    pos = 1
    while pos + np_ - 1 <= len(text):
        k = text[pos - 1 :].find(pat[:np_]) + 1
        if k == 0:
            break
        ncnt += 1
        pos = pos + k - 1 + np_
    return ncnt
