"""Hand-written Python port of legacy.f (labelled-state-machine expression lexer)."""

from __future__ import annotations

_OPERATORS = "+-*/^=(),"


def iclass(c):
    c = (c + " ")[:1]
    cls = 0
    if "0" <= c <= "9":
        cls = 1
    if "A" <= c <= "Z":
        cls = 2
    if "a" <= c <= "z":
        cls = 2
    if c in _OPERATORS:
        cls = 3
    if c == ".":
        cls = 4
    return cls


def token(line, maxtok, kind, istart, iend, ntok, ierr):
    ntok = 0
    n = len(line)
    while n > 0 and line[n - 1] == " ":
        n -= 1
    i = 1
    while i <= n:
        if line[i - 1] == " ":
            i += 1
            continue
        if ntok >= maxtok:
            return ntok, -1
        cls = iclass(line[i - 1])
        if cls == 0:
            return ntok, i
        if cls == 4 and (i == n or iclass(line[i]) != 1):
            return ntok, i
        ntok += 1
        istart[ntok - 1] = i
        if cls == 3:
            kind[ntok - 1] = 4
            iend[ntok - 1] = i
            i += 1
            continue
        j = i + 1
        if cls == 2:
            kind[ntok - 1] = 3
            while j <= n and iclass(line[j - 1]) in (1, 2):
                j += 1
        else:
            kind[ntok - 1] = 1 if cls == 1 else 2
            if cls == 1:
                while j <= n and iclass(line[j - 1]) == 1:
                    j += 1
            if cls == 4 or (j <= n and iclass(line[j - 1]) == 4):
                kind[ntok - 1] = 2
                if cls == 1:
                    j += 1
                while j <= n and iclass(line[j - 1]) == 1:
                    j += 1
        iend[ntok - 1] = j - 1
        i = j
    return ntok, 0


def tokint(line, i1, i2, ival):
    ival = 0
    for i in range(i1, i2 + 1):
        ival = ival * 10 + ord(line[i - 1]) - ord("0")
    return ival
