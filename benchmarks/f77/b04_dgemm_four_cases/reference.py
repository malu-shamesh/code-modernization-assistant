"""Python port of DGEMM (C := alpha*op(A)*op(B) + beta*C with leading dimensions)."""

from __future__ import annotations

import numpy as np


def _char1(c: str) -> str:
    return (c + " ")[:1]


def lsame(ca: str, cb: str) -> bool:
    ca = _char1(ca)
    cb = _char1(cb)
    if ca == cb:
        return True
    inta = ord(ca)
    intb = ord(cb)
    if 97 <= inta <= 122:
        inta = inta - 32
    if 97 <= intb <= 122:
        intb = intb - 32
    return inta == intb


def dgescl(m: int, n: int, beta: float, c: np.ndarray, ldc: int) -> None:
    if beta == 1.0:
        return
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if beta == 0.0:
                c[i - 1, j - 1] = 0.0
            else:
                c[i - 1, j - 1] = beta * c[i - 1, j - 1]


def dgemm(
    transa: str,
    transb: str,
    m: int,
    n: int,
    k: int,
    alpha: float,
    a: np.ndarray,
    lda: int,
    b: np.ndarray,
    ldb: int,
    beta: float,
    c: np.ndarray,
    ldc: int,
    info: int,
) -> int:
    nota = lsame(transa, "N")
    notb = lsame(transb, "N")
    nrowa = m if nota else k
    nrowb = k if notb else n
    info = 0
    if not nota and not lsame(transa, "C") and not lsame(transa, "T"):
        info = 1
    elif not notb and not lsame(transb, "C") and not lsame(transb, "T"):
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < max(1, nrowa):
        info = 8
    elif ldb < max(1, nrowb):
        info = 10
    elif ldc < max(1, m):
        info = 13
    if info != 0:
        return info
    if m == 0 or n == 0 or ((alpha == 0.0 or k == 0) and beta == 1.0):
        return info
    dgescl(m, n, beta, c, ldc)
    if alpha == 0.0 or k == 0:
        return info
    if notb:
        if nota:
            for j in range(1, n + 1):
                for l in range(1, k + 1):
                    if b[l - 1, j - 1] != 0.0:
                        temp = alpha * b[l - 1, j - 1]
                        for i in range(1, m + 1):
                            c[i - 1, j - 1] = c[i - 1, j - 1] + temp * a[i - 1, l - 1]
        else:
            for j in range(1, n + 1):
                for i in range(1, m + 1):
                    temp = 0.0
                    for l in range(1, k + 1):
                        temp = temp + a[l - 1, i - 1] * b[l - 1, j - 1]
                    c[i - 1, j - 1] = c[i - 1, j - 1] + alpha * temp
    else:
        if nota:
            for j in range(1, n + 1):
                for l in range(1, k + 1):
                    if b[j - 1, l - 1] != 0.0:
                        temp = alpha * b[j - 1, l - 1]
                        for i in range(1, m + 1):
                            c[i - 1, j - 1] = c[i - 1, j - 1] + temp * a[i - 1, l - 1]
        else:
            for j in range(1, n + 1):
                for i in range(1, m + 1):
                    temp = 0.0
                    for l in range(1, k + 1):
                        temp = temp + a[l - 1, i - 1] * b[j - 1, l - 1]
                    c[i - 1, j - 1] = c[i - 1, j - 1] + alpha * temp
    return info
