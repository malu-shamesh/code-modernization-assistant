"""Reference port of HISTO, PRTHST and the PROGRAM HISTDR (histogram with a text report)."""

from __future__ import annotations

import math

import numpy as np


def _nint(x: float) -> int:
    # half away from zero, unlike Python's round()
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def histo(
    n: int, x: np.ndarray, xmin: float, xmax: float, nb: int, icnt: np.ndarray, nout: int
) -> int:
    if nb < 1 or xmax <= xmin:
        nout = -1
        return nout
    w = (xmax - xmin) / float(nb)
    nout = 0
    for k in range(1, nb + 1):
        icnt[k - 1] = 0
    for i in range(1, n + 1):
        if x[i - 1] < xmin or x[i - 1] > xmax:
            nout += 1
        else:
            k = int((x[i - 1] - xmin) / w) + 1
            if k > nb:
                k = nb
            icnt[k - 1] += 1
    return nout


def prthst(nb: int, icnt: np.ndarray, xmin: float, xmax: float, nout: int) -> None:
    if nb < 1:
        print(" NO BINS")
        return
    w = (xmax - xmin) / float(nb)
    maxc = 0
    ntot = 0
    for k in range(1, nb + 1):
        maxc = max(maxc, int(icnt[k - 1]))
        ntot += int(icnt[k - 1])
    print(" BIN     LOW    HIGH  |          |COUNT")
    for k in range(1, nb + 1):
        count = int(icnt[k - 1])
        length = 0
        if maxc > 0:
            length = _nint(10.0 * float(count) / float(maxc))
        bar = ("*" * length).ljust(10)
        xlo = xmin + float(k - 1) * w
        print(f" {k:3d}{xlo:8.2f}{xlo + w:8.2f}  |{bar}|{count:5d}")
    print(f" TOTAL={ntot:6d}  OUTSIDE ={nout:5d}")


def main() -> None:
    icnt = np.zeros(5, dtype=np.int64)
    x = np.array([0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 9.0, 10.0, 11.0], dtype=np.float32)
    nout = histo(12, x, 0.0, 10.0, 5, icnt, 0)
    prthst(5, icnt, 0.0, 10.0, nout)


if __name__ == "__main__":
    main()
