"""Python port of the strided level 1 vector kernels (DAXPY, DCOPY, DSCAL, DDOT)."""

from __future__ import annotations

import numpy as np


def _start(n: int, inc: int) -> int:
    """1-based index of the first element touched for stride ``inc``."""
    return (-n + 1) * inc + 1 if inc < 0 else 1


def daxpy(n: int, da: float, dx: np.ndarray, incx: int, dy: np.ndarray, incy: int) -> None:
    if n <= 0:
        return
    if da == 0.0:
        return
    ix = _start(n, incx)
    iy = _start(n, incy)
    for _ in range(n):
        dy[iy - 1] = dy[iy - 1] + da * dx[ix - 1]
        ix += incx
        iy += incy


def dcopy(n: int, dx: np.ndarray, incx: int, dy: np.ndarray, incy: int) -> None:
    if n <= 0:
        return
    ix = _start(n, incx)
    iy = _start(n, incy)
    for _ in range(n):
        dy[iy - 1] = dx[ix - 1]
        ix += incx
        iy += incy


def dscal(n: int, da: float, dx: np.ndarray, incx: int) -> None:
    if n <= 0 or incx <= 0:
        return
    if incx != 1:
        for i in range(1, n * incx + 1, incx):
            dx[i - 1] = da * dx[i - 1]
        return
    m = n % 5
    for i in range(1, m + 1):
        dx[i - 1] = da * dx[i - 1]
    if n < 5:
        return
    for i in range(m + 1, n + 1, 5):
        for k in range(5):
            dx[i + k - 1] = da * dx[i + k - 1]


def ddot(n: int, dx: np.ndarray, incx: int, dy: np.ndarray, incy: int) -> float:
    dtemp = 0.0
    if n <= 0:
        return 0.0
    if incx == 1 and incy == 1:
        m = n % 5
        for i in range(1, m + 1):
            dtemp = dtemp + dx[i - 1] * dy[i - 1]
        if n >= 5:
            for i in range(m + 1, n + 1, 5):
                dtemp = (
                    dtemp
                    + dx[i - 1] * dy[i - 1]
                    + dx[i] * dy[i]
                    + dx[i + 1] * dy[i + 1]
                    + dx[i + 2] * dy[i + 2]
                    + dx[i + 3] * dy[i + 3]
                )
        return float(dtemp)
    ix = _start(n, incx)
    iy = _start(n, incy)
    for _ in range(n):
        dtemp = dtemp + dx[ix - 1] * dy[iy - 1]
        ix += incx
        iy += incy
    return float(dtemp)
