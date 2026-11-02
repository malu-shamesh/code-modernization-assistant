"""Small, dependency-free statistics for comparing pass rates on a modest number of modules."""

from __future__ import annotations

import math
from collections.abc import Sequence

Z_95 = 1.959963984540054  # two-sided 95% standard normal quantile


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation it stays inside [0, 1] and behaves for small ``n`` and for
    rates of exactly 0 or 1. With ``n == 0`` nothing is known, so the interval is (0, 1).
    """
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test_pvalue(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test on discordant pairs (McNemar's exact test).

    ``only_a`` / ``only_b`` count the pairs where exactly one of the two systems succeeded. Under
    the null both directions are equally likely. No discordant pairs means no evidence: 1.0.
    """
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must not be negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def paired_counts(baseline: Sequence[bool], other: Sequence[bool]) -> tuple[int, int]:
    """``(baseline passes and other fails, other passes and baseline fails)``."""
    if len(baseline) != len(other):
        raise ValueError("paired outcomes must have the same length")
    only_baseline = sum(1 for a, b in zip(baseline, other, strict=True) if a and not b)
    only_other = sum(1 for a, b in zip(baseline, other, strict=True) if b and not a)
    return only_baseline, only_other


def reduction_pct(baseline: float, value: float) -> float | None:
    """Percentage by which ``value`` is below ``baseline`` (negative when it is higher).

    None when the baseline is zero, because a relative change from nothing is undefined.
    """
    if baseline <= 0:
        return None
    return (baseline - value) / baseline * 100.0
