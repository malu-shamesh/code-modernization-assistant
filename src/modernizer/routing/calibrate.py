"""Pick routing thresholds from observed difficulty scores."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence


def _check_scores(scores: Sequence[float]) -> None:
    if any(math.isnan(s) for s in scores):
        raise ValueError("scores must not contain NaN")


def open_fraction(scores: Sequence[float], threshold: float) -> float:
    """Fraction of modules routed OPEN (score <= threshold). 0.0 for empty input."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s <= threshold) / len(scores)


def calibrate_threshold(scores: Sequence[float], target_open_fraction: float) -> float:
    """Threshold under which about ``target_open_fraction`` of the modules route OPEN.

    Ties make the exact fraction unreachable in general, so the candidate whose resulting open
    fraction is closest to the target wins; on an equal distance the lower threshold (fewer open
    modules) wins. A target of 0 returns a value just below the smallest score, so no module
    routes OPEN, and a target of 1 returns the largest score. Empty input returns 0.0.
    """
    if not 0.0 <= target_open_fraction <= 1.0:
        raise ValueError("target_open_fraction must be in [0, 1]")
    _check_scores(scores)
    if not scores:
        return 0.0

    ordered = sorted(scores)
    n = len(ordered)
    best = math.nextafter(ordered[0], -math.inf)
    best_gap = target_open_fraction
    for t in dict.fromkeys(ordered):
        gap = abs(bisect_right(ordered, t) / n - target_open_fraction)
        if gap < best_gap:
            best, best_gap = t, gap
    return best


def calibrate_from_outcomes(
    scores: Sequence[float], open_succeeded: Sequence[bool], min_pass_rate: float = 0.95
) -> float:
    """Largest observed score ``t`` such that modules with score <= t passed at >= min_pass_rate.

    ``open_succeeded[i]`` is whether the open model converted module ``i`` successfully; it must
    be measured by running the open model on every module of a calibration run, otherwise the
    pass rate is biased by earlier routing. Returns 0.0 if no threshold qualifies.

    Statistical caveat: with few modules the observed pass rate is a noisy estimate. Among 20
    modules, 19 passes already meets 0.95 but is statistically compatible with a true rate far
    lower, and taking the largest qualifying threshold picks the most optimistic cut among many
    that were tried. Treat the result as a point estimate, use a sizeable calibration set, and
    validate on held-out modules before trusting the quality claim.
    """
    if len(scores) != len(open_succeeded):
        raise ValueError("scores and open_succeeded must have the same length")
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be in [0, 1]")
    _check_scores(scores)

    ranked = sorted(zip(scores, open_succeeded, strict=True), key=lambda p: p[0])
    best = 0.0
    passed = 0
    for i, (score, ok) in enumerate(ranked):
        passed += bool(ok)
        last_of_tie = i + 1 == len(ranked) or ranked[i + 1][0] != score
        if last_of_tie and passed / (i + 1) >= min_pass_rate - 1e-12:
            best = score
    return best
