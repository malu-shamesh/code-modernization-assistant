"""Transparent difficulty score for a parsed Fortran 77 module.

The score estimates how likely a cheaper model is to get a conversion wrong. It is a weighted sum
of saturating (log-scaled) signals squashed into [0, 1):

    x = w.density       * min(weighted_feature_score / max(sloc, 1), w.density_cap)
      + w.volume        * ln(1 + weighted_feature_score)
      + w.semantic      * ln(1 + semantic_hits)
      + w.tangle        * ln(1 + tangle_hits)
      + w.size          * ln(1 + sloc / 100)
      + w.cyclomatic    * ln(1 + (cyclomatic - 1) / 10)
      + w.nesting       * ln(1 + max(max_nesting - 2, 0))
      + w.syntax_errors * [has_syntax_errors]

    score = 1 - exp(-x)

``tangle_hits`` counts the constructs that make control flow or storage non-local (GOTO in all
its forms, arithmetic IF, shared DO terminals, COMMON, EQUIVALENCE, ENTRY, alternate RETURN,
sequence association). Those are the hazards that grow superlinearly: one GOTO is a local
rewrite, a web of forty is a state machine the model has to reconstruct.

All terms are non-negative and non-decreasing in their input, so adding a hit, or raising
cyclomatic complexity, nesting or the weighted feature total, never lowers the score. The
density term is the one place a ratio appears: padding a module with clean lines while keeping
its hits fixed dilutes the density and can lower the score slightly, which is intended (the
module really is less legacy-dense). Size growth at constant density never lowers the score.

Semantic hits are counted in absolute terms rather than as a share of all hits, because a share
would fall when a mechanical hit is added.

Calibration. The default weights are set by structural reasoning on typical F77 code, not fitted
to outcomes: a 20-40 line routine with a few mechanical hits and simple loops lands near 0.1-0.2,
a 60 line array routine with arithmetic IF and labeled DOs near 0.5, and a few hundred lines of
GOTO/COMMON/EQUIVALENCE near 0.85-0.95, so scores spread over the unit interval instead of
saturating. Any fixed threshold on the score is only a starting point: thresholds are meant to be
calibrated per corpus, either to a target open share for a batch (``Router.calibrated_for_batch``)
or to observed pass rates (``calibrate_from_outcomes``), never hard-coded.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from modernizer.models import ParsedModule

# Constructs that make control flow or storage non-local (see the module docstring).
TANGLE_FEATURES: frozenset[str] = frozenset(
    {
        "goto_unconditional",
        "goto_computed",
        "goto_assigned",
        "if_arithmetic",
        "do_shared_terminal",
        "alternate_return",
        "entry_statement",
        "common_block",
        "equivalence",
        "sequence_association",
    }
)


@dataclass(frozen=True)
class DifficultyWeights:
    """Coefficients of the difficulty formula (see the module docstring)."""

    density: float = 0.06
    density_cap: float = 3.0
    volume: float = 0.02
    semantic: float = 0.05
    tangle: float = 0.28
    size: float = 0.3
    cyclomatic: float = 0.1
    nesting: float = 0.08
    syntax_errors: float = 1.5

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"weight {field.name} must be finite and non-negative, got {value}"
                )


DEFAULT_WEIGHTS = DifficultyWeights()


def _terms(module: ParsedModule, w: DifficultyWeights) -> list[tuple[float, str]]:
    """Return (contribution to x, human-readable reason) for every signal."""
    m = module.metrics
    weighted = max(m.weighted_feature_score, 0.0)
    tangle_hits = sum(1 for h in module.hits if h.feature_id in TANGLE_FEATURES)
    density = weighted / max(m.sloc, 1)

    return [
        (
            w.density * min(density, w.density_cap),
            f"legacy construct density {density:.2f} weighted points per line",
        ),
        (
            w.volume * math.log1p(weighted),
            f"{m.n_feature_hits} legacy construct hits ({m.distinct_features} distinct, "
            f"weighted total {weighted:.1f})",
        ),
        (
            w.semantic * math.log1p(max(m.semantic_hits, 0)),
            f"{m.semantic_hits} semantic hits (behaviour can change silently)",
        ),
        (
            w.tangle * math.log1p(tangle_hits),
            f"{tangle_hits} control-flow or storage tangles (GOTO, arithmetic IF, COMMON ...)",
        ),
        (w.size * math.log1p(max(m.sloc, 0) / 100), f"{m.sloc} source lines"),
        (
            w.cyclomatic * math.log1p(max(m.cyclomatic - 1, 0) / 10),
            f"cyclomatic complexity {m.cyclomatic}",
        ),
        (
            w.nesting * math.log1p(max(m.max_nesting - 2, 0)),
            f"nesting depth {m.max_nesting}",
        ),
        (
            w.syntax_errors if m.has_syntax_errors else 0.0,
            "syntax errors beyond known legacy syntax",
        ),
    ]


def difficulty_score(module: ParsedModule, weights: DifficultyWeights = DEFAULT_WEIGHTS) -> float:
    """Difficulty in [0, 1]. Pure and deterministic."""
    x = sum(value for value, _ in _terms(module, weights))
    return min(max(1.0 - math.exp(-x), 0.0), 1.0)


def explain_difficulty(
    module: ParsedModule, weights: DifficultyWeights = DEFAULT_WEIGHTS, *, min_share: float = 0.02
) -> list[str]:
    """Reasons behind the score, largest contribution first.

    Signals contributing less than ``min_share`` of the total are omitted. Each reason carries
    its additive contribution to the pre-squash sum, e.g. ``"... (+0.35)"``.
    """
    terms = _terms(module, weights)
    total = sum(value for value, _ in terms)
    if total <= 0.0:
        return ["no difficulty signals"]
    ranked = sorted(
        (t for t in terms if t[0] > 0.0 and t[0] >= min_share * total), key=lambda t: -t[0]
    )
    return [f"{reason} (+{value:.2f})" for value, reason in ranked]
