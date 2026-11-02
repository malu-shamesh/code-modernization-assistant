"""Benchmark and evaluation harness: honest numbers for routing and conversion quality.

Every figure in a report comes from a run: pass rates from the pipeline and from held-out
tests, tokens and cost from the OpenTelemetry meter (cross-checked), comparisons from paired
statistics. ``OracleLLM`` supports dry runs of the plumbing and nothing else.
"""

from __future__ import annotations

from .accounting import EvaluationConsistencyError
from .calibration import calibrate_open_threshold
from .compare import compare_results
from .corpus import BenchmarkCase, CorpusError, corpus_summary, load_corpus, split_calibration
from .oracle import OracleLLM, oracle_model_set
from .policies import PolicySpec, default_policies
from .results import (
    Comparison,
    EvaluationReport,
    ModuleRow,
    PassComparison,
    PolicyResult,
    TokenCounts,
)
from .runner import run_evaluation
from .stats import paired_counts, reduction_pct, sign_test_pvalue, wilson_interval

__all__ = [
    "BenchmarkCase",
    "Comparison",
    "CorpusError",
    "EvaluationConsistencyError",
    "EvaluationReport",
    "ModuleRow",
    "OracleLLM",
    "PassComparison",
    "PolicyResult",
    "PolicySpec",
    "TokenCounts",
    "calibrate_open_threshold",
    "compare_results",
    "corpus_summary",
    "default_policies",
    "load_corpus",
    "oracle_model_set",
    "paired_counts",
    "reduction_pct",
    "run_evaluation",
    "sign_test_pvalue",
    "split_calibration",
    "wilson_interval",
]
