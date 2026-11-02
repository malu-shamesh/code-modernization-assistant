"""Hand-built ParsedModule objects (no dependency on the parsing package)."""

from __future__ import annotations

from collections.abc import Iterable

from modernizer.features import FEATURES
from modernizer.models import FeatureHit, ModuleMetrics, ParsedModule


def make_module(
    hit_ids: Iterable[str] = (),
    *,
    sloc: int = 50,
    cyclomatic: int = 1,
    max_nesting: int = 0,
    has_syntax_errors: bool = False,
    path: str = "m.f",
) -> ParsedModule:
    """Build a module whose metrics are consistent with its hits."""
    hits = [FeatureHit(fid, i + 1, i + 1, f"snippet {fid}") for i, fid in enumerate(hit_ids)]
    metrics = ModuleMetrics(
        loc=sloc,
        sloc=sloc,
        max_nesting=max_nesting,
        cyclomatic=cyclomatic,
        n_feature_hits=len(hits),
        distinct_features=len({h.feature_id for h in hits}),
        weighted_feature_score=sum(FEATURES[h.feature_id].weight for h in hits),
        semantic_hits=sum(1 for h in hits if FEATURES[h.feature_id].severity == "semantic"),
        has_syntax_errors=has_syntax_errors,
    )
    return ParsedModule(path=path, source="", units=[], hits=hits, metrics=metrics)


def mechanical_module() -> ParsedModule:
    """Ten purely mechanical hits in ten lines."""
    return make_module(["double_precision"] * 10, sloc=10)


def tangled_module() -> ParsedModule:
    """Long module with COMMON, GOTO webs and arithmetic IF, but no forced-frontier feature."""
    hits = ["common_block"] * 4 + ["goto_unconditional"] * 12 + ["if_arithmetic"] * 6
    hits += ["format_statement"] * 6 + ["do_labeled"] * 10 + ["double_precision"] * 20
    return make_module(hits, sloc=300, cyclomatic=45, max_nesting=5)


def small_routine() -> ParsedModule:
    """A 15 line mechanical routine: typed arguments, one labeled DO, one MOD call."""
    hits = [
        "double_precision",
        "do_labeled",
        "intrinsic_specific",
        "io_list_directed",
        "intrinsic_semantics",
        "implicit_statement",
    ]
    return make_module(hits, sloc=15, cyclomatic=4, max_nesting=1, path="small.f")


def medium_array_routine() -> ParsedModule:
    """A 60 line array routine with arithmetic IF, labeled DOs and a pair of 2-D arrays."""
    hits = ["do_labeled"] * 4 + ["if_arithmetic"] * 2 + ["array_multidim"] * 2
    hits += ["array_adjustable", "implicit_typing", "double_precision", "data_statement"]
    hits += ["intrinsic_specific"] * 2 + ["logical_dotted", "arg_output", "goto_unconditional"]
    hits += ["fixed_form_source"]
    return make_module(hits, sloc=60, cyclomatic=18, max_nesting=3, path="medium.f")


def large_goto_module() -> ParsedModule:
    """A 250 line module built from GOTO webs, COMMON and EQUIVALENCE, formatted I/O."""
    hits = ["common_block"] * 6 + ["equivalence"] * 3 + ["goto_unconditional"] * 10
    hits += ["goto_computed"] * 2 + ["if_arithmetic"] * 5 + ["do_shared_terminal"] * 2
    hits += ["format_statement"] * 8 + ["io_formatted"] * 6 + ["array_multidim"] * 6
    hits += ["implicit_typing"] * 6 + ["data_statement"] * 6 + ["do_labeled"] * 12
    hits += ["double_precision"] * 10 + ["intrinsic_specific"] * 10 + ["arg_output"] * 4
    hits += ["save_statement"] * 2 + ["fixed_form_source"] * 10
    return make_module(hits, sloc=250, cyclomatic=70, max_nesting=5, path="large.f")
