from __future__ import annotations

import dataclasses
import itertools

import pytest

from modernizer.features import FEATURE_IDS
from modernizer.routing import DifficultyWeights, difficulty_score, explain_difficulty
from tests.routing.helpers import (
    large_goto_module,
    make_module,
    mechanical_module,
    medium_array_routine,
    small_routine,
    tangled_module,
)


def test_mechanical_module_is_clearly_easy() -> None:
    assert difficulty_score(mechanical_module()) < 0.30


def test_tangled_module_is_clearly_hard() -> None:
    assert difficulty_score(tangled_module()) > 0.6


def test_realistic_fixtures_are_ordered_and_banded() -> None:
    small = difficulty_score(small_routine())
    medium = difficulty_score(medium_array_routine())
    large = difficulty_score(large_goto_module())
    assert small < medium < large
    assert small < 0.25
    assert 0.35 <= medium <= 0.65
    assert large > 0.8


def test_scores_spread_instead_of_saturating() -> None:
    trivial = make_module(["double_precision"], sloc=12, cyclomatic=2)
    giant = make_module(
        ["goto_unconditional"] * 60 + ["common_block"] * 20,
        sloc=1500,
        cyclomatic=300,
        max_nesting=7,
    )
    assert difficulty_score(trivial) < 0.1
    assert 0.9 < difficulty_score(giant) < 1.0
    ordinary = [
        difficulty_score(m) for m in (small_routine(), medium_array_routine(), large_goto_module())
    ]
    assert max(ordinary) - min(ordinary) > 0.5


def test_empty_module_scores_zero() -> None:
    assert difficulty_score(make_module([], sloc=0)) == 0.0


def test_score_is_deterministic_and_bounded() -> None:
    module = tangled_module()
    first = difficulty_score(module)
    assert difficulty_score(module) == first
    for fid in FEATURE_IDS:
        assert 0.0 <= difficulty_score(make_module([fid] * 7, sloc=3)) <= 1.0


def test_extreme_metrics_stay_in_unit_interval() -> None:
    module = make_module(
        ["goto_unconditional"] * 500,
        sloc=10,
        cyclomatic=10_000,
        max_nesting=200,
        has_syntax_errors=True,
    )
    assert 0.0 <= difficulty_score(module) <= 1.0


@pytest.mark.parametrize("fid", FEATURE_IDS)
def test_adding_a_hit_never_lowers_score(fid: str) -> None:
    base_hits: list[str] = []
    previous = difficulty_score(make_module(base_hits, sloc=40, cyclomatic=4, max_nesting=2))
    for _ in range(12):
        base_hits.append(fid)
        score = difficulty_score(make_module(base_hits, sloc=40, cyclomatic=4, max_nesting=2))
        assert score >= previous
        previous = score


def test_adding_a_mechanical_hit_to_a_semantic_module_never_lowers_score() -> None:
    hits = ["equivalence", "common_block"]
    previous = difficulty_score(make_module(hits, sloc=80))
    for _ in range(10):
        hits.append("double_precision")
        score = difficulty_score(make_module(hits, sloc=80))
        assert score >= previous
        previous = score


def test_increasing_complexity_never_lowers_score() -> None:
    hits = ["double_precision", "do_labeled", "integer_division"]
    previous = -1.0
    for cyclomatic, nesting in itertools.product(range(1, 60, 3), range(0, 10, 2)):
        score = difficulty_score(
            make_module(hits, sloc=100, cyclomatic=cyclomatic, max_nesting=nesting)
        )
        assert score >= difficulty_score(
            make_module(hits, sloc=100, cyclomatic=max(cyclomatic - 3, 1), max_nesting=nesting)
        )
        assert score >= difficulty_score(
            make_module(hits, sloc=100, cyclomatic=cyclomatic, max_nesting=max(nesting - 2, 0))
        )
        previous = score
    assert previous > difficulty_score(make_module(hits, sloc=100))


def test_increasing_size_at_constant_density_never_lowers_score() -> None:
    previous = -1.0
    for scale in range(1, 40):
        module = make_module(
            ["double_precision", "goto_unconditional"] * scale, sloc=20 * scale, cyclomatic=scale
        )
        score = difficulty_score(module)
        assert score >= previous
        previous = score


def test_syntax_errors_raise_score() -> None:
    clean = make_module(["double_precision"], sloc=20)
    broken = make_module(["double_precision"], sloc=20, has_syntax_errors=True)
    assert difficulty_score(broken) > difficulty_score(clean)


def test_tangles_outweigh_mechanical_hits_of_same_count() -> None:
    tangled = make_module(["goto_unconditional", "common_block"], sloc=60)
    plain = make_module(["double_precision"] * 2, sloc=60)
    assert difficulty_score(tangled) > difficulty_score(plain)


def test_tangle_weight_only_counts_tangle_features() -> None:
    only_tangle = DifficultyWeights(
        density=0,
        volume=0,
        semantic=0,
        tangle=1.0,
        size=0,
        cyclomatic=0,
        nesting=0,
        syntax_errors=0,
    )
    assert difficulty_score(make_module(["double_precision"] * 9), only_tangle) == 0.0
    assert difficulty_score(make_module(["equivalence"]), only_tangle) > 0.0


def test_custom_weights_change_score_and_defaults_are_frozen() -> None:
    module = tangled_module()
    zero = DifficultyWeights(
        density=0,
        volume=0,
        semantic=0,
        tangle=0,
        size=0,
        cyclomatic=0,
        nesting=0,
        syntax_errors=0,
    )
    assert difficulty_score(module, zero) == 0.0
    assert difficulty_score(module, DifficultyWeights(semantic=1.0)) > difficulty_score(module)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DifficultyWeights().size = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_weights_reject_negative_or_non_finite_values(bad: float) -> None:
    with pytest.raises(ValueError):
        DifficultyWeights(size=bad)


def test_explain_orders_by_contribution_and_names_signals() -> None:
    reasons = explain_difficulty(tangled_module())
    assert reasons
    values = [float(r.rsplit("(+", 1)[1].rstrip(")")) for r in reasons]
    assert values == sorted(values, reverse=True)
    assert any("tangles" in r for r in reasons)


def test_explain_for_trivial_module_is_non_empty() -> None:
    assert explain_difficulty(make_module([], sloc=0)) == ["no difficulty signals"]
    assert explain_difficulty(mechanical_module())


def test_explain_reports_syntax_errors() -> None:
    reasons = explain_difficulty(make_module([], sloc=10, has_syntax_errors=True))
    assert reasons[0].startswith("syntax errors")
