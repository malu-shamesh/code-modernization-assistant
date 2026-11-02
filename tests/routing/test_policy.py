from __future__ import annotations

import threading

import pytest

from modernizer.models import RoutingDecision, Tier
from modernizer.routing import FORCE_FRONTIER_FEATURES, Router, difficulty_score
from tests.routing.helpers import make_module, mechanical_module, tangled_module

FRONTIER = "frontier-model"
OPEN = "open-model"


def make_router(**kwargs: object) -> Router:
    return Router(FRONTIER, OPEN, **kwargs)  # type: ignore[arg-type]


def test_complexity_routes_easy_open_and_hard_frontier() -> None:
    router = make_router()
    easy = router.route(mechanical_module())
    hard = router.route(tangled_module())
    assert (easy.tier, easy.model) == (Tier.OPEN, OPEN)
    assert (hard.tier, hard.model) == (Tier.FRONTIER, FRONTIER)
    assert easy.policy == "complexity" and easy.threshold == 0.35
    assert easy.difficulty == difficulty_score(mechanical_module())
    assert easy.reasons and hard.reasons


def test_threshold_boundary_is_inclusive_for_open() -> None:
    module = make_module(["double_precision"] * 5, sloc=40)
    score = difficulty_score(module)
    assert make_router(threshold=score).route(module).tier is Tier.OPEN
    assert make_router(threshold=score - 1e-9).route(module).tier is Tier.FRONTIER


def test_always_policies_ignore_difficulty_and_forced_rule() -> None:
    for module in (mechanical_module(), tangled_module()):
        frontier = make_router(policy="always_frontier").route(module)
        open_ = make_router(policy="always_open").route(module)
        assert (frontier.tier, frontier.model, frontier.threshold) == (
            Tier.FRONTIER,
            FRONTIER,
            None,
        )
        assert (open_.tier, open_.model, open_.threshold) == (Tier.OPEN, OPEN, None)
        assert frontier.policy == "always_frontier" and open_.policy == "always_open"
        assert frontier.reasons and open_.reasons


def test_default_forced_set_is_the_documented_fortran_set() -> None:
    assert FORCE_FRONTIER_FEATURES == frozenset(
        {
            "equivalence",
            "goto_assigned",
            "entry_statement",
            "alternate_return",
            "sequence_association",
        }
    )
    assert make_router().force_frontier_features == FORCE_FRONTIER_FEATURES


@pytest.mark.parametrize("fid", sorted(FORCE_FRONTIER_FEATURES))
def test_forced_frontier_features_override_low_difficulty(fid: str) -> None:
    module = make_module([fid], sloc=500)
    router = make_router(threshold=1.0)
    decision = router.route(module)
    assert decision.tier is Tier.FRONTIER
    assert decision.difficulty <= 1.0
    assert any("forced" in r and fid in r for r in decision.reasons)


def test_forced_rule_can_be_disabled() -> None:
    module = make_module(["equivalence"], sloc=500)
    router = make_router(threshold=1.0, force_frontier_on_features=False)
    assert router.route(module).tier is Tier.OPEN


def test_old_flag_name_is_gone() -> None:
    with pytest.raises(TypeError):
        make_router(force_frontier_if_semantic_bytes=False)


def test_other_semantic_features_are_not_forced() -> None:
    module = make_module(["common_block", "goto_computed"], sloc=500)
    assert make_router(threshold=1.0).route(module).tier is Tier.OPEN


def test_forced_feature_set_is_configurable() -> None:
    module = make_module(["common_block"], sloc=500)
    router = make_router(threshold=1.0, force_frontier_features=frozenset({"common_block"}))
    assert router.route(module).tier is Tier.FRONTIER
    assert make_router(threshold=1.0).route(module).tier is Tier.OPEN
    empty = make_router(threshold=1.0, force_frontier_features=frozenset())
    assert empty.route(make_module(["equivalence"])).tier is Tier.OPEN


def test_unknown_forced_feature_id_rejected() -> None:
    with pytest.raises(ValueError, match="not_a_feature"):
        make_router(force_frontier_features=frozenset({"not_a_feature"}))


def test_escalate_open_goes_to_frontier_keeping_difficulty() -> None:
    router = make_router()
    decision = router.route(mechanical_module())
    escalated = router.escalate(decision)
    assert escalated is not None
    assert escalated.tier is Tier.FRONTIER and escalated.model == FRONTIER
    assert escalated.difficulty == decision.difficulty
    assert escalated.policy == decision.policy
    assert escalated.reasons[-1] == "escalated after failures"
    assert escalated.reasons[:-1] == decision.reasons
    assert decision.tier is Tier.OPEN and "escalated after failures" not in decision.reasons


def test_escalate_frontier_returns_none() -> None:
    router = make_router()
    assert router.escalate(router.route(tangled_module())) is None
    assert (
        make_router(policy="always_frontier").escalate(
            make_router(policy="always_frontier").route(mechanical_module())
        )
        is None
    )


def test_escalate_works_for_always_open_baseline() -> None:
    router = make_router(policy="always_open")
    escalated = router.escalate(router.route(mechanical_module()))
    assert escalated is not None and escalated.tier is Tier.FRONTIER


def test_escalate_accepts_foreign_decision() -> None:
    foreign = RoutingDecision(Tier.OPEN, "x", 0.2, 0.35, "complexity", [])
    escalated = make_router().escalate(foreign)
    assert escalated is not None and escalated.model == FRONTIER


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError):
        make_router(policy="cheapest")
    with pytest.raises(ValueError):
        make_router(threshold=float("nan"))


def test_router_is_safe_to_share_across_threads() -> None:
    router = make_router()
    modules = [mechanical_module(), tangled_module()]
    expected = [router.route(m).tier for m in modules]
    failures: list[str] = []

    def worker() -> None:
        for _ in range(200):
            if [router.route(m).tier for m in modules] != expected:
                failures.append("mismatch")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures
