from __future__ import annotations

import logging

import pytest

from modernizer.models import ParsedModule, Tier
from modernizer.routing import Router, batch_open_share, difficulty_score
from tests.routing.helpers import make_module

FRONTIER = "frontier-model"
OPEN = "open-model"


def make_router(**kwargs: object) -> Router:
    return Router(FRONTIER, OPEN, **kwargs)  # type: ignore[arg-type]


def graded_batch(n: int) -> list[ParsedModule]:
    """n modules with strictly increasing difficulty and no forced-frontier feature."""
    return [
        make_module(["do_labeled"] * (i % 7), sloc=20 + 15 * i, cyclomatic=1 + 2 * i, path=f"{i}.f")
        for i in range(n)
    ]


def open_paths(router: Router, modules: list[ParsedModule]) -> set[str]:
    return {m.path for m in modules if router.route(m).tier is Tier.OPEN}


@pytest.mark.parametrize("target", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_calibrated_share_matches_target_without_ties(target: float) -> None:
    modules = graded_batch(20)
    scores = [difficulty_score(m) for m in modules]
    assert len(set(scores)) == len(scores)
    calibrated = make_router().calibrated_for_batch(modules, target)
    assert batch_open_share(calibrated, modules) == pytest.approx(target)


def test_easiest_modules_go_open() -> None:
    modules = graded_batch(10)
    calibrated = make_router().calibrated_for_batch(modules, 0.3)
    easiest = {m.path for m in sorted(modules, key=difficulty_score)[:3]}
    assert open_paths(calibrated, modules) == easiest


def test_returns_new_router_with_same_configuration() -> None:
    base = make_router(
        threshold=0.5,
        force_frontier_on_features=False,
        force_frontier_features=frozenset({"common_block"}),
    )
    calibrated = base.calibrated_for_batch(graded_batch(10), 0.5)
    assert calibrated is not base
    assert base.threshold == 0.5
    assert calibrated.threshold != 0.5
    assert (calibrated.frontier_model, calibrated.open_model) == (FRONTIER, OPEN)
    assert calibrated.policy == "complexity"
    assert calibrated.force_frontier_on_features is False
    assert calibrated.force_frontier_features == frozenset({"common_block"})
    assert calibrated.weights is base.weights


def test_forced_modules_count_as_frontier_and_leave_the_pool() -> None:
    easy_forced = [make_module(["equivalence"], sloc=10, path=f"f{i}.f") for i in range(4)]
    others = graded_batch(6)
    modules = easy_forced + others
    calibrated = make_router().calibrated_for_batch(modules, 0.5)
    assert batch_open_share(calibrated, modules) == pytest.approx(0.5)
    assert open_paths(calibrated, modules) <= {m.path for m in others}
    assert len(open_paths(calibrated, modules)) == 5


def test_pool_smaller_than_target_opens_everything_eligible_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    forced = [make_module(["alternate_return"], path=f"f{i}.f") for i in range(6)]
    others = graded_batch(4)
    modules = forced + others
    with caplog.at_level(logging.WARNING, logger="modernizer.routing.policy"):
        calibrated = make_router().calibrated_for_batch(modules, 0.8)
    assert any("cannot reach open target" in r.getMessage() for r in caplog.records)
    assert open_paths(calibrated, modules) == {m.path for m in others}
    assert batch_open_share(calibrated, modules) == pytest.approx(0.4)


def test_all_forced_batch_warns_and_keeps_router(caplog: pytest.LogCaptureFixture) -> None:
    modules = [make_module(["entry_statement"], path=f"f{i}.f") for i in range(3)]
    router = make_router()
    with caplog.at_level(logging.WARNING, logger="modernizer.routing.policy"):
        assert router.calibrated_for_batch(modules, 0.5) is router
    assert caplog.records
    assert batch_open_share(router, modules) == 0.0


def test_no_warning_when_target_is_reachable(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="modernizer.routing.policy"):
        make_router().calibrated_for_batch(graded_batch(10), 0.5)
    assert not caplog.records


def test_ties_are_routed_alike_and_closest_count_wins() -> None:
    same = [make_module(["do_labeled"], sloc=30, path=f"t{i}.f") for i in range(4)]
    hard = [
        make_module(["goto_unconditional"] * 8, sloc=200, cyclomatic=40, path=f"h{i}.f")
        for i in range(6)
    ]
    modules = same + hard
    # k = 5: reachable open counts are 0, 4, 10 -> 4 is the closest
    calibrated = make_router().calibrated_for_batch(modules, 0.5)
    assert open_paths(calibrated, modules) == {m.path for m in same}
    # k = 7: 4 and 10 are equally far (3), the smaller count wins
    calibrated = make_router().calibrated_for_batch(modules, 0.7)
    assert batch_open_share(calibrated, modules) == pytest.approx(0.4)
    # k = 8: 10 is closer than 4
    calibrated = make_router().calibrated_for_batch(modules, 0.8)
    assert batch_open_share(calibrated, modules) == pytest.approx(1.0)


def test_target_zero_routes_nothing_open_even_for_zero_scores() -> None:
    modules = [make_module([], sloc=0, path=f"z{i}.f") for i in range(3)]
    calibrated = make_router().calibrated_for_batch(modules, 0.0)
    assert batch_open_share(calibrated, modules) == 0.0


def test_rounding_is_half_up() -> None:
    modules = graded_batch(10)
    # 0.25 * 10 = 2.5 -> k = 3
    share = batch_open_share(make_router().calibrated_for_batch(modules, 0.25), modules)
    assert share == pytest.approx(0.3)


@pytest.mark.parametrize("policy", ["always_frontier", "always_open"])
def test_other_policies_return_self_unchanged(policy: str) -> None:
    router = make_router(policy=policy)
    assert router.calibrated_for_batch(graded_batch(5), 0.5) is router


def test_empty_batch_returns_self() -> None:
    router = make_router()
    assert router.calibrated_for_batch([], 0.5) is router
    assert batch_open_share(router, []) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_target_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        make_router().calibrated_for_batch(graded_batch(3), bad)


def test_calibration_is_independent_of_input_order() -> None:
    modules = graded_batch(12)
    forward = make_router().calibrated_for_batch(modules, 0.5)
    backward = make_router().calibrated_for_batch(modules[::-1], 0.5)
    assert forward.threshold == backward.threshold


def test_batch_open_share_respects_baseline_policies() -> None:
    modules = graded_batch(5)
    assert batch_open_share(make_router(policy="always_open"), modules) == 1.0
    assert batch_open_share(make_router(policy="always_frontier"), modules) == 0.0


def test_forced_modules_join_the_pool_when_the_rule_is_disabled() -> None:
    forced = [make_module(["equivalence"], sloc=10, path=f"f{i}.f") for i in range(4)]
    others = graded_batch(6)
    modules = forced + others
    router = make_router(force_frontier_on_features=False)
    calibrated = router.calibrated_for_batch(modules, 0.4)
    assert open_paths(calibrated, modules) == {
        m.path for m in sorted(modules, key=difficulty_score)[:4]
    }
    assert batch_open_share(calibrated, modules) == pytest.approx(0.4)
