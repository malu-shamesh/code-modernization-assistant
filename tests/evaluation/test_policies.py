from __future__ import annotations

import pytest

from modernizer.evaluation import PolicySpec, default_policies


def test_default_policies_have_no_target_unless_asked():
    policies = {p.name: p for p in default_policies(0.4)}
    assert set(policies) == {"always_frontier", "always_open", "complexity"}
    assert all(p.target_open_fraction is None for p in policies.values())
    assert policies["complexity"].threshold == 0.4
    assert policies["complexity"].escalate
    assert not policies["always_open"].escalate and not policies["always_frontier"].escalate


def test_the_target_goes_to_the_complexity_arm_only():
    policies = {p.name: p for p in default_policies(0.4, target_open_fraction=0.7)}
    assert policies["complexity"].target_open_fraction == 0.7
    assert policies["always_open"].target_open_fraction is None
    assert policies["always_frontier"].target_open_fraction is None


def test_specs_round_trip_through_dicts():
    for spec in default_policies(0.4, 0.5):
        assert PolicySpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_target_must_be_a_share(bad):
    with pytest.raises(ValueError, match="target_open_fraction"):
        PolicySpec("c", "complexity", 0.4, True, bad)


def test_a_target_only_makes_sense_for_the_complexity_policy():
    with pytest.raises(ValueError, match="complexity"):
        PolicySpec("o", "always_open", None, False, 0.5)
    assert PolicySpec("c", "complexity", None, True, 0.0).target_open_fraction == 0.0
    assert PolicySpec("c", "complexity", None, True, 1.0).target_open_fraction == 1.0
