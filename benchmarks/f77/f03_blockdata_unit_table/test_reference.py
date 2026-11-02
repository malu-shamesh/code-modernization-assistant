import pytest

from target_module import ucnt, ufind, uadd, uconv, ureset


@pytest.fixture(autouse=True)
def restore_table_after_test():
    # No reset before the test: the first test below must see the BLOCK DATA state.
    yield
    ureset()


def test_block_data_state_before_any_call():
    assert ucnt() == 6
    assert ufind("FT") == 6
    assert uconv(1.0, "KM", "M", 9) == (1000.0, 0)


@pytest.mark.parametrize(
    "name, index",
    [("M", 1), ("KM", 2), ("CM", 3), ("MM", 4), ("IN", 5), ("FT", 6)],
)
def test_builtin_units_are_found(name, index):
    assert ufind(name) == index
    assert ufind(name.ljust(4)) == index
    assert ufind(name.ljust(9)) == index


def test_unknown_and_case_sensitive_names():
    assert ucnt() == 6
    assert ufind("YD") == 0
    assert ufind("km") == 0
    assert ufind("") == 0
    assert ufind("   ") == 0


def test_name_longer_than_four_is_not_truncated_for_lookup():
    assert ufind("KM   X") == 0
    assert ufind("MMM") == 0


def test_simple_conversions():
    assert uconv(2.0, "KM", "M", 9) == pytest.approx((2000.0, 0), rel=1e-9)
    assert uconv(1.0, "FT", "IN", 9) == pytest.approx((12.0, 0), rel=1e-9)
    assert uconv(254.0, "MM", "IN", 9) == pytest.approx((10.0, 0), rel=1e-9)
    assert uconv(1.0, "CM", "KM", 9) == pytest.approx((1.0e-5, 0), rel=1e-9)


def test_conversion_to_same_unit_and_zero():
    assert uconv(5.5, "IN", "IN", 9) == pytest.approx((5.5, 0), rel=1e-9)
    assert uconv(0.0, "FT", "M", 9) == (0.0, 0)


def test_unknown_source_has_priority_over_unknown_target():
    assert uconv(1.0, "XX", "YY", 9) == (0.0, 1)
    assert uconv(1.0, "XX", "M", 9) == (0.0, 1)


def test_unknown_target():
    assert uconv(1.0, "M", "XX", 9) == (0.0, 2)


def test_add_unit_and_use_it():
    assert uadd("MI", 1609.344, 9) == 0
    assert ucnt() == 7
    assert ufind("MI") == 7
    assert uconv(1.0, "MI", "KM", 9) == pytest.approx((1.609344, 0), rel=1e-9)
    assert uconv(10.0, "KM", "MI", 9) == pytest.approx((10000.0 / 1609.344, 0), rel=1e-9)
    assert uconv(1.0, "KM", "M", 9) == pytest.approx((1000.0, 0), rel=1e-9)


def test_add_error_priorities():
    assert uadd("KM", 5.0, 9) == 3
    assert uadd("ZZ", 0.0, 9) == 2
    assert uadd("ZZ", -1.0, 9) == 2
    assert uadd("KM", -1.0, 9) == 2
    assert ucnt() == 6
    assert ufind("ZZ") == 0


def test_table_full():
    assert uadd("A1", 1.0, 9) == 0
    assert uadd("A2", 2.0, 9) == 0
    assert ucnt() == 8
    assert uadd("A3", 3.0, 9) == 1
    assert uadd("A1", 1.0, 9) == 3
    assert ucnt() == 8
    assert ufind("A2") == 8


def test_stored_name_is_truncated_to_four_characters():
    assert uadd("MILE5", 1609.344, 9) == 0
    assert ufind("MILE") == 7
    assert ufind("MILE5") == 0
    assert uadd("MILE", 1.0, 9) == 3
    # the long spelling is not found, so it is added again as a second 'MILE'
    assert uadd("MILEAGE", 2.0, 9) == 0
    assert ucnt() == 8
    assert ufind("MILE") == 7


def test_reset_restores_builtin_table():
    uadd("MI", 1609.344, 9)
    uadd("YD", 0.9144, 9)
    assert ucnt() == 8
    ureset()
    assert ucnt() == 6
    assert ufind("MI") == 0
    assert ufind("YD") == 0
    assert uconv(1.0, "FT", "CM", 9) == pytest.approx((30.48, 0), rel=1e-9)
    assert uadd("YD", 0.9144, 9) == 0
    assert ufind("YD") == 7
