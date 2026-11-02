import numpy as np
import pytest

import target_module as tm


def post(itype, amt):
    return tm.posttx(itype, amt, 0.0)


def status():
    return tm.acstat(0.0, 0, 0)


def run(kinds, amts, n=None):
    k = np.array(kinds, dtype=np.int64)
    a = np.array(amts, dtype=np.float64)
    return tm.runled(len(kinds) if n is None else n, k, a, 0, 0, 0, 0, 0.0)


def test_deposit_and_withdrawal():
    tm.openac(100.0, 50.0)
    assert post(1, 25.0) == pytest.approx((125.0, 0), rel=1e-9)
    assert post(2, 25.0) == pytest.approx((100.0, 0), rel=1e-9)


def test_withdrawal_exactly_at_overdraft_limit_is_allowed():
    tm.openac(125.0, 50.0)
    assert post(2, 175.0) == pytest.approx((-50.0, 0), rel=1e-9)


def test_withdrawal_beyond_overdraft_is_declined_with_return_2():
    tm.openac(125.0, 50.0)
    newbal, alt = post(2, 200.0)
    assert alt == 2
    assert newbal == pytest.approx(125.0, rel=1e-9)
    assert status()[0] == pytest.approx(125.0, rel=1e-9)


def test_zero_withdrawal_is_declined_when_already_below_the_limit():
    tm.openac(-60.0, 50.0)
    newbal, alt = post(2, 0.0)
    assert alt == 2
    assert newbal == pytest.approx(-60.0, rel=1e-9)


def test_fee_is_never_refused():
    tm.openac(5.0, 0.0)
    newbal, alt = post(3, 100.0)
    assert alt == 0
    assert newbal == pytest.approx(-95.0, rel=1e-9)


def test_interest_only_on_positive_balance():
    tm.openac(1000.0, 0.0)
    assert post(4, 2.5) == pytest.approx((1025.0, 0), rel=1e-9)
    tm.openac(0.0, 0.0)
    assert post(4, 10.0) == pytest.approx((0.0, 0), rel=1e-9)
    tm.openac(-40.0, 100.0)
    assert post(4, 10.0) == pytest.approx((-40.0, 0), rel=1e-9)


@pytest.mark.parametrize("itype, amt", [(9, 1.0), (0, 1.0), (-1, 5.0), (1, -5.0), (3, -0.5)])
def test_invalid_requests_return_1_and_change_nothing(itype, amt):
    tm.openac(80.0, 10.0)
    newbal, alt = post(itype, amt)
    assert alt == 1
    assert newbal == pytest.approx(80.0, rel=1e-9)


def test_counters_split_posted_and_rejected():
    tm.openac(100.0, 50.0)
    outcomes = [
        post(1, 25.0)[1],
        post(2, 200.0)[1],
        post(2, 175.0)[1],
        post(3, 10.0)[1],
        post(4, 5.0)[1],
        post(9, 1.0)[1],
        post(1, -5.0)[1],
        post(2, 0.0)[1],
    ]
    assert outcomes == [0, 2, 0, 0, 0, 1, 1, 2]
    curbal, npost, nrej = status()
    assert curbal == pytest.approx(-60.0, rel=1e-9)
    assert (npost, nrej) == (4, 4)


def test_openac_resets_counters_and_balance():
    tm.openac(10.0, 0.0)
    post(1, 5.0)
    post(9, 1.0)
    tm.openac(7.5, 3.0)
    curbal, npost, nrej = status()
    assert curbal == pytest.approx(7.5, rel=1e-9)
    assert (npost, nrej) == (0, 0)


def test_batch_stops_after_second_decline():
    tm.openac(100.0, 0.0)
    nok, ninv, ndec, ndone, final = run([1, 2, 2, 1, 2, 2, 1], [50, 200, 100, 20, 500, 10, 5])
    assert (nok, ninv, ndec, ndone) == (3, 0, 2, 5)
    assert final == pytest.approx(70.0, rel=1e-9)
    assert status() == pytest.approx((70.0, 3, 2), rel=1e-9)


def test_batch_with_invalid_entries_does_not_count_them_as_declines():
    tm.openac(0.0, 0.0)
    nok, ninv, ndec, ndone, final = run([1, 7, 3, 4], [10, 5, 2, 10])
    assert (nok, ninv, ndec, ndone) == (3, 1, 0, 4)
    assert final == pytest.approx(8.8, rel=1e-9)


def test_batch_one_decline_continues_to_the_end():
    tm.openac(10.0, 0.0)
    nok, ninv, ndec, ndone, final = run([2, 2, 1, 2], [30, 4, 1, 7])
    assert (nok, ninv, ndec, ndone) == (3, 0, 1, 4)
    assert final == pytest.approx(0.0, abs=1e-12)


def test_batch_only_examines_n_entries():
    tm.openac(0.0, 0.0)
    nok, ninv, ndec, ndone, final = run([1, 1, 1], [1, 2, 4], n=2)
    assert (nok, ninv, ndec, ndone) == (2, 0, 0, 2)
    assert final == pytest.approx(3.0, rel=1e-9)


def test_empty_batch():
    tm.openac(42.0, 0.0)
    assert run([1], [1.0], n=0) == pytest.approx((0, 0, 0, 0, 42.0), rel=1e-9)
