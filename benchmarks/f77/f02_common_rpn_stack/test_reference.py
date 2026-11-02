import numpy as np
import pytest

from target_module import stkclr, stkdep, stkpop, stkpsh, stkrpn

ADD, SUB, MUL, DIV, MOD = -1, -2, -3, -4, -5


def evaluate(tokens):
    arr = np.array(tokens, dtype=np.int64)
    return stkrpn(len(tokens), arr, -1, 7)


def test_push_pop_lifo_order():
    stkclr()
    assert stkpsh(5, 7) == 0
    assert stkpsh(9, 7) == 0
    assert stkdep() == 2
    assert stkpop(55, 7) == (9, 0)
    assert stkpop(55, 7) == (5, 0)
    assert stkdep() == 0


def test_pop_empty_reports_underflow_with_zero_value():
    stkclr()
    assert stkpop(55, 7) == (0, 2)
    assert stkdep() == 0


def test_push_overflow_at_depth_eight():
    stkclr()
    for v in range(1, 9):
        assert stkpsh(v, 7) == 0
    assert stkdep() == 8
    assert stkpsh(99, 7) == 1
    assert stkdep() == 8
    assert stkpop(55, 7) == (8, 0)


def test_clear_empties_the_shared_stack():
    stkclr()
    stkpsh(1, 7)
    stkpsh(2, 7)
    stkclr()
    assert stkdep() == 0
    assert stkpop(55, 7)[1] == 2


def test_add_and_multiply():
    # 3 4 + 2 * = 14
    assert evaluate([3, 4, ADD, 2, MUL]) == (14, 0)


def test_subtraction_operand_order():
    # 10 3 - = 7 (second popped is the left operand)
    assert evaluate([10, 3, SUB]) == (7, 0)


def test_division_truncates_toward_zero():
    # 0 7 - 2 / = -7 / 2 = -3
    assert evaluate([0, 7, SUB, 2, DIV]) == (-3, 0)
    assert evaluate([7, 2, DIV]) == (3, 0)
    assert evaluate([0, 7, SUB, 0, 2, SUB, DIV]) == (3, 0)


def test_mod_takes_sign_of_dividend():
    # MOD(-7, 2) = -1 and MOD(7, -2) = 1 and -7 mod -2 = -1
    assert evaluate([0, 7, SUB, 2, MOD]) == (-1, 0)
    assert evaluate([7, 0, 2, SUB, MOD]) == (1, 0)
    assert evaluate([0, 7, SUB, 0, 2, SUB, MOD]) == (-1, 0)


@pytest.mark.parametrize("op", [DIV, MOD])
def test_division_by_zero(op):
    assert evaluate([5, 0, op]) == (0, 3)


def test_underflow_in_expression():
    assert evaluate([1, ADD]) == (0, 2)
    assert evaluate([ADD]) == (0, 2)


def test_unknown_operator_code():
    assert evaluate([1, 2, -6]) == (0, 5)
    assert evaluate([1, 2, -99]) == (0, 5)


def test_leftover_operands_give_bad_final_depth():
    assert evaluate([1, 2]) == (0, 4)
    assert evaluate([]) == (0, 4)


def test_expression_overflow_nine_operands():
    assert evaluate(list(range(1, 10))) == (0, 1)
    # eight operands fit, but the result is then rejected for depth
    assert evaluate(list(range(1, 9))) == (0, 4)


def test_evaluator_clears_previous_stack_content():
    stkclr()
    stkpsh(1000, 7)
    stkpsh(2000, 7)
    assert evaluate([6, 7, MUL]) == (42, 0)
    assert stkdep() == 0


def test_only_first_n_tokens_are_used():
    arr = np.array([2, 3, ADD, 100, MUL], dtype=np.int64)
    assert stkrpn(3, arr, -1, 7) == (5, 0)


def test_deep_expression_uses_full_stack():
    # 1 2 3 4 5 6 7 8 + + + + + + + = 36, exactly fills the eight slots
    assert evaluate(list(range(1, 9)) + [ADD] * 7) == (36, 0)
