from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from modernizer.llm import ScriptedLLM, ScriptExhaustedError
from modernizer.protocols import LLMClient


def test_implements_protocol():
    assert isinstance(ScriptedLLM(["x"]), LLMClient)


def test_replays_in_order_and_records_calls():
    llm = ScriptedLLM(["first", "second"], name="m")
    a = llm.generate("p1", system="sys", tags={"stage": "convert"})
    b = llm.generate("p2")
    assert (a.text, b.text) == ("first", "second")
    assert a.model == "m"
    assert llm.calls == [
        {"prompt": "p1", "system": "sys", "tags": {"stage": "convert"}},
        {"prompt": "p2", "system": None, "tags": {}},
    ]


def test_token_estimate_is_ceil_len_over_four():
    llm = ScriptedLLM(["abcde"])
    resp = llm.generate("12345678901")
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 2
    assert ScriptedLLM([""]).generate("").usage.total == 0


def test_exhausted_script_raises_clear_error_but_records_call():
    llm = ScriptedLLM(["only"])
    llm.generate("a")
    with pytest.raises(ScriptExhaustedError, match="only 1 responses"):
        llm.generate("b")
    assert len(llm.calls) == 2


def test_callable_responses_receive_prompt():
    llm = ScriptedLLM(lambda p: p.upper())
    assert llm.generate("abc").text == "ABC"
    assert llm.generate("abc").text == "ABC"


def test_thread_safety_each_scripted_response_used_exactly_once():
    n = 200
    llm = ScriptedLLM([str(i) for i in range(n)])
    with ThreadPoolExecutor(max_workers=16) as pool:
        texts = list(pool.map(lambda i: llm.generate(f"p{i}").text, range(n)))
    assert sorted(texts, key=int) == [str(i) for i in range(n)]
    assert len(llm.calls) == n
    with pytest.raises(ScriptExhaustedError):
        llm.generate("extra")


def test_bare_string_script_is_rejected():
    with pytest.raises(TypeError):
        ScriptedLLM("abc")
