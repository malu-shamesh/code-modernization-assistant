from __future__ import annotations

import ast

import pytest

from modernizer.conversion import SYSTEM_PROMPT, ConversionError, convert_module
from modernizer.models import ContextSnippet, TokenUsage

from .helpers import FakeLLM, axpy_module, fenced, state_module

CONTEXT = [ContextSnippet("c1", "DO loops", "Compute the trip count once.", 0.9)]
AXPY_PY = (
    "from f77rt import do_range\n\n\n"
    "def axpy(n, a, x, y, ierr):\n"
    "    ierr = 0\n"
    "    for i in do_range(1, n):\n"
    "        y[i - 1] = a * x[i - 1] + y[i - 1]\n"
    "    return ierr"
)
PRELUDE_PY = "import numpy as np\n\n\nclass _Cnt:\n    ncalls = 0\n\n\ncnt = _Cnt()"
INIT_PY = "from f77rt import iand\n\n\ndef init():\n    pass"
BUMP_PY = (
    "import numpy as np\nfrom f77rt import idiv\n\n\n"
    "class _Cnt:\n    ncalls = 99\n\n\ncnt = _Cnt()\n\n\n"
    "def bump(n):\n    cnt.ncalls = cnt.ncalls + n"
)
TOTAL_PY = "from f77rt import nint\n\n\ndef total():\n    return cnt.ncalls"
SMALL = 40  # every unit of state_module() is larger, so each ends up in its own chunk


def chunked_replies():
    return [fenced(PRELUDE_PY), fenced(INIT_PY), fenced(BUMP_PY), fenced(TOTAL_PY), fenced(AXPY_PY)]


def test_single_call_for_small_module():
    llm = FakeLLM([f"Sure!\n{fenced(AXPY_PY)}"])
    out = convert_module(axpy_module(), CONTEXT, llm, tags={"module": "lib/demo.f"})
    assert out.source == AXPY_PY + "\n"
    assert out.calls == 1 and out.chunks == 1
    assert out.usage == TokenUsage(10, 5)
    assert out.raw_responses == [llm.replies[0]]
    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["temperature"] == 0.0
    assert call["max_output_tokens"] == 8192
    assert call["tags"] == {"module": "lib/demo.f", "stage": "convert"}
    assert "Compute the trip count once." in call["prompt"]
    assert "python: def axpy(n, a, x, y, ierr)" in call["prompt"]


def test_max_output_tokens_forwarded_and_caller_tags_not_mutated():
    tags = {"module": "m", "stage": "ignored"}
    llm = FakeLLM([fenced(AXPY_PY)])
    convert_module(axpy_module(), [], llm, max_output_tokens=123, tags=tags)
    assert llm.calls[0]["max_output_tokens"] == 123
    assert llm.calls[0]["tags"] == {"module": "m", "stage": "convert"}
    assert tags == {"module": "m", "stage": "ignored"}


def test_module_within_limit_is_one_call_even_with_many_units():
    module = state_module()
    llm = FakeLLM([fenced(AXPY_PY)])
    out = convert_module(module, [], llm, max_chunk_chars=len(module.source))
    assert out.calls == 1 and out.chunks == 1
    assert "one slice of a larger file" not in llm.calls[0]["prompt"]


def test_oversized_but_single_unit_module_is_one_call():
    llm = FakeLLM([fenced(AXPY_PY)])
    out = convert_module(axpy_module(), [], llm, max_chunk_chars=5)
    assert out.calls == 1 and out.chunks == 1


def test_repair_mode_is_one_call_with_previous_failures_and_repair_tag():
    llm = FakeLLM([fenced(AXPY_PY)])
    out = convert_module(
        state_module(),
        [],
        llm,
        feedback="- [syntax_error] line 3: invalid syntax\nFAILED test_total",
        previous="def total(:\n    pass\n",
        max_chunk_chars=10,
        tags={"module": "m"},
    )
    assert out.calls == 1 and out.chunks == 1
    call = llm.calls[0]
    assert call["tags"] == {"module": "m", "stage": "repair"}
    prompt = call["prompt"]
    assert prompt.startswith("Repair task:")
    assert "invalid syntax" in prompt and "FAILED test_total" in prompt
    assert "def total(:" in prompt
    assert "INTEGER FUNCTION TOTAL()" in prompt and "BLOCK DATA INIT" in prompt
    assert "Fortran source is the authority" in prompt


def test_unusable_reply_raises_conversion_error_with_raw_response_and_usage():
    llm = FakeLLM(["I cannot do that."])
    with pytest.raises(ConversionError) as info:
        convert_module(axpy_module(), [], llm)
    assert info.value.raw_response == "I cannot do that."
    assert info.value.raw_responses == ["I cannot do that."]
    assert info.value.usage == TokenUsage(10, 5)
    assert info.value.chunk_index is None


@pytest.mark.parametrize("reply", ["", "```fortran\n      END\n```"])
def test_blank_reply_and_foreign_only_block_raise(reply):
    with pytest.raises(ConversionError):
        convert_module(axpy_module(), [], FakeLLM([reply]))


def test_state_prelude_call_then_chunks_merge_in_order():
    llm = FakeLLM(chunked_replies())
    out = convert_module(state_module(), CONTEXT, llm, max_chunk_chars=SMALL, tags={"module": "m"})

    assert out.calls == 5 and out.chunks == 4 and len(llm.calls) == 5
    assert out.usage == TokenUsage(50, 25)
    assert out.raw_responses == llm.replies
    assert all(c["tags"] == {"module": "m", "stage": "convert"} for c in llm.calls)
    assert all(c["system"] == SYSTEM_PROMPT for c in llm.calls)

    assert out.source == (
        "import numpy as np\nfrom f77rt import iand, idiv, nint, do_range\n\n\n"
        "class _Cnt:\n    ncalls = 0\n\n\ncnt = _Cnt()\n\n\n"
        "def init():\n    pass\n\n\n"
        "def bump(n):\n    cnt.ncalls = cnt.ncalls + n\n\n\n"
        "def total():\n    return cnt.ncalls\n\n\n"
        "def axpy(n, a, x, y, ierr):\n"
        "    ierr = 0\n"
        "    for i in do_range(1, n):\n"
        "        y[i - 1] = a * x[i - 1] + y[i - 1]\n"
        "    return ierr\n"
    )
    ast.parse(out.source)


def test_prelude_prompt_is_first_and_chunk_prompts_receive_it():
    llm = FakeLLM(chunked_replies())
    convert_module(state_module(), [], llm, max_chunk_chars=SMALL)

    prelude_prompt = llm.calls[0]["prompt"]
    assert "COMMON /CNT/ NCALLS" in prelude_prompt
    assert "SUBROUTINE AXPY" not in prelude_prompt
    assert "## API contract" not in prelude_prompt

    for call in llm.calls[1:]:
        prompt = call["prompt"]
        assert "## Shared module-level state (already converted)" in prompt
        assert "cnt = _Cnt()" in prompt and "class _Cnt:" in prompt
        assert "one slice of a larger file" in prompt


def test_chunk_prompts_scope_hits_api_and_numbering_to_the_chunk():
    llm = FakeLLM(chunked_replies())
    convert_module(state_module(), [], llm, max_chunk_chars=SMALL)
    bump_prompt = llm.calls[2]["prompt"]
    assert "SUBROUTINE bump" in bump_prompt and "SUBROUTINE axpy\n" not in bump_prompt
    assert "line 8: COMMON /CNT/ NCALLS" in bump_prompt
    assert "do_labeled" not in bump_prompt
    assert "   6 |       SUBROUTINE BUMP(N)" in bump_prompt
    assert "SUBROUTINE AXPY(N" not in bump_prompt
    # signatures of the other chunks are listed, not redefined
    assert "- def axpy(n, a, x, y, ierr)" in bump_prompt
    assert "- def total()" in bump_prompt and "- def bump(n)" not in bump_prompt

    axpy_prompt = llm.calls[4]["prompt"]
    assert "lines 22-24: DO 10 I = 1, N" in axpy_prompt
    assert "  17 |       SUBROUTINE AXPY(N, A, X, Y, IERR)" in axpy_prompt


def test_no_prelude_without_shared_state():
    module = state_module()
    module.hits = [h for h in module.hits if h.feature_id == "do_labeled"]
    module.units = [u for u in module.units if u.kind != "block_data"]
    module.source = "\n\n".join(u.source for u in module.units)
    llm = FakeLLM([fenced(BUMP_PY), fenced(TOTAL_PY), fenced(AXPY_PY)])
    out = convert_module(module, [], llm, max_chunk_chars=SMALL)
    assert out.calls == 3 and out.chunks == 3
    assert all("Shared module-level state" not in c["prompt"] for c in llm.calls)
    assert out.source.index("def bump") < out.source.index("def total")
    assert out.source.index("def total") < out.source.index("def axpy")


def test_units_are_packed_up_to_the_limit():
    module = state_module()
    first, second = module.units[0].source, module.units[1].source
    llm = FakeLLM(chunked_replies())
    out = convert_module(module, [], llm, max_chunk_chars=len(first) + 2 + len(second))
    assert out.chunks == 3 and out.calls == 4
    packed = llm.calls[1]["prompt"]
    assert "BLOCK DATA INIT" in packed and "SUBROUTINE BUMP(N)" in packed


def test_error_in_a_chunk_reports_chunk_index_and_partial_usage():
    llm = FakeLLM([fenced(PRELUDE_PY), fenced("def a():\n    pass"), "no code here, sorry."])
    with pytest.raises(ConversionError) as info:
        convert_module(state_module(), [], llm, max_chunk_chars=SMALL)
    assert info.value.chunk_index == 1
    assert info.value.raw_response == "no code here, sorry."
    assert len(info.value.raw_responses) == 3
    assert info.value.usage == TokenUsage(30, 15)
    assert "chunk 2" in str(info.value)


def test_error_in_the_prelude_has_no_chunk_index():
    llm = FakeLLM(["I am sorry, I cannot convert this file."])
    with pytest.raises(ConversionError) as info:
        convert_module(state_module(), [], llm, max_chunk_chars=SMALL)
    assert info.value.chunk_index is None
    assert "prelude" in str(info.value)
    assert info.value.usage == TokenUsage(10, 5)


def test_tags_are_copied_per_call():
    class MutatingLLM(FakeLLM):
        def generate(self, prompt, **kwargs):
            kwargs["tags"]["stage"] = "mutated"
            return super().generate(prompt, **kwargs)

    llm = MutatingLLM(chunked_replies())
    convert_module(state_module(), [], llm, max_chunk_chars=SMALL)
    assert [c["tags"]["stage"] for c in llm.calls] == ["mutated"] * 5


def test_prompts_are_deterministic_across_runs():
    first, second = FakeLLM(chunked_replies()), FakeLLM(chunked_replies())
    a = convert_module(state_module(), CONTEXT, first, max_chunk_chars=SMALL)
    b = convert_module(state_module(), CONTEXT, second, max_chunk_chars=SMALL)
    assert [c["prompt"] for c in first.calls] == [c["prompt"] for c in second.calls]
    assert a.source == b.source
