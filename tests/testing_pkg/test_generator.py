from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modernizer.conventions import CALLING_CONVENTION
from modernizer.models import (
    ApiParam,
    ApiSymbol,
    FeatureHit,
    LLMResponse,
    ModuleMetrics,
    ParsedModule,
    TokenUsage,
)
from modernizer.protocols import LLMClient
from modernizer.testing import (
    TEST_SYSTEM_PROMPT,
    GeneratedTests,
    TestGenerationError,
    build_test_prompt,
    generate_tests,
)
from modernizer.testing.generator import render_api_contract

LEGACY = (
    "      SUBROUTINE SCALE(N, A, S, TOTAL)\n"
    "      INTEGER N\n"
    "      REAL A(N), S, TOTAL\n"
    "      TOTAL = 0.0\n"
    "      DO 10 I = 1, N\n"
    "         A(I) = A(I) * S\n"
    "         TOTAL = TOTAL + A(I)\n"
    "   10 CONTINUE\n"
    "      RETURN\n"
    "      END\n"
    "      INTEGER FUNCTION IHALF(K)\n"
    "      INTEGER K\n"
    "      IHALF = K / 2\n"
    "      RETURN\n"
    "      END\n"
)
CONVERTED = (
    "from f77rt import idiv\n"
    "\n"
    "def scale(n, a, s, total):\n"
    "    return SECRET_IMPLEMENTATION_DETAIL\n"
    "\n"
    "def _private(x):\n"
    "    return x\n"
    "\n"
    "def ihalf(k):\n"
    "    return idiv(k, 2)\n"
)
API = (
    ApiSymbol(
        "scale",
        "subroutine",
        (
            ApiParam("n", "integer"),
            ApiParam("a", "real", rank=1, intent="inout"),
            ApiParam("s", "real"),
            ApiParam("total", "real", intent="out"),
        ),
        returns=("total",),
    ),
    ApiSymbol("ihalf", "function", (ApiParam("k", "integer"),), result_type="integer"),
)
GOOD_TESTS = (
    "import numpy as np\n"
    "import pytest\n"
    "from target_module import ihalf\n"
    "\n"
    "def test_ihalf_truncates():\n"
    "    assert ihalf(-7) == -3\n"
)


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *replies: str, usage: TokenUsage | None = None) -> None:
        self.replies = list(replies)
        self.usage = usage or TokenUsage(120, 45)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        tags: Mapping[str, str] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "tags": dict(tags or {}),
            }
        )
        return LLMResponse(text=self.replies.pop(0), usage=self.usage, model=self.name)


@pytest.fixture
def module() -> ParsedModule:
    return ParsedModule(
        path="lib/scale.f",
        source=LEGACY,
        units=[],
        hits=[
            FeatureHit("integer_division", 13, 13, "K / 2"),
            FeatureHit("arg_output", 4, 4, "TOTAL = 0.0"),
            FeatureHit("integer_division", 13, 13, "K / 2"),
            FeatureHit("do_labeled", 5, 8, "DO 10 I = 1, N"),
        ],
        metrics=ModuleMetrics(),
        api=API,
    )


def test_llm_client_protocol_is_satisfied():
    assert isinstance(ScriptedLLM("x"), LLMClient)


def test_prompt_contains_numbered_fortran_source_and_no_implementation(module: ParsedModule):
    prompt = build_test_prompt(module, CONVERTED)
    assert "   1|       SUBROUTINE SCALE(N, A, S, TOTAL)" in prompt
    assert "  15|       END" in prompt
    assert "never reproduce it" in prompt
    assert "SECRET_IMPLEMENTATION_DETAIL" not in prompt
    assert "_private" not in prompt
    assert "target_module" in prompt


def test_prompt_contains_the_full_calling_convention(module: ParsedModule):
    assert CALLING_CONVENTION in build_test_prompt(module, CONVERTED)


def test_prompt_renders_the_per_unit_contract(module: ParsedModule):
    prompt = build_test_prompt(module, CONVERTED)
    assert "SUBROUTINE scale: def scale(n, a, s, total)" in prompt
    assert "a: rank-1 array of REAL" in prompt
    assert "numpy.ndarray, dtype=float; modified IN PLACE by the routine (never returned)" in prompt
    assert "total: REAL (float, single precision in the Fortran source) scalar; OUTPUT" in prompt
    assert "returns: the single value total (real)" in prompt
    assert "FUNCTION ihalf: def ihalf(k)" in prompt
    assert "returns: the single value function result (integer)" in prompt


def test_contract_return_shapes():
    function = ApiSymbol(
        "fsolve",
        "function",
        (ApiParam("x", "double"), ApiParam("ierr", "integer", intent="out")),
        result_type="double",
        returns=("ierr",),
        alt_returns=1,
    )
    text = render_api_contract((function,))
    assert (
        "returns: a tuple (function result (double), ierr (integer), "
        "alternate-return number (int, 0 for a normal RETURN))"
    ) in text
    assert "1 alternate-return ('*') dummy argument(s)" in text

    nothing = ApiSymbol("noop", "subroutine", (ApiParam("n", "integer"),))
    assert render_api_contract((nothing,)).endswith("returns: None")

    two = ApiSymbol(
        "two",
        "subroutine",
        (ApiParam("a", "integer", intent="out"), ApiParam("b", "integer", intent="inout")),
        returns=("a", "b"),
    )
    text = render_api_contract((two,))
    assert "returns: a tuple (a (integer), b (integer))" in text
    assert "its input value is used and the updated value is returned" in text


def test_scalar_role_follows_returns_not_intent():
    symbol = ApiSymbol(
        "f",
        "subroutine",
        (ApiParam("a", "integer", intent="out"), ApiParam("b", "integer", intent="out")),
        returns=("b",),
    )
    text = render_api_contract((symbol,))
    assert "a: INTEGER (int) scalar; input" in text
    assert "b: INTEGER (int) scalar; OUTPUT" in text
    assert "returns: the single value b (integer)" in text


def test_prompt_explains_fixed_form_columns(module: ParsedModule):
    assert "anything past column 72 is ignored" in build_test_prompt(module, CONVERTED)


def test_contract_for_program_block_data_and_special_arguments():
    api = (
        ApiSymbol("driver", "program"),
        ApiSymbol("init", "block_data"),
        ApiSymbol(
            "names",
            "subroutine",
            (
                ApiParam("tab", "character", rank=1, intent="out"),
                ApiParam("grid", "integer", rank=2),
                ApiParam("f", "external"),
            ),
        ),
    )
    text = render_api_contract(api)
    assert "PROGRAM driver: def main(), no arguments, returns None" in text
    assert "capsys" in text
    assert "BLOCK DATA init:" in text and "not callable" in text
    assert "tab: rank-1 array of CHARACTER (str): a Python list of str; modified IN PLACE" in text
    assert "grid: rank-2 array of INTEGER (int): numpy.ndarray, dtype=int; read only" in text
    assert "f: EXTERNAL procedure (a Python callable" in text


def test_prompt_requires_a_test_for_every_callable_unit(module: ParsedModule):
    prompt = build_test_prompt(module, CONVERTED, min_tests=11)
    assert "at least one test per program unit (scale, ihalf)" in prompt
    assert "at least 11 test cases in total" in prompt
    module.api = (*API, ApiSymbol("init", "block_data"))
    assert "(scale, ihalf)" in build_test_prompt(module, CONVERTED)


def test_prompt_states_the_fortran_testing_rules(module: ParsedModule):
    prompt = build_test_prompt(module, CONVERTED)
    for phrase in [
        "pytest.approx(expected, rel=1e-6",
        "rel=1e-12",
        "absolute tolerance",
        "numpy.testing.assert_allclose",
        "truncates toward zero",
        "max(0, (limit - start + step)",
        "MOD takes the sign of the first argument",
        "NINT rounds half away from zero",
        "blank padded",
        "a[i-1, j-1]",
        "zero-trip DO loops",
        "n = 1",
        "negative numbers",
        "empty or all-blank strings",
        "positional arguments in Fortran order",
        "from f77rt import",
        "FortranStop",
        "capsys",
        "explicit FORMAT",
        "list-directed",
        "tmp_path",
        "never read, write or reset them directly",
        "no randomness",
        "no network",
        "subprocess",
        "parametrize",
        "one ```python fenced block",
    ]:
        assert phrase in prompt, phrase


def test_prompt_has_no_em_dash(module: ParsedModule):
    dash = chr(0x2014)
    assert dash not in build_test_prompt(module, CONVERTED)
    assert dash not in TEST_SYSTEM_PROMPT


def test_prompt_default_min_tests(module: ParsedModule):
    assert "at least 8 test cases" in build_test_prompt(module, CONVERTED)


def test_prompt_lists_semantic_features_once_and_skips_mechanical(module: ParsedModule):
    prompt = build_test_prompt(module, CONVERTED)
    assert prompt.count("- integer_division:") == 1
    assert "- arg_output:" in prompt
    assert "- do_labeled:" not in prompt


def test_prompt_without_semantic_features(module: ParsedModule):
    module.hits = [FeatureHit("do_labeled", 5, 8, "DO 10 I = 1, N")]
    assert "easy to get wrong" not in build_test_prompt(module, CONVERTED)


def test_prompt_falls_back_to_converted_signatures_without_a_contract(module: ParsedModule):
    module.api = ()
    prompt = build_test_prompt(module, CONVERTED)
    assert "def scale(n, a, s, total)" in prompt
    assert "def ihalf(k)" in prompt
    assert "_private" not in prompt
    assert "at least one test per program unit of the source" in prompt


def test_prompt_without_any_api_info(module: ParsedModule):
    module.api = ()
    assert "no public names could be determined" in build_test_prompt(module, "def broken(:\n")


def test_generate_tests_happy_path(module: ParsedModule):
    llm = ScriptedLLM(f"```python\n{GOOD_TESTS}```", usage=TokenUsage(500, 200))
    out = generate_tests(
        module,
        CONVERTED,
        llm,
        min_tests=5,
        max_output_tokens=1234,
        tags={"module": "lib/scale.f", "stage": "ignored"},
    )
    assert isinstance(out, GeneratedTests)
    assert out.source == GOOD_TESTS
    assert out.usage == TokenUsage(500, 200)
    assert out.raw_response.startswith("```python")
    (call,) = llm.calls
    assert call["system"] == TEST_SYSTEM_PROMPT
    assert call["max_output_tokens"] == 1234
    assert call["temperature"] == 0.0
    assert call["tags"] == {"module": "lib/scale.f", "stage": "test_gen"}
    assert "at least 5 test cases" in call["prompt"]


def test_generate_tests_default_tags(module: ParsedModule):
    llm = ScriptedLLM(f"```python\n{GOOD_TESTS}```")
    generate_tests(module, CONVERTED, llm)
    assert llm.calls[0]["tags"] == {"stage": "test_gen"}
    assert llm.calls[0]["max_output_tokens"] == 8192


@pytest.mark.parametrize("reply", ["", "   \n", "```python\n```"])
def test_empty_reply_raises(module: ParsedModule, reply: str):
    with pytest.raises(TestGenerationError, match="no test code") as info:
        generate_tests(module, CONVERTED, ScriptedLLM(reply))
    assert info.value.raw_response == reply


def test_unsafe_tests_raise_with_raw_response(module: ParsedModule):
    reply = "```python\nimport subprocess\nimport target_module\ndef test_a():\n    pass\n```"
    with pytest.raises(TestGenerationError, match="forbidden") as info:
        generate_tests(module, CONVERTED, ScriptedLLM(reply))
    assert info.value.raw_response == reply


def test_third_party_import_in_reply_raises(module: ParsedModule):
    reply = "```python\nimport scipy\nimport target_module\ndef test_a():\n    pass\n```"
    with pytest.raises(TestGenerationError, match="not allowed") as info:
        generate_tests(module, CONVERTED, ScriptedLLM(reply))
    assert info.value.raw_response == reply


def test_prose_only_reply_raises(module: ParsedModule):
    reply = "I cannot write tests for this module."
    with pytest.raises(TestGenerationError) as info:
        generate_tests(module, CONVERTED, ScriptedLLM(reply))
    assert info.value.raw_response == reply
