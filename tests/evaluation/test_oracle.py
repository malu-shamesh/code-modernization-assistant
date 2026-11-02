from __future__ import annotations

from dataclasses import replace

import pytest

from modernizer.conversion import (
    SYSTEM_PROMPT,
    build_conversion_prompt,
    convert_module,
    extract_code,
)
from modernizer.evaluation import OracleLLM, load_corpus, oracle_model_set
from modernizer.evaluation.oracle import OracleLookupError
from modernizer.llm import LLMError
from modernizer.parsing import analyze, validate_conversion
from modernizer.testing import TEST_SYSTEM_PROMPT, build_test_prompt, sanitize_tests

from .helpers import BENCHMARK_DIR

EASY, MEDIUM, HARD = "a01_horner_poly", "a08_taylor_series", "a10_calendar_loop_traps"
SHARED_STATE = "f02_common_rpn_stack"


@pytest.fixture(scope="module")
def cases():
    return load_corpus(BENCHMARK_DIR, ids=[EASY, MEDIUM, HARD, SHARED_STATE])


@pytest.fixture(scope="module")
def by_id(cases):
    return {case.id: case for case in cases}


def lf(text):
    """The oracle replies with LF line endings whatever the checkout used."""
    return text.replace("\r\n", "\n").strip()


def parsed(case):
    return analyze(f"{case.id}.f", case.legacy_source)


def conversion_prompt(case, **kwargs):
    return build_conversion_prompt(parsed(case), [], **kwargs)


def test_conversion_prompts_get_the_reference_port(cases):
    oracle = OracleLLM(cases, name="o")
    for case in cases:
        response = oracle.generate(conversion_prompt(case), system=SYSTEM_PROMPT)
        assert response.text.startswith("```python\n")
        assert extract_code(response.text).strip() == lf(case.reference_source)
        assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0


def test_repair_prompts_are_recognised_too(by_id):
    oracle = OracleLLM(by_id.values())
    case = by_id[MEDIUM]
    prompt = conversion_prompt(
        case, feedback="3 failed, 0 errors, 5 passed", previous="def broken(:\n    pass"
    )
    assert "## Fortran 77 source (the authority)" in prompt
    response = oracle.generate(prompt, system=SYSTEM_PROMPT)
    assert extract_code(response.text).strip() == lf(case.reference_source)


def test_test_prompts_get_the_held_out_tests(by_id):
    oracle = OracleLLM(by_id.values())
    for case in by_id.values():
        prompt = build_test_prompt(parsed(case), case.reference_source)
        response = oracle.generate(prompt, system=TEST_SYSTEM_PROMPT)
        held_out = case.reference_tests.read_bytes().decode("utf-8")
        assert sanitize_tests(response.text).strip() == lf(held_out)


def test_lookup_ignores_trailing_blanks_and_line_endings(by_id):
    case = by_id[EASY]
    prompt = conversion_prompt(case)
    sloppy = replace(case, legacy_source=case.legacy_source.replace("\n", "   \r\n") + "\n\n")
    oracle = OracleLLM([sloppy])
    response = oracle.generate(prompt, system=SYSTEM_PROMPT)
    assert extract_code(response.text).strip() == lf(case.reference_source)


def test_a_changed_source_line_is_not_matched(by_id):
    case = by_id[EASY]
    oracle = OracleLLM([case])
    lines = case.legacy_source.split("\n")
    index = next(i for i, line in enumerate(lines) if line.strip() and line[0] not in "Cc*")
    lines[index] += " + 1"
    other = replace(case, legacy_source="\n".join(lines))
    with pytest.raises(OracleLookupError, match="matches no corpus case"):
        oracle.generate(conversion_prompt(other), system=SYSTEM_PROMPT)


def test_the_gutter_is_not_confused_with_source_text(by_id):
    case = by_id[EASY]
    tricky = "C    12 | looks like a gutter\n" + case.legacy_source
    oracle = OracleLLM([replace(case, legacy_source=tricky)])
    response = oracle.generate(
        conversion_prompt(replace(case, legacy_source=tricky)), system=SYSTEM_PROMPT
    )
    assert extract_code(response.text).strip() == lf(case.reference_source)


def test_token_counts_are_length_over_four(cases):
    oracle = OracleLLM(cases)
    prompt = conversion_prompt(cases[0])
    response = oracle.generate(prompt, system=SYSTEM_PROMPT)
    assert response.usage.input_tokens == len(SYSTEM_PROMPT + prompt) // 4
    assert response.usage.output_tokens == len(response.text) // 4


def test_fail_difficulties_return_the_fortran_source_as_uncompilable_python(by_id):
    oracle = OracleLLM(by_id.values(), fail_difficulties=frozenset({"hard"}))
    easy, hard = by_id[EASY], by_id[HARD]
    assert easy.difficulty == "easy" and hard.difficulty == "hard"
    ok = oracle.generate(conversion_prompt(easy), system=SYSTEM_PROMPT)
    failed = oracle.generate(conversion_prompt(hard), system=SYSTEM_PROMPT)
    assert extract_code(ok.text).strip() == lf(easy.reference_source)
    assert extract_code(failed.text).strip() == lf(hard.legacy_source)
    assert validate_conversion(parsed(easy), extract_code(ok.text)).ok
    assert not validate_conversion(parsed(hard), extract_code(failed.text)).ok
    # A failing repair answer is as bad as the first one, so the tier exhausts and escalates.
    repair = oracle.generate(
        conversion_prompt(hard, feedback="static validation failed", previous="x"),
        system=SYSTEM_PROMPT,
    )
    assert extract_code(repair.text).strip() == lf(hard.legacy_source)


def test_failing_oracle_still_writes_tests(by_id):
    oracle = OracleLLM(by_id.values(), fail_difficulties=frozenset({"hard"}))
    hard = by_id[HARD]
    prompt = build_test_prompt(parsed(hard), hard.reference_source)
    assert oracle.generate(prompt, system=TEST_SYSTEM_PROMPT).text


def test_unknown_prompts_are_harness_errors_not_model_errors(cases):
    oracle = OracleLLM(cases)
    with pytest.raises(OracleLookupError):
        oracle.generate("an unrelated prompt", system=SYSTEM_PROMPT)
    with pytest.raises(OracleLookupError):
        oracle.generate("an unrelated prompt", system=TEST_SYSTEM_PROMPT)
    with pytest.raises(OracleLookupError):
        oracle.generate(conversion_prompt(cases[0]), system="something else")
    assert not issubclass(OracleLookupError, LLMError)


def test_a_test_prompt_is_not_a_conversion_prompt(cases):
    oracle = OracleLLM(cases)
    prompt = build_test_prompt(parsed(cases[0]), cases[0].reference_source)
    with pytest.raises(OracleLookupError):
        oracle.generate(prompt, system=SYSTEM_PROMPT)


def test_chunked_and_prelude_prompts_raise_lookup_errors(by_id):
    for case_id in (EASY, SHARED_STATE):
        case = by_id[case_id]
        oracle = OracleLLM(by_id.values())
        # 300 characters forces slices (and for the COMMON module a shared-state prelude call).
        with pytest.raises(OracleLookupError):
            convert_module(parsed(case), [], oracle, max_chunk_chars=300)
        assert oracle.lookup_failures == 1


def test_call_count_and_construction_rules(cases):
    oracle = OracleLLM(cases)
    assert oracle.call_count == 0
    oracle.generate(conversion_prompt(cases[0]), system=SYSTEM_PROMPT)
    assert oracle.call_count == 1
    with pytest.raises(ValueError):
        OracleLLM([])


def test_cases_with_identical_sources_are_rejected(by_id):
    case = by_id[EASY]
    with pytest.raises(ValueError, match="same Fortran source"):
        OracleLLM([case, replace(case, id="twin")])


def test_model_set_names_and_failure_default(cases):
    models = oracle_model_set(cases)
    assert models.frontier.name == "oracle-frontier"
    assert models.open.name == "oracle-open"
    assert models.frontier.fail_difficulties == frozenset()
    assert models.open.fail_difficulties == frozenset({"hard"})
    assert (
        oracle_model_set(cases, fail_difficulties=frozenset()).open.fail_difficulties == frozenset()
    )


def test_the_whole_corpus_can_be_indexed_without_collisions():
    oracle = OracleLLM(load_corpus(BENCHMARK_DIR))
    assert oracle.call_count == 0


def test_lookup_failures_are_counted(cases):
    oracle = OracleLLM(cases)
    assert oracle.lookup_failures == 0
    with pytest.raises(OracleLookupError):
        oracle.generate("an unrelated prompt", system=SYSTEM_PROMPT)
    with pytest.raises(OracleLookupError):
        oracle.generate(conversion_prompt(cases[0]), system="something else")
    assert oracle.lookup_failures == 2
    assert oracle.call_count == 2
