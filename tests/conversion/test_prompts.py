from __future__ import annotations

import pytest

from modernizer.conventions import CALLING_CONVENTION
from modernizer.conversion import prompts
from modernizer.conversion.prompts import (
    SYSTEM_PROMPT,
    build_conversion_prompt,
    build_state_prelude_prompt,
    format_api_contract,
    format_checklist,
    format_numbered,
    format_signatures,
)
from modernizer.features import FEATURES
from modernizer.models import (
    ApiParam,
    ApiSymbol,
    ContextSnippet,
    FeatureHit,
    ModuleMetrics,
    ParsedModule,
)

from .helpers import AXPY, AXPY_API, axpy_module, state_module

CATALOGUE = "idiv(a, b) -> int  # truncating integer division\nnint(x) -> int  # round half away"
CONTEXT = [
    ContextSnippet("n1", "Arithmetic IF", "Rewrite the arithmetic IF as if/elif/else.", 0.9),
    ContextSnippet("n2", "DO loops", "Compute the trip count once.", 0.8),
]


@pytest.fixture(autouse=True)
def fixed_catalogue(monkeypatch):
    monkeypatch.setattr(prompts, "api_doc", lambda: CATALOGUE)


def test_system_prompt_states_machine_validation_and_fidelity():
    lowered = SYSTEM_PROMPT.lower()
    assert "fortran 77" in lowered
    assert "machine-validated" in lowered and "tests" in lowered
    assert "faithful" in lowered and "behaviour-preserving" in lowered
    assert "```python" in SYSTEM_PROMPT


def test_sections_are_present_in_order():
    prompt = build_conversion_prompt(axpy_module(), CONTEXT)
    markers = [
        "Task: port the Fortran 77",
        "## Calling convention",
        "## Runtime helpers (module `f77rt`)",
        "## API contract (follow exactly)",
        "## Legacy constructs detected by the parser",
        "## Migration notes",
        "## Fortran 77 source to convert",
        "Output rules:",
    ]
    positions = [prompt.index(m) for m in markers]
    assert positions == sorted(positions)
    assert CALLING_CONVENTION.strip() in prompt
    assert CATALOGUE in prompt
    assert "ONLY names from this catalogue" in prompt
    assert "Rewrite the arithmetic IF as if/elif/else." in prompt
    assert "Note 2 - DO loops" in prompt
    assert "machine-validated" in prompt and "behaviour-preserving" in prompt


def test_output_rules_content():
    prompt = build_conversion_prompt(axpy_module(), [])
    rules = prompt[prompt.index("Output rules:") :]
    assert "ONE ```python fenced block" in rules
    assert "Fortran comments as Python comments" in rules
    assert "No placeholders" in rules and "TODO" in rules
    assert "No scipy" in rules
    assert "gutter" in rules
    assert "(no migration notes were retrieved)" in prompt


def test_prompt_is_deterministic_and_independent_of_hit_order():
    module = axpy_module()
    first = build_conversion_prompt(module, CONTEXT)
    assert build_conversion_prompt(module, CONTEXT) == first
    module.hits = list(reversed(module.hits))
    assert build_conversion_prompt(module, CONTEXT) == first


def test_prompt_has_no_em_dash():
    assert chr(0x2014) not in build_conversion_prompt(axpy_module(), CONTEXT)


def test_empty_runtime_catalogue_is_handled(monkeypatch):
    monkeypatch.setattr(prompts, "api_doc", lambda: "")
    prompt = build_conversion_prompt(axpy_module(), [])
    assert "no helpers are available" in prompt


# --------------------------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------------------------


def test_subroutine_contract_arrays_and_outputs():
    text = format_api_contract((AXPY_API,))
    assert "- SUBROUTINE axpy" in text
    assert "python: def axpy(n, a, x, y, ierr)" in text
    assert "n: int scalar, input only" in text
    assert "x: ARRAY of rank 1, numpy.ndarray of float, allocated by the caller, read only" in text
    assert "y: ARRAY of rank 1, numpy.ndarray of float" in text and "modified IN PLACE" in text
    assert "ierr: int scalar, OUTPUT, assigned by the unit and RETURNED" in text
    assert "returns: the single value ierr (output argument) (not wrapped in a tuple)" in text


def test_subroutine_without_outputs_returns_none():
    sym = ApiSymbol("bump", "subroutine", (ApiParam("n", "integer"),))
    text = format_api_contract((sym,))
    assert "returns: None" in text


def test_subroutine_with_several_outputs_is_a_tuple_in_parameter_order():
    sym = ApiSymbol(
        "stats",
        "subroutine",
        (
            ApiParam("x", "double", rank=2),
            ApiParam("mean", "double", intent="out"),
            ApiParam("n", "integer", intent="inout"),
        ),
        returns=("mean", "n"),
    )
    text = format_api_contract((sym,))
    assert "x: ARRAY of rank 2, numpy.ndarray of float" in text
    assert "returns: a tuple in exactly this order: (mean, n)" in text
    assert "n: int scalar, OUTPUT, assigned by the unit and RETURNED (its input value" in text


def test_function_result_comes_first_in_the_tuple():
    sym = ApiSymbol(
        "solve",
        "function",
        (ApiParam("a", "real", rank=2), ApiParam("info", "integer", intent="out")),
        result_type="double",
        returns=("info",),
    )
    text = format_api_contract((sym,))
    assert "- FUNCTION solve" in text
    assert "returns: a tuple in exactly this order: (solve, info)" in text
    assert "solve (the function result, float)" in text
    assert "result type: float" in text


def test_function_without_outputs_returns_bare_result():
    sym = ApiSymbol("total", "function", result_type="integer")
    text = format_api_contract((sym,))
    assert "parameters: none" in text
    assert "returns: the single value total (the function result, int)" in text


def test_alternate_returns_append_an_int():
    sym = ApiSymbol(
        "chk",
        "subroutine",
        (ApiParam("x", "real"), ApiParam("ok", "logical", intent="out")),
        returns=("ok",),
        alt_returns=2,
    )
    text = format_api_contract((sym,))
    assert "python: def chk(x, ok)" in text
    assert "returns: a tuple in exactly this order: (ok, alt)" in text
    assert "alt (int: number of the alternate RETURN taken, 0 for a normal RETURN)" in text
    assert "2 '*' dummy argument(s) are not Python parameters" in text


def test_star_dummies_never_appear_in_the_signature():
    sym = ApiSymbol("s", "subroutine", (ApiParam("x", "real"), ApiParam("*", "unknown")))
    assert "python: def s(x)" in format_api_contract((sym,))


def test_external_character_and_program_and_block_data():
    sub = ApiSymbol(
        "integ",
        "subroutine",
        (
            ApiParam("f", "external"),
            ApiParam("names", "character", rank=1),
            ApiParam("s", "character"),
        ),
    )
    text = format_api_contract((sub, ApiSymbol("main", "program"), ApiSymbol("init", "block_data")))
    assert "f: callable (Fortran EXTERNAL procedure" in text
    assert "names: ARRAY of rank 1, list of str" in text
    assert "s: str scalar, input only" in text
    assert "python: def main()" in text
    assert "BLOCK DATA init: no function" in text


def test_contract_section_without_units_and_signatures_helper():
    assert "(no program units were detected)" in build_conversion_prompt(
        ParsedModule(path="x.f", source="", units=[], hits=[], metrics=ModuleMetrics()),
        [],
    )
    sigs = format_signatures((AXPY_API, ApiSymbol("total", "function", result_type="real")))
    assert "- def axpy(n, a, x, y, ierr)  # returns the single value ierr" in sigs
    assert "- def total()  # returns the single value total" in sigs


def test_contract_in_prompt_uses_preserved_api_when_given():
    prompt = build_conversion_prompt(
        axpy_module(), [], preserved_api=(ApiSymbol("other", "subroutine"),)
    )
    assert "SUBROUTINE other" in prompt and "SUBROUTINE axpy" not in prompt


# --------------------------------------------------------------------------------------------
# Checklist and source
# --------------------------------------------------------------------------------------------


def test_checklist_groups_by_feature_with_original_lines_and_descriptions():
    text = format_checklist(axpy_module().hits)
    lines = text.splitlines()
    assert lines[0].startswith("1. [semantic] arg_output:")
    assert FEATURES["arg_output"].description in lines[0]
    assert lines[0].endswith("(1 occurrence)")
    assert "   - line 5: IERR = 0" in lines
    assert any(ln.startswith("2. [mechanical] do_labeled:") for ln in lines)
    assert "   - lines 6-8: DO 10 I = 1, N" in lines
    assert any(ln.startswith("3. [semantic] implicit_typing:") for ln in lines)


def test_checklist_caps_occurrences_and_flattens_snippets():
    hits = [FeatureHit("format_statement", n, n, f"FORMAT (1X,  I{n})\n") for n in range(1, 13)]
    text = format_checklist(hits)
    assert "(12 occurrences)" in text
    assert "line 8: FORMAT (1X, I8)" in text
    assert "line 9:" not in text
    assert "... and 4 more" in text
    long = format_checklist([FeatureHit("hollerith", 1, 1, "X" * 300)])
    assert "..." in long and len(long.splitlines()[1]) < 130


def test_checklist_unknown_feature_and_empty():
    assert "custom_thing: custom_thing" in format_checklist([FeatureHit("custom_thing", 3, 3, "x")])
    assert "No legacy constructs were detected" in format_checklist([])


def test_source_listing_has_line_number_gutter():
    prompt = build_conversion_prompt(axpy_module(), [])
    assert "```fortran\n" in prompt
    assert "   1 |       SUBROUTINE AXPY(N, A, X, Y, IERR)" in prompt
    assert "   2 | C     Y = A*X + Y" in prompt
    assert "  10 |       END" in prompt
    assert "NOT part of the source" in prompt
    assert AXPY not in prompt  # only the numbered form is shown


def test_format_numbered_width_and_blank_gutter():
    text = format_numbered([(9, "A"), (None, ""), (12345, "B  ")])
    assert text.splitlines() == ["    9 | A", "      |", "12345 | B"]


def test_sliced_source_keeps_original_line_numbers():
    module = state_module()
    bump, total = module.units[1], module.units[2]
    chunk = state_module()
    chunk.units = [bump, total]
    chunk.hits = [h for h in module.hits if 6 <= h.line <= 15]
    source = bump.source + "\n\n" + total.source
    prompt = build_conversion_prompt(chunk, [], source_override=source)
    assert "   6 |       SUBROUTINE BUMP(N)" in prompt
    assert "  10 |       END" in prompt
    assert "     |\n" in prompt  # separator line has an empty gutter
    assert "  12 |       INTEGER FUNCTION TOTAL()" in prompt
    assert "line 8: COMMON /CNT/ NCALLS" in prompt


def test_sliced_source_falls_back_to_plain_numbering_when_units_do_not_match():
    prompt = build_conversion_prompt(axpy_module(), [], source_override="      X = 1\n      END")
    assert "   1 |       X = 1" in prompt and "   2 |       END" in prompt


# --------------------------------------------------------------------------------------------
# Repair, slices, prelude
# --------------------------------------------------------------------------------------------


def test_repair_prompt_shows_previous_attempt_and_failures():
    prompt = build_conversion_prompt(
        axpy_module(),
        CONTEXT,
        feedback="- [fortran_remnant] line 4: GOTO\nFAILED test_axpy - assert 1 == 2",
        previous="def axpy(n):\n    goto 10\n",
    )
    assert prompt.startswith("Repair task:")
    for marker in (
        "## Your previous attempt",
        "def axpy(n):\n    goto 10",
        "## Failures to fix",
        "FAILED test_axpy - assert 1 == 2",
        "## Fortran 77 source (the authority)",
    ):
        assert marker in prompt
    assert prompt.index("## Your previous attempt") < prompt.index("## Failures to fix")
    assert prompt.index("## Failures to fix") < prompt.index("## Fortran 77 source (the authority)")
    assert "Fortran source is the authority" in prompt
    assert "follow the Fortran source" in prompt
    assert CALLING_CONVENTION.strip() in prompt
    assert "Task: port" not in prompt


def test_whitespace_feedback_is_not_repair():
    prompt = build_conversion_prompt(axpy_module(), [], feedback="  \n", previous="x = 1")
    assert prompt.startswith("Task: port") and "previous attempt" not in prompt


def test_other_chunks_note_and_state_prelude_sections():
    prompt = build_conversion_prompt(
        axpy_module(),
        [],
        source_override=AXPY,
        other_chunks_note="- def bump(n)  # returns None",
        state_prelude="class _Cnt:\n    ncalls = 0\n\ncnt = _Cnt()",
    )
    assert "one slice of a larger file" in prompt
    assert "- def bump(n)  # returns None" in prompt
    assert "## Shared module-level state (already converted)" in prompt
    assert "cnt = _Cnt()" in prompt and "Do NOT redefine them" in prompt
    assert prompt.index("## API contract") < prompt.index("## Shared module-level state")
    assert prompt.index("## Shared module-level state") < prompt.index("## Legacy constructs")


def test_slice_note_is_absent_for_whole_module():
    prompt = build_conversion_prompt(axpy_module(), [], other_chunks_note="- def bump(n)")
    assert "one slice of a larger file" not in prompt


def test_state_prelude_prompt():
    module = state_module()
    lines = [(1, "      BLOCK DATA INIT"), (2, "      COMMON /CNT/ NCALLS")]
    notes = [
        ContextSnippet(
            "c", "COMMON", "Model each COMMON block as a class.", 0.9, ("common_block",)
        ),
        ContextSnippet("d", "Arithmetic IF", "Use if/elif.", 0.5, ("if_arithmetic",)),
    ]
    prompt = build_state_prelude_prompt(module, notes, lines)
    assert "CALLING CONVENTION" in prompt and CATALOGUE in prompt
    assert "common_block" in prompt and "do_labeled" not in prompt
    assert "Model each COMMON block as a class." in prompt and "Use if/elif." not in prompt
    assert "   1 |       BLOCK DATA INIT" in prompt
    assert "Do NOT define any function" in prompt
    assert "block_data init" in prompt and "function total" in prompt
    assert "machine-validated" in prompt


def test_empty_fortran_routines_may_be_empty_functions():
    prompt = build_conversion_prompt(axpy_module(), [])
    assert "an empty Fortran routine may be an empty function" in prompt


def test_prelude_prompt_warns_about_commons_declared_differently_per_unit():
    prompt = build_state_prelude_prompt(state_module(), [], [(2, "      COMMON /CNT/ NCALLS")])
    assert "different variable names" in prompt and "associated by position" in prompt
