"""Prompt construction for the Fortran 77 to Python conversion step.

Everything here is a pure function of its inputs: ordering follows the order of the arguments
(or line numbers), never set or dict iteration order of unordered data.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from ..conventions import CALLING_CONVENTION, RUNTIME_MODULE
from ..features import FEATURES
from ..models import (
    ApiParam,
    ApiSymbol,
    CodeUnit,
    ContextSnippet,
    FeatureHit,
    ParsedModule,
)
from ..runtime import api_doc

SYSTEM_PROMPT = (
    "You are an expert in numerical software who ports legacy FORTRAN 77 code to Python (3.10 or "
    "newer, numpy). Your port must be faithful and behaviour-preserving: the same results for "
    "the same inputs, including integer truncation, DO-loop trip counts, column-major array "
    "indexing, fixed-length string semantics and the order of floating point operations. "
    "Your output is machine-validated and then executed under automatically generated tests: it "
    "must parse as Python, contain no Fortran remnants (GOTO, labels, FORMAT statements), and "
    "keep exactly the function signatures and return shapes of the API contract you are given. "
    "Reply with the complete converted module in a single ```python fenced block and nothing "
    "else."
)

MAX_OCCURRENCES_PER_FEATURE = 8
MAX_SNIPPET_CHARS = 100

# Features whose constructs describe state that outlives a single call or is shared between units.
STATE_FEATURES: tuple[str, ...] = (
    "common_block",
    "block_data",
    "data_statement",
    "save_statement",
    "equivalence",
)

_PY_TYPES = {
    "integer": "int",
    "real": "float",
    "double": "float",
    "complex": "complex",
    "logical": "bool",
    "character": "str",
    "external": "callable",
}

_OUTPUT_RULES = (
    "Output rules:",
    "- Return the COMPLETE converted Python module (not a diff, not a fragment, nothing elided) "
    "in ONE ```python fenced block and no prose before or after it.",
    "- The port must be faithful and behaviour-preserving. Do not refactor, vectorise, reorder "
    "floating point operations or 'improve' the algorithm; keep the Fortran variable names "
    "(lower case) and the order of the program units.",
    "- Follow the API contract exactly: function names, parameter names and order, and the "
    "return shape (including the order of returned outputs).",
    "- Preserve the Fortran comments as Python comments (a 'C' or '*' in column 1 and '!' "
    "comments become '#').",
    "- No placeholders: no TODO or FIXME stubs and no 'pass', '...' or NotImplementedError "
    "standing in for code that exists in the source (an empty Fortran routine may be an empty "
    "function).",
    f"- Imports: the standard library, numpy and the helpers listed above, imported only as "
    f"'from {RUNTIME_MODULE} import ...'. No scipy and no other third-party packages.",
    "- Do not reproduce the line-number gutter of the source listing and do not mention it.",
)

_REPAIR_RULES = (
    "- The Fortran source is the authority. If a failing test disagrees with what the Fortran "
    "source does, follow the Fortran source.",
    "- Change only what the failures require; keep everything that already works.",
)


# --------------------------------------------------------------------------------------------
# API contract
# --------------------------------------------------------------------------------------------


def _py_type(type_name: str) -> str:
    return _PY_TYPES.get(type_name, "a type to infer from the Fortran declarations")


def _is_output(sym: ApiSymbol, param: ApiParam) -> bool:
    return param.rank == 0 and param.name in sym.returns


def _param_line(sym: ApiSymbol, param: ApiParam) -> str:
    if param.type == "external":
        return f"{param.name}: callable (Fortran EXTERNAL procedure, same calling convention)"
    if param.rank > 0:
        if param.type == "character":
            kind = "list of str"
        else:
            kind = f"numpy.ndarray of {_py_type(param.type)}"
        use = (
            "modified IN PLACE, never returned" if param.intent in ("out", "inout") else "read only"
        )
        return f"{param.name}: ARRAY of rank {param.rank}, {kind}, allocated by the caller, {use}"
    kind = f"{_py_type(param.type)} scalar"
    if _is_output(sym, param):
        how = "OUTPUT, assigned by the unit and RETURNED"
        if param.intent == "inout":
            how += " (its input value is also read)"
        return f"{param.name}: {kind}, {how}"
    return f"{param.name}: {kind}, input only"


def _return_elements(sym: ApiSymbol) -> list[str]:
    elements: list[str] = []
    if sym.kind == "function":
        elements.append(f"{sym.name} (the function result, {_py_type(sym.result_type or '')})")
    elements.extend(f"{name} (output argument)" for name in sym.returns)
    if sym.alt_returns:
        elements.append("alt (int: number of the alternate RETURN taken, 0 for a normal RETURN)")
    return elements


def return_shape(sym: ApiSymbol) -> str:
    """One-line description of what the Python function returns, in exact tuple order."""
    elements = _return_elements(sym)
    if sym.kind == "program":
        return "None"
    if not elements:
        return "None"
    if len(elements) == 1:
        return f"the single value {elements[0]} (not wrapped in a tuple)"
    return (
        "a tuple in exactly this order: (" + ", ".join(e.split(" ", 1)[0] for e in elements) + ")"
    )


def python_signature(sym: ApiSymbol) -> str:
    """The ``def`` line the converted unit must have."""
    if sym.kind == "program":
        return "def main()"
    names = [p.name for p in sym.params if p.name != "*"]
    return f"def {sym.name}({', '.join(names)})"


def format_api_contract(symbols: Sequence[ApiSymbol]) -> str:
    """Render the full per-unit API contract (signature, parameters, outputs, return shape)."""
    blocks: list[str] = []
    for sym in symbols:
        if sym.kind == "block_data":
            blocks.append(
                f"- BLOCK DATA {sym.name}: no function. Initialise the module-level state of "
                "the COMMON blocks it names (at import time)."
            )
            continue
        lines = [f"- {sym.kind.upper()} {sym.name}"]
        lines.append(f"  python: {python_signature(sym)}")
        real_params = [p for p in sym.params if p.name != "*"]
        if real_params:
            lines.append("  parameters:")
            lines.extend(f"    - {_param_line(sym, p)}" for p in real_params)
        else:
            lines.append("  parameters: none")
        lines.append(f"  returns: {return_shape(sym)}")
        if len(_return_elements(sym)) > 1:
            lines.extend(f"    - {e}" for e in _return_elements(sym))
        if sym.alt_returns:
            lines.append(
                f"  alternate returns: {sym.alt_returns} '*' dummy argument(s) are not Python "
                "parameters; 'RETURN n' is reported through the trailing int."
            )
        if sym.kind == "function":
            lines.append(
                f"  result type: {_py_type(sym.result_type or '')} "
                f"(assigned to the name '{sym.name}' in the Fortran source)"
            )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def format_signatures(symbols: Sequence[ApiSymbol]) -> str:
    """Compact one-line-per-unit signatures, used to describe units that live in other slices."""
    lines: list[str] = []
    for sym in symbols:
        if sym.kind == "block_data":
            lines.append(f"- block data {sym.name} (module-level state only)")
        else:
            lines.append(f"- {python_signature(sym)}  # returns {return_shape(sym)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Checklist, notes, source
# --------------------------------------------------------------------------------------------


def _snippet_text(snippet: str) -> str:
    flat = " ".join(snippet.split())
    if len(flat) > MAX_SNIPPET_CHARS:
        flat = flat[: MAX_SNIPPET_CHARS - 3] + "..."
    return flat


def format_checklist(hits: Sequence[FeatureHit]) -> str:
    """Numbered checklist of detected constructs, grouped by feature, with original line numbers."""
    if not hits:
        return (
            "No legacy constructs were detected automatically. Still read the whole source "
            "carefully (implicit typing, integer division, DO-loop semantics, array indexing) "
            "before porting it."
        )
    by_feature: dict[str, list[FeatureHit]] = defaultdict(list)
    for hit in hits:
        by_feature[hit.feature_id].append(hit)
    groups = sorted(
        ((min(h.line for h in group), fid, group) for fid, group in by_feature.items()),
        key=lambda item: (item[0], item[1]),
    )
    lines: list[str] = []
    for number, (_, feature_id, group) in enumerate(groups, start=1):
        spec = FEATURES.get(feature_id)
        description = spec.description if spec else feature_id
        severity = f"[{spec.severity}] " if spec else ""
        count = len(group)
        noun = "occurrence" if count == 1 else "occurrences"
        lines.append(f"{number}. {severity}{feature_id}: {description} ({count} {noun})")
        ordered = sorted(group, key=lambda h: (h.line, h.end_line, h.snippet))
        for hit in ordered[:MAX_OCCURRENCES_PER_FEATURE]:
            where = (
                f"line {hit.line}"
                if hit.end_line <= hit.line
                else f"lines {hit.line}-{hit.end_line}"
            )
            lines.append(f"   - {where}: {_snippet_text(hit.snippet)}")
        if count > MAX_OCCURRENCES_PER_FEATURE:
            lines.append(f"   - ... and {count - MAX_OCCURRENCES_PER_FEATURE} more")
    return "\n".join(lines)


def _context_section(context: Sequence[ContextSnippet]) -> str:
    if not context:
        return "(no migration notes were retrieved)"
    parts = [f"Note {i} - {snip.title}\n{snip.text.strip()}" for i, snip in enumerate(context, 1)]
    return "\n\n".join(parts)


def format_numbered(lines: Iterable[tuple[int | None, str]]) -> str:
    """Render ``(line_number, text)`` pairs with a line-number gutter; ``None`` leaves it blank."""
    rows = [(n, text.rstrip()) for n, text in lines]
    numbers = [n for n, _ in rows if n is not None]
    width = max(4, len(str(max(numbers)))) if numbers else 4
    out = []
    for number, text in rows:
        gutter = " " * width if number is None else f"{number:>{width}}"
        out.append(f"{gutter} | {text}".rstrip() if text else f"{gutter} |")
    return "\n".join(out)


def _numbered_lines(source: str, units: Sequence[CodeUnit], *, from_units: bool) -> str:
    """Gutter-number ``source``: 1-based for a whole file, original unit numbers for a slice."""
    lines = source.rstrip("\n").split("\n")
    numbers: list[int | None] = list(range(1, len(lines) + 1))
    if from_units and units:
        located: list[int | None] = [None] * len(lines)
        cursor = 0
        for unit in units:
            text = unit.source.rstrip("\n")
            index = source.find(text, cursor)
            if index < 0:
                break
            first = source.count("\n", 0, index)
            for offset in range(text.count("\n") + 1):
                if first + offset < len(located):
                    located[first + offset] = unit.start_line + offset
            cursor = index + len(text)
        else:
            numbers = located
    return format_numbered(zip(numbers, lines, strict=True))


def _source_block(numbered: str) -> str:
    return f"```fortran\n{numbered}\n```"


def _python_block(code: str) -> str:
    return f"```python\n{code.strip(chr(10))}\n```"


def _runtime_section() -> str:
    catalogue = api_doc().strip()
    if not catalogue:
        return (
            f"## Runtime helpers (module `{RUNTIME_MODULE}`)\n"
            "(no helpers are available: use only numpy and the standard library)"
        )
    return (
        f"## Runtime helpers (module `{RUNTIME_MODULE}`)\n"
        f"Import ONLY names from this catalogue, as 'from {RUNTIME_MODULE} import name1, name2'. "
        "Prefer these helpers for Fortran-specific semantics (integer division, MOD, SIGN, NINT, "
        "DO trip counts, character handling, FORMAT I/O) instead of re-implementing them.\n"
        f"```\n{catalogue}\n```"
    )


def _convention_section() -> str:
    return "## Calling convention\n" + CALLING_CONVENTION.strip()


# --------------------------------------------------------------------------------------------
# Public builders
# --------------------------------------------------------------------------------------------


def build_conversion_prompt(
    module: ParsedModule,
    context: Sequence[ContextSnippet],
    *,
    feedback: str | None = None,
    previous: str | None = None,
    source_override: str | None = None,
    preserved_api: Sequence[ApiSymbol] | None = None,
    other_chunks_note: str | None = None,
    state_prelude: str | None = None,
) -> str:
    """Build the AST-guided conversion prompt, or a repair prompt when ``feedback`` is given.

    ``source_override`` replaces ``module.source`` (one slice of a large module; its line-number
    gutter then follows the original line numbers of ``module.units``); ``preserved_api``
    defaults to ``module.api``; ``other_chunks_note`` describes the units that live in the other
    slices; ``state_prelude`` is the already converted shared module-level state.
    """
    sliced = source_override is not None
    source = source_override if source_override is not None else module.source
    api = module.api if preserved_api is None else tuple(preserved_api)
    feedback_text = (feedback or "").strip()
    repair = bool(feedback_text)

    sections: list[str] = []
    if repair:
        sections.append(
            f"Repair task: your previous Python port of the Fortran 77 file `{module.path}` "
            "failed validation or testing. Fix the concrete failures below and return the "
            "corrected, complete module. The output is machine-validated and executed under "
            "tests, and it must remain a faithful, behaviour-preserving port."
        )
    else:
        sections.append(
            f"Task: port the Fortran 77 (fixed-form) file `{module.path}` to a self-contained "
            "Python module using numpy. The output is machine-validated (it must parse, contain "
            "no Fortran remnants and keep the exact API contract below) and then executed under "
            "automatically generated pytest tests, so it must be a faithful, "
            "behaviour-preserving port."
        )

    if sliced and other_chunks_note:
        sections.append(
            "This source is one slice of a larger file. Slices are converted separately and "
            "merged afterwards (duplicate top-level imports are removed, the shared state "
            "below is placed first). Convert only the units in this slice. The other slices "
            "define the following module-level functions, which you may call but must not "
            "redefine:\n" + other_chunks_note.strip()
        )

    sections.append(_convention_section())
    sections.append(_runtime_section())
    sections.append(
        "## API contract (follow exactly)\n"
        + (format_api_contract(api) if api else "(no program units were detected)")
    )
    if state_prelude and state_prelude.strip():
        sections.append(
            "## Shared module-level state (already converted)\n"
            "This code was produced separately for the COMMON, DATA, BLOCK DATA and SAVE state of "
            "the whole file and is placed at the top of the final module, before your code. Use "
            "these names exactly to read and write the shared variables. Do NOT redefine them and "
            "do not repeat this code in your reply.\n" + _python_block(state_prelude)
        )
    sections.append(
        "## Legacy constructs detected by the parser (handle every one)\n"
        "Line numbers refer to the source listing below.\n" + format_checklist(module.hits)
    )
    sections.append("## Migration notes\n" + _context_section(context))

    listing = _source_block(_numbered_lines(source, module.units, from_units=sliced))
    gutter_note = (
        "The left gutter (line number and '|') is NOT part of the source; the Fortran text "
        "starts after it. Fixed form: 'C', 'c' or '*' in column 1 is a comment, columns 1-5 hold "
        "a label, a character in column 6 continues the previous line, statements are in "
        "columns 7-72."
    )
    if repair:
        if previous and previous.strip():
            sections.append("## Your previous attempt\n" + _python_block(previous))
        sections.append("## Failures to fix\n" + feedback_text)
        sections.append("## Fortran 77 source (the authority)\n" + gutter_note + "\n" + listing)
    else:
        sections.append("## Fortran 77 source to convert\n" + gutter_note + "\n" + listing)

    rules = list(_OUTPUT_RULES)
    if repair:
        rules[1:1] = list(_REPAIR_RULES)
    sections.append("\n".join(rules))
    return "\n\n".join(sections) + "\n"


def build_state_prelude_prompt(
    module: ParsedModule,
    context: Sequence[ContextSnippet],
    state_lines: Sequence[tuple[int, str]],
) -> str:
    """Prompt for the shared-state prelude: only COMMON/DATA/BLOCK DATA/SAVE state, as Python.

    ``state_lines`` are the ``(original_line_number, text)`` pairs of the state declarations of
    all units (see ``chunking.state_lines``).
    """
    state_hits = [h for h in module.hits if h.feature_id in STATE_FEATURES]
    notes = [c for c in context if not c.feature_ids or set(c.feature_ids) & set(STATE_FEATURES)]
    units = ", ".join(f"{u.kind} {u.name}" for u in module.units) or "(none)"
    sections = [
        f"Task: the Fortran 77 file `{module.path}` is too large to convert in one piece. Before "
        "its program units are converted separately, write the module-level Python state that "
        "they will share: the COMMON blocks, the BLOCK DATA initial values, the DATA "
        "initialisation of COMMON variables, and the SAVE-d or DATA-initialised locals that "
        "persist between calls. The output is machine-validated and executed under tests.",
        "Program units of the file: " + units,
        _convention_section(),
        _runtime_section(),
        "## State constructs detected by the parser (handle every one)\n"
        "Line numbers refer to the listing below.\n" + format_checklist(state_hits),
        "## Migration notes\n" + _context_section(notes),
        "## State declarations extracted from all program units\n"
        "The left gutter (line number and '|') is NOT part of the source. Only declarations are "
        "shown; ignore every statement that is not shared or persistent state.\n"
        + _source_block(format_numbered(state_lines)),
        "Output rules:\n"
        "- Return ONE ```python fenced block and no prose.\n"
        "- Write only imports, constants and module-level state (for example one small class "
        "instance per COMMON block with the Fortran variable names as attributes, arrays "
        "allocated with the declared extents, initial values from DATA and BLOCK DATA applied at "
        "import time). Do NOT define any function for a program unit.\n"
        "- The same COMMON block may be declared with different variable names, types or "
        "shapes in different units (storage is associated by position). Define each block once "
        "and add a comment with the name mapping of every unit, or per-unit alias names that "
        "refer to the same storage.\n"
        "- Name persistent locals of a unit '<unit>_<variable>' so that the unit can use them.\n"
        "- Use exactly the same module-level names that every unit would need; they are shown "
        "to the unit converters as fixed context.\n"
        "- Preserve the Fortran comments as Python comments. No placeholders or TODO stubs.\n"
        f"- Imports: the standard library, numpy and the helpers above ('from {RUNTIME_MODULE} "
        "import ...'). No other third-party packages. Do not reproduce the gutter.",
    ]
    return "\n\n".join(sections) + "\n"
