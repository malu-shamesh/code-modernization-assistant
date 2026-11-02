"""Run the LLM conversion, chunking large modules and merging the converted chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..models import ContextSnippet, ParsedModule, TokenUsage
from ..protocols import LLMClient
from .chunking import (
    build_chunks,
    chunk_api,
    has_shared_state,
    other_chunks_note,
    pack_units,
    state_lines,
)
from .extract import extract_code
from .merge import merge_sources
from .prompts import SYSTEM_PROMPT, build_conversion_prompt, build_state_prelude_prompt


class ConversionError(Exception):
    """The model reply contained no usable code. Carries what was received so far."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        raw_responses: Sequence[str] = (),
        usage: TokenUsage | None = None,
        chunk_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.raw_responses = list(raw_responses)
        self.usage = usage or TokenUsage()
        self.chunk_index = chunk_index


@dataclass
class ConversionOutput:
    source: str
    usage: TokenUsage
    calls: int
    chunks: int  # converted slices of program units (the state prelude call is not a chunk)
    raw_responses: list[str] = field(default_factory=list)


def convert_module(
    module: ParsedModule,
    context: Sequence[ContextSnippet],
    llm: LLMClient,
    *,
    feedback: str | None = None,
    previous: str | None = None,
    max_chunk_chars: int = 12000,
    max_output_tokens: int = 8192,
    tags: Mapping[str, str] | None = None,
) -> ConversionOutput:
    """Convert a Fortran module with ``llm``, or repair ``previous`` when ``feedback`` is given.

    Modules up to ``max_chunk_chars`` characters use a single call. Larger ones are split on
    program-unit boundaries; when COMMON or BLOCK DATA state couples the units, one extra call
    first converts that state into a module-level prelude that every chunk prompt receives.
    The converted pieces are merged (prelude first). Repair always converts the whole module in
    one call. Raises :class:`ConversionError` if a reply has no extractable code.
    """
    repair = bool((feedback or "").strip())
    call_tags = {**(tags or {}), "stage": "repair" if repair else "convert"}
    usage = TokenUsage()
    raw: list[str] = []

    def call(prompt: str, where: str, chunk_index: int | None) -> str:
        nonlocal usage
        response = llm.generate(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            tags=dict(call_tags),
        )
        usage = usage + response.usage
        raw.append(response.text)
        code = extract_code(response.text)
        if not code:
            raise ConversionError(
                f"model reply{where} contained no extractable Python code",
                raw_response=response.text,
                raw_responses=raw,
                usage=usage,
                chunk_index=chunk_index,
            )
        return code

    groups = (
        pack_units(module, max_chunk_chars)
        if not repair and len(module.source) > max_chunk_chars
        else []
    )
    if len(groups) <= 1:
        prompt = build_conversion_prompt(module, context, feedback=feedback, previous=previous)
        source = call(prompt, "", None)
        return ConversionOutput(source, usage, len(raw), 1, raw)

    chunks = build_chunks(module, groups)

    prelude: str | None = None
    if has_shared_state(module):
        lines = state_lines(module)
        if lines:
            prompt = build_state_prelude_prompt(module, context, lines)
            prelude = call(prompt, " for the shared-state prelude", None)

    converted = [
        call(
            build_conversion_prompt(
                chunk.module,
                context,
                source_override=chunk.source,
                preserved_api=chunk_api(module, chunk),
                other_chunks_note=other_chunks_note(module, chunks, index),
                state_prelude=prelude,
            ),
            f" for chunk {index + 1}",
            index,
        )
        for index, chunk in enumerate(chunks)
    ]
    sources = [prelude, *converted] if prelude else converted
    merged = merge_sources(sources, prelude=prelude is not None)
    return ConversionOutput(merged, usage, len(raw), len(chunks), raw)
