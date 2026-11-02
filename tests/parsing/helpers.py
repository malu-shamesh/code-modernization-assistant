"""Shared helpers for the parsing tests."""

from __future__ import annotations

import re
import textwrap

from modernizer.models import ParsedModule
from modernizer.parsing import analyze, detect_features

_COMMENT = re.compile(r"[Cc*!](?:\s|$)")
_LABELLED = re.compile(r"(\d+)\s+(.*)")


def fixed(text: str) -> str:
    """Turn a compact listing into fixed-form source.

    Lines are written without the six-column margin: ``10 CONTINUE`` becomes a labelled
    statement, ``& more`` a continuation line and ``C text`` a comment line.
    """
    out: list[str] = []
    for line in textwrap.dedent(text).strip("\n").splitlines():
        labelled = _LABELLED.match(line)
        if not line.strip():
            out.append("")
        elif _COMMENT.match(line):
            out.append(line)
        elif labelled:
            out.append(f"{labelled.group(1):>5} {labelled.group(2)}")
        elif line.startswith("&"):
            out.append("     &" + line[1:])
        else:
            out.append("      " + line)
    return "\n".join(out) + "\n"


def feature_ids(source: str) -> set[str]:
    return {h.feature_id for h in detect_features(source)}


def parse(text: str, path: str = "t.f") -> ParsedModule:
    return analyze(path, fixed(text))
