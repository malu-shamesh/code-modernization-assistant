"""Pull Python source out of a free-form LLM reply."""

from __future__ import annotations

import keyword
import re

_FENCE_MARK = "```"
_OPEN_FENCE = re.compile(r"^\s*(`{3,})\s*([\w+.-]*)\s*$")
_CLOSE_FENCE = re.compile(r"^\s*`{3,}\s*$")
_PYTHON_TAGS = frozenset({"", "python", "py", "python3", "py3"})
_KEYWORDS = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)
_PROSE_END = (":", ".", "!", "?")


def _fenced_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Return ``(language_tag, body_lines)`` for every fenced block, tolerating a missing close."""
    blocks: list[tuple[str, list[str]]] = []
    tag: str | None = None
    body: list[str] = []
    for line in lines:
        if tag is None:
            match = _OPEN_FENCE.match(line)
            if match:
                tag, body = match.group(2).lower(), []
            continue
        if _CLOSE_FENCE.match(line):
            blocks.append((tag, body))
            tag, body = None, []
            continue
        match = _OPEN_FENCE.match(line)
        if match and match.group(2):
            # A new tagged fence while one is still open means the first was never closed.
            blocks.append((tag, body))
            tag, body = match.group(2).lower(), []
            continue
        body.append(line)
    if tag is not None:
        blocks.append((tag, body))
    return blocks


def _looks_like_prose(line: str) -> bool:
    if not line or line[0].isspace() or not line[0].isalpha():
        return False
    stripped = line.rstrip()
    words = stripped.split()
    first = re.match(r"[A-Za-z_]+", words[0])
    if first is None or first.group(0) in _KEYWORDS:
        return False
    return len(words) >= 3 and stripped.endswith(_PROSE_END) and "=" not in stripped


def _strip_prose(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and (not lines[start].strip() or _looks_like_prose(lines[start])):
        start += 1
    while end > start and (not lines[end - 1].strip() or _looks_like_prose(lines[end - 1])):
        end -= 1
    return lines[start:end]


def _finish(lines: list[str]) -> str:
    kept = [ln for ln in lines if not ln.lstrip().startswith(_FENCE_MARK)]
    code = "\n".join(kept).strip("\n")
    if not code.strip() or _FENCE_MARK in code:
        return ""
    return code + "\n"


def extract_code(text: str) -> str:
    """Extract Python source from an LLM reply.

    Prefers the longest fenced block tagged python/py (or untagged), tolerates a missing closing
    fence, and without any fence drops leading/trailing lines that are clearly prose. Returns an
    empty string when nothing usable is found. The result never contains fence markers.
    """
    if not text or not text.strip():
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = _fenced_blocks(lines)
    if blocks:
        candidates = [_finish(body) for tag, body in blocks if tag in _PYTHON_TAGS]
        best = max(candidates, key=len, default="")
        foreign = any(tag not in _PYTHON_TAGS for tag, _ in blocks)
        if best or foreign:
            return best
    return _finish(_strip_prose(lines))
