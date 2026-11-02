"""Reading Fortran sources from disk and writing converted output."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

SOURCE_SUFFIXES = (".f", ".for", ".f77", ".ftn")
_SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules", ".tox"}


def read_source(path: Path) -> str:
    """Decode a source file: UTF-8 first, then latin-1, which legacy Fortran often used."""
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def collect_sources(paths: Iterable[Path]) -> dict[str, str]:
    """Map relative posix path to source for every Fortran file under the given files or dirs."""
    sources: dict[str, str] = {}
    for root in paths:
        if root.is_file():
            sources[root.name] = read_source(root)
            continue
        for file in sorted(root.rglob("*")):
            relative = file.relative_to(root)
            if file.suffix.lower() not in SOURCE_SUFFIXES or not file.is_file():
                continue
            if any(part in _SKIP_DIRS for part in relative.parts):
                continue
            sources[relative.as_posix()] = read_source(file)
    return sources


def write_output(out_dir: Path, relative_path: str, text: str) -> Path:
    target = (out_dir / relative_path).resolve()
    if out_dir.resolve() not in target.parents:
        raise ValueError(f"refusing to write outside the output directory: {relative_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target
