"""Curated Fortran 77 -> Python migration knowledge: the documents the retriever indexes."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..features import FEATURES

DEFAULT_CORPUS_PACKAGE = "modernizer.retrieval"
DEFAULT_CORPUS_RELPATH = ("data", "f77_to_python.jsonl")
DEFAULT_RENDER_CHARS = 1800


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


@dataclass(frozen=True)
class KnowledgeEntry:
    """One migration pattern: a Fortran 77 snippet, its Python equivalent and the pitfalls."""

    id: str
    title: str
    feature_ids: tuple[str, ...]
    legacy: str
    modern: str
    notes: str = ""

    def embedding_text(self) -> str:
        """Text that represents this entry in the vector index.

        The modern code is left out on purpose: queries are legacy source, so the legacy
        side, the title, the construct descriptions and the notes carry the signal.
        """
        descriptions = [FEATURES[f].description for f in self.feature_ids if f in FEATURES]
        parts = [self.title, *descriptions, self.legacy, self.notes]
        return "\n".join(p for p in parts if p)

    def render(self, max_chars: int = DEFAULT_RENDER_CHARS) -> str:
        """Prompt-ready block. The result never exceeds ``max_chars``.

        Notes are sacrificed first, then the code blocks are clipped evenly.
        """
        legacy = self.legacy.strip("\n")
        modern = self.modern.strip("\n")
        notes = self.notes.strip()

        def build(legacy_text: str, modern_text: str, notes_text: str) -> str:
            lines = [
                f"### {self.title}",
                "Fortran 77:",
                "```fortran",
                legacy_text,
                "```",
                "Python:",
                "```python",
                modern_text,
                "```",
            ]
            if notes_text:
                lines.append(f"Notes: {notes_text}")
            return "\n".join(lines)

        full = build(legacy, modern, notes)
        if len(full) <= max_chars:
            return full

        no_notes = build(legacy, modern, "")
        if len(no_notes) <= max_chars:
            budget = max_chars - len(no_notes) - len("\nNotes: ")
            return build(legacy, modern, _clip(notes, budget)) if budget > 0 else no_notes

        overhead = len(build("", "", ""))
        budget = max(max_chars - overhead, 0)
        each = budget // 2
        clipped = build(_clip(legacy, each), _clip(modern, each), "")
        return clipped[:max_chars]


def _entry_from_record(record: object, where: str) -> KnowledgeEntry:
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(record).__name__}")
    for key in ("id", "title", "legacy", "modern"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where}: field '{key}' must be a non-empty string")
    notes = record.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError(f"{where}: field 'notes' must be a string")
    feature_ids = record.get("feature_ids", [])
    if not isinstance(feature_ids, list) or not all(isinstance(f, str) for f in feature_ids):
        raise ValueError(f"{where}: field 'feature_ids' must be a list of strings")
    unknown = [f for f in feature_ids if f not in FEATURES]
    if unknown:
        raise ValueError(f"{where}: unknown feature id(s) {unknown}")
    return KnowledgeEntry(
        id=record["id"],
        title=record["title"],
        feature_ids=tuple(dict.fromkeys(feature_ids)),
        legacy=record["legacy"],
        modern=record["modern"],
        notes=notes,
    )


def _entry_to_record(entry: KnowledgeEntry) -> dict[str, object]:
    return {
        "id": entry.id,
        "title": entry.title,
        "feature_ids": list(entry.feature_ids),
        "legacy": entry.legacy,
        "modern": entry.modern,
        "notes": entry.notes,
    }


def _dump_line(entry: KnowledgeEntry) -> str:
    return json.dumps(_entry_to_record(entry), ensure_ascii=False)


class KnowledgeBase:
    """Ordered, id-unique collection of :class:`KnowledgeEntry`. Safe to share across threads."""

    def __init__(self, entries: Iterable[KnowledgeEntry] = ()) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}
        self._lock = threading.RLock()
        for entry in entries:
            self.add(entry)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> KnowledgeBase:
        path = Path(path)
        with path.open(encoding="utf-8-sig") as handle:
            return cls._from_lines(handle, str(path))

    @classmethod
    def default(cls) -> KnowledgeBase:
        """Load the corpus packaged with the library."""
        resource = resources.files(DEFAULT_CORPUS_PACKAGE)
        for part in DEFAULT_CORPUS_RELPATH:
            resource = resource / part
        if not resource.is_file():
            raise FileNotFoundError(
                f"packaged corpus not found: {'/'.join(DEFAULT_CORPUS_RELPATH)} "
                f"in {DEFAULT_CORPUS_PACKAGE}"
            )
        text = resource.read_text(encoding="utf-8-sig")
        # str.splitlines() would also split on U+2028, U+0085 and similar characters that JSON
        # permits unescaped inside strings; only a newline separates records.
        return cls._from_lines(text.split("\n"), "/".join(DEFAULT_CORPUS_RELPATH))

    @classmethod
    def _from_lines(cls, lines: Iterable[str], source: str) -> KnowledgeBase:
        kb = cls()
        first_seen: dict[str, int] = {}
        for lineno, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue
            where = f"{source}:{lineno}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
            entry = _entry_from_record(record, where)
            if entry.id in first_seen:
                raise ValueError(
                    f"{where}: duplicate id '{entry.id}' (first defined on line "
                    f"{first_seen[entry.id]})"
                )
            first_seen[entry.id] = lineno
            kb.add(entry)
        return kb

    def add(self, entry: KnowledgeEntry) -> None:
        unknown = [f for f in entry.feature_ids if f not in FEATURES]
        if unknown:
            raise ValueError(f"entry '{entry.id}': unknown feature id(s) {unknown}")
        with self._lock:
            if entry.id in self._entries:
                raise ValueError(f"duplicate knowledge entry id '{entry.id}'")
            self._entries[entry.id] = entry

    def get(self, entry_id: str) -> KnowledgeEntry:
        """Return the entry with this id; raises ``KeyError`` if absent."""
        with self._lock:
            return self._entries[entry_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, entry_id: object) -> bool:
        with self._lock:
            return entry_id in self._entries

    def __iter__(self) -> Iterator[KnowledgeEntry]:
        return iter(self.entries)

    @property
    def entries(self) -> list[KnowledgeEntry]:
        """Snapshot of all entries in insertion order."""
        with self._lock:
            return list(self._entries.values())

    def by_feature(self, feature_id: str) -> list[KnowledgeEntry]:
        with self._lock:
            return [e for e in self._entries.values() if feature_id in e.feature_ids]

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(_dump_line(e) + "\n" for e in self.entries)
        path.write_text(text, encoding="utf-8", newline="\n")

    def append_jsonl(self, path: str | Path, entry: KnowledgeEntry) -> None:
        """Add ``entry`` to this knowledge base and append it to the JSONL file at ``path``."""
        path = Path(path)
        # Serialise before touching the knowledge base so an unencodable entry fails cleanly.
        line = (_dump_line(entry) + "\n").encode("utf-8")
        with self._lock:
            self.add(entry)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                needs_newline = False
                if path.exists() and path.stat().st_size > 0:
                    with path.open("rb") as existing:
                        existing.seek(-1, 2)
                        needs_newline = existing.read(1) != b"\n"
                with path.open("ab") as handle:
                    handle.write((b"\n" if needs_newline else b"") + line)
            except OSError:
                del self._entries[entry.id]
                raise
