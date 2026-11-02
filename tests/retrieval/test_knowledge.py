from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from modernizer.retrieval import KnowledgeBase, KnowledgeEntry
from modernizer.retrieval import knowledge as knowledge_module

GOTO_LEGACY = "      GO TO (10, 20), K\n   10 CONTINUE\n   20 CONTINUE"
GOTO_MODERN = "if k == 1:\n    pass\nelif k == 2:\n    pass"


def make_entry(
    entry_id: str = "e1", features: tuple[str, ...] = ("goto_computed",)
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        title="computed GO TO dispatch",
        feature_ids=features,
        legacy=GOTO_LEGACY,
        modern=GOTO_MODERN,
        notes="An index outside 1..n falls through to the next statement.",
    )


def record(entry_id: str = "e1", **overrides) -> dict:
    base = {
        "id": entry_id,
        "title": "t",
        "feature_ids": ["goto_computed"],
        "legacy": "l",
        "modern": "m",
        "notes": "n",
    }
    base.update(overrides)
    return base


def write_jsonl(path: Path, lines: list[object]) -> Path:
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_embedding_text_contains_legacy_side_and_feature_descriptions():
    text = make_entry().embedding_text()
    assert "computed GO TO dispatch" in text
    assert "computed GO TO (l1, l2, ...), i" in text
    assert "GO TO (10, 20), K" in text
    assert "An index outside 1..n" in text
    assert "elif k == 2" not in text


def test_render_contains_both_code_blocks_and_notes():
    rendered = make_entry().render()
    assert rendered.index("Fortran 77:") < rendered.index("Python:")
    assert rendered.index("```fortran") < rendered.index("```python")
    assert "GO TO (10, 20), K" in rendered and "elif k == 2" in rendered
    assert "Notes: An index outside 1..n falls through to the next statement." in rendered
    assert rendered.count("```") == 4
    assert "Python 2" not in rendered


def test_render_keeps_fixed_form_leading_blanks_but_strips_surrounding_newlines():
    legacy = "\n\n      GO TO 10\n   10 CONTINUE\n\n"
    entry = KnowledgeEntry("x", "t", (), legacy, "\n\nx = 1\n\n")
    lines = entry.render().split("\n")
    start = lines.index("```fortran")
    assert lines[start + 1 : start + 3] == ["      GO TO 10", "   10 CONTINUE"]
    assert lines[start + 3] == "```"


def test_render_without_notes_has_no_notes_line():
    entry = KnowledgeEntry("x", "t", (), "a", "b", "")
    assert "Notes" not in entry.render()


@pytest.mark.parametrize("limit", [20, 60, 120, 200, 260, 400])
def test_render_never_exceeds_max_chars(limit):
    entry = KnowledgeEntry(
        "x", "title", (), "legacy line\n" * 30, "modern line\n" * 30, "note " * 50
    )
    assert len(entry.render(max_chars=limit)) <= limit


def test_render_drops_notes_before_code():
    entry = KnowledgeEntry("x", "title", (), "old()", "new()", "n" * 500)
    rendered = entry.render(max_chars=150)
    assert "old()" in rendered and "new()" in rendered
    assert len(rendered) <= 150


def test_jsonl_round_trip(tmp_path):
    kb = KnowledgeBase([make_entry("a"), make_entry("b", ("format_statement", "io_formatted"))])
    path = tmp_path / "out" / "kb.jsonl"
    kb.save_jsonl(path)
    loaded = KnowledgeBase.from_jsonl(path)
    assert loaded.entries == kb.entries
    assert [e.id for e in loaded.entries] == ["a", "b"]


def test_jsonl_preserves_non_ascii(tmp_path):
    entry = KnowledgeEntry("u", "unicode", (), "      S = 'caf\u00e9'", "s = 'caf\u00e9'", "")
    path = tmp_path / "kb.jsonl"
    KnowledgeBase([entry]).save_jsonl(path)
    assert "caf\u00e9" in path.read_text(encoding="utf-8")
    assert KnowledgeBase.from_jsonl(path).get("u") == entry


def test_from_jsonl_skips_blank_lines_and_defaults_notes(tmp_path):
    rec = record()
    del rec["notes"]
    path = write_jsonl(tmp_path / "kb.jsonl", ["", rec, "   ", record("e2")])
    kb = KnowledgeBase.from_jsonl(path)
    assert len(kb) == 2
    assert kb.get("e1").notes == ""


def test_unknown_feature_id_reports_line_number(tmp_path):
    path = write_jsonl(
        tmp_path / "kb.jsonl", [record("a"), record("b", feature_ids=["no_such_feature"])]
    )
    with pytest.raises(ValueError, match=r"kb\.jsonl:2.*no_such_feature"):
        KnowledgeBase.from_jsonl(path)


def test_invalid_json_reports_line_number(tmp_path):
    path = write_jsonl(tmp_path / "kb.jsonl", [record("a"), "{not json"])
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        KnowledgeBase.from_jsonl(path)


@pytest.mark.parametrize("missing", ["id", "title", "legacy", "modern"])
def test_missing_required_field_is_rejected(tmp_path, missing):
    rec = record()
    del rec[missing]
    path = write_jsonl(tmp_path / "kb.jsonl", [rec])
    with pytest.raises(ValueError, match=rf":1: field '{missing}'"):
        KnowledgeBase.from_jsonl(path)


@pytest.mark.parametrize(
    "overrides",
    [{"feature_ids": "goto_computed"}, {"feature_ids": [1]}, {"notes": 3}, {"legacy": ""}],
)
def test_wrongly_typed_fields_are_rejected(tmp_path, overrides):
    path = write_jsonl(tmp_path / "kb.jsonl", [record(**overrides)])
    with pytest.raises(ValueError, match=":1:"):
        KnowledgeBase.from_jsonl(path)


def test_non_object_line_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "kb.jsonl", ["[1, 2]"])
    with pytest.raises(ValueError, match=r":1: expected a JSON object"):
        KnowledgeBase.from_jsonl(path)


def test_duplicate_id_reports_both_lines(tmp_path):
    path = write_jsonl(tmp_path / "kb.jsonl", [record("a"), record("b"), record("a")])
    with pytest.raises(ValueError, match=r":3: duplicate id 'a'.*line 1"):
        KnowledgeBase.from_jsonl(path)


def test_add_validates_features_and_duplicates():
    kb = KnowledgeBase()
    kb.add(make_entry("a"))
    with pytest.raises(ValueError, match="duplicate"):
        kb.add(make_entry("a"))
    with pytest.raises(ValueError, match="unknown feature"):
        kb.add(make_entry("b", ("bogus",)))
    assert len(kb) == 1


def test_get_by_feature_contains_and_entries_snapshot():
    kb = KnowledgeBase(
        [
            make_entry("a"),
            make_entry("b", ("format_statement",)),
            make_entry("c", ("goto_computed", "format_statement")),
        ]
    )
    assert kb.get("b").id == "b"
    with pytest.raises(KeyError):
        kb.get("zzz")
    assert [e.id for e in kb.by_feature("format_statement")] == ["b", "c"]
    assert kb.by_feature("equivalence") == []
    assert "a" in kb and "zzz" not in kb
    snapshot = kb.entries
    snapshot.clear()
    assert len(kb) == 3


def test_append_jsonl_adds_to_kb_and_file(tmp_path):
    path = tmp_path / "kb.jsonl"
    kb = KnowledgeBase()
    kb.append_jsonl(path, make_entry("a"))
    kb.append_jsonl(path, make_entry("b"))
    assert [e.id for e in KnowledgeBase.from_jsonl(path).entries] == ["a", "b"]
    assert len(kb) == 2


def test_append_jsonl_repairs_missing_trailing_newline(tmp_path):
    path = tmp_path / "kb.jsonl"
    path.write_text(json.dumps(record("a")), encoding="utf-8")
    KnowledgeBase([make_entry("seed")]).append_jsonl(path, make_entry("b"))
    assert [e.id for e in KnowledgeBase.from_jsonl(path).entries] == ["a", "b"]


def test_append_jsonl_duplicate_leaves_file_untouched(tmp_path):
    path = tmp_path / "kb.jsonl"
    kb = KnowledgeBase()
    kb.append_jsonl(path, make_entry("a"))
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        kb.append_jsonl(path, make_entry("a"))
    assert path.read_text(encoding="utf-8") == before


def test_concurrent_adds_are_all_kept():
    kb = KnowledgeBase()

    def worker(start: int) -> None:
        for i in range(start, start + 50):
            kb.add(make_entry(f"e{i}"))

    threads = [threading.Thread(target=worker, args=(n * 50,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(kb) == 400


def test_default_raises_clear_error_when_corpus_is_missing(monkeypatch):
    monkeypatch.setattr(
        knowledge_module, "DEFAULT_CORPUS_RELPATH", ("data", "does_not_exist.jsonl")
    )
    with pytest.raises(FileNotFoundError, match="does_not_exist.jsonl"):
        KnowledgeBase.default()


def test_default_loads_packaged_corpus_when_present():
    try:
        kb = KnowledgeBase.default()
    except FileNotFoundError:
        pytest.skip("packaged corpus not present")
    assert len(kb) > 0


def test_default_does_not_split_records_on_unicode_line_separators(tmp_path, monkeypatch):
    odd = make_entry("odd")
    odd = KnowledgeEntry(
        odd.id, odd.title, odd.feature_ids, "      S = '\u2028'", "b\x85c", "n\u2029"
    )
    corpus = tmp_path / "data" / "f77_to_python.jsonl"
    KnowledgeBase([odd, make_entry("plain")]).save_jsonl(corpus)
    monkeypatch.setattr(knowledge_module.resources, "files", lambda _package: tmp_path)
    kb = KnowledgeBase.default()
    assert [e.id for e in kb.entries] == ["odd", "plain"]
    assert kb.get("odd") == odd
    assert KnowledgeBase.from_jsonl(corpus).get("odd") == odd


def test_append_jsonl_unencodable_entry_is_rejected_without_side_effects(tmp_path):
    path = tmp_path / "kb.jsonl"
    kb = KnowledgeBase()
    kb.append_jsonl(path, make_entry("a"))
    before = path.read_bytes()
    bad = KnowledgeEntry("bad", "t", (), "lone surrogate \ud800", "m")
    with pytest.raises(UnicodeEncodeError):
        kb.append_jsonl(path, bad)
    assert "bad" not in kb
    assert path.read_bytes() == before
