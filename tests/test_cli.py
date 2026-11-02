from __future__ import annotations

import json

import pytest

from modernizer.cli import main

SCALE = """\
      SUBROUTINE SCALE(N, A, X)
      INTEGER N
      REAL A, X(*)
      INTEGER I
      DO 10 I = 1, N
         X(I) = A * X(I)
   10 CONTINUE
      RETURN
      END
"""


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "src"
    (src / "lib").mkdir(parents=True)
    (src / "lib" / "scale.f").write_text(SCALE, encoding="utf-8")
    return src


def test_analyze_prints_table(source_dir, capsys):
    assert main(["analyze", str(source_dir)]) == 0
    out = capsys.readouterr().out
    assert "lib/scale.f" in out
    assert "difficulty" in out


def test_analyze_json(source_dir, capsys):
    assert main(["analyze", "--json", str(source_dir)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["path"] == "lib/scale.f"
    assert "do_labeled" in rows[0]["features"]
    assert rows[0]["tier"] in {"open", "frontier"}


def test_analyze_without_fortran_files_is_an_error(tmp_path, capsys):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert main(["analyze", str(tmp_path)]) == 2
    assert "no Fortran files" in capsys.readouterr().err


def test_convert_with_stub_provider_reports_failure_and_ships_runtime(
    source_dir, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("MODERNIZER_PROVIDER", "mock")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    code = main(
        [
            "convert",
            str(source_dir),
            "--out",
            str(out),
            "--no-tests",
            "--open-fraction",
            "1.0",
            "--report",
            str(report),
        ]
    )
    assert code == 1, "the stub model produces code that fails validation"
    assert (out / "f77rt.py").is_file()
    assert (out / "_failed" / "lib" / "scale.py").is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["modules"] == 1
    assert payload["modules"][0]["success"] is False
    assert "planned open share" in capsys.readouterr().out


def test_open_fraction_must_be_a_fraction(source_dir, tmp_path, capsys):
    code = main(["convert", str(source_dir), "--out", str(tmp_path / "o"), "--open-fraction", "70"])
    assert code == 2
    assert "between 0 and 1" in capsys.readouterr().err
