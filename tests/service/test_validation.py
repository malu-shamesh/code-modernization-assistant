from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from modernizer.service.schemas import validate_path

from .fakes import LEGACY

ENDPOINTS = ["/v1/modernize", "/v1/analyze"]


def file(path: str = "a.f", source: str = LEGACY) -> dict[str, str]:
    return {"path": path, "source": source}


@pytest.mark.parametrize(
    "path",
    [
        "a.f",
        "a.for",
        "a.f77",
        "a.ftn",
        "A.F",
        "SRC/LAPACK.FOR",
        "pkg/mod.f",
        "a/b/c_d-e.f77",
        "with space.f",
        "x.y.ftn",
        "a..f",
        "mixed.Ftn",
        "dir.py/a.f",
    ],
)
def test_valid_paths_are_accepted(path: str) -> None:
    assert validate_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd.f",
        "../a.f",
        "a/../b.f",
        "a/./b.f",
        "a//b.f",
        "dir/",
        "a\\b.f",
        "..\\a.f",
        "C:/a.f",
        "C:\\a.f",
        "a.py",  # the old input language is not accepted
        "a.txt",
        "a.fpp",
        "a.f90",
        "a.fo",
        "a.ff",
        "a.f.txt",
        "noext",
        "f",
        ".f",
        "dir/.f",
        "dir/.for",
        ".F77",
        "a\x00.f",
        "a\n.f",
        "a\x85b.f",  # C1 control character
        "a\u2028b.f",  # line separator
        "a\u202eb.f",  # bidi override
        "a\u200bb.f",  # zero-width space
        "x" * 300 + ".f",
    ],
)
def test_bad_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        validate_path(path)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "path", ["/abs.f", "../up.f", "a/../b.f", "a\\b.f", "C:\\x.f", "notes.txt", "old.py", ""]
)
def test_invalid_paths_give_422(client: TestClient, endpoint: str, path: str) -> None:
    response = client.post(endpoint, json={"files": [file(path)]})
    assert response.status_code == 422


def test_the_path_error_names_the_accepted_fortran_extensions(client: TestClient) -> None:
    response = client.post("/v1/analyze", json={"files": [file("old.py")]})
    assert response.status_code == 422
    text = response.text
    assert "Fortran" in text and ".f77" in text and ".ftn" in text


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_duplicate_paths_give_422(client: TestClient, endpoint: str) -> None:
    response = client.post(endpoint, json={"files": [file("a.f"), file("b.f"), file("a.f")]})
    assert response.status_code == 422
    assert "duplicate" in json.dumps(response.json())


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"files": []},
        {"files": "a.f"},
        {"files": [{"path": "a.f"}]},
        {"files": [file()], "unknown": 1},
        {"files": [{"path": "a.f", "source": "x", "mode": "y"}]},
    ],
)
def test_malformed_bodies_give_422(client: TestClient, endpoint: str, payload: object) -> None:
    assert client.post(endpoint, json=payload).status_code == 422


@pytest.mark.parametrize(
    "options",
    [
        {"policy": "cheapest"},
        {"threshold": 1.5},
        {"threshold": -0.1},
        {"threshold": "high"},
        {"open_fraction": 1.5},
        {"open_fraction": -0.1},
        {"open_fraction": "most"},
        {"run_tests": "maybe"},
    ],
)
def test_bad_options_give_422(client: TestClient, options: dict[str, object]) -> None:
    response = client.post("/v1/modernize", json={"files": [file()], **options})
    assert response.status_code == 422


@pytest.mark.parametrize("name", ["threshold", "open_fraction"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(client: TestClient, name: str, value: str) -> None:
    raw = f'{{"files": [{{"path": "a.f", "source": "x"}}], "{name}": {value}}}'
    response = client.post(
        "/v1/modernize", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("fraction", [0, 0.0, 0.5, 1, 1.0])
def test_open_fraction_bounds_are_inclusive(client: TestClient, fraction: float) -> None:
    response = client.post(
        "/v1/modernize", json={"files": [file()], "open_fraction": fraction, "run_tests": False}
    )
    assert response.status_code == 200


def test_analyze_takes_no_open_fraction(client: TestClient) -> None:
    """Analysis routes every file on its own, so a batch target makes no sense there."""
    response = client.post("/v1/analyze", json={"files": [file()], "open_fraction": 0.5})
    assert response.status_code == 422


def test_422_does_not_echo_the_submitted_source(client: TestClient) -> None:
    marker = "MARKER_SOURCE_1f3a"
    response = client.post(
        "/v1/analyze", json={"files": [file("../x.f", f"      PRINT *, '{marker}'")]}
    )
    assert response.status_code == 422
    assert marker not in response.text


def test_invalid_json_gives_422(client: TestClient) -> None:
    response = client.post(
        "/v1/analyze", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_too_many_files_give_413(
    make_client, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_FILES", "2")
    client = make_client()
    files = [file(f"m{i}.f") for i in range(3)]
    response = client.post(endpoint, json={"files": files})
    assert response.status_code == 413
    assert "too many files" in response.json()["detail"]
    assert client.post(endpoint, json={"files": files[:2]}).status_code == 200


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_oversized_file_gives_413(
    make_client, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_FILE_CHARS", "50")
    client = make_client()
    response = client.post(endpoint, json={"files": [file("big.f", "      X = 1\n" * 10)]})
    assert response.status_code == 413
    assert "big.f" in response.json()["detail"]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_oversized_total_gives_413(
    make_client, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_FILE_CHARS", "100")
    monkeypatch.setenv("MODERNIZER_MAX_TOTAL_CHARS", "150")
    client = make_client()
    files = [file("a.f", "      X = 1\n" * 8), file("b.f", "      Y = 2\n" * 8)]
    response = client.post(endpoint, json={"files": files})
    assert response.status_code == 413
    assert "request is too large" in response.json()["detail"]


def test_huge_content_length_is_refused_before_parsing(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_TOTAL_CHARS", "10")
    client = make_client()
    payload = json.dumps({"files": [file("a.f", "x" * 300_000)]})
    response = client.post(
        "/v1/analyze", content=payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "request body is too large"


def test_limits_do_not_apply_to_the_probes(make_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_FILES", "1")
    client = make_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/analyze", json={"files": [file("a.f", LEGACY)]}).status_code == 200
