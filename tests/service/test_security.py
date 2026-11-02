from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from .fakes import LEGACY, Harness

BODY = {"files": [{"path": "sumsq.f", "source": LEGACY}], "run_tests": False}
KEY = "dev-key-123"
V1 = [("/v1/modernize", BODY), ("/v1/analyze", {"files": BODY["files"]})]


def test_no_key_configured_means_open(client: TestClient) -> None:
    assert client.post("/v1/analyze", json={"files": BODY["files"]}).status_code == 200


@pytest.fixture
def secured(make_client, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MODERNIZER_API_KEY", KEY)
    return make_client()


@pytest.mark.parametrize(("path", "payload"), V1)
def test_v1_requires_the_bearer_key(secured: TestClient, path: str, payload: dict) -> None:
    missing = secured.post(path, json=payload)
    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    for header in (
        f"Bearer {KEY}x",
        "Bearer wrong",
        "Bearer ",
        f"Basic {KEY}",
        KEY,
    ):
        assert (
            secured.post(path, json=payload, headers={"Authorization": header}).status_code == 401
        )
    assert KEY not in missing.text


@pytest.mark.parametrize(("path", "payload"), V1)
def test_v1_accepts_the_right_key(secured: TestClient, path: str, payload: dict) -> None:
    response = secured.post(path, json=payload, headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200
    lower = secured.post(path, json=payload, headers={"Authorization": f"bearer {KEY}"})
    assert lower.status_code == 200


def test_probes_stay_open_with_a_key(secured: TestClient) -> None:
    assert secured.get("/healthz").status_code == 200
    assert secured.get("/readyz").status_code == 200


def test_auth_is_checked_before_the_body(secured: TestClient) -> None:
    assert secured.post("/v1/modernize", json={"files": "garbage"}).status_code == 401
    garbage = secured.post(
        "/v1/modernize", json={"files": "garbage"}, headers={"Authorization": f"Bearer {KEY}"}
    )
    assert garbage.status_code == 422


def test_blank_key_counts_as_unset(make_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERNIZER_API_KEY", "   ")
    client = make_client()
    assert client.post("/v1/analyze", json={"files": BODY["files"]}).status_code == 200


def test_unexpected_error_gives_generic_500_and_is_logged(
    client: TestClient, harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    harness.build_error = RuntimeError("password=hunter2 /srv/secret/path.f")
    with caplog.at_level(logging.ERROR):
        response = client.post("/v1/modernize", json=BODY)

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "internal server error"
    for leak in ("hunter2", "Traceback", "RuntimeError", "secret"):
        assert leak not in response.text
    # The details are in the server log, tied to the client-visible id.
    record = next(r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info)
    assert data["error_id"] in record.getMessage()
    assert "hunter2" in str(record.exc_info[1])


def test_pipeline_crash_is_a_generic_500(client: TestClient, harness: Harness) -> None:
    harness.run_error = ValueError("internal detail")
    response = client.post("/v1/modernize", json=BODY)
    assert response.status_code == 500
    assert "internal detail" not in response.text
    assert harness.flushes == 1, "telemetry is flushed even when the run blows up"


def test_unknown_route_is_a_plain_404(client: TestClient) -> None:
    assert client.get("/v1/nothing").status_code == 404
    assert client.get("/v1/modernize").status_code == 405


def test_non_ascii_key_matches_the_bytes_the_client_sent(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "sch\u00fcssel-\u00e9"
    monkeypatch.setenv("MODERNIZER_API_KEY", key)
    client = make_client()
    payload = {"files": BODY["files"]}

    def post(token: bytes) -> int:
        return client.post(
            "/v1/analyze", json=payload, headers={"Authorization": b"Bearer " + token}
        ).status_code

    assert post(key.encode("utf-8")) == 200
    assert post("sch\u00fcssel-e".encode()) == 401
    assert post(b"\xff\xfe") == 401, "undecodable bytes are an authentication failure, not a 500"
