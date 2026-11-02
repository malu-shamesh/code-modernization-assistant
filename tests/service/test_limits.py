from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from modernizer.service import ConcurrencyGate, LimitExceeded, Limits
from modernizer.service import app as app_module

from .fakes import GOOD, LEGACY, WAIT_S, Harness

BODY = {"files": [{"path": "sumsq.f", "source": LEGACY}], "run_tests": False}


def test_limit_defaults_and_env_overrides() -> None:
    defaults = Limits.from_env({})
    assert (defaults.max_files, defaults.max_file_chars) == (25, 150_000)
    assert (defaults.max_total_chars, defaults.max_concurrent) == (400_000, 2)

    custom = Limits.from_env(
        {
            "MODERNIZER_MAX_FILES": "3",
            "MODERNIZER_MAX_FILE_CHARS": " 10 ",
            "MODERNIZER_MAX_TOTAL_CHARS": "20",
            "MODERNIZER_MAX_CONCURRENT": "1",
            "MODERNIZER_RETRY_AFTER_S": "9",
        }
    )
    assert (custom.max_files, custom.max_file_chars, custom.max_total_chars) == (3, 10, 20)
    assert (custom.max_concurrent, custom.retry_after_s) == (1, 9)


@pytest.mark.parametrize("value", ["abc", "0", "-3", "1.5"])
def test_bad_limit_values_name_the_variable(value: str) -> None:
    with pytest.raises(ValueError, match="MODERNIZER_MAX_FILES"):
        Limits.from_env({"MODERNIZER_MAX_FILES": value})


def test_check_sources_boundaries() -> None:
    limits = Limits(max_files=2, max_file_chars=5, max_total_chars=8)
    limits.check_sources([("a.f", "12345"), ("b.f", "123")])  # exactly at every limit
    with pytest.raises(LimitExceeded, match="too many files"):
        limits.check_sources([("a.f", ""), ("b.f", ""), ("c.f", "")])
    with pytest.raises(LimitExceeded, match="too large"):
        limits.check_sources([("a.f", "123456")])
    with pytest.raises(LimitExceeded, match="request is too large"):
        limits.check_sources([("a.f", "12345"), ("b.f", "1234")])


def test_gate_is_non_blocking_and_reusable() -> None:
    gate = ConcurrencyGate(2)
    assert gate.try_acquire() and gate.try_acquire()
    assert not gate.try_acquire()
    gate.release()
    assert gate.try_acquire()
    gate.release()
    gate.release()
    with pytest.raises(ValueError):
        gate.release()  # more releases than acquires is a bug, not a silent extra slot
    with pytest.raises(ValueError):
        ConcurrencyGate(0)


def test_excess_concurrent_modernize_requests_get_429(
    make_client, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_CONCURRENT", "1")
    monkeypatch.setenv("MODERNIZER_RETRY_AFTER_S", "7")
    client: TestClient = make_client()
    harness.hold.clear()  # the open model now blocks until released

    with ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(client.post, "/v1/modernize", json=BODY)
        try:
            assert harness.entered.wait(WAIT_S), "the first request never reached the model"

            refused = client.post("/v1/modernize", json=BODY)
            assert refused.status_code == 429
            assert refused.headers["Retry-After"] == "7"
            assert "stack" not in refused.text.lower()

            # Only /v1/modernize is gated: probes and analysis stay available.
            assert client.get("/healthz").status_code == 200
            assert client.post("/v1/analyze", json={"files": BODY["files"]}).status_code == 200
        finally:
            harness.hold.set()
        first = slow.result(timeout=WAIT_S)

    assert first.status_code == 200
    assert first.json()["results"][0]["success"] is True

    harness.open_script = [GOOD]
    again = client.post("/v1/modernize", json=BODY)
    assert again.status_code == 200, "the slot is free again once the first request finished"


def test_slot_is_released_when_the_pipeline_crashes(
    make_client, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_CONCURRENT", "1")
    client = make_client()
    harness.run_error = RuntimeError("boom")
    assert client.post("/v1/modernize", json=BODY).status_code == 500
    harness.run_error = None
    assert client.post("/v1/modernize", json=BODY).status_code == 200


def test_slot_is_released_when_validation_fails_before_running(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODERNIZER_MAX_CONCURRENT", "1")
    monkeypatch.setenv("MODERNIZER_MAX_FILES", "1")
    client = make_client()
    two = {"files": [{"path": "a.f", "source": "x"}, {"path": "b.f", "source": "y"}]}
    assert client.post("/v1/modernize", json=two).status_code == 413
    assert client.post("/v1/modernize", json=BODY).status_code == 200


def test_slot_release_is_idempotent() -> None:
    gate = ConcurrencyGate(1)
    slot = gate.acquire_slot()
    assert slot is not None
    assert gate.acquire_slot() is None
    slot.release()
    slot.release()  # a second release must not hand out a phantom extra place
    again = gate.acquire_slot()
    assert again is not None
    assert gate.acquire_slot() is None
    again.release()


def test_slot_is_released_when_the_worker_never_starts(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handler cancelled while waiting for a free thread must not leak its slot."""
    monkeypatch.setenv("MODERNIZER_MAX_CONCURRENT", "1")
    client = make_client()
    real = app_module.run_in_threadpool

    async def never_runs(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no thread available")

    monkeypatch.setattr(app_module, "run_in_threadpool", never_runs)
    assert client.post("/v1/modernize", json=BODY).status_code == 500

    monkeypatch.setattr(app_module, "run_in_threadpool", real)
    assert client.post("/v1/modernize", json=BODY).status_code == 200
