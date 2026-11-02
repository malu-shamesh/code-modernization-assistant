from __future__ import annotations

import httpx
import pytest
from google.api_core import exceptions as gax
from google.auth import exceptions as auth_exceptions
from google.genai import errors as genai_errors

from modernizer.llm import (
    LLMEmptyResponseError,
    LLMRequestError,
    LLMRetryExhaustedError,
    RetryPolicy,
    call_with_retry,
    is_transient,
)


class Flaky:
    def __init__(self, failures: list[BaseException], result: str = "ok") -> None:
        self.failures = list(failures)
        self.result = result
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


def policy(sleeps: list[float], **kw) -> RetryPolicy:
    kw.setdefault("jitter", False)
    return RetryPolicy(sleep=sleeps.append, **kw)


def test_transient_then_success_backs_off_exponentially():
    sleeps: list[float] = []
    fn = Flaky([LLMRequestError("x", status_code=503), LLMRequestError("x", status_code=429)])
    assert call_with_retry(fn, policy(sleeps)) == "ok"
    assert fn.calls == 3
    assert sleeps == [1.0, 2.0]


def test_delay_is_capped():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, jitter=False)
    assert [p.delay_for(n) for n in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_jitter_stays_in_upper_half_of_interval():
    p = RetryPolicy(base_delay_s=4.0, jitter=True)
    assert all(2.0 <= p.delay_for(1) <= 4.0 for _ in range(100))


def test_delay_does_not_overflow_for_huge_attempt_numbers():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0, jitter=False)
    assert p.delay_for(5000) == 30.0


def test_retry_after_hint_raises_the_wait_up_to_the_cap():
    p = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter=False)
    assert p.delay_for(1, retry_after_s=7.0) == 7.0
    assert p.delay_for(1, retry_after_s=99.0) == 10.0


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_transient_status_fails_immediately(status):
    sleeps: list[float] = []
    fn = Flaky([LLMRequestError("bad", status_code=status)])
    with pytest.raises(LLMRequestError) as info:
        call_with_retry(fn, policy(sleeps))
    assert info.value.status_code == status
    assert fn.calls == 1
    assert sleeps == []


def test_exhausted_retries_raise_typed_error_with_cause():
    sleeps: list[float] = []
    fn = Flaky([LLMRequestError("down", status_code=503)] * 10)
    with pytest.raises(LLMRetryExhaustedError) as info:
        call_with_retry(fn, policy(sleeps, max_attempts=3))
    assert info.value.attempts == 3
    assert info.value.status_code == 503
    assert isinstance(info.value.__cause__, LLMRequestError)
    assert fn.calls == 3
    assert len(sleeps) == 2


def test_single_attempt_policy_never_sleeps():
    sleeps: list[float] = []
    fn = Flaky([LLMRequestError("down", status_code=500)])
    with pytest.raises(LLMRetryExhaustedError):
        call_with_retry(fn, policy(sleeps, max_attempts=1))
    assert sleeps == []


def test_programming_errors_propagate_untouched():
    fn = Flaky([KeyError("oops")])
    with pytest.raises(KeyError):
        call_with_retry(fn, policy([]))
    assert fn.calls == 1


def test_other_llm_errors_are_not_retried():
    fn = Flaky([LLMEmptyResponseError("empty")])
    with pytest.raises(LLMEmptyResponseError):
        call_with_retry(fn, policy([]))
    assert fn.calls == 1


def test_policy_validation():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_s=-1)


def genai_error(cls, code: int) -> genai_errors.APIError:
    return cls(code, {"error": {"code": code, "message": "m", "status": "S"}})


@pytest.mark.parametrize(
    "exc",
    [
        genai_error(genai_errors.ServerError, 503),
        genai_error(genai_errors.ClientError, 429),
        genai_error(genai_errors.ClientError, 408),
        gax.ServiceUnavailable("x"),
        gax.ResourceExhausted("x"),
        gax.DeadlineExceeded("x"),
        gax.InternalServerError("x"),
        httpx.ReadTimeout("t"),
        httpx.ConnectError("c"),
        httpx.RemoteProtocolError("r"),
        ConnectionResetError(),
        TimeoutError(),
        auth_exceptions.TransportError("t"),
        auth_exceptions.RefreshError("r", retryable=True),
    ],
)
def test_transient_classification(exc):
    assert is_transient(exc)


@pytest.mark.parametrize(
    "exc",
    [
        genai_error(genai_errors.ClientError, 400),
        genai_error(genai_errors.ClientError, 401),
        genai_error(genai_errors.ClientError, 403),
        gax.InvalidArgument("x"),
        gax.PermissionDenied("x"),
        gax.Unauthenticated("x"),
        ValueError("x"),
        LLMEmptyResponseError("x"),
        auth_exceptions.RefreshError("r"),
        auth_exceptions.DefaultCredentialsError("d"),
    ],
)
def test_non_transient_classification(exc):
    assert not is_transient(exc)


def test_non_transient_provider_error_is_wrapped_with_status():
    fn = Flaky([genai_error(genai_errors.ClientError, 403)])
    with pytest.raises(LLMRequestError) as info:
        call_with_retry(fn, policy([]))
    assert info.value.status_code == 403
    assert isinstance(info.value.__cause__, genai_errors.ClientError)
