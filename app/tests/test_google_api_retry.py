from __future__ import annotations

import logging
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.google.api_retry import GoogleApiRetryPolicy, execute_with_retry

try:
    from googleapiclient.errors import HttpError
    from httplib2 import Response as Httplib2Response  # type: ignore
except ImportError:
    HttpError = None  # type: ignore
    Httplib2Response = None  # type: ignore


LOGGER = logging.getLogger("test-google-api-retry")


class _StubTransientError(RuntimeError):
    def __init__(self, status_code: int, resource_id: str, original: Exception) -> None:
        super().__init__(f"transient status={status_code} resource={resource_id}")
        self.status_code = status_code
        self.resource_id = resource_id
        self.original = original


def _make_http_error(status: int) -> Exception:
    """Build a realistic HttpError with a specific status code."""
    if HttpError is not None and Httplib2Response is not None:
        resp = Httplib2Response({"status": str(int(status))})
        message: bytes = f"http error {int(status)}".encode("utf-8")
        return HttpError(resp, message)
    error = Exception(f"http error {int(status)}")
    error.resp = MagicMock()  # type: ignore[attr-defined]
    error.resp.status = int(status)  # type: ignore[attr-defined]
    return error


def _make_policy(
    *,
    connection_errors: tuple[type[BaseException], ...] = (),
    factory: Any = None,
) -> GoogleApiRetryPolicy:
    return GoogleApiRetryPolicy(
        max_retries=4,
        base_delay_sec=15.0,
        max_delay_sec=90.0,
        transient_status_codes=frozenset({500, 502, 503, 504}),
        connection_errors=connection_errors,
        log_prefix="docs",
        resource_label="document_id",
        transient_error_factory=factory or _StubTransientError,
    )


def _run(
    policy: GoogleApiRetryPolicy,
    request_callable: Any,
    *,
    request_count: Optional[int] = None,
) -> Any:
    return execute_with_retry(
        policy=policy,
        logger=LOGGER,
        operation_name="op",
        resource_id="res-1",
        request_callable=request_callable,
        request_count=request_count,
    )


class TestComputeDelay:
    def test_grows_by_powers_of_two(self) -> None:
        policy = _make_policy()
        assert policy.compute_delay(1) == 15.0
        assert policy.compute_delay(2) == 30.0
        assert policy.compute_delay(3) == 60.0

    def test_capped_at_max(self) -> None:
        policy = _make_policy()
        assert policy.compute_delay(4) == 90.0
        assert policy.compute_delay(10) == 90.0


class TestHttpStatusHandling:
    @patch("app.google.api_retry.time.sleep")
    def test_429_sleeps_once_then_succeeds(self, mock_sleep: MagicMock) -> None:
        policy = _make_policy()
        request = MagicMock(side_effect=[_make_http_error(429), "ok"])

        assert _run(policy, request) == "ok"

        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == policy.base_delay_sec

    @patch("app.google.api_retry.time.sleep")
    def test_502_uses_transient_error_factory_without_sleeping(
        self, mock_sleep: MagicMock
    ) -> None:
        factory = MagicMock(
            side_effect=lambda status_code, resource_id, original: _StubTransientError(
                status_code, resource_id, original
            )
        )
        policy = _make_policy(factory=factory)
        error_502 = _make_http_error(502)

        with pytest.raises(_StubTransientError) as exc_info:
            _run(policy, MagicMock(side_effect=error_502))

        factory.assert_called_once_with(502, "res-1", error_502)
        assert exc_info.value.status_code == 502
        assert exc_info.value.original is error_502
        mock_sleep.assert_not_called()

    @patch("app.google.api_retry.time.sleep")
    def test_400_raises_immediately(self, mock_sleep: MagicMock) -> None:
        policy = _make_policy()
        request = MagicMock(side_effect=_make_http_error(400))

        with pytest.raises(Exception):
            _run(policy, request)

        assert request.call_count == 1
        mock_sleep.assert_not_called()

    @patch("app.google.api_retry.time.sleep")
    def test_429_exhausts_max_retries(self, mock_sleep: MagicMock) -> None:
        policy = _make_policy()
        request = MagicMock(side_effect=_make_http_error(429))

        with pytest.raises(Exception):
            _run(policy, request)

        assert request.call_count == policy.max_retries
        assert mock_sleep.call_count == policy.max_retries - 1


class TestConnectionErrorHandling:
    @patch("app.google.api_retry.time.sleep")
    def test_retried_when_declared(self, mock_sleep: MagicMock) -> None:
        policy = _make_policy(connection_errors=(OSError,))
        request = MagicMock(side_effect=[ConnectionResetError(10054, "reset"), "ok"])

        assert _run(policy, request) == "ok"

        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == policy.base_delay_sec

    @patch("app.google.api_retry.time.sleep")
    def test_raised_immediately_when_not_declared(self, mock_sleep: MagicMock) -> None:
        policy = _make_policy(connection_errors=())
        request = MagicMock(side_effect=ConnectionResetError(10054, "reset"))

        with pytest.raises(ConnectionResetError):
            _run(policy, request)

        assert request.call_count == 1
        mock_sleep.assert_not_called()


class TestLogMessageFormats:
    """Guards the byte-for-byte log formats inherited from the per-client loops."""

    def _messages(self, caplog: pytest.LogCaptureFixture) -> List[str]:
        return [record.getMessage() for record in caplog.records]

    @patch("app.google.api_retry.time.sleep")
    def test_docs_formats(
        self, _mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        policy = _make_policy(connection_errors=(OSError,))
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            _run(policy, MagicMock(side_effect=[_make_http_error(429), "ok"]))
            _run(
                policy,
                MagicMock(side_effect=[_make_http_error(429), "ok"]),
                request_count=7,
            )
            _run(policy, MagicMock(side_effect=[ConnectionResetError(10054, "reset"), "ok"]))

        assert self._messages(caplog) == [
            "docs_request_429 operation=op attempt=1/4 delay_sec=15.0 document_id=res-1",
            "docs_batch_update_429 attempt=1/4 delay_sec=15.0 document_id=res-1 requests_count=7",
            "docs_request_connection_error operation=op attempt=1/4 delay_sec=15.0"
            " document_id=res-1 error=ConnectionResetError: [Errno 10054] reset",
        ]
        assert all(
            record.warning_category == "informational" for record in caplog.records
        )

    @patch("app.google.api_retry.time.sleep")
    def test_drive_format(
        self, _mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        policy = GoogleApiRetryPolicy(
            max_retries=4,
            base_delay_sec=15.0,
            max_delay_sec=90.0,
            transient_status_codes=frozenset({500, 502, 503, 504}),
            connection_errors=(),
            log_prefix="drive",
            resource_label="file_id",
            transient_error_factory=_StubTransientError,
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER.name):
            _run(policy, MagicMock(side_effect=[_make_http_error(429), "ok"]))

        assert self._messages(caplog) == [
            "drive_request_429 operation=op attempt=1/4 delay_sec=15.0 file_id=res-1"
        ]
