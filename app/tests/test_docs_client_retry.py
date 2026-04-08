from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.google.docs_client import (
    GoogleDocsClient,
    _compute_retry_delay,
    _DOCS_WRITE_BASE_DELAY_SEC,
    _DOCS_WRITE_MAX_DELAY_SEC,
    _DOCS_WRITE_MAX_RETRIES,
)

try:
    from googleapiclient.errors import HttpError
    from httplib2 import Response as Httplib2Response  # type: ignore
except ImportError:
    HttpError = None  # type: ignore
    Httplib2Response = None  # type: ignore


def _make_http_429_error() -> Exception:
    """Build a realistic HttpError 429."""
    if HttpError is not None and Httplib2Response is not None:
        resp = Httplib2Response({"status": "429"})
        return HttpError(resp, b"quota exceeded")
    # Fallback for environments without googleapiclient
    error = Exception("quota exceeded")
    error.resp = MagicMock()  # type: ignore[attr-defined]
    error.resp.status = 429  # type: ignore[attr-defined]
    return error


def _make_http_400_error() -> Exception:
    """Build a non-retryable HttpError."""
    if HttpError is not None and Httplib2Response is not None:
        resp = Httplib2Response({"status": "400"})
        return HttpError(resp, b"bad request")
    error = Exception("bad request")
    error.resp = MagicMock()  # type: ignore[attr-defined]
    error.resp.status = 400  # type: ignore[attr-defined]
    return error


class TestComputeRetryDelay:
    def test_attempt_1(self) -> None:
        assert _compute_retry_delay(1) == _DOCS_WRITE_BASE_DELAY_SEC

    def test_attempt_2(self) -> None:
        assert _compute_retry_delay(2) == _DOCS_WRITE_BASE_DELAY_SEC * 2

    def test_attempt_3(self) -> None:
        assert _compute_retry_delay(3) == _DOCS_WRITE_BASE_DELAY_SEC * 4

    def test_capped_at_max(self) -> None:
        # Large attempt should not exceed max
        assert _compute_retry_delay(10) == _DOCS_WRITE_MAX_DELAY_SEC

    def test_exponential_growth(self) -> None:
        delays = [_compute_retry_delay(i) for i in range(1, 5)]
        # Each delay should be >= the previous one
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]


class TestBatchUpdateRetry:
    @patch("app.google.docs_client.time.sleep")
    def test_succeeds_on_first_try(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        client = GoogleDocsClient(mock_service)
        client.batch_update("doc123", [{"insertText": {}}])
        mock_sleep.assert_not_called()
        mock_service.documents().batchUpdate.assert_called_once()

    @patch("app.google.docs_client.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        error_429 = _make_http_429_error()
        # First call raises 429, second call succeeds
        mock_service.documents().batchUpdate().execute.side_effect = [
            error_429,
            None,
        ]
        client = GoogleDocsClient(mock_service)
        client.batch_update("doc123", [{"insertText": {}}])
        assert mock_sleep.call_count == 1
        delay_used = mock_sleep.call_args[0][0]
        assert delay_used == _DOCS_WRITE_BASE_DELAY_SEC

    @patch("app.google.docs_client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        error_429 = _make_http_429_error()
        mock_service.documents().batchUpdate().execute.side_effect = error_429
        client = GoogleDocsClient(mock_service)
        with pytest.raises(Exception):
            client.batch_update("doc123", [{"insertText": {}}])
        # Should have retried MAX_RETRIES - 1 times (last attempt raises without sleep)
        assert mock_sleep.call_count == _DOCS_WRITE_MAX_RETRIES - 1

    @patch("app.google.docs_client.time.sleep")
    def test_non_429_error_raises_immediately(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        error_400 = _make_http_400_error()
        mock_service.documents().batchUpdate().execute.side_effect = error_400
        client = GoogleDocsClient(mock_service)
        with pytest.raises(Exception):
            client.batch_update("doc123", [{"insertText": {}}])
        mock_sleep.assert_not_called()
