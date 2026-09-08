from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.google.docs_client import (
    GoogleDocsClient,
    GoogleDocsTransientError,
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


def _make_http_error(status: int) -> Exception:
    """Build a realistic HttpError with a specific status code."""
    if HttpError is not None and Httplib2Response is not None:
        resp = Httplib2Response({"status": str(int(status))})
        message: bytes = f"http error {int(status)}".encode("utf-8")
        return HttpError(resp, message)
    # Fallback for environments without googleapiclient
    error = Exception(f"http error {int(status)}")
    error.resp = MagicMock()  # type: ignore[attr-defined]
    error.resp.status = int(status)  # type: ignore[attr-defined]
    return error


def _make_http_429_error() -> Exception:
    return _make_http_error(429)


def _make_http_400_error() -> Exception:
    return _make_http_error(400)


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


class TestGetDocumentRetry:
    @patch("app.google.docs_client.time.sleep")
    def test_succeeds_on_first_try(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        expected_doc = {"documentId": "doc123", "body": {"content": []}}
        mock_service.documents().get().execute.return_value = expected_doc
        mock_service.documents().get.reset_mock()
        client = GoogleDocsClient(mock_service)

        result = client.get_document("doc123")

        assert result == expected_doc
        mock_sleep.assert_not_called()
        mock_service.documents().get.assert_called_once_with(documentId="doc123")

    @patch("app.google.docs_client.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        expected_doc = {"documentId": "doc123"}
        mock_service.documents().get().execute.side_effect = [
            _make_http_error(429),
            expected_doc,
        ]
        client = GoogleDocsClient(mock_service)

        result = client.get_document("doc123")

        assert result == expected_doc
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == _DOCS_WRITE_BASE_DELAY_SEC

    @patch("app.google.docs_client.time.sleep")
    def test_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.documents().get().execute.side_effect = _make_http_error(502)
        client = GoogleDocsClient(mock_service)

        with pytest.raises(GoogleDocsTransientError) as exc_info:
            client.get_document("doc-502")

        assert exc_info.value.status_code == 502
        assert exc_info.value.document_id == "doc-502"
        mock_sleep.assert_not_called()

    @patch("app.google.docs_client.time.sleep")
    def test_400_raises_immediately(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.documents().get().execute.side_effect = _make_http_error(400)
        client = GoogleDocsClient(mock_service)

        with pytest.raises(Exception):
            client.get_document("doc400")

        mock_sleep.assert_not_called()


class TestCreateDocumentRetry:
    @patch("app.google.docs_client.time.sleep")
    def test_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.documents().create().execute.side_effect = _make_http_error(502)
        client = GoogleDocsClient(mock_service)

        with pytest.raises(GoogleDocsTransientError) as exc_info:
            client.create_document("title")

        assert exc_info.value.status_code == 502
        assert exc_info.value.document_id == "<new_document>"
        mock_sleep.assert_not_called()

    @patch("app.google.docs_client.time.sleep")
    def test_connection_reset_is_retried_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.documents().create().execute.side_effect = [
            ConnectionResetError(10054, "Remote host forcibly closed the connection"),
            {"documentId": "doc-after-reset"},
        ]
        client = GoogleDocsClient(mock_service)

        assert client.create_document("title") == "doc-after-reset"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == _DOCS_WRITE_BASE_DELAY_SEC

    @patch("app.google.docs_client.time.sleep")
    def test_connection_error_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.documents().create().execute.side_effect = ConnectionResetError(10054, "reset")
        client = GoogleDocsClient(mock_service)

        with pytest.raises(ConnectionResetError):
            client.create_document("title")

        assert mock_service.documents().create().execute.call_count == _DOCS_WRITE_MAX_RETRIES
        assert mock_sleep.call_count == _DOCS_WRITE_MAX_RETRIES - 1

    @patch("app.google.docs_client.time.sleep")
    def test_remote_disconnected_on_batch_update_is_retried(self, mock_sleep: MagicMock) -> None:
        import http.client

        mock_service = MagicMock()
        mock_service.documents().batchUpdate().execute.side_effect = [
            http.client.RemoteDisconnected("Remote end closed connection without response"),
            {"replies": []},
        ]
        client = GoogleDocsClient(mock_service)

        client.batch_update("doc123", [{"insertText": {}}])

        assert mock_sleep.call_count == 1


class TestBatchUpdate5xxBehavior:
    @patch("app.google.docs_client.time.sleep")
    def test_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.documents().batchUpdate().execute.side_effect = _make_http_error(502)
        client = GoogleDocsClient(mock_service)

        with pytest.raises(GoogleDocsTransientError) as exc_info:
            client.batch_update("doc-502", [{"insertText": {}}])

        assert exc_info.value.status_code == 502
        assert exc_info.value.document_id == "doc-502"
        mock_sleep.assert_not_called()
