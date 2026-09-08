from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.google.drive_client import (
    GoogleDriveClient,
    GoogleDriveTransientError,
    _compute_drive_retry_delay,
    _DRIVE_BASE_DELAY_SEC,
    _DRIVE_MAX_DELAY_SEC,
)

try:
    from googleapiclient.errors import HttpError
    from httplib2 import Response as Httplib2Response  # type: ignore
except ImportError:
    HttpError = None  # type: ignore
    Httplib2Response = None  # type: ignore


def _make_http_error(status: int) -> Exception:
    if HttpError is not None and Httplib2Response is not None:
        resp = Httplib2Response({"status": str(int(status))})
        message: bytes = f"http error {int(status)}".encode("utf-8")
        return HttpError(resp, message)
    error = Exception(f"http error {int(status)}")
    error.resp = MagicMock()  # type: ignore[attr-defined]
    error.resp.status = int(status)  # type: ignore[attr-defined]
    return error


class TestComputeDriveRetryDelay:
    def test_attempt_1(self) -> None:
        assert _compute_drive_retry_delay(1) == _DRIVE_BASE_DELAY_SEC

    def test_attempt_2(self) -> None:
        assert _compute_drive_retry_delay(2) == _DRIVE_BASE_DELAY_SEC * 2

    def test_attempt_3(self) -> None:
        assert _compute_drive_retry_delay(3) == _DRIVE_BASE_DELAY_SEC * 4

    def test_capped_at_max(self) -> None:
        assert _compute_drive_retry_delay(10) == _DRIVE_MAX_DELAY_SEC


class TestEnsureFolderRetry:
    @patch("app.google.drive_client.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.side_effect = [
            _make_http_error(429),
            {"files": [{"id": "folder-1", "name": "news"}]},
        ]
        client = GoogleDriveClient(mock_service)

        folder_id = client.ensure_folder(parent_folder_id="parent-1", folder_name="news")

        assert folder_id == "folder-1"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == _DRIVE_BASE_DELAY_SEC

    @patch("app.google.drive_client.time.sleep")
    def test_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.side_effect = _make_http_error(502)
        client = GoogleDriveClient(mock_service)

        with pytest.raises(GoogleDriveTransientError) as exc_info:
            client.ensure_folder(parent_folder_id="parent-1", folder_name="news")

        assert exc_info.value.status_code == 502
        assert exc_info.value.file_id == "parent-1"
        mock_sleep.assert_not_called()


class TestSetAnyonePermissionRetry:
    @patch("app.google.drive_client.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.permissions().create().execute.side_effect = [
            _make_http_error(429),
            {"id": "perm-1"},
        ]
        client = GoogleDriveClient(mock_service)

        client.set_anyone_permission(file_id="file-1", role="reader")

        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == _DRIVE_BASE_DELAY_SEC

    @patch("app.google.drive_client.time.sleep")
    def test_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.permissions().create().execute.side_effect = _make_http_error(502)
        client = GoogleDriveClient(mock_service)

        with pytest.raises(GoogleDriveTransientError) as exc_info:
            client.set_anyone_permission(file_id="file-1", role="reader")

        assert exc_info.value.status_code == 502
        assert exc_info.value.file_id == "file-1"
        mock_sleep.assert_not_called()


class TestMoveFileToFolderRetry:
    @patch("app.google.drive_client.time.sleep")
    def test_get_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.files().get().execute.side_effect = _make_http_error(502)
        client = GoogleDriveClient(mock_service)

        with pytest.raises(GoogleDriveTransientError) as exc_info:
            client.move_file_to_folder(file_id="file-1", folder_id="folder-1")

        assert exc_info.value.status_code == 502
        assert exc_info.value.file_id == "file-1"
        mock_sleep.assert_not_called()

    @patch("app.google.drive_client.time.sleep")
    def test_update_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.files().get().execute.return_value = {"parents": ["old-parent"]}
        mock_service.files().update().execute.side_effect = _make_http_error(502)
        client = GoogleDriveClient(mock_service)

        with pytest.raises(GoogleDriveTransientError) as exc_info:
            client.move_file_to_folder(file_id="file-1", folder_id="folder-1")

        assert exc_info.value.status_code == 502
        assert exc_info.value.file_id == "file-1"
        mock_sleep.assert_not_called()
