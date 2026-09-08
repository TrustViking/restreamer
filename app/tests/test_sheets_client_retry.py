from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.google.sheets_client import (
    GoogleSheetsClient,
    GoogleSheetsTransientError,
    _compute_sheets_retry_delay,
    _SHEETS_BASE_DELAY_SEC,
    _SHEETS_MAX_DELAY_SEC,
    _SHEETS_MAX_RETRIES,
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


def _values_response() -> Dict[str, Any]:
    values: List[List[str]] = [
        ["Links", "Date", "Time"],
        ["https://youtu.be/abc", "2026-09-08", "10:00"],
    ]
    return {"values": values}


class TestComputeSheetsRetryDelay:
    def test_attempt_1(self) -> None:
        assert _compute_sheets_retry_delay(1) == _SHEETS_BASE_DELAY_SEC

    def test_attempt_2(self) -> None:
        assert _compute_sheets_retry_delay(2) == _SHEETS_BASE_DELAY_SEC * 2

    def test_attempt_3(self) -> None:
        assert _compute_sheets_retry_delay(3) == _SHEETS_BASE_DELAY_SEC * 4

    def test_capped_at_max(self) -> None:
        assert _compute_sheets_retry_delay(10) == _SHEETS_MAX_DELAY_SEC


class TestPingAccessRetry:
    @patch("app.google.sheets_client.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.spreadsheets().get().execute.side_effect = [
            _make_http_error(429),
            {"spreadsheetId": "sheet-1", "properties": {"title": "Plan"}},
        ]
        client = GoogleSheetsClient(mock_service)

        assert client.ping_access("sheet-1") == ("sheet-1", "Plan")
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == _SHEETS_BASE_DELAY_SEC

    @patch("app.google.sheets_client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.spreadsheets().get().execute.side_effect = _make_http_error(429)
        client = GoogleSheetsClient(mock_service)

        with pytest.raises(Exception):
            client.ping_access("sheet-1")

        assert mock_sleep.call_count == _SHEETS_MAX_RETRIES - 1


class TestReadRowsRetry:
    @patch("app.google.sheets_client.time.sleep")
    def test_read_timeout_is_retried_then_succeeds(self, mock_sleep: MagicMock) -> None:
        # Сценарий из прогона 20260908_155025: read timeout при обращении к Sheets.
        mock_service = MagicMock()
        mock_service.spreadsheets().values().get().execute.side_effect = [
            TimeoutError("The read operation timed out"),
            _values_response(),
        ]
        client = GoogleSheetsClient(mock_service)

        rows = client.read_rows(spreadsheet_id="sheet-1", range_name="Sheet1!A:F")

        assert len(rows) == 1
        assert rows[0].link == "https://youtu.be/abc"
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == _SHEETS_BASE_DELAY_SEC

    @patch("app.google.sheets_client.time.sleep")
    def test_400_raises_immediately(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.spreadsheets().values().get().execute.side_effect = _make_http_error(400)
        client = GoogleSheetsClient(mock_service)

        with pytest.raises(Exception):
            client.read_rows(spreadsheet_id="sheet-1", range_name="Sheet1!A:F")

        mock_sleep.assert_not_called()


class TestUpdateCellStringRetry:
    @patch("app.google.sheets_client.time.sleep")
    def test_502_raises_transient_error(self, mock_sleep: MagicMock) -> None:
        mock_service = MagicMock()
        mock_service.spreadsheets().values().update().execute.side_effect = _make_http_error(502)
        client = GoogleSheetsClient(mock_service)

        with pytest.raises(GoogleSheetsTransientError) as exc_info:
            client.update_cell_string(
                spreadsheet_id="sheet-502",
                cell_a1="F2",
                value="done",
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.spreadsheet_id == "sheet-502"
        mock_sleep.assert_not_called()
