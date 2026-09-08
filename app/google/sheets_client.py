from __future__ import annotations

import http.client
import re
# Kept imported here on purpose: tests patch "app.google.sheets_client.time.sleep",
# which resolves through this name into the shared retry engine.
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import SheetRow
from app.google.api_retry import GoogleApiRetryPolicy, execute_with_retry

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore


LOGGER = _get_logger_impl(__name__)

_SHEETS_MAX_RETRIES: int = 4
_SHEETS_BASE_DELAY_SEC: float = 15.0
_SHEETS_MAX_DELAY_SEC: float = 90.0
_SHEETS_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})
# Transport-level failures below the HTTP layer: read timeouts, connection reset,
# SSL errors, remote disconnect. Retried in place with the same backoff as 429.
_SHEETS_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)


def _compute_sheets_retry_delay(attempt: int) -> float:
    """Exponential backoff: base * 2^(attempt-1), capped at max."""
    return _SHEETS_RETRY_POLICY.compute_delay(attempt)


class GoogleSheetsTransientError(RuntimeError):
    """Raised when Google Sheets API returns a transient server-side error."""

    def __init__(self, status_code: int, spreadsheet_id: str, original: Exception) -> None:
        super().__init__(
            f"Google Sheets transient error status={status_code} spreadsheet_id={spreadsheet_id}"
        )
        self.status_code = status_code
        self.spreadsheet_id = spreadsheet_id
        self.original = original


_SHEETS_RETRY_POLICY: GoogleApiRetryPolicy = GoogleApiRetryPolicy(
    max_retries=_SHEETS_MAX_RETRIES,
    base_delay_sec=_SHEETS_BASE_DELAY_SEC,
    max_delay_sec=_SHEETS_MAX_DELAY_SEC,
    transient_status_codes=_SHEETS_TRANSIENT_STATUS_CODES,
    connection_errors=_SHEETS_CONNECTION_ERRORS,
    log_prefix="sheets",
    resource_label="spreadsheet_id",
    transient_error_factory=lambda status_code, spreadsheet_id, original: (
        GoogleSheetsTransientError(
            status_code=status_code,
            spreadsheet_id=spreadsheet_id,
            original=original,
        )
    ),
)


def _execute_sheets_with_retry(
    operation_name: str,
    spreadsheet_id: str,
    request_callable: Callable[[], Any],
) -> Any:
    return execute_with_retry(
        policy=_SHEETS_RETRY_POLICY,
        logger=LOGGER,
        operation_name=operation_name,
        resource_id=spreadsheet_id,
        request_callable=request_callable,
    )


class GoogleSheetsClient:
    def __init__(self, sheets_service: Any) -> None:
        self._sheets_service: Any = sheets_service

    def ping_access(self, spreadsheet_id: str) -> Tuple[str, str]:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            _execute_sheets_with_retry(
                operation_name="ping_access",
                spreadsheet_id=spreadsheet_id,
                request_callable=lambda: self._sheets_service.spreadsheets()
                .get(
                    spreadsheetId=spreadsheet_id,
                    fields="spreadsheetId,properties(title)",
                )
                .execute(),
            ),
        )
        resolved_id: str = str(response.get("spreadsheetId") or spreadsheet_id).strip()
        properties: Dict[str, Any] = cast(
            Dict[str, Any], response.get("properties", {})
        )
        title: str = str(properties.get("title") or "unknown").strip() or "unknown"
        return (resolved_id, title)

    def read_rows(self, spreadsheet_id: str, range_name: str) -> List[SheetRow]:
        try:
            response: Dict[str, Any] = cast(
                Dict[str, Any],
                _execute_sheets_with_retry(
                    operation_name="read_rows",
                    spreadsheet_id=spreadsheet_id,
                    request_callable=lambda: self._sheets_service.spreadsheets()
                    .values()
                    .get(spreadsheetId=spreadsheet_id, range=range_name)
                    .execute(),
                ),
            )
        except HttpError as error:
            error_text: str = str(error)
            if "Unable to parse range" not in error_text:
                raise
            LOGGER.warning(
                "Range %s is invalid for spreadsheet %s; fallback to A:F on first sheet.",
                range_name,
                spreadsheet_id,
            )
            response = cast(
                Dict[str, Any],
                _execute_sheets_with_retry(
                    operation_name="read_rows_fallback_range",
                    spreadsheet_id=spreadsheet_id,
                    request_callable=lambda: self._sheets_service.spreadsheets()
                    .values()
                    .get(spreadsheetId=spreadsheet_id, range="A:F")
                    .execute(),
                ),
            )
        values: List[List[str]] = cast(List[List[str]], response.get("values", []))
        if not values:
            LOGGER.warning("Google Sheets range is empty: %s", range_name)
            return []

        header: List[str] = [str(value).strip() for value in values[0]]
        normalized_header: List[str] = [_normalize_header_name(item) for item in header]

        links_index: Optional[int] = _find_header_index(
            normalized_header=normalized_header,
            aliases=("links", "link", "url", "video", "youtube"),
        )
        date_index: Optional[int] = _find_header_index(
            normalized_header=normalized_header,
            aliases=("date", "дата", "day"),
        )
        time_index: Optional[int] = _find_header_index(
            normalized_header=normalized_header,
            aliases=("time", "время", "hour"),
        )
        LOGGER.info(
            "Sheets header detected: links_index=%s date_index=%s time_index=%s",
            links_index,
            date_index,
            time_index,
        )
        if links_index is None or date_index is None or time_index is None:
            raise RuntimeError(
                "Google Sheets header не распознан. "
                f"Found columns: {header!r}. "
                "Нужны колонки для Links/Date/Time."
            )

        rows: List[SheetRow] = []
        for row_number, row_values in enumerate(values[1:], start=2):
            rows.append(
                SheetRow(
                    row_number=row_number,
                    link=_value_from_row(row_values=row_values, index=links_index),
                    date_raw=_value_from_row(row_values=row_values, index=date_index),
                    time_raw=_value_from_row(row_values=row_values, index=time_index),
                    links_column_index=links_index,
                )
            )
        return rows

    def update_cell_string(
        self,
        *,
        spreadsheet_id: str,
        cell_a1: str,
        value: str,
    ) -> None:
        _execute_sheets_with_retry(
            operation_name="update_cell_string",
            spreadsheet_id=spreadsheet_id,
            request_callable=lambda: self._sheets_service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=cell_a1,
                valueInputOption="RAW",
                body={"values": [[str(value or "")]]},
            )
            .execute(),
        )


def _value_from_row(row_values: List[str], index: int) -> str:
    if index >= len(row_values):
        return ""
    return str(row_values[index]).strip()


def _normalize_header_name(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", value.strip().lower(), flags=re.IGNORECASE)


def _find_header_index(
    normalized_header: List[str],
    aliases: Tuple[str, ...],
) -> Optional[int]:
    normalized_aliases: Tuple[str, ...] = tuple(
        _normalize_header_name(alias) for alias in aliases
    )
    for index, name in enumerate(normalized_header):
        if name in normalized_aliases:
            return index
    for index, name in enumerate(normalized_header):
        if any(alias in name for alias in normalized_aliases if alias):
            return index
    return None
