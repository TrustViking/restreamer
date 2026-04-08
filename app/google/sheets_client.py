from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import SheetRow

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore


LOGGER = _get_logger_impl(__name__)


class GoogleSheetsClient:
    def __init__(self, sheets_service: Any) -> None:
        self._sheets_service: Any = sheets_service

    def ping_access(self, spreadsheet_id: str) -> Tuple[str, str]:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._sheets_service.spreadsheets()
            .get(
                spreadsheetId=spreadsheet_id,
                fields="spreadsheetId,properties(title)",
            )
            .execute(),
        )
        resolved_id: str = str(response.get("spreadsheetId") or spreadsheet_id).strip()
        properties: Dict[str, Any] = cast(
            Dict[str, Any], response.get("properties", {})
        )
        title: str = str(properties.get("title") or "unknown").strip() or "unknown"
        return (resolved_id, title)

    def read_rows(self, spreadsheet_id: str, range_name: str) -> List[SheetRow]:
        try:
            response: Dict[str, Any] = (
                self._sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )
        except HttpError as error:
            error_text: str = str(error)
            if "Unable to parse range" not in error_text:
                raise
            LOGGER.warning(
                "Range %s is invalid for spreadsheet %s; fallback to A:D on first sheet.",
                range_name,
                spreadsheet_id,
            )
            response = (
                self._sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range="A:D")
                .execute()
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
        self._sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=cell_a1,
            valueInputOption="RAW",
            body={"values": [[str(value or "")]]},
        ).execute()


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
