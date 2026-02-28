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
                "Range %s is invalid for spreadsheet %s; fallback to A:C on first sheet.",
                range_name,
                spreadsheet_id,
            )
            response = (
                self._sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range="A:C")
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
        merge_index: Optional[int] = None
        merge_aliases_specific: Tuple[str, ...] = (
            "Merge (ua/en/ru)",
            "Translate/Overwrite (ua/en/ru)",
        )
        merge_aliases_generic: Tuple[str, ...] = (
            "Merge",
            "Translate/Overwrite",
            "Translate",
            "Overwrite",
            "Перевод",
            "Переклад",
            "Замена",
            "Перезапись",
        )
        merge_aliases_specific_normalized: Tuple[str, ...] = tuple(
            _normalize_header_name(alias) for alias in merge_aliases_specific
        )
        merge_aliases_generic_normalized: Tuple[str, ...] = tuple(
            _normalize_header_name(alias) for alias in merge_aliases_generic
        )

        for idx, name in enumerate(normalized_header):
            if name in merge_aliases_specific_normalized:
                merge_index = idx
                break
        if merge_index is None:
            for idx, name in enumerate(normalized_header):
                if any(
                    alias and alias in name
                    for alias in merge_aliases_specific_normalized
                ):
                    merge_index = idx
                    break
        if merge_index is None:
            for idx, name in enumerate(normalized_header):
                if name in merge_aliases_generic_normalized:
                    merge_index = idx
                    break
        if merge_index is None:
            for idx, name in enumerate(normalized_header):
                if any(
                    alias and alias in name
                    for alias in merge_aliases_generic_normalized
                ):
                    merge_index = idx
                    break

        merge_header_text: Optional[str] = (
            header[merge_index]
            if merge_index is not None and merge_index < len(header)
            else None
        )
        LOGGER.info(
            "Sheets header detected: links_index=%s date_index=%s time_index=%s merge_index=%s merge_header=%r",
            links_index,
            date_index,
            time_index,
            merge_index,
            merge_header_text,
        )
        if merge_index is None:
            possible_translate_headers: List[str] = [
                original_name
                for original_name, normalized_name in zip(header, normalized_header)
                if "translate" in normalized_name or "overwrite" in normalized_name
            ]
            if possible_translate_headers:
                LOGGER.warning(
                    "Merge column not detected. Found possible translate/overwrite column in headers: %r",
                    possible_translate_headers,
                )
        if links_index is None or date_index is None or time_index is None:
            raise RuntimeError(
                "Google Sheets header не распознан. "
                f"Found columns: {header!r}. "
                "Нужны колонки для Links/Date/Time."
            )

        rows: List[SheetRow] = []
        for row_number, row_values in enumerate(values[1:], start=2):
            merge_raw: str = (
                _value_from_row(row_values=row_values, index=merge_index)
                if merge_index is not None
                else ""
            )
            merge_languages: List[str] = _parse_merge_languages(merge_raw)
            if merge_raw.strip():
                LOGGER.info(
                    "Row %d: merge column parsed raw=%r tokens=%s",
                    row_number,
                    merge_raw,
                    merge_languages,
                )
                valid_tokens: set[str] = {"ua", "uk", "en", "ru"}
                raw_tokens: List[str] = [
                    item.strip()
                    for item in re.split(r"\s*[;,|]\s*", merge_raw)
                    if item.strip()
                ]
                for token in raw_tokens:
                    if token.lower() not in valid_tokens:
                        LOGGER.warning(
                            "Row %d: invalid Merge token %r ignored",
                            row_number,
                            token,
                        )
            rows.append(
                SheetRow(
                    row_number=row_number,
                    link=_value_from_row(row_values=row_values, index=links_index),
                    date_raw=_value_from_row(row_values=row_values, index=date_index),
                    time_raw=_value_from_row(row_values=row_values, index=time_index),
                    merge_raw=merge_raw,
                    merge_languages=merge_languages,
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


def _parse_merge_languages(value: str) -> List[str]:
    token_map: Dict[str, str] = {
        "ua": "uk",
        "uk": "uk",
        "en": "en",
        "ru": "ru",
    }
    stable_order: Tuple[str, ...] = ("uk", "en", "ru")
    selected: set[str] = set()
    for raw_token in re.split(r"\s*[;,|]\s*", str(value or "")):
        token: str = raw_token.strip().lower()
        if not token:
            continue
        mapped: Optional[str] = token_map.get(token)
        if mapped:
            selected.add(mapped)
    return [language for language in stable_order if language in selected]
