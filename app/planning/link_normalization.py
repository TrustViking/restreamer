from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from app.core.models import LinkNormalizationCandidate
from app.google.sheets_client import GoogleSheetsClient


@dataclass(frozen=True)
class LinkWritebackOutcome:
    status: str
    writeback: str
    reason: str = ""


def column_index_to_letters(column_index_zero_based: int) -> str:
    if column_index_zero_based < 0:
        raise ValueError("column index must be >= 0")
    value: int = column_index_zero_based + 1
    symbols: List[str] = []
    while value > 0:
        value, rem = divmod(value - 1, 26)
        symbols.append(chr(ord("A") + rem))
    return "".join(reversed(symbols))


def sheet_name_from_range(range_name: str) -> Optional[str]:
    cleaned: str = str(range_name or "").strip()
    if "!" not in cleaned:
        return None
    raw_sheet_name: str = cleaned.split("!", 1)[0].strip()
    if not raw_sheet_name:
        return None
    return raw_sheet_name.strip("'")


def build_sheet_cell_a1(
    *,
    sheet_name: Optional[str],
    row_index: int,
    col_index_zero_based: int,
) -> str:
    column_letters: str = column_index_to_letters(col_index_zero_based)
    if row_index < 1:
        raise ValueError("row index must be >= 1")
    if not sheet_name:
        return f"{column_letters}{row_index}"
    escaped_sheet_name: str = str(sheet_name).replace("'", "''")
    return f"'{escaped_sheet_name}'!{column_letters}{row_index}"


def handle_normalized_link_writeback(
    *,
    logger: logging.Logger,
    summarize_error: Callable[[Exception], str],
    sheets_client: GoogleSheetsClient,
    spreadsheet_id: str,
    sheet_name_for_writeback: Optional[str],
    row_number: int,
    links_column_index: int,
    old_link: str,
    normalized_link: str,
    writeback_enabled: bool,
    normalization_candidates: Optional[List[LinkNormalizationCandidate]] = None,
    links_column_label: Optional[str] = None,
) -> LinkWritebackOutcome:
    old_link_value: str = str(old_link or "").strip()
    new_link_value: str = str(normalized_link or "").strip()
    if not new_link_value:
        return LinkWritebackOutcome(
            status="invalid",
            writeback="not_applicable",
            reason="empty_normalized_link",
        )
    if new_link_value == old_link_value:
        logger.debug(
            'Row %d: link_validated changed=false url="%s"',
            row_number,
            new_link_value,
        )
        return LinkWritebackOutcome(
            status="unchanged",
            writeback="not_applicable",
        )
    if normalization_candidates is not None:
        normalization_candidates.append(
            LinkNormalizationCandidate(
                row_index=row_number,
                column_ref=str(links_column_label or "").strip()
                or column_index_to_letters(links_column_index),
                old_value=old_link_value,
                new_value=new_link_value,
            )
        )
    if not writeback_enabled:
        logger.info(
            'Row %d: link_changed writeback=disabled old="%s" new="%s"',
            row_number,
            old_link_value,
            new_link_value,
        )
        return LinkWritebackOutcome(
            status="changed",
            writeback="no",
            reason="writeback_disabled",
        )
    try:
        link_cell_a1: str = build_sheet_cell_a1(
            sheet_name=sheet_name_for_writeback,
            row_index=row_number,
            col_index_zero_based=links_column_index,
        )
        sheets_client.update_cell_string(
            spreadsheet_id=spreadsheet_id,
            cell_a1=link_cell_a1,
            value=new_link_value,
        )
        logger.info(
            'Row %d: link_changed writeback=applied old="%s" new="%s"',
            row_number,
            old_link_value,
            new_link_value,
        )
        return LinkWritebackOutcome(
            status="changed",
            writeback="yes",
            reason="writeback_applied",
        )
    except Exception as writeback_error:
        reason_text: str = summarize_error(writeback_error)
        logger.warning(
            'Row %d: link_changed writeback=failed old="%s" new="%s" reason=%s',
            row_number,
            old_link_value,
            new_link_value,
            reason_text,
        )
        return LinkWritebackOutcome(
            status="changed",
            writeback="no",
            reason=f"writeback_failed:{reason_text}",
        )


def log_link_forensic_event(
    *,
    logger: logging.Logger,
    row_number: int,
    original_url: str,
    normalized_url: str,
    status: str,
    writeback: str,
    reason: str = "",
) -> None:
    logger.info(
        'link_forensic row=%d original_url="%s" normalized_url="%s" status=%s writeback=%s reason=%s',
        row_number,
        _compact_single_line(original_url, max_len=220),
        _compact_single_line(normalized_url, max_len=220),
        status,
        writeback,
        reason or "none",
    )


def _compact_single_line(value: object, *, max_len: int = 300) -> str:
    compact: str = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len].rstrip()}..."


def log_link_normalization_report(
    *,
    logger: logging.Logger,
    normalization_candidates: List[LinkNormalizationCandidate],
    writeback_enabled: bool,
    report_limit: int,
) -> None:
    logger.info("links_normalized_total=%d", len(normalization_candidates))
    logger.info("links_writeback=%s", "enabled" if writeback_enabled else "disabled")
    if not normalization_candidates:
        return
    if writeback_enabled:
        logger.info(
            "links_normalized_report_details=omitted total=%d",
            len(normalization_candidates),
        )
        return
    for item in normalization_candidates[:report_limit]:
        logger.info(
            "row=%d col=%s old=%s -> new=%s",
            item.row_index,
            item.column_ref,
            _compact_single_line(item.old_value, max_len=220),
            _compact_single_line(item.new_value, max_len=220),
        )
    if len(normalization_candidates) > report_limit:
        logger.info(
            "links_normalized_report_truncated=%d",
            len(normalization_candidates) - report_limit,
        )
