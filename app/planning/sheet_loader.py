from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import List, Optional
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.config.validators import now_filter_timezone
from app.core.env_flags import (
    sheets_autoexpand_range_from_env,
    sheets_link_writeback_enabled_from_env,
)
from app.core.models import LinkNormalizationCandidate, SheetRow
from app.pipeline.runtime_services import BatchServices
from app.planning import (
    expand_sheet_range_to_af,
    merge_semantics_from_env,
    sheet_name_from_range,
    sheet_range_includes_merge_column,
)


@dataclass(frozen=True)
class BatchSheetState:
    rows: List[SheetRow]
    sheet_name_for_writeback: Optional[str]
    sheets_link_writeback_enabled: bool
    link_normalization_candidates: List[LinkNormalizationCandidate]
    now_for_filter: datetime
    merge_semantics: str


def load_sheet_state(
    *,
    logger: logging.Logger,
    config: AppConfig,
    services: BatchServices,
    kiev_tz: ZoneInfo,
    run_id: str,
) -> BatchSheetState:
    configured_range: str = config.google.sheets_range
    effective_range: str = configured_range
    if not sheet_range_includes_merge_column(configured_range):
        logger.warning(
            "Sheets range %r does not include Merge column (E); merge settings will be ignored. Use A:F.",
            configured_range,
        )
        if sheets_autoexpand_range_from_env():
            effective_range = expand_sheet_range_to_af(configured_range)
            logger.warning(
                "STG_SHEETS_AUTOEXPAND_RANGE=1 -> using expanded range %r",
                effective_range,
            )
    logger.info(
        "run_id=%s Reading Google Sheets: spreadsheet=%s range=%s",
        run_id,
        config.google.sheets_id,
        effective_range,
    )
    rows: List[SheetRow] = services.sheets_client.read_rows(
        spreadsheet_id=config.google.sheets_id,
        range_name=effective_range,
    )
    logger.info("Rows loaded from sheet: %d", len(rows))
    now_filter_tz: timezone | ZoneInfo = now_filter_timezone(
        config.processing.now_tz_mode,
        kiev_tz,
    )
    now_for_filter: datetime = datetime.now(now_filter_tz)
    logger.info(
        "now_tz=%s now=%s",
        config.processing.now_tz_mode,
        now_for_filter.isoformat(),
    )
    merge_semantics: str = merge_semantics_from_env()
    logger.info("merge_semantics=%s", merge_semantics)
    return BatchSheetState(
        rows=rows,
        sheet_name_for_writeback=sheet_name_from_range(effective_range),
        sheets_link_writeback_enabled=sheets_link_writeback_enabled_from_env(),
        link_normalization_candidates=[],
        now_for_filter=now_for_filter,
        merge_semantics=merge_semantics,
    )

