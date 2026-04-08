from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List

from app.config.settings import AppConfig, AppTemplates
from app.core.models import PlannedVideo
from app.publish.telegram_batch_sender import TelegramDateBatch, send_telegram_date_batch
from app.telegram.bot_client import TelegramBotClient

from .slot_processing import SlotProcessResult


@dataclass(frozen=True)
class DailyTelegramPublishResult:
    sent_count: int
    failed_count: int
    skipped_count: int


def publish_daily_telegram(
    *,
    logger: logging.Logger,
    config: AppConfig,
    telegram_client: TelegramBotClient,
    templates: AppTemplates,
    slot_results: List[SlotProcessResult],
    date_key: str,
    doc_url: str,
    header_context: Dict[str, str],
    dry_run: bool,
    processing_mode: str,
    branch_label: str,
) -> DailyTelegramPublishResult:
    if not slot_results:
        result = DailyTelegramPublishResult(sent_count=0, failed_count=0, skipped_count=1)
        logger.info(
            "[%s] telegram_publish_start date_key=%s slot_count=0 video_count=0 status=skipped reason=empty_slot_results",
            branch_label,
            date_key,
        )
        logger.info(
            "[%s] telegram_publish_finish date_key=%s status=skipped",
            branch_label,
            date_key,
        )
        logger.info(
            "[%s] telegram_publish_forensic date_key=%s messages_sent=%d skipped_messages=%d doc_url_attached=%s overall_status=skipped",
            branch_label,
            date_key,
            result.sent_count,
            result.skipped_count,
            "yes" if str(doc_url or "").strip() else "no",
        )
        return result

    combined_day_videos: List[PlannedVideo] = [
        item for slot in slot_results for item in slot.day_videos
    ]

    slot_keys: List[str] = [slot.slot_key for slot in slot_results]
    slots_list_text: str = f"[{','.join(slot_keys)}]"
    logger.info(
        "[%s] telegram_publish_start date_key=%s doc_url=%s video_count=%d slots=%s send_mode=per_date",
        branch_label,
        date_key,
        doc_url,
        len(combined_day_videos),
        slots_list_text,
    )
    logger.info(
        "[%s] telegram_batch_start date_key=%s doc_url=%s video_count=%d slots=%s send_mode=per_date",
        branch_label,
        date_key,
        doc_url,
        len(combined_day_videos),
        slots_list_text,
    )

    if dry_run or not config.telegram.enabled:
        result = DailyTelegramPublishResult(sent_count=0, failed_count=0, skipped_count=1)
        logger.info(
            "[%s] telegram_publish_finish date_key=%s status=%s video_count=%d",
            branch_label,
            date_key,
            "dry_run" if dry_run else "disabled",
            len(combined_day_videos),
        )
        logger.info(
            "[%s] telegram_publish_forensic date_key=%s messages_sent=%d skipped_messages=%d doc_url_attached=%s overall_status=%s",
            branch_label,
            date_key,
            result.sent_count,
            result.skipped_count,
            "yes" if str(doc_url or "").strip() else "no",
            "dry_run" if dry_run else "disabled",
        )
        return result

    telegram_send_ok: bool = True
    try:
        send_telegram_date_batch(
            logger=logger,
            config=config,
            telegram_client=telegram_client,
            templates=templates,
            batch=TelegramDateBatch(
                slot_results=slot_results,
                header_context=header_context,
                doc_url=doc_url,
                dry_run=dry_run,
                date_key=date_key,
                processing_mode=processing_mode,
            ),
        )
    except Exception as telegram_error:
        telegram_send_ok = False
        logger.error(
            "[%s] telegram_send_failed date_key=%s error=%s",
            branch_label,
            date_key,
            telegram_error,
        )

    final_status: str = "sent" if telegram_send_ok else "failed"
    logger.info(
        "[%s] telegram_publish_finish date_key=%s status=%s video_count=%d",
        branch_label,
        date_key,
        final_status,
        len(combined_day_videos),
    )
    result: DailyTelegramPublishResult = (
        DailyTelegramPublishResult(sent_count=1, failed_count=0, skipped_count=0)
        if telegram_send_ok
        else DailyTelegramPublishResult(sent_count=0, failed_count=1, skipped_count=0)
    )
    logger.info(
        "[%s] telegram_publish_forensic date_key=%s messages_sent=%d skipped_messages=%d doc_url_attached=%s overall_status=%s",
        branch_label,
        date_key,
        result.sent_count,
        result.skipped_count,
        "yes" if str(doc_url or "").strip() else "no",
        final_status,
    )
    return result

