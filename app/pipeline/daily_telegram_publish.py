from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Set

from app.config.settings import AppConfig, AppTemplates
from app.core.models import LanguageMergeAttempt, MergedLanguageContent, PlannedVideo
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

    combined_day_videos: List[PlannedVideo] = []
    combined_merged_content_by_language: Dict[str, MergedLanguageContent] = {}
    combined_merge_audit_by_language: Dict[str, LanguageMergeAttempt] = {}
    merge_language_collisions: Set[str] = set()
    for slot in slot_results:
        combined_day_videos.extend(slot.day_videos)
        for language, merged_content in slot.merged_content_by_language.items():
            if (
                language in merge_language_collisions
                or language in combined_merged_content_by_language
            ):
                merge_language_collisions.add(language)
                combined_merged_content_by_language.pop(language, None)
                combined_merge_audit_by_language.pop(language, None)
                continue
            combined_merged_content_by_language[language] = merged_content
        for language, merge_attempt in slot.merge_audit_by_language.items():
            if language in merge_language_collisions:
                continue
            if language in combined_merge_audit_by_language:
                merge_language_collisions.add(language)
                combined_merged_content_by_language.pop(language, None)
                combined_merge_audit_by_language.pop(language, None)
                continue
            combined_merge_audit_by_language[language] = merge_attempt
    if merge_language_collisions:
        logger.info(
            "[%s] telegram merge map collision: date_key=%s disabled_languages=%s",
            branch_label,
            date_key,
            sorted(merge_language_collisions),
        )

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
    send_telegram_date_batch(
        logger=logger,
        config=config,
        telegram_client=telegram_client,
        templates=templates,
        batch=TelegramDateBatch(
            day_videos=combined_day_videos,
            header_context=header_context,
            doc_url=doc_url,
            merged_content_by_language=combined_merged_content_by_language,
            merge_audit_by_language=combined_merge_audit_by_language,
            dry_run=dry_run,
            slot_key=f"date:{date_key}",
            processing_mode=processing_mode,
        ),
    )
    logger.info(
        "[%s] telegram_publish_finish date_key=%s status=%s video_count=%d",
        branch_label,
        date_key,
        "dry_run" if dry_run else ("disabled" if not config.telegram.enabled else "sent"),
        len(combined_day_videos),
    )
    if dry_run or not config.telegram.enabled:
        result = DailyTelegramPublishResult(sent_count=0, failed_count=0, skipped_count=1)
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
    result = DailyTelegramPublishResult(sent_count=1, failed_count=0, skipped_count=0)
    logger.info(
        "[%s] telegram_publish_forensic date_key=%s messages_sent=%d skipped_messages=%d doc_url_attached=%s overall_status=sent",
        branch_label,
        date_key,
        result.sent_count,
        result.skipped_count,
        "yes" if str(doc_url or "").strip() else "no",
    )
    return result

