from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as datetime_time
from typing import Dict, List, Optional, TYPE_CHECKING

from app.config.settings import AppConfig, AppTemplates
from app.core.language_display import language_to_flag_emoji
from app.core.models import PlannedVideo
from app.planning import language_sort_key
from app.publish.telegram_renderer import (
    build_telegram_header_text,
    build_telegram_key_form_reminder,
    build_telegram_language_block,
    build_telegram_language_digest_block,
    build_telegram_language_merged_block,
    build_telegram_language_nomerge_block,
    build_telegram_post_header_text,
)
from app.telegram.bot_client import TelegramBotClient

if TYPE_CHECKING:
    from app.pipeline.slot_processing import SlotProcessResult


@dataclass(frozen=True)
class TelegramDateBatch:
    slot_results: List["SlotProcessResult"]
    header_context: Dict[str, str]
    doc_url: str
    dry_run: bool
    date_key: str
    processing_mode: str


def _slot_language_flag(language: str) -> str:
    return language_to_flag_emoji(language)


def _slot_language_separator(language: str, config: AppConfig) -> str:
    flag_symbol: str = _slot_language_flag(language)
    return flag_symbol * max(1, int(config.telegram.flag_repeat_count))


def send_telegram_date_batch(
    *,
    logger: logging.Logger,
    config: AppConfig,
    telegram_client: TelegramBotClient,
    templates: Optional[AppTemplates],
    batch: TelegramDateBatch,
) -> None:
    if not config.telegram.enabled:
        logger.info("Telegram disabled by TELEGRAM_ENABLED=0.")
        return
    logger.info("Telegram send for date=%s", batch.date_key)

    date_separator: str = config.telegram.symbol_separator * max(
        1, int(config.telegram.separator_repeat_count)
    )
    start_separator: str = config.telegram.symbol_separator_start * max(
        1, int(config.telegram.separator_start_repeat_count)
    )
    header_message: str = build_telegram_header_text(
        context=batch.header_context,
        generated_doc_url=batch.doc_url,
        config=config,
    )
    if batch.dry_run:
        logger.info("DRY RUN Telegram date separator start:\n%s", start_separator)
        logger.info("DRY RUN Telegram header:\n%s", header_message)
    else:
        telegram_client.send_text(start_separator)
        telegram_client.send_text(header_message)

    def _send_slot_key_form(*, context: Dict[str, str]) -> None:
        key_form_reminder: str = build_telegram_key_form_reminder(
            context,
            config=config,
        )
        if batch.dry_run:
            logger.info("DRY RUN Telegram key form reminder block:\n%s", key_form_reminder)
            return
        telegram_client.send_text(key_form_reminder)

    grouped_for_digest: Dict[str, List[PlannedVideo]] = {}
    for slot in batch.slot_results:
        language: str = slot.language
        if language == "unknown":
            language = next(
                (key for key, value in slot.language_groups.items() if value),
                language,
            )
        source_items: List[PlannedVideo] = list(
            slot.day_videos or slot.language_groups.get(language, [])
        )
        items: List[PlannedVideo] = sorted(
            source_items,
            key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
        )
        if not items:
            continue
        slot_context: Dict[str, str] = slot.header_context or batch.header_context
        slot_separator: str = _slot_language_separator(language, config)
        if batch.dry_run:
            logger.info(
                "DRY RUN Telegram slot separator (%s):\n%s",
                language,
                slot_separator,
            )
        else:
            telegram_client.send_text(slot_separator)
        grouped_for_digest.setdefault(language, []).extend(items)
        merged_for_language = slot.merged_content_by_language.get(language)
        merge_attempt = slot.merge_audit_by_language.get(language)
        logger.info(
            "telegram_send slot_key=%s language=%s item_count=%d processing_mode=%s",
            slot.slot_key,
            language,
            len(items),
            batch.processing_mode,
        )
        if len(items) > 1 and merged_for_language is not None:
            merged_block_text: str = build_telegram_language_merged_block(
                language=language,
                videos=items,
                merged_content=merged_for_language,
                merge_attempt=merge_attempt,
                config=config,
                templates=templates,
            )
            if batch.dry_run:
                logger.info(
                    "DRY RUN Telegram merged block (%s):\n%s",
                    language,
                    merged_block_text,
                )
                for item in items:
                    if item.local_thumbnail_path is None:
                        raise RuntimeError(
                            f"Row {item.row_number}: local thumbnail path is not set."
                        )
                    logger.info(
                        "DRY RUN Telegram merged preview file row %d: %s",
                        item.row_number,
                        item.local_thumbnail_path.name,
                    )
            else:
                telegram_client.send_text(merged_block_text)
                for item in items:
                    if item.local_thumbnail_path is None:
                        raise RuntimeError(
                            f"Row {item.row_number}: local thumbnail path is not set."
                        )
                    telegram_client.send_photo_as_file_bytes(
                        photo_bytes=item.thumbnail.bytes_data,
                        filename=item.local_thumbnail_path.name,
                        mime_type=item.thumbnail.mime_type,
                    )
            _send_slot_key_form(context=slot_context)
            continue
        if len(items) > 1 and batch.processing_mode == "nomerge":
            nomerge_block_text: str = build_telegram_language_nomerge_block(
                language=language,
                videos=items,
                config=config,
                templates=templates,
            )
            if batch.dry_run:
                logger.info(
                    "DRY RUN Telegram nomerge block (%s):\n%s",
                    language,
                    nomerge_block_text,
                )
                for item in items:
                    if item.local_thumbnail_path is None:
                        raise RuntimeError(
                            f"Row {item.row_number}: local thumbnail path is not set."
                        )
                    logger.info(
                        "DRY RUN Telegram nomerge preview file row %d: %s",
                        item.row_number,
                        item.local_thumbnail_path.name,
                    )
            else:
                telegram_client.send_text(nomerge_block_text)
                for item in items:
                    if item.local_thumbnail_path is None:
                        raise RuntimeError(
                            f"Row {item.row_number}: local thumbnail path is not set."
                        )
                    telegram_client.send_photo_as_file_bytes(
                        photo_bytes=item.thumbnail.bytes_data,
                        filename=item.local_thumbnail_path.name,
                        mime_type=item.thumbnail.mime_type,
                    )
            _send_slot_key_form(context=slot_context)
            continue
        for item in items:
            block_text: str = build_telegram_language_block(
                item,
                config=config,
                templates=templates,
            )
            if batch.dry_run:
                logger.info(
                    "DRY RUN Telegram block for row %d:\n%s",
                    item.row_number,
                    block_text,
                )
                continue
            if item.local_thumbnail_path is None:
                raise RuntimeError(
                    f"Row {item.row_number}: local thumbnail path is not set."
                )
            telegram_client.send_text(block_text)
            telegram_client.send_photo_as_file_bytes(
                photo_bytes=item.thumbnail.bytes_data,
                filename=item.local_thumbnail_path.name,
                mime_type=item.thumbnail.mime_type,
            )
        _send_slot_key_form(context=slot_context)

    sparkle_separator_message: str = config.templates.telegram_sparkle_separator
    if batch.dry_run:
        logger.info(
            "DRY RUN Telegram sparkle separator block:\n%s",
            sparkle_separator_message,
        )
    else:
        telegram_client.send_text(sparkle_separator_message)

    post_header_message: str = build_telegram_post_header_text(
        header_context=batch.header_context,
        config=config,
    )
    if batch.dry_run:
        logger.info(
            "DRY RUN Telegram post-date header block:\n%s",
            post_header_message,
        )
    else:
            telegram_client.send_text(post_header_message)

    def _digest_language_sort_key(lang: str) -> tuple:
        lang_items: List[PlannedVideo] = grouped_for_digest.get(lang, [])
        earliest_time = min(
            (item.scheduled_at_kiev.time() for item in lang_items),
            default=None,
        )
        return (earliest_time or datetime_time(), language_sort_key(lang))

    ordered_digest_languages: List[str] = sorted(
        grouped_for_digest.keys(),
        key=_digest_language_sort_key,
    )
    for language in ordered_digest_languages:
        items: List[PlannedVideo] = sorted(
            grouped_for_digest.get(language, []),
            key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
        )
        if not items:
            continue
        digest_text: str = build_telegram_language_digest_block(
            language=language,
            videos=items,
            context=batch.header_context,
            config=config,
        )
        if batch.dry_run:
            logger.info(
                "DRY RUN Telegram language digest block (%s):\n%s",
                language,
                digest_text,
            )
            continue
        telegram_client.send_text(digest_text)
        for item in items:
            if item.local_thumbnail_path is None:
                raise RuntimeError(
                    f"Row {item.row_number}: local thumbnail path is not set."
                )
            telegram_client.send_photo_as_file_bytes(
                photo_bytes=item.thumbnail.bytes_data,
                filename=item.local_thumbnail_path.name,
                mime_type=item.thumbnail.mime_type,
            )

    if batch.dry_run:
        logger.info("DRY RUN Telegram date separator end:\n%s", date_separator)
    else:
        telegram_client.send_text(date_separator)

