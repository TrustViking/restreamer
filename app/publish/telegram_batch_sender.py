from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config.settings import AppConfig, AppTemplates
from app.core.models import LanguageMergeAttempt, MergedLanguageContent, PlannedVideo
from app.planning import planned_video_block_language
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


@dataclass(frozen=True)
class TelegramDateBatch:
    day_videos: List[PlannedVideo]
    header_context: Dict[str, str]
    doc_url: str
    merged_content_by_language: Optional[Dict[str, MergedLanguageContent]]
    merge_audit_by_language: Optional[Dict[str, LanguageMergeAttempt]]
    dry_run: bool
    slot_key: str
    processing_mode: str


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
    logger.info("Telegram send for slot=%s", batch.slot_key)

    date_separator: str = config.telegram.symbol_separator * max(
        1, int(config.telegram.separator_repeat_count)
    )
    header_message: str = build_telegram_header_text(
        context=batch.header_context,
        generated_doc_url=batch.doc_url,
        config=config,
    )
    if batch.dry_run:
        logger.info("DRY RUN Telegram date separator start:\n%s", date_separator)
        logger.info("DRY RUN Telegram header:\n%s", header_message)
    else:
        telegram_client.send_text(date_separator)
        telegram_client.send_text(header_message)

    grouped: Dict[str, List[PlannedVideo]] = {
        "uk": [],
        "en": [],
        "ru": [],
        "other": [],
    }
    merged_map: Dict[str, MergedLanguageContent] = (
        batch.merged_content_by_language or {}
    )
    merge_attempt_map: Dict[str, LanguageMergeAttempt] = (
        batch.merge_audit_by_language or {}
    )
    for video in batch.day_videos:
        grouped[planned_video_block_language(video)].append(video)
    logger.info(
        "telegram_send batch_key=%s items_uk=%d en=%d ru=%d other=%d processing_mode=%s",
        batch.slot_key,
        len(grouped["uk"]),
        len(grouped["en"]),
        len(grouped["ru"]),
        len(grouped["other"]),
        batch.processing_mode,
    )

    for language in ("uk", "en", "ru", "other"):
        items: List[PlannedVideo] = sorted(
            grouped[language],
            key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
        )
        merged_for_language: Optional[MergedLanguageContent] = merged_map.get(language)
        if len(items) > 1 and merged_for_language is not None:
            merged_block_text: str = build_telegram_language_merged_block(
                language=language,
                videos=items,
                merged_content=merged_for_language,
                merge_attempt=merge_attempt_map.get(language),
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

    key_form_reminder: str = build_telegram_key_form_reminder(
        batch.header_context,
        config=config,
    )
    if batch.dry_run:
        logger.info("DRY RUN Telegram key form reminder block:\n%s", key_form_reminder)
    else:
        telegram_client.send_text(key_form_reminder)

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

    for language in ("uk", "en", "ru", "other"):
        items = sorted(
            grouped[language],
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

