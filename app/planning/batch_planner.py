from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.core.error_summary import summarize_error
from app.core.models import (
    NormalizedImage,
    PlannedVideo,
    PreparedVideo,
    RowVideoCharacteristics,
    VideoMetadata,
)
from app.google import GoogleDriveClient
from app.ingest.youtube_metadata import YouTubeMetadataFetcher, normalize_youtube_link
from app.media.image_normalizer import normalize_thumbnail
from app.paths.name_builder import NamePathBuilder, build_drive_preview_path_segments
from app.pipeline.runtime_services import BatchServices
from app.planning import (
    column_index_to_letters,
    deduplicate_planned_videos_within_slot_language,
    handle_normalized_link_writeback,
    log_link_forensic_event,
    parse_sheet_datetime,
    planned_video_block_language,
    planned_video_slot_key,
)
from app.planning.sheet_loader import BatchSheetState


def build_prepared_videos(
    *,
    logger: logging.Logger,
    config: AppConfig,
    services: BatchServices,
    sheet_state: BatchSheetState,
    metadata_fetcher: YouTubeMetadataFetcher,
    http_client: Any,
    kiev_tz: ZoneInfo,
) -> List[PreparedVideo]:
    prepared_videos: List[PreparedVideo] = []
    metadata_cache: Dict[str, VideoMetadata] = {}
    thumbnail_cache: Dict[str, NormalizedImage] = {}
    language_cache: Dict[str, str] = {}
    for row in sheet_state.rows:
        logger.info("Row %d: read", row.row_number)
        if not row.link:
            logger.warning("Row %d: skipped, empty Links", row.row_number)
            continue
        if not row.date_raw or not row.time_raw:
            logger.warning("Row %d: skipped, missing Date/Time", row.row_number)
            continue
        try:
            scheduled_at: datetime = parse_sheet_datetime(
                date_raw=row.date_raw,
                time_raw=row.time_raw,
                tz=kiev_tz,
            )
            if scheduled_at < sheet_state.now_for_filter:
                logger.info(
                    "Row %d: skipped, already in the past (%s)",
                    row.row_number,
                    scheduled_at.isoformat(),
                )
                continue
            normalized_link: Optional[str] = normalize_youtube_link(row.link)
            if not normalized_link:
                log_link_forensic_event(
                    logger=logger,
                    row_number=row.row_number,
                    original_url=row.link,
                    normalized_url="",
                    status="invalid",
                    writeback="not_applicable",
                    reason="cannot_extract_video_id",
                )
                logger.warning(
                    "Row %d: skipped, cannot extract YouTube video id from Links=%r",
                    row.row_number,
                    row.link,
                )
                continue
            writeback_outcome = handle_normalized_link_writeback(
                logger=logger,
                summarize_error=summarize_error,
                sheets_client=services.sheets_client,
                spreadsheet_id=config.google_sheets_id,
                sheet_name_for_writeback=sheet_state.sheet_name_for_writeback,
                row_number=row.row_number,
                links_column_index=row.links_column_index,
                old_link=row.link,
                normalized_link=normalized_link,
                writeback_enabled=sheet_state.sheets_link_writeback_enabled,
                normalization_candidates=sheet_state.link_normalization_candidates,
                links_column_label=(
                    f"Links/{column_index_to_letters(row.links_column_index)}"
                ),
            )
            log_link_forensic_event(
                logger=logger,
                row_number=row.row_number,
                original_url=row.link,
                normalized_url=normalized_link,
                status=writeback_outcome.status,
                writeback=writeback_outcome.writeback,
                reason=writeback_outcome.reason,
            )
            metadata: Optional[VideoMetadata] = metadata_cache.get(normalized_link)
            if metadata is None:
                metadata = metadata_fetcher.fetch(video_url=normalized_link)
                metadata_cache[normalized_link] = metadata
                logger.info("Row %d: metadata fetched", row.row_number)
            else:
                logger.info(
                    "Row %d: metadata reused from cache for %s",
                    row.row_number,
                    normalized_link,
                )
            language: Optional[str] = language_cache.get(normalized_link)
            if language is None:
                from app.core.language import detect_language_decision, log_language_decision

                language_decision = detect_language_decision(metadata)
                language = language_decision.final_language
                log_language_decision(
                    row_number=row.row_number,
                    decision=language_decision,
                )
                language_cache[normalized_link] = language
            else:
                logger.info(
                    "Row %d: language=%s (cached)",
                    row.row_number,
                    language,
                )
            normalized_thumbnail: Optional[NormalizedImage] = thumbnail_cache.get(
                normalized_link
            )
            if normalized_thumbnail is None:
                thumbnail_bytes: bytes = http_client.get_bytes(metadata.thumbnail_url)
                normalized_thumbnail = normalize_thumbnail(
                    thumbnail_bytes,
                    logger=logger,
                )
                thumbnail_cache[normalized_link] = normalized_thumbnail
                logger.info("Row %d: thumbnail fetched", row.row_number)
            else:
                logger.info(
                    "Row %d: thumbnail reused from cache for %s",
                    row.row_number,
                    normalized_link,
                )
            prepared_videos.append(
                PreparedVideo(
                    row_number=row.row_number,
                    original_link=row.link,
                    normalized_link=normalized_link,
                    date_raw=row.date_raw,
                    time_raw=row.time_raw,
                    scheduled_at_kiev=scheduled_at,
                    date_key=scheduled_at.strftime("%d%m%y"),
                    date_display=scheduled_at.strftime("%d.%m.%Y"),
                    language=language,
                    metadata=metadata,
                    thumbnail=normalized_thumbnail,
                    local_thumbnail_path=None,
                    merge_raw=row.merge_raw,
                    merge_languages=list(row.merge_languages),
                )
            )
        except Exception as error:
            logger.exception(
                "Row %d: shared preparation failed. reason=%s",
                row.row_number,
                error,
            )
    return prepared_videos


def materialize_prepared_previews(
    *,
    logger: logging.Logger,
    config: AppConfig,
    drive_client: GoogleDriveClient,
    name_builder: NamePathBuilder,
    prepared_videos: List[PreparedVideo],
    dry_run: bool,
) -> List[PreparedVideo]:
    preview_root_folder_id: Optional[str] = (
        config.google_drive_preview_folder_id or config.google_drive_folder_id
    )
    language_positions: Dict[Tuple[str, str], int] = {}
    drive_folder_cache: Dict[Tuple[str, str], Optional[str]] = {}
    materialized_videos: List[PreparedVideo] = []
    for prepared in sorted(
        prepared_videos,
        key=lambda item: (
            item.date_key,
            item.language,
            item.scheduled_at_kiev.time(),
            item.row_number,
        ),
    ):
        language_key: Tuple[str, str] = (prepared.date_key, prepared.language)
        next_position: int = language_positions.get(language_key, 0) + 1
        language_positions[language_key] = next_position
        local_image_path: Path = name_builder.build_image_path(
            language=prepared.language,
            date_key=prepared.date_key,
            language_position=next_position,
            title=prepared.metadata.title,
            extension=prepared.thumbnail.extension,
        )
        local_image_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_image_path.exists():
            local_image_path.write_bytes(prepared.thumbnail.bytes_data)
            logger.info(
                "Row %d: shared preview saved to %s",
                prepared.row_number,
                local_image_path,
            )
        else:
            logger.info(
                "Row %d: shared preview reused at %s",
                prepared.row_number,
                local_image_path,
            )

        preview_target_folder_id: Optional[str] = preview_root_folder_id
        drive_folder_key: Tuple[str, str] = (prepared.date_key, prepared.language)
        if not dry_run and preview_root_folder_id:
            if drive_folder_key not in drive_folder_cache:
                try:
                    preview_path_parts: List[str] = build_drive_preview_path_segments(
                        template=config.google_drive_preview_path_template,
                        language=prepared.language,
                        date_key=prepared.date_key,
                    )
                    if preview_path_parts:
                        preview_target_folder_id = drive_client.ensure_folder_path(
                            parent_folder_id=preview_root_folder_id,
                            path_parts=preview_path_parts,
                        )
                except Exception as error:
                    logger.warning(
                        "Date %s language=%s: failed to ensure shared Drive preview path under %s. template=%r. Fallback to parent folder. reason=%s",
                        prepared.date_key,
                        prepared.language,
                        preview_root_folder_id,
                        config.google_drive_preview_path_template,
                        summarize_error(error),
                    )
                drive_folder_cache[drive_folder_key] = preview_target_folder_id
            preview_target_folder_id = drive_folder_cache[drive_folder_key]
            if preview_target_folder_id and local_image_path.exists():
                try:
                    drive_client.upload_image_and_make_public(
                        image_path=local_image_path,
                        folder_id=preview_target_folder_id,
                        mime_type=prepared.thumbnail.mime_type,
                    )
                    logger.info(
                        "Row %d: shared preview uploaded to Google Drive (%s)",
                        prepared.row_number,
                        local_image_path.name,
                    )
                except Exception as error:
                    logger.warning(
                        "Row %d: shared preview upload to Google Drive failed. reason=%s",
                        prepared.row_number,
                        summarize_error(error),
                    )
        materialized_videos.append(
            dataclasses.replace(
                prepared,
                local_thumbnail_path=local_image_path,
            )
        )
    return materialized_videos


def derive_planned_videos(
    *,
    logger: logging.Logger,
    prepared_videos: List[PreparedVideo],
    processing_mode: str,
    merge_semantics: str,
) -> List[PlannedVideo]:
    logger.info(
        "Planning stage: prepared_items=%d processing_mode=%s",
        len(prepared_videos),
        processing_mode,
    )
    processed: List[PlannedVideo] = []
    for prepared in prepared_videos:
        base_video: PlannedVideo = _build_base_planned_video(
            prepared=prepared,
            processing_mode=processing_mode,
            logger=logger,
        )
        if processing_mode == "nomerge":
            processed.append(base_video)
            continue
        if prepared.merge_languages and merge_semantics == "override":
            base_block_language: str = planned_video_block_language(base_video)
            if base_block_language in prepared.merge_languages:
                processed.append(base_video)
            for merge_language in prepared.merge_languages:
                if merge_language == base_block_language:
                    continue
                processed.append(
                    dataclasses.replace(
                        base_video,
                        forced_block_language=merge_language,
                    )
                )
            continue
        processed.append(base_video)
        base_block_language = planned_video_block_language(base_video)
        for merge_language in prepared.merge_languages:
            if merge_language == base_block_language:
                logger.warning(
                    "Row %d: merge token %r matches base block %r; skipping clone",
                    prepared.row_number,
                    merge_language,
                    base_block_language,
                )
                continue
            processed.append(
                dataclasses.replace(
                    base_video,
                    forced_block_language=merge_language,
                )
            )
    processed = deduplicate_planned_videos_within_slot_language(processed)
    processed_by_slot: Dict[str, List[PlannedVideo]] = {}
    for item in processed:
        processed_by_slot.setdefault(planned_video_slot_key(item), []).append(item)
    for slot_key in sorted(processed_by_slot.keys()):
        slot_items: List[PlannedVideo] = processed_by_slot[slot_key]
        date_key_for_slot: str = slot_items[0].date_key
        slot_lang_counts: Dict[str, int] = {
            language: sum(
                1
                for slot_item in slot_items
                if planned_video_block_language(slot_item) == language
            )
            for language in ("uk", "en", "ru", "other")
        }
        logger.info(
            "After dedup: mode=%s date=%s slot=%s counts_by_language=uk:%d en:%d ru:%d other:%d",
            processing_mode,
            date_key_for_slot,
            slot_key,
            slot_lang_counts["uk"],
            slot_lang_counts["en"],
            slot_lang_counts["ru"],
            slot_lang_counts["other"],
        )
    return processed


def _build_base_planned_video(
    *,
    prepared: PreparedVideo,
    processing_mode: str,
    logger: logging.Logger,
) -> PlannedVideo:
    base_video: PlannedVideo = PlannedVideo(
        row_number=prepared.row_number,
        original_link=prepared.original_link,
        normalized_link=prepared.normalized_link,
        scheduled_at_kiev=prepared.scheduled_at_kiev,
        date_key=prepared.date_key,
        date_display=prepared.date_display,
        language=prepared.language,
        metadata=prepared.metadata,
        thumbnail=prepared.thumbnail,
        local_thumbnail_path=prepared.local_thumbnail_path,
    )
    base_block_lang: str = planned_video_block_language(base_video)
    row_merge_languages: List[str] = (
        list(prepared.merge_languages) if processing_mode == "merge" else []
    )
    if processing_mode == "nomerge" and (
        bool(prepared.merge_languages) or bool(str(prepared.merge_raw or "").strip())
    ):
        logger.info(
            'Row %d: merge column ignored due to processing_mode=nomerge (merge_langs_raw="%s")',
            prepared.row_number,
            prepared.merge_raw,
        )
    row_characteristics: RowVideoCharacteristics = RowVideoCharacteristics(
        row_index=prepared.row_number,
        raw_link=prepared.original_link,
        normalized_link=prepared.normalized_link,
        date=prepared.date_raw,
        time=prepared.time_raw,
        detected_source_language=prepared.language,
        merge_languages=row_merge_languages,
        base_block_language=base_block_lang,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Row %d prepared metadata: normalized_link=%r detected_source_language=%s merge_languages=%s base_block_language=%s",
            prepared.row_number,
            row_characteristics.normalized_link,
            row_characteristics.detected_source_language,
            row_characteristics.merge_languages,
            row_characteristics.base_block_language,
        )
    return dataclasses.replace(
        base_video,
        row_characteristics=row_characteristics,
    )
