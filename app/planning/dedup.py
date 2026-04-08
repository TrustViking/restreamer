from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import PlannedVideo
from app.planning.merge_rules import planned_video_block_language
from app.planning.slots import planned_video_slot_key


LOGGER = _get_logger_impl(__name__)


def deduplicate_planned_videos_within_date_language(
    videos: List[PlannedVideo],
) -> List[PlannedVideo]:
    sorted_by_row: List[PlannedVideo] = sorted(
        videos,
        key=lambda item: (
            item.date_key,
            planned_video_block_language(item),
            item.row_number,
        ),
    )
    kept_by_key: Dict[Tuple[str, str, str], PlannedVideo] = {}
    deduped: List[PlannedVideo] = []
    for video in sorted_by_row:
        block_language: str = planned_video_block_language(video)
        normalized_url: str = str(video.normalized_link or "").strip()
        dedup_key: Tuple[str, str, str] = (
            video.date_key,
            block_language,
            normalized_url,
        )
        existing: Optional[PlannedVideo] = kept_by_key.get(dedup_key)
        if existing is None:
            kept_by_key[dedup_key] = video
            deduped.append(video)
            continue
        LOGGER.warning(
            "Duplicate link skipped: date=%s lang=%s url=%s kept_row=%d skipped_row=%d",
            video.date_key,
            block_language,
            normalized_url,
            existing.row_number,
            video.row_number,
        )
    return deduped


def deduplicate_planned_videos_within_slot_global(
    videos: List[PlannedVideo],
) -> List[PlannedVideo]:
    kept_rows_by_key: Dict[Tuple[str, str], int] = {}

    for item in videos:
        slot_key: str = planned_video_slot_key(item)
        url_key: str = item.normalized_link
        group_key: Tuple[str, str] = (slot_key, url_key)

        current_best_row: Optional[int] = kept_rows_by_key.get(group_key)
        if current_best_row is None:
            kept_rows_by_key[group_key] = item.row_number
            continue
        if item.row_number < current_best_row:
            kept_rows_by_key[group_key] = item.row_number

    skipped_rows_logged: Set[Tuple[str, str, int, int]] = set()
    filtered: List[PlannedVideo] = []
    for item in videos:
        slot_key = planned_video_slot_key(item)
        url_key = item.normalized_link
        group_key = (slot_key, url_key)
        kept_row: int = kept_rows_by_key[group_key]
        if item.row_number == kept_row:
            filtered.append(item)
            continue

        log_key: Tuple[str, str, int, int] = (
            slot_key,
            url_key,
            kept_row,
            item.row_number,
        )
        if log_key not in skipped_rows_logged:
            skipped_rows_logged.add(log_key)
            LOGGER.info(
                "Duplicate URL skipped in slot: slot=%s url=%s kept_row=%d skipped_row=%d reason=%s",
                slot_key,
                url_key,
                kept_row,
                item.row_number,
                "kept_earliest_row",
            )
    return filtered


def deduplicate_planned_videos_within_slot_language(
    videos: List[PlannedVideo],
) -> List[PlannedVideo]:
    sorted_by_row: List[PlannedVideo] = sorted(
        videos,
        key=lambda item: (
            planned_video_slot_key(item),
            planned_video_block_language(item),
            item.row_number,
        ),
    )
    kept_by_key: Dict[Tuple[str, str, str], PlannedVideo] = {}
    deduped: List[PlannedVideo] = []
    for video in sorted_by_row:
        block_language: str = planned_video_block_language(video)
        normalized_url: str = str(video.normalized_link or "").strip()
        slot_key: str = planned_video_slot_key(video)
        dedup_key: Tuple[str, str, str] = (
            slot_key,
            block_language,
            normalized_url,
        )
        existing: Optional[PlannedVideo] = kept_by_key.get(dedup_key)
        if existing is None:
            kept_by_key[dedup_key] = video
            deduped.append(video)
            continue
        LOGGER.warning(
            "Duplicate URL skipped in group: slot=%s lang=%s url=%s kept_row=%d skipped_row=%d",
            slot_key,
            block_language,
            normalized_url,
            existing.row_number,
            video.row_number,
        )
    return deduped
